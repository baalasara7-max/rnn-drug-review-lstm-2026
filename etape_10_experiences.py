"""Neuf experiences reproductibles, selection sur validation, puis test final.

Les sous-ensembles reduisent le cout CPU. Ne jamais choisir les reglages sur le test.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from etape_03_nettoyer import nettoyer_texte, categorie
from etape_08_entrainer import preparer_dataset, evaluer
from etape_09_variantes import ModeleLSTM


def echantillon_stratifie(tableau, taille, graine=42):
    if taille >= len(tableau):
        return tableau.sample(frac=1, random_state=graine).reset_index(drop=True)
    effectifs = tableau["sentiment"].value_counts().sort_index()
    quotas = effectifs / len(tableau) * taille
    nombres = quotas.astype(int)
    reste = taille - nombres.sum()
    for nom in (quotas - nombres).sort_values(ascending=False).index[:reste]:
        nombres[nom] += 1
    morceaux = [tableau.loc[tableau.sentiment.eq(nom)].sample(n=int(n), random_state=graine)
                for nom, n in nombres.items()]
    return pd.concat(morceaux).sample(frac=1, random_state=graine).reset_index(drop=True)


def nettoyer_test(chemin, textes_interdits):
    brut = pd.read_csv(chemin, sep="\t")
    donnees = brut[["review", "rating"]].copy()
    donnees["rating"] = pd.to_numeric(donnees["rating"], errors="coerce")
    valide = donnees.review.notna() & donnees.rating.between(1, 10) & donnees.rating.mod(1).eq(0)
    donnees = donnees.loc[valide].copy()
    donnees["review"] = donnees.review.apply(nettoyer_texte)
    donnees = donnees.loc[donnees.review.ne("")].copy()
    donnees["sentiment"] = donnees.rating.apply(categorie)
    conflit = donnees.groupby("review").sentiment.transform("nunique").gt(1)
    conflits = int(conflit.sum())
    donnees = donnees.loc[~conflit].drop_duplicates("review")
    avant = len(donnees)
    donnees = donnees.loc[~donnees.review.isin(textes_interdits)].copy()
    return donnees, {"raw": len(brut), "conflicting_rows": conflits,
                     "overlapping_unique_texts_removed": avant - len(donnees), "eligible": len(donnees)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=4000)
    args = parser.parse_args()
    if min(vars(args).values()) < 1:
        parser.error("Toutes les tailles et epochs doivent etre positives.")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dossier = Path(__file__).resolve().parent
    data = dossier / "data"
    sortie = dossier / "resultats"
    sortie.mkdir(exist_ok=True)
    configuration = json.loads((data / "vocabulaire.json").read_text(encoding="utf-8"))
    train_pool = pd.read_csv(data / "train.tsv", sep="\t")
    val_pool = pd.read_csv(data / "validation.tsv", sep="\t")
    train = echantillon_stratifie(train_pool, args.train_size)
    validation = echantillon_stratifie(val_pool, args.validation_size)
    for nom, tableau in [("train_experience", train), ("validation_experience", validation)]:
        tableau.to_csv(sortie / f"{nom}.tsv", sep="\t", index=False)
    ds_train = preparer_dataset(train, configuration)
    ds_val = preparer_dataset(validation, configuration)
    val_loader = DataLoader(ds_val, batch_size=128)
    effectifs = torch.bincount(ds_train.tensors[2], minlength=3).float()
    poids = len(train) / (3 * effectifs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configurations = [(0.001, 128), (0.003, 128), (0.001, 64)]
    registre = []
    selections = {}
    print(f"Experiences : train={len(train)}, validation={len(validation)}, epochs={args.epochs}, device={device}", flush=True)
    for architecture in ("vanilla", "cifg", "wmc"):
        meilleur_arch = -1
        for lr, batch in configurations:
            identifiant = f"{architecture}_lr{lr}_batch{batch}"
            historique_path = sortie / f"{identifiant}.json"
            checkpoint_path = sortie / f"{identifiant}.pt"
            # Reprendre uniquement une experience achevee avec les memes reglages.
            if historique_path.exists() and checkpoint_path.exists():
                ancien = json.loads(historique_path.read_text())
                if ancien.get("arguments") == vars(args) and ancien.get("complete"):
                    registre.append(ancien)
                    if ancien["best_validation_macro_f1"] > meilleur_arch:
                        meilleur_arch = ancien["best_validation_macro_f1"]
                        selections[architecture] = ancien
                    print("Experience deja terminee :", identifiant, flush=True)
                    continue
            torch.manual_seed(42)
            modele = ModeleLSTM(len(configuration["vocabulaire"]), architecture=architecture).to(device)
            loader = DataLoader(ds_train, batch_size=batch, shuffle=True,
                                generator=torch.Generator().manual_seed(42))
            optimiseur = torch.optim.Adam(modele.parameters(), lr=lr)
            perte = nn.CrossEntropyLoss(weight=poids.to(device))
            meilleur = -1
            bilan = {"id": identifiant, "architecture": architecture, "learning_rate": lr,
                     "batch_size": batch, "parameters": sum(p.numel() for p in modele.parameters()),
                     "arguments": vars(args), "epochs": [], "complete": False}
            print("DEBUT", identifiant, flush=True)
            for epoque in range(1, args.epochs + 1):
                debut = time.perf_counter()
                modele.train()
                total = 0
                for ids, longueurs, etiquettes in loader:
                    optimiseur.zero_grad()
                    scores = modele(ids.to(device), longueurs.to(device))
                    erreur = perte(scores, etiquettes.to(device))
                    erreur.backward()
                    nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
                    optimiseur.step()
                    total += erreur.item()
                acc, f1, cm = evaluer(modele, val_loader, device)
                epoch_result = {"epoch": epoque, "mean_batch_loss": total / len(loader),
                                "validation_accuracy": acc, "validation_macro_f1": f1,
                                "seconds": time.perf_counter() - debut}
                bilan["epochs"].append(epoch_result)
                if f1 > meilleur:
                    meilleur = f1
                    bilan["best_epoch"] = epoque
                    bilan["best_validation_macro_f1"] = f1
                    torch.save({"model_state_dict": modele.state_dict(), "configuration": configuration,
                                "architecture": architecture, "epoch": epoque,
                                "validation_macro_f1": f1}, checkpoint_path)
                historique_path.write_text(json.dumps(bilan, indent=2), encoding="utf-8")
                print(f"{identifiant} epoque {epoque}: val F1={f1:.4f}, acc={acc:.4f}, {epoch_result['seconds']:.1f}s", flush=True)
            bilan["complete"] = True
            historique_path.write_text(json.dumps(bilan, indent=2), encoding="utf-8")
            registre.append(bilan)
            if meilleur > meilleur_arch:
                meilleur_arch = meilleur
                selections[architecture] = bilan

    # Seulement APRES selection de toutes les architectures, preparer le test.
    interdits = set(train_pool.review) | set(val_pool.review)
    test_pool, audit_test = nettoyer_test(data / "drugsComTest_raw.tsv", interdits)
    test = echantillon_stratifie(test_pool, args.test_size)
    assert not (set(train.review) & set(validation.review))
    assert not (set(test.review) & interdits)
    test.to_csv(sortie / "test_experience.tsv", sep="\t", index=False)
    test_loader = DataLoader(preparer_dataset(test, configuration), batch_size=128)
    finaux = []
    for architecture, choix in selections.items():
        checkpoint = torch.load(sortie / f"{choix['id']}.pt", map_location=device, weights_only=False)
        modele = ModeleLSTM(len(configuration["vocabulaire"]), architecture=architecture).to(device)
        modele.load_state_dict(checkpoint["model_state_dict"])
        accuracy, f1, cm = evaluer(modele, test_loader, device)
        from shutil import copyfile
        copyfile(sortie / f"{choix['id']}.pt", sortie / f"meilleur_{architecture}.pt")
        predictions = []
        with torch.no_grad():
            for ids, longueurs, _ in test_loader:
                predictions.extend(modele(ids.to(device), longueurs.to(device)).argmax(1).cpu().tolist())
        pd.DataFrame({"actual": test.sentiment.map(configuration["categories"]), "predicted": predictions}).to_csv(
            sortie / f"predictions_{architecture}.csv", index=False)
        finaux.append({"architecture": architecture, "learning_rate": choix["learning_rate"],
                       "batch_size": choix["batch_size"], "parameters": choix["parameters"],
                       "best_epoch": choix["best_epoch"], "validation_macro_f1": choix["best_validation_macro_f1"],
                       "test_accuracy": accuracy, "test_macro_f1": f1, "confusion_matrix": cm})
        print("TEST", architecture, accuracy, f1, flush=True)
    majoritaire = train.sentiment.value_counts().idxmax()
    proportion = float(test.sentiment.eq(majoritaire).mean())
    baseline = {"class": majoritaire, "test_accuracy": proportion,
                "test_macro_f1": (2 * proportion / (1 + proportion)) / 3}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [data / "drugsComTrain_raw.tsv", data / "drugsComTest_raw.tsv", data / "vocabulaire.json"]}
    synthese = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                                "device": str(device), "platform": platform.platform()},
                "seed": 42, "arguments": vars(args), "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
                "class_counts": {nom: t.sentiment.value_counts().to_dict() for nom, t in [("train", train), ("validation", validation), ("test", test)]},
                "test_audit": audit_test, "sha256": hashes, "experiments": registre,
                "final_results": finaux, "majority_baseline": baseline}
    (sortie / "synthese.json").write_text(json.dumps(synthese, indent=2), encoding="utf-8")
    pd.DataFrame([{k:v for k,v in r.items() if k != "confusion_matrix"} for r in finaux]).to_csv(sortie / "comparaison.csv", index=False)
    print("TERMINE : resultats/synthese.json", flush=True)


if __name__ == "__main__":
    main()
