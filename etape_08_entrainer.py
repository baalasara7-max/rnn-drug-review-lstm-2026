"""Entrainer le Vanilla LSTM et choisir son checkpoint sur la validation.

--verification execute seulement un petit essai technique, pas une experience finale.
Le fichier de test n'est jamais utilise ici.
"""

from pathlib import Path
import argparse
import json
import time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from etape_05_vocabulaire import encoder
from etape_07_modele_vanilla import ModeleVanillaLSTM


def preparer_dataset(tableau, configuration):
    """Un exemple contient les identifiants, la longueur utile et la bonne categorie."""
    encodages = [encoder(texte, configuration["vocabulaire"], configuration["longueur_max"])
                 for texte in tableau["review"]]
    ids = torch.tensor([resultat[0] for resultat in encodages], dtype=torch.long)
    longueurs = torch.tensor([resultat[1] for resultat in encodages], dtype=torch.long)
    etiquettes = torch.tensor(tableau["sentiment"].map(configuration["categories"]).tolist(), dtype=torch.long)
    return TensorDataset(ids, longueurs, etiquettes)


def evaluer(modele, chargeur, device):
    """Mesurer sans apprendre : lignes = vraies categories, colonnes = predictions."""
    modele.eval()
    confusion = torch.zeros(3, 3, dtype=torch.long)
    with torch.no_grad():
        for ids, longueurs, etiquettes in chargeur:
            scores = modele(ids.to(device), longueurs.to(device))
            predictions = scores.argmax(dim=1).cpu()
            indices = 3 * etiquettes + predictions
            confusion += torch.bincount(indices, minlength=9).reshape(3, 3)
    vrais_positifs = confusion.diag().float()
    # F1 de chaque categorie = 2 TP / (nombre reel + nombre predit).
    denominateur = confusion.sum(dim=0) + confusion.sum(dim=1)
    f1 = 2 * vrais_positifs / denominateur.clamp_min(1)
    accuracy = (vrais_positifs.sum() / confusion.sum().clamp_min(1)).item()
    return accuracy, f1.mean().item(), confusion.tolist()


def main():
    parser = argparse.ArgumentParser(description="Entrainer notre Vanilla LSTM.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--verification", action="store_true", help="Petit essai technique : 96 avis train, 48 validation, une epoque.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size et lr doivent etre strictement positifs.")

    torch.manual_seed(42)
    # Limiter les threads evite le surcout CPU pour les petites matrices recurrentes.
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dossier = Path(__file__).resolve().parent
    configuration = json.loads((dossier / "data/vocabulaire.json").read_text(encoding="utf-8"))
    train = pd.read_csv(dossier / "data/train.tsv", sep="\t")
    validation = pd.read_csv(dossier / "data/validation.tsv", sep="\t")
    if args.verification:
        # Echantillons par categorie afin de verifier les trois classes.
        train = pd.concat([g.sample(n=min(32, len(g)), random_state=42)
                           for _, g in train.groupby("sentiment")])
        validation = pd.concat([g.sample(n=min(16, len(g)), random_state=42)
                                for _, g in validation.groupby("sentiment")])
    epochs = 1 if args.verification else args.epochs
    destination = dossier / "entrainements" / ("verification" if args.verification else "vanilla")
    destination.mkdir(parents=True, exist_ok=True)
    print("Mode :", "VERIFICATION TECHNIQUE, pas de resultats finaux" if args.verification else "ENTRAINEMENT", flush=True)
    print("Materiel :", device, "| Train :", len(train), "| Validation :", len(validation), flush=True)
    print("Preparation des entrees numeriques...", flush=True)
    dataset_train = preparer_dataset(train, configuration)
    dataset_validation = preparer_dataset(validation, configuration)
    generateur = torch.Generator().manual_seed(42)
    chargeur_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, generator=generateur)
    chargeur_validation = DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=False)

    modele = ModeleVanillaLSTM(len(configuration["vocabulaire"])).to(device)
    # Donner plus de poids aux classes rares ; calcul sur l'entrainement seulement.
    effectifs = torch.bincount(dataset_train.tensors[2], minlength=3).float()
    if bool((effectifs == 0).any()):
        raise SystemExit("Chaque categorie doit etre presente dans l'entrainement.")
    poids = (len(train) / (3 * effectifs)).to(device)
    fonction_erreur = nn.CrossEntropyLoss(weight=poids)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=args.lr)
    print("Poids des categories (negatif, neutre, positif) :", poids.tolist(), flush=True)
    print("Parametres :", sum(p.numel() for p in modele.parameters()), flush=True)

    historique = []
    meilleur_f1 = -1.0
    for epoque in range(1, epochs + 1):
        debut = time.perf_counter()
        modele.train()
        somme_erreurs = 0.0
        for numero, (ids, longueurs, etiquettes) in enumerate(chargeur_train, start=1):
            ids, longueurs, etiquettes = ids.to(device), longueurs.to(device), etiquettes.to(device)
            # 1. Effacer les gradients de l'iteration precedente.
            optimiseur.zero_grad()
            # 2. Predire puis mesurer l'erreur face aux bonnes categories.
            scores = modele(ids, longueurs)
            erreur = fonction_erreur(scores, etiquettes)
            # 3. Calculer les gradients, puis mettre a jour les parametres.
            erreur.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), max_norm=1.0)
            optimiseur.step()
            somme_erreurs += erreur.item()
            if numero == 1 or numero % 100 == 0:
                print(f"Epoque {epoque}/{epochs} | lot {numero}/{len(chargeur_train)} | erreur {erreur.item():.4f}", flush=True)

        accuracy, macro_f1, confusion = evaluer(modele, chargeur_validation, device)
        bilan = {"epoque": epoque, "erreur_moyenne_par_lot": somme_erreurs / len(chargeur_train),
                 "validation_accuracy": accuracy, "validation_macro_f1": macro_f1,
                 "matrice_confusion": confusion, "secondes": time.perf_counter() - debut}
        historique.append(bilan)
        print(f"Validation : accuracy={accuracy:.2%}, macro-F1={macro_f1:.4f} | {bilan['secondes']:.1f} s", flush=True)
        if macro_f1 > meilleur_f1:
            meilleur_f1 = macro_f1
            torch.save({"model_state_dict": modele.state_dict(), "configuration": configuration,
                        "dimension_embedding": 64, "dimension_cachee": 64,
                        "epoque": epoque, "validation_macro_f1": macro_f1,
                        "verification": args.verification, "reglages": vars(args)}, destination / "meilleur_modele.pt")
            print("Meilleure version sauvegardee.", flush=True)
        (destination / "historique.json").write_text(json.dumps(historique, indent=2), encoding="utf-8")

    print("\nMeilleur macro-F1 de validation :", round(meilleur_f1, 4))
    print("Dossier des resultats :", destination)
    if args.verification:
        print("ATTENTION : petit essai technique, a ne pas presenter comme performance finale.")
    print("Aucun resultat de test : le test reste reserve a l'evaluation finale.")


if __name__ == "__main__":
    main()
