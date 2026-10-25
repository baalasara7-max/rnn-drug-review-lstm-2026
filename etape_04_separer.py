"""Separer les donnees nettoyees en entrainement et validation."""

from pathlib import Path
import pandas as pd


def main():
    dossier = Path(__file__).resolve().parent / "data"
    source = dossier / "train_nettoye.tsv"
    if not source.is_file():
        raise SystemExit("Execute d'abord etape_03_nettoyer.py.")
    donnees = pd.read_csv(source, sep="\t")

    # La graine fixe rend la separation reproductible.
    graine = 42
    proportion_validation = 0.15
    morceaux_validation = []

    # Tirer 15 % des avis de CHAQUE categorie : separation stratifiee.
    for sentiment, groupe in donnees.groupby("sentiment"):
        if len(groupe) < 2:
            raise SystemExit(f"Pas assez d'exemples pour la categorie {sentiment}.")
        nombre = max(1, min(len(groupe) - 1, round(len(groupe) * proportion_validation)))
        selection = groupe.sample(n=nombre, random_state=graine)
        morceaux_validation.append(selection)

    validation = pd.concat(morceaux_validation)
    # Les avis non selectionnes pour la validation servent a l'entrainement.
    entrainement = donnees.drop(index=validation.index)

    # Melanger les lignes pour ne pas les laisser regroupees par categorie.
    entrainement = entrainement.sample(frac=1, random_state=graine).reset_index(drop=True)
    validation = validation.sample(frac=1, random_state=graine).reset_index(drop=True)

    # Verifier que les memes textes ne sont pas presents dans les deux parties.
    if set(entrainement["review"]) & set(validation["review"]):
        raise SystemExit("Erreur : des textes sont communs aux deux parties.")
    if len(entrainement) + len(validation) != len(donnees):
        raise SystemExit("Erreur : des lignes ont ete perdues.")

    entrainement.to_csv(dossier / "train.tsv", sep="\t", index=False)
    validation.to_csv(dossier / "validation.tsv", sep="\t", index=False)

    print("Total des avis nettoyes :", len(donnees))
    print(f"Entrainement : {len(entrainement)} avis ({len(entrainement) / len(donnees):.1%})")
    print(f"Validation   : {len(validation)} avis ({len(validation) / len(donnees):.1%})")
    for nom, partie in [("Entrainement", entrainement), ("Validation", validation)]:
        print(f"\n{nom} : repartition des categories")
        effectifs = partie["sentiment"].value_counts().reindex(["negatif", "neutre", "positif"])
        print(pd.DataFrame({"nombre": effectifs, "pourcentage": (100 * effectifs / len(partie)).round(2)}).to_string())
    print("\nAucun texte commun entre entrainement et validation.")
    print("Fichiers crees :", dossier / "train.tsv", "et", dossier / "validation.tsv")
    print("Le fichier de test n'a pas ete utilise. Aucun modele entraine.")


if __name__ == "__main__":
    main()
