"""Creer et examiner les categories sur le fichier d'entrainement uniquement."""

from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Explorer les categories de sentiment.")
    parser.add_argument(
        "fichier", nargs="?",
        default=str(Path(__file__).parent / "data" / "drugsComTrain_raw.tsv"),
    )
    args = parser.parse_args()
    chemin = Path(args.fichier)
    df = pd.read_csv(chemin, sep="\t")
    if not {"review", "rating"}.issubset(df.columns):
        raise SystemExit("Le fichier doit contenir review et rating.")

    # Seuls le texte et la note sont utiles pour notre tache.
    donnees = df[["review", "rating"]].copy()
    donnees["rating"] = pd.to_numeric(donnees["rating"], errors="coerce")

    # Ecarter les textes absents/vides et les notes non entieres ou hors de 1 a 10.
    texte_valide = donnees["review"].fillna("").str.strip().ne("")
    note_valide = donnees["rating"].between(1, 10) & donnees["rating"].mod(1).eq(0)
    donnees = donnees.loc[texte_valide & note_valide].copy()
    print("Avis charges :", len(df))
    print("Avis exclus (texte ou note invalide) :", len(df) - len(donnees))

    # Ces seuils sont notre choix de definition du sentiment.
    def categorie(note):
        if note <= 4:
            return "negatif"
        if note <= 6:
            return "neutre"
        return "positif"

    donnees["sentiment"] = donnees["rating"].apply(categorie)

    print("\nNombre d'avis par note :")
    print(donnees["rating"].value_counts().sort_index().to_string())

    # reindex impose un ordre lisible plutot qu'un tri par effectif.
    effectifs = donnees["sentiment"].value_counts().reindex(
        ["negatif", "neutre", "positif"], fill_value=0
    )
    resume = pd.DataFrame({
        "nombre": effectifs,
        "pourcentage": (effectifs / len(donnees) * 100).round(2) if len(donnees) else 0,
    })
    print("\nRepartition des categories :")
    print(resume.to_string())

    # Afficher un exemple court par categorie, sans modifier les textes sources.
    print("\nUn exemple par categorie (texte raccourci pour l'affichage) :")
    exemples = donnees.groupby("sentiment", sort=False).head(1).copy()
    exemples["review"] = exemples["review"].str.replace(r"\s+", " ", regex=True).str.slice(0, 180)
    print(exemples.to_string(index=False))
    print("\nAucun modele entraine. Aucun fichier source modifie.")


if __name__ == "__main__":
    main()
