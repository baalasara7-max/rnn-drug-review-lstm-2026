"""Nettoyer les avis d'entrainement et sauvegarder leurs categories."""

from pathlib import Path
import argparse
import html
import re
import pandas as pd


def nettoyer_texte(texte):
    # Decoder les entites HTML : I&#039;m devient I'm ; &amp; devient &.
    texte = html.unescape(texte)
    # Uniformiser les majuscules/minuscules.
    texte = texte.lower()
    # Remplacer les retours a la ligne et espaces multiples par un seul espace.
    texte = re.sub(r"\s+", " ", texte).strip()
    # Conserver la ponctuation : elle peut aider a comprendre le sentiment.
    return texte


def categorie(note):
    if note <= 4:
        return "negatif"
    if note <= 6:
        return "neutre"
    return "positif"


def main():
    dossier = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Nettoyer le fichier d'entrainement.")
    parser.add_argument("fichier", nargs="?", default=str(dossier / "data/drugsComTrain_raw.tsv"))
    args = parser.parse_args()
    df = pd.read_csv(args.fichier, sep="\t")
    if not {"review", "rating"}.issubset(df.columns):
        raise SystemExit("Les colonnes review et rating sont requises.")
    print("Avis charges :", len(df))
    donnees = df[["review", "rating"]].copy()
    donnees["rating"] = pd.to_numeric(donnees["rating"], errors="coerce")
    valide = donnees["review"].notna() & donnees["rating"].between(1, 10) & donnees["rating"].mod(1).eq(0)
    donnees = donnees.loc[valide].copy()
    print("Avis exclus pour texte absent ou note invalide :", len(df) - len(donnees))

    avant = donnees["review"].iloc[0] if len(donnees) else ""
    donnees["review"] = donnees["review"].apply(nettoyer_texte)
    apres = donnees["review"].iloc[0] if len(donnees) else ""
    vides = donnees["review"].eq("")
    print("Textes vides apres nettoyage :", int(vides.sum()))
    donnees = donnees.loc[~vides].copy()
    donnees["sentiment"] = donnees["rating"].apply(categorie)

    # Un meme texte associe a plusieurs categories est ambigu : on l'ecarte.
    # Ce choix evite de conserver arbitrairement l'une de ses categories.
    nombres_categories = donnees.groupby("review")["sentiment"].transform("nunique")
    conflits = nombres_categories.gt(1)
    print("Lignes avec texte identique mais categories contradictoires :", int(conflits.sum()))
    donnees = donnees.loc[~conflits].copy()
    avant_dedup = len(donnees)
    donnees = donnees.drop_duplicates(subset="review").reset_index(drop=True)
    print("Doublons supplementaires retires :", avant_dedup - len(donnees))
    print("Avis conserves :", len(donnees))

    print("\nExemple avant nettoyage :\n", avant[:250])
    print("\nExemple apres nettoyage :\n", apres[:250])
    print("\nExemple de decodage HTML :")
    print("I&#039;m happy &amp; relieved ->", nettoyer_texte("I&#039;m happy &amp; relieved"))
    print("\nCategories apres nettoyage :")
    print(donnees["sentiment"].value_counts().to_string())

    destination = dossier / "data/train_nettoye.tsv"
    donnees.to_csv(destination, sep="\t", index=False)
    print("\nFichier cree :", destination)
    print("Les fichiers originaux restent inchanges. Aucun modele entraine.")


if __name__ == "__main__":
    main()
