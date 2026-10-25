"""Premiere etape : charger et observer les donnees, sans entrainer de modele."""

from pathlib import Path
import argparse
import pandas as pd


def main():
    # Un chemin peut etre fourni dans le terminal de VS Code.
    parser = argparse.ArgumentParser(description="Inspecter le fichier d'entrainement Drug Review.")
    parser.add_argument("fichier", nargs="?", help="Chemin du fichier CSV ou TSV")
    args = parser.parse_args()

    # Sans chemin, ouvrir une fenetre pour choisir le fichier sur le PC.
    if args.fichier:
        chemin = Path(args.fichier)
    else:
        import tkinter as tk
        from tkinter.filedialog import askopenfilename

        fenetre = tk.Tk()
        fenetre.withdraw()
        selection = askopenfilename(
            title="Choisir le fichier d'entrainement Drug Review",
            filetypes=[("Donnees CSV ou TSV", "*.csv *.tsv"), ("Tous les fichiers", "*.*")],
        )
        fenetre.destroy()
        if not selection:
            print("Aucun fichier selectionne.")
            return
        chemin = Path(selection)

    if not chemin.is_file():
        raise SystemExit(f"Fichier introuvable : {chemin}")
    if chemin.suffix.lower() not in {".csv", ".tsv"}:
        raise SystemExit("Choisis un CSV ou un TSV. Decompresse d'abord les fichiers ZIP.")

    # Un TSV separe les colonnes par des tabulations, un CSV par des virgules.
    separateur = "\t" if chemin.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(chemin, sep=separateur)

    # Chaque ligne est un avis ; chaque colonne contient une information.
    print("\nFichier :", chemin.name)
    print("Nombre de lignes :", df.shape[0])
    print("Nombre de colonnes :", df.shape[1])
    print("\nNoms des colonnes :")
    print(df.columns.tolist())
    print("\nLes cinq premieres lignes :")
    print(df.head().to_string(index=False))
    print("\nTypes des colonnes :")
    print(df.dtypes)
    print("\nValeurs manquantes par colonne :")
    print(df.isna().sum())

    # Observation seulement : on ne supprime ni ne transforme encore de lignes.
    if {"review", "rating"}.issubset(df.columns):
        print("\nExemples de textes et de notes :")
        print(df[["review", "rating"]].head().to_string(index=False))
    else:
        print("\nLes colonnes review/rating ne sont pas toutes presentes : verifier le fichier.")


if __name__ == "__main__":
    main()
