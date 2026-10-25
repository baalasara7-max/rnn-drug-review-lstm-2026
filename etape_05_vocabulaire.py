"""Construire le vocabulaire sur train seulement et encoder des exemples."""

from pathlib import Path
from collections import Counter
import json
import re
import pandas as pd

TAILLE_MAX_VOCABULAIRE = 12000  # Inclut PAD et UNK.
LONGUEUR_MAX = 128  # Choix initial ajustable, pas une exigence du sujet.


def tokeniser(texte):
    # Garder les mots (avec apostrophes internes) et la ponctuation separee.
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", texte.lower())


def encoder(texte, vocabulaire, longueur_max=LONGUEUR_MAX):
    tokens = tokeniser(texte)
    identifiants = [vocabulaire.get(token, vocabulaire["<UNK>"]) for token in tokens[:longueur_max]]
    longueur = len(identifiants)
    # PAD complete les sequences courtes ; longueur permet d'ignorer ce remplissage.
    identifiants += [vocabulaire["<PAD>"]] * (longueur_max - longueur)
    return identifiants, longueur


def main():
    dossier = Path(__file__).resolve().parent / "data"
    source = dossier / "train.tsv"
    if not source.is_file():
        raise SystemExit("Execute d'abord etape_04_separer.py.")
    train = pd.read_csv(source, sep="\t")

    # La validation et le test ne participent PAS au choix des mots connus.
    compteur = Counter()
    longueurs = []
    for texte in train["review"]:
        tokens = tokeniser(texte)
        compteur.update(tokens)
        longueurs.append(len(tokens))

    vocabulaire = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in compteur.most_common(TAILLE_MAX_VOCABULAIRE - 2):
        vocabulaire[token] = len(vocabulaire)

    configuration = {
        "vocabulaire": vocabulaire,
        "longueur_max": LONGUEUR_MAX,
        "categories": {"negatif": 0, "neutre": 1, "positif": 2},
    }
    destination = dossier / "vocabulaire.json"
    destination.write_text(json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Avis utilises pour creer le vocabulaire :", len(train))
    print("Tokens distincts observes :", len(compteur))
    print("Taille du vocabulaire conserve :", len(vocabulaire))
    print("Longueur maximale choisie :", LONGUEUR_MAX)
    coupes = sum(longueur > LONGUEUR_MAX for longueur in longueurs)
    print(f"Avis d'entrainement qui seront tronques : {coupes} ({100 * coupes / len(train):.2f} %)")
    print("\nLes 10 tokens les plus frequents :")
    print(compteur.most_common(10))

    texte = "this medicine helped me!"
    ids, longueur = encoder(texte, vocabulaire)
    print("\nExemple :", texte)
    print("Tokens :", tokeniser(texte))
    print("Identifiants utiles :", ids[:longueur])
    print("Longueur utile :", longueur)
    print("Debut de la sequence completee :", ids[:15])
    print("PAD = 0 : remplissage ; UNK = 1 : token absent du vocabulaire.")
    print("Les identifiants des mots ne sont ni des notes ni des scores de sentiment.")
    print("\nFichier cree :", destination)
    print("Aucun modele entraine. Le test n'a pas ete utilise.")


if __name__ == "__main__":
    main()
