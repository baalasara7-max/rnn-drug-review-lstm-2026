"""Observer la transformation identifiants -> vecteurs, sans entrainement."""

from pathlib import Path
import json
import torch
from torch import nn
from etape_05_vocabulaire import encoder, tokeniser


def main():
    source = Path(__file__).resolve().parent / "data/vocabulaire.json"
    if not source.is_file():
        raise SystemExit("Execute d'abord etape_05_vocabulaire.py.")
    configuration = json.loads(source.read_text(encoding="utf-8"))
    vocabulaire = configuration["vocabulaire"]
    longueur_max = configuration["longueur_max"]

    # Fixer la graine rend cette demonstration reproductible.
    torch.manual_seed(42)
    dimension_embedding = 64

    # Une ligne de 64 nombres par token du vocabulaire.
    # padding_idx=0 reserve une ligne nulle pour PAD et empeche sa mise a jour
    # par les gradients de cette couche.
    embedding = nn.Embedding(
        num_embeddings=len(vocabulaire),
        embedding_dim=dimension_embedding,
        padding_idx=vocabulaire["<PAD>"],
    )

    texte = "this medicine helped me!"
    ids, longueur = encoder(texte, vocabulaire, longueur_max)
    # long = entiers, requis pour les indices de nn.Embedding.
    # unsqueeze ajoute une dimension pour le lot : ici un seul avis.
    entrees = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    vecteurs = embedding(entrees)

    print("Texte :", texte)
    print("Tokens :", tokeniser(texte))
    print("Identifiants utiles :", ids[:longueur])
    print("Longueur utile :", longueur)
    print("\nTaille de la table Embedding :", tuple(embedding.weight.shape))
    print("Forme des entrees :", tuple(entrees.shape), "= (avis, positions)")
    print("Forme des sorties :", tuple(vecteurs.shape), "= (avis, positions, nombres par vecteur)")
    print("\nLes 8 premiers nombres du vecteur de chaque token utile :")
    for position, token in enumerate(tokeniser(texte)[:longueur]):
        print(token, "->", vecteurs[0, position, :8].detach().tolist())

    print("\nVecteur PAD entierement nul :", bool(torch.all(embedding.weight[0] == 0)))
    print("Nombre de parametres de la table :", embedding.weight.numel())
    print("\nCes vecteurs sont initialises aleatoirement : aucun sens n'a encore ete appris.")
    print("Ils seront ajustes pendant l'entrainement du futur modele.")
    print("Le LSTM devra aussi ignorer les positions PAD grace aux longueurs utiles.")
    print("Cette demonstration ne sauvegarde ni n'entraine de modele.")


if __name__ == "__main__":
    main()
