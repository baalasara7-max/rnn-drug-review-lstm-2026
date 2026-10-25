"""Predire un avis avec un modele entraine, sans reutiliser sa note."""
from pathlib import Path
import argparse
import torch
from etape_03_nettoyer import nettoyer_texte
from etape_05_vocabulaire import encoder
from etape_09_variantes import ModeleLSTM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("texte", help="Avis en anglais, entre guillemets")
    parser.add_argument("--architecture", choices=["vanilla", "cifg", "wmc"], default="cifg")
    args = parser.parse_args()
    torch.set_num_threads(4)
    chemin = Path(__file__).resolve().parent / "resultats" / f"meilleur_{args.architecture}.pt"
    if not chemin.exists():
        raise SystemExit("Execute d'abord etape_10_experiences.py.")
    # Charger seulement un checkpoint de confiance produit par ce projet.
    sauvegarde = torch.load(chemin, map_location="cpu", weights_only=False)
    configuration = sauvegarde["configuration"]
    modele = ModeleLSTM(len(configuration["vocabulaire"]), architecture=args.architecture)
    modele.load_state_dict(sauvegarde["model_state_dict"])
    texte = nettoyer_texte(args.texte)
    ids, longueur = encoder(texte, configuration["vocabulaire"], configuration["longueur_max"])
    if longueur == 0:
        raise SystemExit("Le texte doit contenir au moins un token.")
    modele.eval()
    with torch.no_grad():
        scores = modele(torch.tensor([ids]), torch.tensor([longueur]))
        probabilites = torch.softmax(scores, dim=1)[0]
    for nom, indice in configuration["categories"].items():
        print(f"{nom} : {probabilites[indice].item():.2%}")
    categorie = {v:k for k,v in configuration["categories"].items()}[int(probabilites.argmax())]
    print("Prediction :", categorie)
    print("Probabilites du modele, sans garantie de calibration. Texte limite aux premiers tokens configures.")


if __name__ == "__main__":
    main()
