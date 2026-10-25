from pathlib import Path
import json
import torch
from torch import nn
from etape_05_vocabulaire import encoder


class CelluleVanillaLSTM(nn.Module):
    """Traiter UN token : (x_t, h_precedent, c_precedent) -> (h_t, c_t)."""

    def __init__(self, dimension_entree=64, dimension_cachee=64):
        super().__init__()
        self.dimension_cachee = dimension_cachee

        # Chaque paire calcule Wx*x + Wh*h + b.
        # Le biais est dans la transformation de x uniquement : un seul b par porte.
        self.x_i = nn.Linear(dimension_entree, dimension_cachee)
        self.h_i = nn.Linear(dimension_cachee, dimension_cachee, bias=False)
        self.x_f = nn.Linear(dimension_entree, dimension_cachee)
        self.h_f = nn.Linear(dimension_cachee, dimension_cachee, bias=False)
        self.x_o = nn.Linear(dimension_entree, dimension_cachee)
        self.h_o = nn.Linear(dimension_cachee, dimension_cachee, bias=False)
        self.x_g = nn.Linear(dimension_entree, dimension_cachee)
        self.h_g = nn.Linear(dimension_cachee, dimension_cachee, bias=False)

    def forward(self, x_t, h_precedent, c_precedent):
        # 1. OUBLI : quelle part de l'ancienne memoire conserver ?
        f_t = torch.sigmoid(self.x_f(x_t) + self.h_f(h_precedent))

        # 2. ENTREE : quelle nouvelle information ajouter ?
        i_t = torch.sigmoid(self.x_i(x_t) + self.h_i(h_precedent))
        g_t = torch.tanh(self.x_g(x_t) + self.h_g(h_precedent))
        # * multiplie les vecteurs element par element (symbole du sujet : odot).
        c_t = f_t * c_precedent + i_t * g_t

        # 3. SORTIE : quelle information transmettre ?
        o_t = torch.sigmoid(self.x_o(x_t) + self.h_o(h_precedent))
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class ModeleVanillaLSTM(nn.Module):
    """Lire un lot d'avis et produire un score par categorie pour chacun."""

    def __init__(self, taille_vocabulaire, dimension_embedding=64,
                 dimension_cachee=64, nombre_categories=3, padding_idx=0):
        super().__init__()
        # Cette couche sera entrainee avec le LSTM. La demonstration de l'etape 6
        # n'avait sauvegarde aucun poids : nous creons ici l'Embedding du modele.
        self.embedding = nn.Embedding(taille_vocabulaire, dimension_embedding,
                                      padding_idx=padding_idx)
        self.cellule = CelluleVanillaLSTM(dimension_embedding, dimension_cachee)
        self.classifieur = nn.Linear(dimension_cachee, nombre_categories)

    def forward(self, identifiants, longueurs):
        # identifiants : (nombre_avis, nombre_positions), entiers.
        # longueurs : nombre de tokens utiles par avis, apres troncature.
        if identifiants.ndim != 2 or longueurs.shape != (identifiants.shape[0],):
            raise ValueError("Entrees attendues : ids (lot, positions), longueurs (lot,).")
        longueurs = longueurs.to(identifiants.device)
        if bool(((longueurs < 1) | (longueurs > identifiants.shape[1])).any()):
            raise ValueError("Chaque longueur doit etre comprise entre 1 et le nombre de positions.")

        vecteurs = self.embedding(identifiants)
        nombre_avis, nombre_positions, _ = vecteurs.shape
        h = vecteurs.new_zeros(nombre_avis, self.cellule.dimension_cachee)
        c = vecteurs.new_zeros(nombre_avis, self.cellule.dimension_cachee)

        # La MEME cellule et les MEMES poids sont reutilises pour tous les tokens.
        for t in range(nombre_positions):
            x_t = vecteurs[:, t, :]
            nouveau_h, nouveau_c = self.cellule(x_t, h, c)

            # Pour un avis deja termine, garder h et c inchanges : ignorer PAD.
            actif = (t < longueurs).unsqueeze(1)
            h = torch.where(actif, nouveau_h, h)
            c = torch.where(actif, nouveau_c, c)

        # h est l'etat du dernier token UTILE de chaque avis.
        # Les logits sont des scores bruts, pas encore des probabilites.
        return self.classifieur(h)


def main():
    source = Path(__file__).resolve().parent / "data/vocabulaire.json"
    if not source.is_file():
        raise SystemExit("Execute d'abord etape_05_vocabulaire.py.")
    configuration = json.loads(source.read_text(encoding="utf-8"))
    vocabulaire = configuration["vocabulaire"]
    torch.manual_seed(42)
    modele = ModeleVanillaLSTM(len(vocabulaire), padding_idx=vocabulaire["<PAD>"])

    texte = "this medicine helped me!"
    ids, longueur = encoder(texte, vocabulaire, configuration["longueur_max"])
    entrees = torch.tensor([ids], dtype=torch.long)
    longueurs = torch.tensor([longueur], dtype=torch.long)

    # eval() + no_grad() : demonstration sans calculer de gradients.
    modele.eval()
    with torch.no_grad():
        scores = modele(entrees, longueurs)
        probabilites = torch.softmax(scores, dim=1)

    print("Texte :", texte)
    print("Forme des entrees :", tuple(entrees.shape))
    print("Longueur utile :", longueur)
    print("Forme des scores :", tuple(scores.shape), "= (avis, categories)")
    print("\nScores et probabilites AVANT entrainement :")
    for nom, indice in sorted(configuration["categories"].items(), key=lambda item: item[1]):
        print(f"{nom:8s} : score={scores[0, indice].item():.4f}, probabilite={probabilites[0, indice].item():.2%}")
    print("\nNombre de parametres du modele :", sum(p.numel() for p in modele.parameters()))
    print("Poids initialises aleatoirement : ces sorties ne mesurent pas la qualite du modele.")
    print("Le modele est construit, mais aucun entrainement ni sauvegarde n'a ete effectue.")
    print("Les fichiers de validation et de test n'ont pas ete utilises.")


if __name__ == "__main__":
    main()
