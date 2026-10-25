"""CIFG et WMC : memes entrees/sorties que le Vanilla et cellules manuelles."""
import torch
from torch import nn
from etape_07_modele_vanilla import CelluleVanillaLSTM, ModeleVanillaLSTM


class CelluleCIFG(CelluleVanillaLSTM):
    def __init__(self, dimension_entree=64, dimension_cachee=64):
        super().__init__(dimension_entree, dimension_cachee)
        # La porte d'entree n'a plus ses propres parametres.
        del self.x_i
        del self.h_i

    def forward(self, x_t, h_precedent, c_precedent):
        f_t = torch.sigmoid(self.x_f(x_t) + self.h_f(h_precedent))
        i_t = 1 - f_t
        g_t = torch.tanh(self.x_g(x_t) + self.h_g(h_precedent))
        c_t = f_t * c_precedent + i_t * g_t
        o_t = torch.sigmoid(self.x_o(x_t) + self.h_o(h_precedent))
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class CelluleWMC(CelluleVanillaLSTM):
    def __init__(self, dimension_entree=64, dimension_cachee=64):
        super().__init__(dimension_entree, dimension_cachee)
        self.c_i = nn.Linear(dimension_cachee, dimension_cachee, bias=False)
        self.c_f = nn.Linear(dimension_cachee, dimension_cachee, bias=False)
        self.c_o = nn.Linear(dimension_cachee, dimension_cachee, bias=False)

    def forward(self, x_t, h_precedent, c_precedent):
        i_t = torch.sigmoid(self.x_i(x_t) + self.h_i(h_precedent) + torch.tanh(self.c_i(c_precedent)))
        f_t = torch.sigmoid(self.x_f(x_t) + self.h_f(h_precedent) + torch.tanh(self.c_f(c_precedent)))
        g_t = torch.tanh(self.x_g(x_t) + self.h_g(h_precedent))
        c_t = f_t * c_precedent + i_t * g_t
        # L'annexe utilise la NOUVELLE memoire pour la porte de sortie WMC.
        o_t = torch.sigmoid(self.x_o(x_t) + self.h_o(h_precedent) + torch.tanh(self.c_o(c_t)))
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class ModeleLSTM(ModeleVanillaLSTM):
    def __init__(self, taille_vocabulaire, architecture="vanilla", dimension_embedding=64,
                 dimension_cachee=64, nombre_categories=3, padding_idx=0):
        super().__init__(taille_vocabulaire, dimension_embedding, dimension_cachee,
                         nombre_categories, padding_idx)
        cellules = {"vanilla": CelluleVanillaLSTM, "cifg": CelluleCIFG, "wmc": CelluleWMC}
        if architecture not in cellules:
            raise ValueError(f"Architecture inconnue : {architecture}")
        if architecture != "vanilla":
            self.cellule = cellules[architecture](dimension_embedding, dimension_cachee)
        self.architecture = architecture


if __name__ == "__main__":
    for nom in ("vanilla", "cifg", "wmc"):
        modele = ModeleLSTM(12000, architecture=nom)
        print(nom, sum(p.numel() for p in modele.parameters()), "parametres")
