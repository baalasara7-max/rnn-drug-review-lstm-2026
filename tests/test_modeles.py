"""Tests des equations, gradients et invariance aux positions PAD."""
import unittest
import torch
from torch.nn import functional as F
from etape_09_variantes import CelluleCIFG, CelluleWMC, ModeleLSTM
from etape_07_modele_vanilla import CelluleVanillaLSTM
from etape_08_entrainer import evaluer


class TestsModeles(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_equations_annexe(self):
        # Reference par produit matriciel explicite, independante des appels Linear.
        for classe in (CelluleVanillaLSTM, CelluleCIFG, CelluleWMC):
            cell = classe(3, 2).double()
            x, h, c = torch.randn(2, 3).double(), torch.randn(2, 2).double(), torch.randn(2, 2).double()
            def affine(nom):
                a, b = getattr(cell, 'x_' + nom), getattr(cell, 'h_' + nom)
                return x @ a.weight.T + h @ b.weight.T + a.bias
            f = affine('f')
            if isinstance(cell, CelluleWMC):
                f = f + torch.tanh(c @ cell.c_f.weight.T)
            f = torch.sigmoid(f)
            if isinstance(cell, CelluleCIFG):
                i = 1 - f
            else:
                i = affine('i')
                if isinstance(cell, CelluleWMC):
                    i = i + torch.tanh(c @ cell.c_i.weight.T)
                i = torch.sigmoid(i)
            attendu_c = f * c + i * torch.tanh(affine('g'))
            o = affine('o')
            if isinstance(cell, CelluleWMC):
                o = o + torch.tanh(attendu_c @ cell.c_o.weight.T)
            attendu_h = torch.sigmoid(o) * torch.tanh(attendu_c)
            obtenu_h, obtenu_c = cell(x, h, c)
            torch.testing.assert_close(obtenu_h, attendu_h)
            torch.testing.assert_close(obtenu_c, attendu_c)

    def test_gradients_numeriques(self):
        for classe in (CelluleVanillaLSTM, CelluleCIFG, CelluleWMC):
            cell = classe(2, 2).double()
            entrees = tuple(torch.randn(1, 2, dtype=torch.double, requires_grad=True) for _ in range(3))
            self.assertTrue(torch.autograd.gradcheck(cell, entrees))

    def test_padding_et_apprentissage(self):
        for architecture in ('vanilla', 'cifg', 'wmc'):
            modele = ModeleLSTM(20, architecture, 3, 2).double()
            ids = torch.tensor([[2, 3, 4], [5, 6, 0]])
            longueurs = torch.tensor([3, 2])
            scores = modele(ids, longueurs)
            complete = F.pad(ids, (0, 5))
            torch.testing.assert_close(scores, modele(complete, longueurs))
            torch.testing.assert_close(scores[1:2], modele(ids[1:2, :2], longueurs[1:2]))
            F.cross_entropy(scores, torch.tensor([0, 2])).backward()
            for p in modele.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            self.assertTrue(modele.embedding.weight.grad[0].eq(0).all())

    def test_nombre_parametres(self):
        for nom, nombre in [('vanilla', 801219), ('cifg', 792963), ('wmc', 813507)]:
            self.assertEqual(sum(p.numel() for p in ModeleLSTM(12000, nom).parameters()), nombre)


if __name__ == '__main__':
    unittest.main()
