# =============================================================================
# THÉORIE — RÉSEAU DE NEURONES AVEC PYTORCH (AUTOGRAD + OPTIMISEUR SGD)
# =============================================================================
#
# MÊME ARCHITECTURE QUE reseau_manuel.py — mais PyTorch fait tout automatiquement
#
# nn.Linear(in, out)
# -------------------
#   Implémente z = W·x + b  (matrice de poids W, vecteur de biais b)
#   PyTorch initialise W et b aléatoirement (distribution de Kaiming par défaut).
#   Linear(2, 3) : 2 entrées → 3 neurones  ⟹  6 poids + 3 biais = 9 paramètres
#   Linear(3, 1) : 3 entrées → 1 neurone   ⟹  3 poids + 1 biais = 4 paramètres
#
# nn.ReLU()
# ----------
#   ReLU(z) = max(0, z)  — appliqué élément par élément sur le vecteur de sortie.
#
# nn.MSELoss()
# -------------
#   L = (1/N) * Σ (y_pred_i − y_i)²
#   Mesure l'écart quadratique moyen entre prédiction et vérité terrain.
#
# torch.optim.SGD(params, lr)
# ----------------------------
#   Stochastic Gradient Descent :
#     w ← w − lr * ∂L/∂w
#     b ← b − lr * ∂L/∂b
#   "Stochastic" car dans la pratique on utilise des mini-batches, pas tout le dataset.
#
# CYCLE D'ENTRAÎNEMENT — les 4 étapes obligatoires
# --------------------------------------------------
#
#   1. optimizer.zero_grad()
#      PyTorch ACCUMULE les gradients par défaut.
#      Sans ce reset, les gradients de l'epoch précédente s'ajoutent au suivant → bug.
#
#   2. y_pred = model(x)
#      Forward pass : PyTorch enregistre toutes les opérations dans un graphe
#      de calcul (computation graph) pour pouvoir dériver ensuite.
#
#   3. loss.backward()
#      Rétropropagation automatique via AUTOGRAD.
#      PyTorch parcourt le graphe en sens inverse et calcule ∂L/∂θ
#      pour chaque paramètre θ (poids et biais).
#      Ces gradients sont stockés dans param.grad.
#
#   4. optimizer.step()
#      Applique la mise à jour : θ ← θ − lr * θ.grad
#      pour chaque paramètre enregistré dans l'optimiseur.
#
# AUTOGRAD — Comment ça marche ?
# --------------------------------
#   Chaque tensor créé avec requires_grad=True (c'est automatique pour nn.Parameter)
#   mémorise les opérations qui l'ont produit.
#   .backward() remonte ce graphe en appliquant la règle de la chaîne
#   de façon symbolique → aucun calcul manuel de dérivée nécessaire.
#
# =============================================================================

import torch
import torch.nn as nn

x = torch.tensor([2.0, 1.0])
y = torch.tensor([5.0])

class Reseau(nn.Module):
    def __init__(self):
        super().__init__()

        self.couche_1 = nn.Linear(2,3)
        self.relu = nn.ReLU()
        self.couche_2 = nn.Linear(3,1)

    def forward(self, x):
        sortie_couche_1 = self.couche_1(x)
        sortie_couche_1_relu = self.relu(sortie_couche_1)
        y_pred = self.couche_2(sortie_couche_1_relu)
        return y_pred


model = Reseau()    
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
mse = nn.MSELoss()

for epoch in range(100):

    # zero_grad() -> signifie que l'on va mettre à zéro les gradients des poids et biais du réseau de neurones
    optimizer.zero_grad()

    # y_pred = model(x) -> signifie que l'on va faire une prédiction avec le réseau de neurones
    y_pred = model(x)

    # loss = mse(y_pred, y) -> signifie que l'on va calculer la perte entre la prédiction et la valeur réelle : formule : 1/n * somme((y_pred - y)²) 
    loss = mse(y_pred, y)

    # backward() -> signifie que l'on va calculer les gradients des poids et biais du réseau de neurones : formule : dw = dloss/dw et db = dloss/db 
    loss.backward()

    # step() -> signifie que l'on va mettre à jour les poids et biais du réseau de neurones : formule : w = w - lr * dw et b = b - lr * db
    optimizer.step()