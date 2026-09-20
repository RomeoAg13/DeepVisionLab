# =============================================================================
# THÉORIE — CNN (Convolutional Neural Network) SUR CIFAR-10
# =============================================================================
#
# POURQUOI UN CNN POUR LES IMAGES ?
# ----------------------------------
#   Un réseau Dense (fully-connected) sur une image 32×32×3 = 3072 entrées
#   → explosion du nombre de paramètres et perte de la structure spatiale.
#   Le CNN exploite la localité et la translation-invariance : un filtre
#   détecte un motif (bord, texture) peu importe où il apparaît dans l'image.
#
# ── COUCHE DE CONVOLUTION — nn.Conv2d ────────────────────────────────────────
#
# OPÉRATION DE CONVOLUTION
# -------------------------
#   Pour chaque filtre k et chaque position (i, j) :
#   Output[k, i, j] = Σ_{c,di,dj} Input[c, i+di, j+dj] * Filtre[k, c, di, dj] + biais[k]
#
#   Un filtre = une petite fenêtre (kernel) glissante qui "balaye" l'image.
#   Chaque filtre apprend à détecter un motif spécifique (vertical, couleur, texture…).
#
# PARAMÈTRES D'UN FILTRE
# -----------------------
#   Nombre de poids par filtre = in_channels * kernel_size * kernel_size
#                               = 3 * 3 * 3 = 27  (+ 1 biais = 28 par filtre)
#   Ici 16 filtres → 16 * 28 = 448 paramètres au total pour conv1.
#
# TAILLE DE LA FEATURE MAP EN SORTIE
# ------------------------------------
#   H_out = (H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
#   W_out = (W_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
#
#   Avec padding=1, kernel=3, stride=1 (défaut), dilation=1 (défaut) :
#     H_out = (32 + 2*1 - 1*(3-1) - 1) / 1 + 1 = (32 + 2 - 2 - 1) / 1 + 1 = 32
#   → La convolution avec padding=1 conserve la taille spatiale (32×32).
#
# ── ACTIVATION ReLU ──────────────────────────────────────────────────────────
#   ReLU(z) = max(0, z)  — appliquée sur chaque pixel de chaque feature map.
#   Introduit la non-linéarité indispensable pour apprendre des représentations complexes.
#
# ── POOLING — nn.MaxPool2d(2) ────────────────────────────────────────────────
#   Fenêtre 2×2, stride=2 (défaut) → divise chaque dimension par 2.
#   32×32 → 16×16
#
#   Objectif :
#     → Réduire la résolution spatiale (moins de calcul, moins de paramètres)
#     → Rendre la représentation invariante aux petits décalages (robustesse)
#   Max-pooling prend la valeur maximale dans chaque fenêtre (garde l'activation la plus forte).
#
# ── FLATTEN ──────────────────────────────────────────────────────────────────
#   Convertit le volume 3D (C, H, W) en vecteur 1D pour la couche Dense.
#   Ici : 16 filtres × 16 × 16 = 4096 valeurs  → vecteur de taille 4096.
#   Formule : Flatten = C * H_final * W_final
#
# ── COUCHE FULLY-CONNECTED — nn.Linear(4096, 10) ─────────────────────────────
#   Classifie parmi 10 classes CIFAR-10 :
#   avion, voiture, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion.
#   Produit 10 "logits" (scores bruts, non normalisés).
#
# ── LOSS — CrossEntropyLoss ───────────────────────────────────────────────────
#   CrossEntropyLoss = Softmax + Log + NLLLoss, en une seule opération.
#
#   SOFTMAX — convertit les logits en probabilités :
#     p_i = exp(z_i) / Σ_j exp(z_j)
#     → chaque p_i ∈ (0,1)  et  Σ p_i = 1
#
#   NEGATIVE LOG-LIKELIHOOD :
#     L = −log(p_{vraie_classe})
#     → Si le modèle est sûr et correct  : p proche de 1 → L proche de 0
#     → Si le modèle se trompe           : p proche de 0 → L → +∞
#
#   torch.argmax(logits, dim=1) → indice de la classe avec le score le plus élevé.
#
# ── CYCLE D'ENTRAÎNEMENT ─────────────────────────────────────────────────────
#   1. optimizer.zero_grad()  — remet les gradients à zéro
#   2. logits = model(x)      — forward pass (convolution → ReLU → pool → flatten → fc)
#   3. loss = criterion(logits, label) — CrossEntropyLoss
#   4. loss.backward()        — rétropropagation (autograd calcule ∂L/∂θ pour chaque θ)
#   5. optimizer.step()       — SGD : θ ← θ − lr * ∂L/∂θ
#
# ── DATASET CIFAR-10 ─────────────────────────────────────────────────────────
#   50 000 images d'entraînement, 10 000 de test, 32×32 pixels, 3 canaux RGB.
#   transforms.ToTensor() : convertit PIL Image (0-255) en Tensor float (0.0-1.0)
#   DataLoader : charge les données en mini-batches (ici batch_size=32) et les mélange.
#
# =============================================================================

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

class CNN(nn.Module): 
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16*8*8, 10)

    def forward(self, x):
        conv1 = self.conv1(x)
        relu = self.relu(conv1)
        pool = self.pool(relu)

        conv2 = self.conv2(pool)
        relu2 = self.relu(conv2)
        pool2 = self.pool(relu2)
        
        flatten = self.flatten(pool2)
        fc = self.fc(flatten)
        return fc

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def train(model, train_loader, criterion, optimizer):
    print("Training...")
    model.train()
    for _ in range(5):
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            predicted = torch.argmax(logits, dim=1)
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        moyenne_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(moyenne_loss)
        print(accuracy)
        print("--------------------------------------------------")



def test(model, test_loader, criterion):
    print("Testing...")
    model.eval()
    with torch.no_grad():
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in test_loader:
            logits = model(images)
            loss = criterion(logits, labels)
            predicted = torch.argmax(logits, dim=1)
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        moyenne_loss = running_loss / len(test_loader)
        accuracy = 100 * correct / total
        print(moyenne_loss)
        print(accuracy)



train(model, train_loader, criterion, optimizer)
test(model, test_loader, criterion)