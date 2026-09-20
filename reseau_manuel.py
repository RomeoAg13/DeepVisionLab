# =============================================================================
# THÉORIE — RÉSEAU DE NEURONES MANUEL + RÉTROPROPAGATION (BACKPROPAGATION)
# =============================================================================
#
# ARCHITECTURE DU RÉSEAU
# -----------------------
#   Entrée  x = (x1, x2)          — 2 features
#   Couche 1 : 3 neurones (z1, z2, z3) + activation ReLU → (a1, a2, a3)
#   Couche 2 : 1 neurone  (y_pred)     — sortie scalaire
#
# ── FORWARD PASS ─────────────────────────────────────────────────────────────
#
# SOMME PONDÉRÉE (pre-activation)
# --------------------------------
#   z = w^T · x + b  =  w[0]*x[0] + w[1]*x[1] + b
#
#   z représente la "valeur brute" du neurone avant activation.
#   C'est une combinaison linéaire des entrées pondérées par les poids.
#
# ACTIVATION — ReLU (Rectified Linear Unit)
# ------------------------------------------
#   ReLU(z) = max(0, z)
#
#   Pourquoi ReLU ?
#     → Introduit la non-linéarité (sans elle, empiler des couches ne sert à rien)
#     → Simple et rapide à calculer
#     → Évite le vanishing gradient des fonctions sigmoïde/tanh pour z > 0
#
# PRÉDICTION COUCHE 2
# --------------------
#   y_pred = w4[0]*a1 + w4[1]*a2 + w4[2]*a3 + b4
#   (pas de ReLU ici : on veut une valeur réelle libre pour la régression)
#
# LOSS — MSE avec N=1
# --------------------
#   L = (y_pred − y)²
#
# ── BACKWARD PASS (Rétropropagation) ─────────────────────────────────────────
#
# RÈGLE DE LA CHAÎNE (Chain Rule)
# --------------------------------
#   Pour calculer ∂L/∂w on "chaîne" les dérivées intermédiaires :
#   ∂L/∂w = ∂L/∂y_pred · ∂y_pred/∂w
#
# GRADIENT SUR LA COUCHE 2
# -------------------------
#   ∂L/∂y_pred     = 2*(y_pred − y)
#   ∂y_pred/∂w4[i] = a_i
#   ∂y_pred/∂b4    = 1
#
#   → ∂L/∂w4[i] = 2*(y_pred − y) * a_i
#   → ∂L/∂b4    = 2*(y_pred − y)
#
# PROPAGATION DU GRADIENT VERS COUCHE 1
# ---------------------------------------
#   ∂L/∂a_i = ∂L/∂y_pred · ∂y_pred/∂a_i
#            = 2*(y_pred − y) * w4[i]
#
# DÉRIVÉE DE ReLU
# ----------------
#   d(ReLU)/dz = 1   si z > 0
#              = 0   si z ≤ 0
#
#   Physiquement : si le neurone était "éteint" (z≤0), son gradient est nul
#   → le signal ne se propage pas (neurone mort).
#
#   → ∂L/∂z_i = ∂L/∂a_i * d(ReLU)/dz_i
#
# GRADIENT SUR LES POIDS DE COUCHE 1
# ------------------------------------
#   ∂z_i/∂w_i[j] = x[j]
#   ∂z_i/∂b_i    = 1
#
#   → ∂L/∂w_i[j] = ∂L/∂z_i * x[j]
#   → ∂L/∂b_i    = ∂L/∂z_i
#
# MISE À JOUR SGD
# ----------------
#   w ← w − lr * ∂L/∂w
#   b ← b − lr * ∂L/∂b
#
#   lr = 0.01 (learning rate) : contrôle la taille du pas de correction.
#
# =============================================================================

x = (2,1)
learning_rate = 0.01
# valeur cible
y = 5

# couche 1 - 3 neurone
w1 = (0.2,0.7)
w2 = (-0.3,0.5)
w3 = (0.4,-0.8)
b1 = 1
b2 = 2
b3 = -2

# couche 2 - 1 neurone
w4 = (0.3, -0.4, 0.8)
b4 = 3

for epoch in range(3):
    # formule de z = w^t*x + b
    z1 = w1[0] * x[0] + w1[1] * x[1] + b1
    print(f"z1 (somme pondérée neurone 1, couche 1) = {z1}")
    z2 = w2[0] * x[0] + w2[1] * x[1] + b2
    print(f"z2 (somme pondérée neurone 2, couche 1) = {z2}")
    z3 = w3[0] * x[0] + w3[1] * x[1] + b3
    print(f"z3 (somme pondérée neurone 3, couche 1) = {z3}")

    # ReLU(z) = max(0,z)
    a1 = max(0,z1)
    a2 = max(0,z2)
    a3 = max(0,z3)

    y_pred = w4[0] * a1 + w4[1] * a2 + w4[2] * a3 + b4
    print(f"y_pred (prédiction du réseau) = {y_pred}")

    # calculer la loss
    # loss = 1/N  Somme((y-y_pred)^2)
    loss = 1 / 1 * (y_pred-y)**2
    print(f"loss (erreur quadratique entre y_pred et y) = {loss}")

    # gradient
    gradient1 = 2 / 1 * (y_pred - y) * a1
    print(f"gradient1 (dloss/dw4[0]) = {gradient1}")
    gradient2 = 2 / 1 * (y_pred - y) * a2
    print(f"gradient2 (dloss/dw4[1]) = {gradient2}")
    gradient3 = 2 / 1 * (y_pred - y) * a3
    print(f"gradient3 (dloss/dw4[2]) = {gradient3}")
    gradient_b = 2* (y_pred - y)
    print(f"gradient_b (dloss/db4) = {gradient_b}")

    grad_a1 = 2 * ( y_pred - y) * w4[0]
    print(f"grad_a1 (dloss/da1, propagé vers la couche 1) = {grad_a1}")
    grad_a2 = 2 * ( y_pred - y) * w4[1]
    print(f"grad_a2 (dloss/da2, propagé vers la couche 1) = {grad_a2}")
    grad_a3 = 2 * ( y_pred - y) * w4[2]
    print(f"grad_a3 (dloss/da3, propagé vers la couche 1) = {grad_a3}")

    relu_derive1 = 1 if z1 > 0 else 0
    relu_derive2 = 1 if z2 > 0 else 0
    relu_derive3 = 1 if z3 > 0 else 0

    grad_z1 = grad_a1 * relu_derive1
    print(f"grad_z1 (dloss/dz1, après dérivée de ReLU) = {grad_z1}")
    grad_z2 = grad_a2 * relu_derive2
    print(f"grad_z2 (dloss/dz2, après dérivée de ReLU) = {grad_z2}")
    grad_z3 = grad_a3 * relu_derive3
    print(f"grad_z3 (dloss/dz3, après dérivée de ReLU) = {grad_z3}")

    grad_w1_1 = grad_z1 * x[0]
    print(f"grad_w1_1 (dloss/dw1[0]) = {grad_w1_1}")
    grad_w1_2 = grad_z1 * x[1]
    print(f"grad_w1_2 (dloss/dw1[1]) = {grad_w1_2}")
    grad_b1   = grad_z1
    print(f"grad_b1 (dloss/db1) = {grad_b1}")

    grad_w2_1 = grad_z2 * x[0]
    print(f"grad_w2_1 (dloss/dw2[0]) = {grad_w2_1}")
    grad_w2_2 = grad_z2 * x[1]
    print(f"grad_w2_2 (dloss/dw2[1]) = {grad_w2_2}")
    grad_b2   = grad_z2
    print(f"grad_b2 (dloss/db2) = {grad_b2}")

    grad_w3_1 = grad_z3 * x[0]
    print(f"grad_w3_1 (dloss/dw3[0]) = {grad_w3_1}")
    grad_w3_2 = grad_z3 * x[1]
    print(f"grad_w3_2 (dloss/dw3[1]) = {grad_w3_2}")
    grad_b3   = grad_z3
    print(f"grad_b3 (dloss/db3) = {grad_b3}")

    w1 = (w1[0] - learning_rate * grad_w1_1, w1[1] - learning_rate * grad_w1_2)
    print(f"w1 (poids du neurone 1 mis à jour) = {w1}")
    b1 = b1 - learning_rate * grad_b1
    print(f"b1 (biais du neurone 1 mis à jour) = {b1}")

    w2 = (w2[0] - learning_rate * grad_w2_1, w2[1] - learning_rate * grad_w2_2)
    print(f"w2 (poids du neurone 2 mis à jour) = {w2}")
    b2 = b2 - learning_rate * grad_b2
    print(f"b2 (biais du neurone 2 mis à jour) = {b2}")

    w3 = (w3[0] - learning_rate * grad_w3_1, w3[1] - learning_rate * grad_w3_2)
    print(f"w3 (poids du neurone 3 mis à jour) = {w3}")
    b3 = b3 - learning_rate * grad_b3
    print(f"b3 (biais du neurone 3 mis à jour) = {b3}")

    # new w
    w4 = (w4[0] - learning_rate * gradient1,  w4[1] - learning_rate * gradient2, w4[2] - learning_rate * gradient3)
    print(f"w4 (poids de la couche 2 mis à jour) = {w4}")

    # new b
    b4 = b4 - learning_rate * gradient_b
    print(f"b4 (biais de la couche 2 mis à jour) = {b4}")
