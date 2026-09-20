# =============================================================================
# THÉORIE — RÉGRESSION LINÉAIRE & DESCENTE DE GRADIENT (BATCH)
# =============================================================================
#
# MODÈLE
# ------
#   y_pred = w * x + b
#   Le réseau apprend une droite : w est la pente, b est l'ordonnée à l'origine.
#
# FONCTION DE PERTE — MSE (Mean Squared Error)
# --------------------------------------------
#   L = (1/N) * Σ (y_pred_i - y_i)²
#
#   Pourquoi le carré ?
#     → Toujours positif (pas d'annulation entre erreurs + et −)
#     → Pénalise davantage les grosses erreurs (effet quadratique)
#     → Dérivable partout (nécessaire pour la descente de gradient)
#
# GRADIENT DE LA LOSS PAR RAPPORT À w
# -------------------------------------
#   ∂L/∂w = (2/N) * Σ (y_pred_i - y_i) * x_i
#
#   Dérivation étape par étape :
#     L = (1/N) * Σ (w*x_i + b - y_i)²
#     ∂L/∂w = (1/N) * Σ 2*(w*x_i + b - y_i) * x_i
#           = (2/N) * Σ (y_pred_i - y_i) * x_i
#
# GRADIENT DE LA LOSS PAR RAPPORT À b
# -------------------------------------
#   ∂L/∂b = (2/N) * Σ (y_pred_i - y_i)
#
#   Dérivation :
#     ∂L/∂b = (1/N) * Σ 2*(w*x_i + b - y_i) * 1
#           = (2/N) * Σ (y_pred_i - y_i)
#
# MISE À JOUR DES PARAMÈTRES — Descente de gradient
# ---------------------------------------------------
#   w ← w − lr * ∂L/∂w
#   b ← b − lr * ∂L/∂b
#
#   Le learning_rate (lr) contrôle la taille du pas :
#     → trop grand  : on oscille / diverge
#     → trop petit  : convergence très lente
#
# BATCH GRADIENT DESCENT
# -----------------------
#   On accumule les gradients sur TOUS les exemples du dataset avant
#   de faire une seule mise à jour de w et b.
#   Contraire du SGD (Stochastic Gradient Descent) qui met à jour après
#   chaque exemple individuel.
#
#   Ici : N=4 exemples, 6 epochs
#   Objectif : trouver w≈3, b≈0  (car Y = 3*X)
#
# =============================================================================

from PIL import Image
import numpy as np
import torch
import math 

# def load__rgb_image(path):
#     image = Image.open(path)
#     image_rgb = image.convert("RGB")
#     return image_rgb


# def resize(image):
#     image_resize = image.resize((224,224))
#     return image_resize


# def normalized(image): 
#     img_convert = np.array(image)
#     normalized = img_convert / 255
#     return normalized


# def tenseur(image):
#     image2 = torch.from_numpy(image)
#     modif = image2.permute(2,0,1)
#     return modif


# def main(path):
#     image = load__rgb_image(path)
#     img_resized = resize(image)
#     img_normalized = normalized(img_resized)
#     img_tenseur = tenseur(img_normalized)
#     return img_tenseur


# image1 = main("./data/raw/image.png")
# image2 = main("./data/raw/image2.png")
# image3 = main("./data/raw/image3.png")


# batch = torch.stack((image1,image2,image3))
# print(batch.shape)


# # X - Input data - Which is the data that we feed into the model
# x = (2,-1, 3, 0.5, 4)

# # Weights - Which are the parameters that the model learns during training
# w1 = (0.2, -0.5, 1.0, 0.3, -0.1)
# w2 = (-0.4, 0.7, 0.2, -0.8, 0.5)
# w3 = (1.2, 0.1, -0.3, 0.4, 0.9)

# # Bias - Which is a constant value added to the linear combination of inputs and weights
# b1 = -2
# b2 = 5
# b3 = 2

# # Neural network linear combination
# z1 = w1[0]*x[0] + w1[1]*x[1]+ w1[2]*x[2] + w1[3]*x[3] + w1[4]*x[4] + b1
# z2 = w2[0]*x[0] + w2[1]*x[1]+ w2[2]*x[2] + w2[3]*x[3] + w2[4]*x[4] + b2
# z3 = w3[0]*x[0] + w3[1]*x[1]+ w3[2]*x[2] + w3[3]*x[3] + w3[4]*x[4] + b3

# # ReLU - Which is an activation function that outputs the input if it is positive, and 0 otherwise
# a1 = max(0, z1)
# a2 = max(0, z2)
# a3 = max(0, z3)

# output = (a1, a2, a3)
# print(output)


# w4 = (0.1, -0.2, 0.3)
# w5 = (-0.3, 0.6, -0.1)

# b4 = 4
# b5 = -1

# z4 = w4[0]*output[0] + w4[1]*output[1]+ w4[2]*output[2] + b4
# z5 = w5[0]*output[0] + w5[1]*output[1]+ w5[2]*output[2] + b5

# a4 = max(0, z4)
# a5 = max(0, z5)


# output2 = (a4, a5)
# print(output2)



# # Softmax: convert raw scores into values between 0 and 1 that sum to 1
# # formule de softmax = e^(z_i) / sum(e^(x_j))
# e1 = math.exp(output2[0])
# e2 = math.exp(output2[1])
# somme = e1 + e2

# p1 = e1 / somme
# p2 = e2 / somme


# print(p1, p2)
# print(p1 + p2)


# loss1 = -math.log(p1)
# loss2 = -math.log(p2)

# print(loss1, loss2)




# x = 2
# w = 1
# y = 6
# learning_rate = 0.01

# for i in range(20):

#     y_pred = w * x

#     loss = (y_pred - y)**2

#     grad = 2 * (y_pred - y) * x

#     w = w - learning_rate * grad

#     print(
#         "iteration:", i,
#         "prediction:", y_pred,
#         "loss:", loss,
#         "weight:", w
#     )



# X = [1,2,3,4]
# Y = [3,6,9,12]
# w = 1
# learning_rate = 0.01

# for epoch in range(100):
#     for i in range(len(X)):
#         y_pred = w * X[i]
#         loss = (y_pred - Y[i])**2
#         grad = 2*(y_pred - Y[i]) * X[i]
#         w = w - learning_rate * grad
#         print(
#             "iteration:", i,
#             "prediction:", y_pred,
#             "loss:", loss,
#             "gradient:", grad,
#             "weight:", w
#         )



# x = (1,2)
# W1 = (0.5, -0.1, 0.2, 0.3)
# b1 = (0, 0.1)
# z1 = W1[0]*x[0] + W1[1]*x[1] + b1[0]
# z2 = W1[2]*x[0] + W1[3]*x[1] + b1[1]

X = [1,2,3,4]
Y = [3,6,9,12]
w = 1
b = 2

learning_rate = 0.01
gradient = 0
gradient_b = 0

for epoch in range(6):
    erreurs_carre = []
    gradient_a = []
    gradient_ab = []
    for i in range(len(X)):
        y_pred = w * X[i] + b
        erreurs_carre.append((Y[i] - y_pred)**2)

        gradient = (y_pred- Y[i]) * X[i]
        gradient_a.append(gradient)

        gradient_b = (y_pred- Y[i])
        gradient_ab.append(gradient_b)

        print(f"pred: {y_pred}")   
        print(f"gradient: {gradient}")   

    mse = sum(erreurs_carre) / len(Y)

    grad = 2 / len(Y) * sum(gradient_a)

    grad_b = 2 / len(Y) * sum(gradient_ab)

    b = b - learning_rate * grad_b
    w = w - learning_rate * grad

    print(f"b: {b}")   
    print(f"grad b: {grad_b}")   
    print(f"mse: {mse}")   
    print(f"grad: {grad}")   
    print(f"w: {w}")   
