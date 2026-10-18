
# ============================================================
# TP MU-MISO : ETAPE 2
# Generation du canal radio H et calcul du precodage ZF.
# Configuration : M = 8 antennes, K = 4 utilisateurs.
# ============================================================

import numpy as np

# ------------------------------------------------------------
# 1. PARAMETRES DU SYSTEME
# ------------------------------------------------------------

# Nombre d'antennes d'emission de la station de base.
M = 8

# Nombre d'utilisateurs.
K = 4

# Fixer la graine aleatoire pour pouvoir reproduire le test.
np.random.seed(42)


# ------------------------------------------------------------
# 2. GENERER LE CANAL DE RAYLEIGH COMPLEXE
# ------------------------------------------------------------

# Generer la partie reelle du canal :
# K lignes (utilisateurs), M colonnes (antennes).
H_reel = np.random.randn(K, M)

# Generer la partie imaginaire independamment.
H_imag = np.random.randn(K, M)

# Construire le canal complexe.
# La division par sqrt(2) donne une puissance moyenne unitaire.
H = (H_reel + 1j * H_imag) / np.sqrt(2)

# Afficher la matrice du canal.
print("Matrice du canal H :")
print(H)

# Verifier ses dimensions.
print("\nDimensions de H :", H.shape)

# Confirmer que H est bien de dimension K x M.
assert H.shape == (K, M), "Dimensions incorrectes pour H."


# ------------------------------------------------------------
# 3. VERIFIER LE RANG DU CANAL
# ------------------------------------------------------------

# Le ZF classique exige que H soit de rang plein en lignes.
rang_H = np.linalg.matrix_rank(H)

print("\nRang de H :", rang_H)

# Pour K utilisateurs, le rang doit etre K.
if rang_H < K:
    raise ValueError(
        "Le canal n'est pas de rang plein. "
        "Le precodage ZF classique n'est pas applicable "
        "avec cette matrice."
    )

print("Le canal est de rang plein en lignes.")


# ------------------------------------------------------------
# 4. CALCULER LA MATRICE DE PRECODAGE ZF
# ------------------------------------------------------------

# Calculer H conjuguee transposee.
H_H = H.conj().T

# Calculer la matrice H fois H conjuguee transposee.
HH_H = H @ H_H

# Calculer son inverse.
inverse_HH_H = np.linalg.inv(HH_H)

# Calculer le precodage Zero-Forcing.
W_ZF = H_H @ inverse_HH_H

# Afficher le precodage.
print("\nMatrice de precodage W_ZF :")
print(W_ZF)

# Verifier ses dimensions : M lignes et K colonnes.
print("\nDimensions de W_ZF :", W_ZF.shape)

assert W_ZF.shape == (M, K), (
    "Dimensions incorrectes pour W_ZF."
)


# ------------------------------------------------------------
# 5. VERIFIER LE PRODUIT G = H W_ZF
# ------------------------------------------------------------

# Le produit G represente le canal effectif apres precodage.
G = H @ W_ZF

print("\nMatrice G = H @ W_ZF :")
print(G)

# Afficher les dimensions de G.
print("\nDimensions de G :", G.shape)

assert G.shape == (K, K), (
    "Dimensions incorrectes pour G."
)

# Construire la matrice identite de taille K.
I_K = np.eye(K)

# Mesurer l'ecart entre G et l'identite.
erreur_ZF = np.linalg.norm(G - I_K, ord="fro")

print("\nEcart entre G et I_K :", erreur_ZF)

# Verifier numeriquement que G est proche de l'identite.
if np.allclose(G, I_K, atol=1e-10):
    print("SUCCES : le produit H @ W_ZF est proche de I_K.")
else:
    print("ATTENTION : G differe de I_K.")
    print("Verifier le rang et le calcul du precodage.")


# ------------------------------------------------------------
# 6. AFFICHER LES INTERFERENCES RESIDUELLES
# ------------------------------------------------------------

# Calculer les modules des coefficients de G.
# La diagonale represente les gains des flux utiles.
# Les elements hors diagonale representent les interferences.
G_module = np.abs(G)

print("\nModule |G| :")
print(G_module)

# Extraire les coefficients hors diagonale.
masque_hors_diagonale = ~np.eye(K, dtype=bool)
interferences = G_module[masque_hors_diagonale]

# Mesurer la plus grande interference hors diagonale.
interference_max = np.max(interferences)

print(
    "\nInterference hors diagonale maximale :",
    interference_max
)

print("\nETAPE 2 TERMINEE.")
