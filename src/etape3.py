
import numpy as np

# ------------------------------------------------------------
# 1. Parametres du systeme
# ------------------------------------------------------------

M = 8  # Nombre d'antennes d'emission
K = 4  # Nombre d'utilisateurs

# Puissance totale moyenne cible.
# Ici, nous choisissons P_T = 1 pour commencer.
P_T = 1.0

# Reproduire le meme canal que dans etape2.py.
np.random.seed(42)

H_reel = np.random.randn(K, M)
H_imag = np.random.randn(K, M)

H = (H_reel + 1j * H_imag) / np.sqrt(2)

# ------------------------------------------------------------
# 2. Calcul du precodage ZF
# ------------------------------------------------------------

W_ZF = H.conj().T @ np.linalg.inv(H @ H.conj().T)

# Verifier les dimensions.
assert H.shape == (K, M)
assert W_ZF.shape == (M, K)

# ------------------------------------------------------------
# 3. Calculer la puissance avant normalisation
# ------------------------------------------------------------

# La norme de Frobenius au carre est la somme des modules
# au carre de tous les coefficients de W_ZF.
P_avant = np.linalg.norm(W_ZF, "fro") ** 2

print("Puissance cible P_T :", P_T)
print("Puissance avant normalisation :", P_avant)

# ------------------------------------------------------------
# 4. Calculer le facteur de normalisation
# ------------------------------------------------------------

# Verifier que la puissance est strictement positive.
if P_avant <= 0:
    raise ValueError("La puissance avant normalisation doit etre positive.")

# Calculer le facteur de normalisation.
alpha = np.sqrt(P_T / P_avant)

# Normaliser la matrice de precodage.
W_norm = alpha * W_ZF

print("\nFacteur de normalisation alpha :", alpha)

# ------------------------------------------------------------
# 5. Verifier la puissance apres normalisation
# ------------------------------------------------------------

P_apres = np.linalg.norm(W_norm, "fro") ** 2

print("Puissance apres normalisation :", P_apres)

# Verifier que la puissance correspond a la cible.
assert np.isclose(P_apres, P_T), (
    "La puissance apres normalisation ne correspond pas a P_T."
)

print("SUCCES : la puissance moyenne cible est respectee.")

# ------------------------------------------------------------
# 6. Verifier le canal effectif apres normalisation
# ------------------------------------------------------------

G_norm = H @ W_norm

print("\nModule de G_norm = H @ W_norm :")
print(np.abs(G_norm))

# La diagonale est mise a l'echelle par alpha.
# Les termes hors diagonale doivent rester proches de zero.
masque = ~np.eye(K, dtype=bool)
interference_max = np.max(np.abs(G_norm[masque]))

print("\nInterference hors diagonale maximale :", interference_max)

# Verifier numeriquement que les interferences restent nulles.
assert np.allclose(G_norm[masque], 0, atol=1e-10)

print("SUCCES : les interferences restent annulees.")
