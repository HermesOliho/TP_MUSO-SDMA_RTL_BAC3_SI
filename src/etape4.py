
# ============================================================
# TP MU-MISO : ETAPE 4
# Heatmap de la matrice effective G = H @ W_norm.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Parametres du systeme
# ------------------------------------------------------------

M = 8  # Nombre d'antennes d'emission
K = 4  # Nombre d'utilisateurs
P_T = 1.0  # Puissance moyenne totale cible

# Utiliser la meme graine pour reproduire le meme canal.
np.random.seed(42)

# ------------------------------------------------------------
# 2. Generer le canal complexe H
# ------------------------------------------------------------

H_reel = np.random.randn(K, M)
H_imag = np.random.randn(K, M)

H = (H_reel + 1j * H_imag) / np.sqrt(2)

# ------------------------------------------------------------
# 3. Calculer le precodage ZF
# ------------------------------------------------------------

# Calculer la transposée conjuguee de H.
H_H = H.conj().T

# Calculer le precodage Zero-Forcing.
W_ZF = H_H @ np.linalg.inv(H @ H_H)

# ------------------------------------------------------------
# 4. Normaliser la puissance totale
# ------------------------------------------------------------

# Calculer la puissance totale moyenne avant normalisation.
P_avant = np.linalg.norm(W_ZF, "fro") ** 2

# Calculer le facteur de normalisation.
alpha = np.sqrt(P_T / P_avant)

# Obtenir la matrice de precodage normalisee.
W_norm = alpha * W_ZF

# Verifier la puissance moyenne obtenue.
P_apres = np.linalg.norm(W_norm, "fro") ** 2

print("Puissance cible :", P_T)
print("Puissance obtenue :", P_apres)

# ------------------------------------------------------------
# 5. Calculer la matrice effective G
# ------------------------------------------------------------

# Calculer le canal effectif apres precodage.
G = H @ W_norm

# Calculer le module des coefficients complexes.
G_module = np.abs(G)

# Afficher les valeurs numeriques.
print("\nModule de G :")
print(np.round(G_module, 6))

# ------------------------------------------------------------
# 6. Tracer la heatmap
# ------------------------------------------------------------

# Creer la figure.
fig, ax = plt.subplots(figsize=(7, 5.5))

# Afficher les modules sous forme de carte de chaleur.
# Chaque case correspond a un couple utilisateur-flux emis.
image = ax.imshow(
    G_module,
    cmap="viridis",
    interpolation="nearest",
    aspect="equal"
)

# Ajouter une barre de couleur.
barre = fig.colorbar(image, ax=ax)
barre.set_label(r"Module $|G_{k,j}|$")

# Nommer les axes.
ax.set_xlabel("Flux de données émis (j)")
ax.set_ylabel("Utilisateur récepteur (k)")

# Numeroter les lignes et colonnes de 1 a K.
ax.set_xticks(range(K))
ax.set_xticklabels([f"Flux {j + 1}" for j in range(K)])

ax.set_yticks(range(K))
ax.set_yticklabels([f"Utilisateur {k + 1}" for k in range(K)])

# Ajouter une valeur numerique dans chaque case.
for k in range(K):
    for j in range(K):
        valeur = G_module[k, j]

        # Utiliser une couleur de texte contrastante.
        couleur_texte = "white" if valeur < 0.6 else "black"

        ax.text(
            j,
            k,
            f"{valeur:.3e}",
            ha="center",
            va="center",
            color=couleur_texte,
            fontsize=9
        )

# Ajouter un titre.
ax.set_title(
    r"Heatmap de $|G|=|H W_{\mathrm{norm}}|$"
    "\n"
    "MU-MISO : 8 antennes et 4 utilisateurs"
)

# Ajuster la disposition.
plt.tight_layout()

# Enregistrer la figure pour le rapport du TP.
plt.savefig(
    "heatmap_G.png",
    dpi=300,
    bbox_inches="tight"
)

# Afficher la heatmap.
plt.show()

print("\nHeatmap enregistrée dans heatmap_G.png")
