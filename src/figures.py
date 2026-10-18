"""Génération des figures du TP (matplotlib).

Toutes les figures sont calculées à partir des résultats des
simulations et enregistrées dans ``figures/``. Titres, axes et légendes
sont en français.

Convention d'affichage des BER observés nuls
--------------------------------------------
Un BER observé nul ne prouve pas que le BER théorique est nul : la
simulation n'a simplement observé aucune erreur sur le nombre de bits
testé. Sur les courbes, un point de BER observé nul est affiché par un
marqueur ouvert (▼) placé à la « limite d'observation »
``1 / (2 N_bits)`` (convention de tracé), avec une annotation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Palette de couleurs des utilisateurs / flux / modulations.
COULEURS_UTILISATEURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                         "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                         "#bcbd22", "#17becf", "#aec7e8", "#ffbb78")


# ------------------------------------------------------------
# Outils d'affichage des BER observés nuls
# ------------------------------------------------------------

def valeurs_ber_affichables(resultats, attribut: str = "ber") -> tuple:
    """Prépare des valeurs de BER pour un tracé semi-logarithmique.

    Les BER observés nuls sont remplacés pour l'affichage par la limite
    d'observation ``1 / (2 N_bits)`` et signalés par un masque.

    Parameters
    ----------
    resultats : sequence
        Objets ``ResultatPoint``.
    attribut : str
        Attribut ou propriété à tracer (``"ber"`` ou ``"ber_moyen"``).

    Returns
    -------
    tuple
        ``(valeurs, masque_nul, bornes)``.
    """
    valeurs, masque_nul, bornes = [], [], []
    for res in resultats:
        valeur = float(getattr(res, attribut))
        borne = max(res.borne_observation, 1e-12)
        if valeur > 0.0:
            valeurs.append(valeur)
            masque_nul.append(False)
        else:
            valeurs.append(0.5 * borne)
            masque_nul.append(True)
        bornes.append(res.borne_observation)
    return np.array(valeurs), np.array(masque_nul), np.array(bornes)


def _marquer_zeros(ax, x, valeurs, masque_nul, etiquette=None):
    """Marque les points de BER observé nul par des ▼ ouverts."""
    if np.any(masque_nul):
        ax.scatter(np.asarray(x)[masque_nul], valeurs[masque_nul], s=60, marker="v",
                   facecolors="none", edgecolors="crimson", zorder=5,
                   label=etiquette or "BER observé nul (limite d'observation)")


# ------------------------------------------------------------
# Figure 1 : schéma fonctionnel du système
# ------------------------------------------------------------

def figure_schema_systeme(chemin: Path, k: int = 4, m: int = 8) -> Path:
    """Dessine le schéma fonctionnel de la chaîne MU-MISO / SDMA."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def boite(x, y, largeur, hauteur, texte, couleur, taille=8):
        ax.add_patch(FancyBboxPatch(
            (x, y), largeur, hauteur, boxstyle="round,pad=0.3",
            linewidth=1.2, edgecolor="black", facecolor=couleur,
        ))
        ax.text(x + largeur / 2, y + hauteur / 2, texte,
                ha="center", va="center", fontsize=taille)

    def fleche(x1, y1, x2, y2, couleur="black", largueur=1.6):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=16, linewidth=largueur, color=couleur,
        ))

    ys_sources = [46 - i * 11 for i in range(k)]
    for i, y in enumerate(ys_sources):
        boite(2, y, 15, 7, f"Image {i + 1}\n(source JPEG)", "#aec7e8")
        fleche(17.5, y + 3.5, 24, 29 + (i - (k - 1) / 2) * 1.2, couleur="#1f77b4")

    boite(24, 25, 13, 9, "Conversion\nen bits\n(8 bits/pixel)", "#ffbb78")
    boite(40, 25, 13, 9, "Modulation\nBPSK / 4-QAM\n/ 16-QAM", "#ffbb78")
    fleche(37.5, 29.5, 40, 29.5)

    boite(56, 22, 16, 15,
          f"Précodage ZF\nW_norm ({m}×{k})\n\n{m} antennes\nde la station de base",
          "#98df8a", taille=8.5)
    fleche(53.5, 29.5, 56, 29.5)

    fleche(72.5, 29.5, 79, 29.5, couleur="#d62728", largueur=2.2)
    ax.text(75.7, 32.5, f"Canal de Rayleigh\nH ({k}×{m}) + bruit AWGN",
            ha="center", va="bottom", fontsize=8, color="#d62728")

    for i, y in enumerate(ys_sources):
        boite(79, y, 19, 7,
              f"Utilisateur {i + 1}\nÉgalisation → détection\n→ image",
              "#c5b0d5")

    ax.text(50, 58,
            "Les K flux partagent les mêmes ressources temps-fréquence : "
            "ils sont séparés dans l'espace par le précodage ZF (SDMA).",
            ha="center", va="center", fontsize=10, style="italic", color="#333333")
    ax.set_title("Chaîne de transmission MU-MISO / SDMA avec précodage Zero-Forcing",
                 fontsize=13)

    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 2 : illustration pédagogique du SDMA
# ------------------------------------------------------------

def figure_partage_sdma(chemin: Path, k: int = 4) -> Path:
    """Illustre le principe SDMA (schéma conceptuel)."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(plt.Circle((0, 0), 0.35, color="#2ca02c", zorder=5))
    ax.text(0, -0.75, "Station de base\n(8 antennes)", ha="center",
            va="top", fontsize=9, fontweight="bold")

    angles = np.linspace(-55, 55, k)
    for i, angle in enumerate(angles):
        theta = np.deg2rad(angle)
        x, y = 4.2 * np.cos(theta), 4.2 * np.sin(theta)
        ax.add_patch(FancyArrowPatch(
            (0, 0), (x * 0.82, y * 0.82), arrowstyle="-|>", mutation_scale=22,
            linewidth=2.4, color=COULEURS_UTILISATEURS[i % len(COULEURS_UTILISATEURS)],
        ))
        ax.add_patch(plt.Circle((x, y), 0.28,
                                color=COULEURS_UTILISATEURS[i % len(COULEURS_UTILISATEURS)]))
        ax.text(x, y + 0.55, f"Utilisateur {i + 1}", ha="center", va="bottom", fontsize=9)

    ax.text(-5.7, -3.0, "Mêmes ressources\ntemps-fréquence\npour les 4 flux :",
            ha="left", va="center", fontsize=9, fontweight="bold")
    for i in range(k):
        y_barre = -2.4 - i * 0.42
        ax.add_patch(FancyBboxPatch(
            (-2.6, y_barre - 0.14), 8.0, 0.28, boxstyle="round,pad=0.02",
            facecolor=COULEURS_UTILISATEURS[i % len(COULEURS_UTILISATEURS)], alpha=0.75,
        ))
    ax.annotate("", xy=(-2.6, -1.9), xytext=(5.4, -1.9),
                arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4", color="black"))
    ax.text(1.4, -1.65, "même fenêtre temps-fréquence", ha="center", va="bottom", fontsize=8)

    ax.set_title(
        "SDMA (Space Division Multiple Access) : séparation spatiale des flux\n"
        "Chaque utilisateur reçoit son flux propre grâce au précodage ZF, "
        "qui annule les interférences inter-utilisateurs.",
        fontsize=11,
    )
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 3 : heatmap de la matrice effective
# ------------------------------------------------------------

def figure_heatmap_g(g: np.ndarray, alpha: float, chemin: Path) -> Path:
    """Trace la heatmap annotée de ``|G| = |H @ W_norm|``."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    g_module = np.abs(np.asarray(g))
    k = g_module.shape[0]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(g_module, cmap="viridis", interpolation="nearest", aspect="equal")
    barre = fig.colorbar(image, ax=ax)
    barre.set_label(r"Module $|G_{k,j}|$ (gain utile si $k=j$, interférence sinon)")

    ax.set_xlabel("Flux émis (j)")
    ax.set_ylabel("Utilisateur récepteur (k)")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"Flux {j + 1}" for j in range(k)])
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Utilisateur {i + 1}" for i in range(k)])

    for i in range(k):
        for j in range(k):
            valeur = g_module[i, j]
            couleur = "white" if valeur < 0.6 * max(g_module.max(), 1e-12) else "black"
            ax.text(j, i, f"{valeur:.3e}", ha="center", va="center",
                    fontsize=8.5, color=couleur)

    ax.set_title(
        r"Heatmap de $|G| = |H\,W_{\mathrm{norm}}|$" "\n"
        f"Diagonale = gain utile $\\alpha$ = {alpha:.6f} ; hors diagonale ≈ 0"
    )
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 4 : constellations (avant / après égalisation, 3 modulations)
# ------------------------------------------------------------

def figure_constellations(
    donnees: list,
    chemin: Path,
    utilisateur: int = 1,
    n_max_points: int = 1200,
) -> Path:
    """Constellations : émises, avant égalisation, après égalisation.

    Une ligne par modulation, une colonne par étape (émis, reçu avant
    égalisation, reçu après égalisation), pour un utilisateur donné.

    Parameters
    ----------
    donnees : list of tuple
        Chaque élément est ``(nom_modulation, s, y_avant, y_apres)`` où
        ``s``, ``y_avant`` et ``y_apres`` sont des tableaux ``(K, T)``.
    chemin : pathlib.Path
        Fichier de sortie (PNG).
    utilisateur : int
        Indice (1-based) de l'utilisateur affiché.
    n_max_points : int
        Nombre maximal de points par sous-figure.

    Returns
    -------
    pathlib.Path
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    lignes = len(donnees)
    fig, axes = plt.subplots(lignes, 3, figsize=(13, 3.8 * lignes), squeeze=False)

    titres_colonnes = ("Symboles émis", "Reçus (avant égalisation)",
                       "Après égalisation ($y_k/g_k$)")
    indice_utilisateur = utilisateur - 1
    couleur = COULEURS_UTILISATEURS[indice_utilisateur % len(COULEURS_UTILISATEURS)]

    for ligne, (nom_modulation, s, y_avant, y_apres) in enumerate(donnees):
        t = s.shape[1]
        indices = np.linspace(0, t - 1, min(n_max_points, t)).astype(int)
        for colonne, (serie, titre) in enumerate(zip((s, y_avant, y_apres), titres_colonnes)):
            ax = axes[ligne, colonne]
            donnees_user = serie[indice_utilisateur, indices]
            ax.scatter(np.real(donnees_user), np.imag(donnees_user), s=6, alpha=0.55,
                       color=couleur)
            ax.axhline(0, color="grey", linewidth=0.6)
            ax.axvline(0, color="grey", linewidth=0.6)
            ax.grid(alpha=0.25)
            ax.set_xlabel("Partie réelle")
            ax.set_ylabel("Partie imaginaire")
            if ligne == 0:
                ax.set_title(titre, fontsize=10)
            if colonne == 0:
                ax.set_ylabel(f"{nom_modulation}\nPartie imaginaire", fontsize=9)

    fig.suptitle(
        f"Constellations de l'utilisateur {utilisateur} : émission, réception "
        "avant et après égalisation (canal complexe + ZF)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 5 : BER en fonction du SNR (ZF vs sans précodage)
# ------------------------------------------------------------

def figure_ber_snr(resultats: dict, chemin: Path) -> Path:
    """Trace le BER en fonction du SNR (ZF vs sans précodage)."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    snr = np.asarray(resultats["snr_db"], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6))

    for cle, etiquette, couleur in (
        ("zf", "Avec précodage ZF", "#1f77b4"),
        ("sans_precodage", "Sans précodage (1 antenne/utilisateur, même $P_T$)", "#ff7f0e"),
    ):
        valeurs, masque_nul, bornes = valeurs_ber_affichables(resultats[cle], attribut="ber_moyen")
        ax.semilogy(snr, valeurs, "o-", color=couleur, label=etiquette)
        _marquer_zeros(ax, snr, valeurs, masque_nul, etiquette=f"{etiquette} : BER observé nul")

    ax.set_xlabel("SNR en dB  (convention : $\\sigma^2 = P_T / 10^{\\mathrm{SNR_{dB}}/10}$)")
    ax.set_ylabel("BER moyen")
    ax.set_title(
        f"BER en fonction du SNR — {resultats['modulation']}, "
        f"M = {resultats['m']} antennes, K = {resultats['k']} utilisateurs"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 6 : BER_k et BER moyen vs SNR, pour chaque M
# ------------------------------------------------------------

def figure_ber_snr_par_m(resultats: dict, chemin: Path) -> Path:
    """BER_k et BER moyen en fonction du SNR, un sous-graphe par M."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    valeurs_m = resultats["valeurs_m"]
    snr = np.asarray(resultats["snr_db"], dtype=float)
    k = resultats["k"]
    modulation = resultats["modulation"]

    n = len(valeurs_m)
    n_colonnes = 3
    n_lignes = int(np.ceil(n / n_colonnes))
    fig, axes = plt.subplots(n_lignes, n_colonnes, figsize=(5 * n_colonnes, 4 * n_lignes),
                             squeeze=False)

    for indice, m in enumerate(valeurs_m):
        ax = axes[indice // n_colonnes][indice % n_colonnes]
        points = resultats["resultats"][m]

        # BER par utilisateur.
        bornes = np.array([p.borne_observation for p in points])
        for utilisateur in range(k):
            ber_u = np.array([p.ber_par_utilisateur[utilisateur] for p in points])
            aff = np.where(ber_u > 0, ber_u, np.where(bornes > 0, 0.5 * bornes, 1e-12))
            ax.semilogy(snr, aff, "o-", markersize=3,
                        color=COULEURS_UTILISATEURS[utilisateur % len(COULEURS_UTILISATEURS)],
                        label=f"BER$_{{{utilisateur + 1}}}$")

        valeurs, masque_nul, _ = valeurs_ber_affichables(points, attribut="ber_moyen")
        ax.semilogy(snr, valeurs, "k-s", linewidth=2, markersize=4, label="BER moyen")
        _marquer_zeros(ax, snr, valeurs, masque_nul)

        ax.set_title(f"M = {m} antennes (K = {k})")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    # Masquer les sous-graphes inutilisés.
    for indice in range(n, n_lignes * n_colonnes):
        axes[indice // n_colonnes][indice % n_colonnes].axis("off")

    fig.suptitle(
        f"BER individuels et BER moyen en fonction du SNR pour différents M "
        f"({modulation}, K = {k} fixé)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 7 : BER moyen vs M, pour les trois modulations
# ------------------------------------------------------------

def figure_ber_moyen_vs_m(resultats: dict, chemin: Path) -> Path:
    """BER moyen en fonction de M (SNR fixé) pour les trois modulations."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    valeurs_m = np.asarray(resultats["valeurs_m"], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6))

    for indice, modulation in enumerate(resultats["modulations"]):
        points = resultats["resultats"][modulation]
        valeurs, masque_nul, _ = valeurs_ber_affichables(points, attribut="ber_moyen")
        ax.semilogy(valeurs_m, valeurs, "o-",
                    color=COULEURS_UTILISATEURS[indice % len(COULEURS_UTILISATEURS)],
                    label=modulation)
        _marquer_zeros(ax, valeurs_m, valeurs, masque_nul)

    ax.set_xlabel("Nombre d'antennes d'émission M")
    ax.set_ylabel("BER moyen")
    ax.set_title(
        f"BER moyen en fonction de M pour BPSK, 4-QAM et 16-QAM\n"
        f"(K = {resultats['k']} fixé, SNR = {resultats['snr_db']} dB)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 8 : BER moyen vs K (M fixé)
# ------------------------------------------------------------

def figure_ber_utilisateurs(resultats: dict, chemin: Path) -> Path:
    """BER moyen en fonction du nombre d'utilisateurs K (M fixé)."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    valeurs_k = np.asarray(resultats["valeurs_k"], dtype=float)
    valeurs, masque_nul, _ = valeurs_ber_affichables(resultats["zf"], attribut="ber_moyen")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(valeurs_k, valeurs, "o-", color="#2ca02c", label="ZF (BER moyen)")
    _marquer_zeros(ax, valeurs_k, valeurs, masque_nul)

    ax.set_xlabel("Nombre d'utilisateurs K")
    ax.set_ylabel("BER moyen")
    ax.set_title(
        f"BER moyen en fonction du nombre d'utilisateurs K — {resultats['modulation']}, "
        f"M = {resultats['m']} antennes, SNR = {resultats['snr_db']} dB\n"
        "Lorsque K se rapproche de M, le conditionnement de H se dégrade et le "
        "bruit de recodage augmente."
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 9 : conditionnement -> BER
# ------------------------------------------------------------

def figure_conditionnement_ber(resultats: dict, chemin: Path) -> Path:
    """Nuage de points kappa(H H^H) en fonction du BER observé."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    kappa = np.asarray(resultats["kappa"], dtype=float)
    ber = np.asarray(resultats["ber"], dtype=float)
    ber_aff = np.where(ber > 0, ber, np.nan)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(kappa, ber_aff, "o", color="#d62728", alpha=0.7)
    ax.set_xlabel(r"Conditionnement $\kappa(H H^{H})$")
    ax.set_ylabel("BER observé")
    ax.set_title(
        f"BER observé en fonction du conditionnement $\\kappa(HH^H)$ — "
        f"{resultats['modulation']}, M = {resultats['m']}, K = {resultats['k']}, "
        f"SNR = {resultats['snr_db']} dB\n"
        f"{resultats['n_realisations']} réalisations indépendantes ; un canal mal "
        "conditionné (kappa élevé) conduit à un BER plus élevé (bruit de recodage)."
    )
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 10 : canaux corrélés (antennes)
# ------------------------------------------------------------

def figure_canaux_correles(resultats: dict, chemin: Path) -> Path:
    """Compare les performances ZF selon la corrélation entre antennes."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    snr = np.asarray(resultats["snr_db"], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6))

    for indice_rho, rho in enumerate(resultats["rho"]):
        serie = resultats["ber"][indice_rho]
        valeurs, masque_nul, _ = valeurs_ber_affichables(serie, attribut="ber_moyen")
        ax.semilogy(snr, valeurs, "o-",
                    color=COULEURS_UTILISATEURS[indice_rho % len(COULEURS_UTILISATEURS)],
                    label=(f"$\\rho$ = {rho:g} " + ("(i.i.d.)" if rho == 0 else "(corrélé)")))
        _marquer_zeros(ax, snr, valeurs, masque_nul)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER moyen")
    ax.set_title(
        "Effet de la corrélation entre antennes d'émission sur le BER du ZF\n"
        r"Modèle exponentiel $R_{m,n} = \rho^{|m-n|}$ (Kronecker côté émission)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 11 : canaux utilisateurs similaires
# ------------------------------------------------------------

def figure_canaux_similaires(resultats: dict, chemin: Path) -> Path:
    """BER moyen lorsque deux canaux utilisateurs deviennent similaires."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    snr = np.asarray(resultats["snr_db"], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 6))

    for indice_beta, beta in enumerate(resultats["beta"]):
        serie = resultats["ber"][indice_beta]
        valeurs, masque_nul, _ = valeurs_ber_affichables(serie, attribut="ber_moyen")
        ax.semilogy(snr, valeurs, "o-",
                    color=COULEURS_UTILISATEURS[indice_beta % len(COULEURS_UTILISATEURS)],
                    label=(f"$\\beta$ = {beta:g} "
                           + ("(canaux indépendants)" if beta == 0 else "(similaires)")))
        _marquer_zeros(ax, snr, valeurs, masque_nul)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER moyen")
    ax.set_title(
        "Effet de la similitude entre deux canaux utilisateurs sur le BER du ZF\n"
        r"$h_2 \leftarrow \sqrt{1-\beta^2}\,h_2 + \beta\,h_1$ : "
        r"$\beta \to 1$ rend $H H^H$ mal conditionnée"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 12 : images originales et reconstruites à plusieurs SNR
# ------------------------------------------------------------

def figure_images_reconstruites(
    image_originale: np.ndarray,
    reconstructions: dict,
    snr_liste: tuple,
    modulations: tuple,
    chemin: Path,
    utilisateur: int = 1,
) -> Path:
    """Grille : image originale (référence) et reconstruite à plusieurs SNR.

    Parameters
    ----------
    image_originale : numpy.ndarray
        Image en niveaux de gris de l'utilisateur affiché.
    reconstructions : dict
        ``reconstructions[(modulation, snr)]`` -> image reconstruite.
    snr_liste : sequence of float
        Valeurs de SNR (colonnes).
    modulations : sequence of str
        Modulations (lignes).
    chemin : pathlib.Path
    utilisateur : int
        Indice (1-based) de l'utilisateur affiché.

    Returns
    -------
    pathlib.Path
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    modulations = list(modulations)
    snr_liste = list(snr_liste)
    n_lignes = len(modulations) + 1   # +1 pour l'originale
    n_colonnes = len(snr_liste)

    fig, axes = plt.subplots(n_lignes, n_colonnes,
                             figsize=(3.2 * n_colonnes, 3.4 * n_lignes), squeeze=False)

    for colonne, snr in enumerate(snr_liste):
        ax = axes[0][colonne]
        ax.imshow(image_originale, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Originale (niveaux de gris)\nutilisateur {utilisateur}", fontsize=9)
        ax.axis("off")

    for ligne, modulation in enumerate(modulations, start=1):
        for colonne, snr in enumerate(snr_liste):
            image = reconstructions[(modulation, snr)]
            ax = axes[ligne][colonne]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            pixels_err = int(np.count_nonzero(image != image_originale))
            ax.set_title(f"{modulation} — SNR {snr:g} dB\n{pixels_err} pixel(s) erroné(s)",
                         fontsize=9)
            ax.axis("off")

    fig.suptitle(
        f"Image originale et reconstruite (utilisateur {utilisateur}) "
        "pour BPSK, 4-QAM et 16-QAM à plusieurs SNR",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin
