"""Génération des figures du TP (matplotlib).

Toutes les figures sont calculées à partir des résultats des
simulations et enregistrées dans ``figures/``. Les titres, axes et
légendes sont en français.

Convention d'affichage des BER observés nuls
--------------------------------------------
Un BER observé nul ne prouve pas que le BER théorique est nul : la
simulation n'a simplement observé aucune erreur sur le nombre de bits
testé. Sur les courbes, un point de BER observé nul est affiché par un
marqueur ouvert (▼) positionné à la « limite d'observation »
``1 / (2 * N_bits)`` (convention de tracé, pas une mesure), avec une
annotation expliquant le nombre de bits testés.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Palette de couleurs des utilisateurs / flux.
COULEURS_UTILISATEURS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                         "#9467bd", "#8c564b")


# ------------------------------------------------------------
# Outil d'affichage des BER observés nuls
# ------------------------------------------------------------

def valeurs_ber_affichables(resultats) -> tuple:
    """Prépare les valeurs de BER pour un tracé semi-logarithmique.

    Les BER observés nuls sont remplacés pour l'affichage par la limite
    d'observation ``1 / (2 * N_bits)`` et signalés par un masque.

    Parameters
    ----------
    resultats : sequence
        objets ``ResultatPoint`` (attributs ``ber`` et ``nb_bits``).

    Returns
    -------
    tuple
        ``(valeurs, masque_nul, bornes)`` : valeurs à tracer, masque
        booléen des points observés à zéro, bornes d'observation.
    """
    valeurs = []
    masque_nul = []
    bornes = []
    for res in resultats:
        if res.ber > 0.0:
            valeurs.append(res.ber)
            masque_nul.append(False)
        else:
            # Ligne d'affichage : milieu de la plage [0, limite d'observation].
            borne = max(res.borne_observation, 1e-12)
            valeurs.append(0.5 * borne)
            masque_nul.append(True)
        bornes.append(res.borne_observation)
    return np.array(valeurs), np.array(masque_nul), np.array(bornes)


def _annoter_zeros_ber(ax, snr, valeurs, masque_nul, bornes, texte_bits=None):
    """Ajoute les marqueurs ouverts et l'annotation des BER nuls."""
    if np.any(masque_nul):
        ax.scatter(
            np.asarray(snr)[masque_nul],
            valeurs[masque_nul],
            s=70, marker="v", facecolors="none", edgecolors="crimson",
            zorder=5,
            label="BER observé nul : ▼ à la limite d'observation",
        )
        annotation = (
            "▼ : aucun bit erroné observé en ces points — le BER réel est "
            "inférieur à la limite d'observation (≈ 1 / N_bits), il n'est pas nul "
            "de façon théorique."
        )
        if texte_bits:
            annotation += f"\n({texte_bits})"
        ax.annotate(
            annotation,
            xy=(0.03, 0.03), xycoords="axes fraction",
            fontsize=8, color="crimson",
        )


# ------------------------------------------------------------
# Figure 1 : schéma fonctionnel du système
# ------------------------------------------------------------

def figure_schema_systeme(chemin: Path, k: int = 4, m: int = 8) -> Path:
    """Dessine le schéma fonctionnel de la chaîne MU-MISO / SDMA.

    Chaîne représentée : images sources -> conversion en bits ->
    modulation BPSK -> précodage ZF -> ``M`` antennes de la station de
    base -> canal de Rayleigh -> ``K`` utilisateurs -> détection BPSK ->
    images reconstruites.

    Parameters
    ----------
    chemin : pathlib.Path
        Fichier de sortie (PNG).
    k : int
        Nombre d'utilisateurs.
    m : int
        Nombre d'antennes d'émission.

    Returns
    -------
    pathlib.Path
        Chemin de la figure enregistrée.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def boite(x, y, largeur, hauteur, texte, couleur, taille=8):
        ax.add_patch(FancyBboxPatch(
            (x, y), largeur, hauteur,
            boxstyle="round,pad=0.3", linewidth=1.2,
            edgecolor="black", facecolor=couleur,
        ))
        ax.text(x + largeur / 2, y + hauteur / 2, texte,
                ha="center", va="center", fontsize=taille)

    def fleche(x1, y1, x2, y2, couleur="black", style="-|>", largueur=1.6):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style,
            mutation_scale=16, linewidth=largueur, color=couleur,
        ))

    # Colonne 1 : les K images sources.
    ys_sources = [46 - i * 11 for i in range(k)]
    for i, y in enumerate(ys_sources):
        boite(2, y, 15, 7, f"Image {i + 1}\n(source JPEG)", "#aec7e8")
        fleche(17.5, y + 3.5, 24, 29 + (i - (k - 1) / 2) * 1.2, couleur="#1f77b4")

    # Colonne 2 : conversion en bits.
    boite(24, 25, 13, 9, "Conversion\nen bits\n(8 bits/pixel)", "#ffbb78")

    # Colonne 3 : modulation BPSK.
    boite(40, 25, 13, 9, "Modulation\nBPSK\n0 → −1,  1 → +1", "#ffbb78")
    fleche(37.5, 29.5, 40, 29.5)

    # Colonne 4 : précodage ZF + antennes.
    boite(56, 22, 16, 15,
          f"Précodage ZF\nW_norm ({m}×{k})\n\n{m} antennes\nde la station de base",
          "#98df8a", taille=8.5)
    fleche(53.5, 29.5, 56, 29.5)

    # Canal radio.
    fleche(72.5, 29.5, 79, 29.5, couleur="#d62728", largueur=2.2)
    ax.text(75.7, 32.5,
            f"Canal de Rayleigh\nH ({k}×{m}) + bruit AWGN",
            ha="center", va="bottom", fontsize=8, color="#d62728")

    # Colonne 5 : les K utilisateurs avec détection.
    for i, y in enumerate(ys_sources):
        boite(79, y, 19, 7,
              f"Utilisateur {i + 1}\nDétection BPSK → image",
              "#c5b0d5")

    # Annotation SDMA.
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
    """Illustre le principe SDMA (schéma conceptuel, pas une mesure).

    Parameters
    ----------
    chemin : pathlib.Path
        Fichier de sortie (PNG).
    k : int
        Nombre d'utilisateurs illustrés.

    Returns
    -------
    pathlib.Path
        Chemin de la figure enregistrée.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Station de base (origine des faisceaux).
    ax.add_patch(plt.Circle((0, 0), 0.35, color="#2ca02c", zorder=5))
    ax.text(0, -0.75, "Station de base\n(8 antennes)", ha="center",
            va="top", fontsize=9, fontweight="bold")

    # Utilisateurs répartis dans l'espace.
    angles = np.linspace(-55, 55, k)
    positions = []
    for i, angle in enumerate(angles):
        theta = np.deg2rad(angle)
        x, y = 4.2 * np.cos(theta), 4.2 * np.sin(theta)
        positions.append((x, y))
        # Faisceau dirigé (flèche colorée).
        ax.add_patch(FancyArrowPatch(
            (0, 0), (x * 0.82, y * 0.82),
            arrowstyle="-|>", mutation_scale=22, linewidth=2.4,
            color=COULEURS_UTILISATEURS[i % len(COULEURS_UTILISATEURS)],
            alpha=0.75,
        ))
        # Utilisateur.
        ax.add_patch(plt.Circle((x, y), 0.28,
                                color=COULEURS_UTILISATEURS[i % len(COULEURS_UTILISATEURS)]))
        ax.text(x, y + 0.55, f"Utilisateur {i + 1}", ha="center",
                va="bottom", fontsize=9)

    # Barre : mêmes ressources temps-fréquence partagées.
    ax.text(-5.7, -3.0, "Mêmes ressources\ntemps-fréquence\npour les 4 flux :",
            ha="left", va="center", fontsize=9, fontweight="bold")
    for i in range(k):
        y_barre = -2.4 - i * 0.42
        ax.add_patch(FancyBboxPatch(
            (-2.6, y_barre - 0.14), 8.0, 0.28,
            boxstyle="round,pad=0.02",
            facecolor=COULEURS_UTILISATEURS[i % len(COULEURS_UTILISATEURS)],
            alpha=0.75,
        ))
    ax.annotate(
        "", xy=(-2.6, -1.9), xytext=(5.4, -1.9),
        arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4", color="black"),
    )
    ax.text(1.4, -1.65, "même fenêtre temps-fréquence", ha="center",
            va="bottom", fontsize=8)

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
    """Trace la heatmap annotée de ``|G| = |H @ W_norm|``.

    La diagonale vaut ``alpha`` (gain utile après normalisation de la
    puissance totale) et non 1 : la normalisation réduit toutes les
    colonnes d'un facteur commun ``alpha``. Les termes hors diagonale
    mesurent les interférences résiduelles (nuls à la précision
    numérique pour un ZF idéal).

    Parameters
    ----------
    g : numpy.ndarray
        Matrice effective ``(K, K)``.
    alpha : float
        Facteur de normalisation de puissance.
    chemin : pathlib.Path
        Fichier de sortie (PNG).

    Returns
    -------
    pathlib.Path
        Chemin de la figure enregistrée.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    g_module = np.abs(np.asarray(g))
    k = g_module.shape[0]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    image = ax.imshow(g_module, cmap="viridis", interpolation="nearest", aspect="equal")
    barre = fig.colorbar(image, ax=ax)
    barre.set_label(r"Module $|G_{k,j}|$ (gain utile si $k=j$, interférence si $k\neq j$)")

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
        r"Heatmap de $|G| = |H\,W_{\mathrm{norm}}|$"
        "\n"
        f"Diagonale = gain utile $\\alpha$ = {alpha:.6f} ; hors diagonale ≈ 0 (interférences annulées)"
    )

    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 4 : constellations BPSK
# ------------------------------------------------------------

def figure_constellations(
    s_bloc: np.ndarray,
    y_sans_bruit: np.ndarray,
    y_bruit: np.ndarray,
    alpha: float,
    chemin: Path,
    n_max_points: int = 1200,
) -> Path:
    """Affiche les constellations BPSK émises et reçues (complexe).

    Le plan complexe est représenté en entier : comme le canal et le
    précodage sont complexes, les composantes réelle et imaginaire du
    signal reçu sont tracées afin de ne pas masquer un éventuel biais de
    phase. En pratique, le ZF annule la composante imaginaire du canal
    effectif (``G = alpha I_K`` réel) : les symboles reçus sans bruit se
    concentrent sur l'axe réel aux valeurs ``± alpha`` ; avec bruit, le
    « nuage » s'étale dans les deux dimensions.

    Parameters
    ----------
    s_bloc : numpy.ndarray
        Symboles émis ``(K, T)``.
    y_sans_bruit : numpy.ndarray
        Symboles reçus sans bruit ``(K, T)``.
    y_bruit : numpy.ndarray
        Symboles reçus avec bruit ``(K, T)``.
    alpha : float
        Gain utile diagonal après normalisation.
    chemin : pathlib.Path
        Fichier de sortie (PNG).
    n_max_points : int
        Nombre maximal de points affichés par utilisateur (sous-échantillonnage
        pour la lisibilité).

    Returns
    -------
    pathlib.Path
        Chemin de la figure enregistrée.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    s_bloc = np.asarray(s_bloc)
    y_sans_bruit = np.asarray(y_sans_bruit)
    y_bruit = np.asarray(y_bruit)
    k, t = s_bloc.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sous-échantillonnage déterministe pour la lisibilité.
    indices = np.linspace(0, t - 1, min(n_max_points, t)).astype(int)

    series = [
        (axes[0], s_bloc[:, indices].real,
         "Symboles émis\n(valors ±1)"),
        (axes[1], y_sans_bruit[:, indices],
         r"Reçus sans bruit : $G\,S = \alpha S$"
         f"\n(gains ≈ ±{alpha:.3f})"),
        (axes[2], y_bruit[:, indices],
         "Reçus avec bruit\n(nuage complexe)"),
    ]

    for axe, donnees, titre in series:
        for i in range(k):
            if donnees.ndim == 2 and np.iscomplexobj(donnees):
                axe.scatter(donnees[i].real, donnees[i].imag, s=4, alpha=0.5,
                            color=COULEURS_UTILISATEURS[i % len(COULEURS_UTILISATEURS)],
                            label=f"Utilisateur {i + 1}")
            else:
                axe.scatter(donnees[i], np.zeros_like(donnees[i]), s=4, alpha=0.5,
                            color=COULEURS_UTILISATEURS[i % len(COULEURS_UTILISATEURS)],
                            label=f"Utilisateur {i + 1}")
        axe.axhline(0, color="grey", linewidth=0.6)
        axe.axvline(0, color="grey", linewidth=0.6)
        axe.set_aspect("equal")
        axe.set_xlabel("Partie réelle")
        axe.set_ylabel("Partie imaginaire")
        axe.set_title(titre)
        axe.grid(alpha=0.25)

    axes[0].legend(loc="upper center", bbox_to_anchor=(1.6, -0.18), ncol=k, fontsize=8)
    fig.suptitle(
        "Constellations BPSK : émission, réception sans bruit et avec bruit "
        "(canal complexe + précodage ZF)",
        fontsize=12,
    )

    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 5 : BER en fonction du SNR
# ------------------------------------------------------------

def figure_ber_snr(resultats: dict, chemin: Path) -> Path:
    """Trace le BER en fonction du SNR (ZF vs sans précodage).

    Parameters
    ----------
    resultats : dict
        Sortie de :func:`simulations.courbe_ber_snr`.
    chemin : pathlib.Path
        Fichier de sortie (PNG).

    Returns
    -------
    pathlib.Path
        Chemin de la figure enregistrée.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    snr = np.asarray(resultats["snr_db"], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    for cle, etiquette, couleur in (
        ("zf", "Avec précodage ZF", "#1f77b4"),
        ("sans_precodage", "Sans précodage (1 antenne/utilisateur, même $P_T$)", "#ff7f0e"),
    ):
        valeurs, masque_nul, bornes = valeurs_ber_affichables(resultats[cle])
        ax.semilogy(snr, valeurs, "o-", color=couleur, label=etiquette)
        _annoter_zeros_ber(ax, snr, valeurs, masque_nul, bornes)

    ax.set_xlabel("SNR en dB  (convention : $\\sigma^2 = P_T / 10^{\\mathrm{SNR_{dB}}/10}$)")
    ax.set_ylabel("BER (taux d'erreur binaire)")
    ax.set_title(
        f"BER en fonction du SNR — MU-MISO : M = {resultats['m']} antennes, "
        f"K = {resultats['k']} utilisateurs"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()

    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 6 : BER en fonction du nombre d'antennes
# ------------------------------------------------------------

def figure_ber_antennes(resultats: dict, k: int, chemin: Path) -> Path:
    """Trace le BER en fonction du nombre d'antennes ``M`` (``K`` fixé).

    Parameters
    ----------
    resultats : dict
        Sortie de :func:`simulations.courbe_ber_antennes`.
    k : int
        Nombre d'utilisateurs maintenu fixe (il faut ``M >= K``).
    chemin : pathlib.Path
        Fichier de sortie (PNG).

    Returns
    -------
    pathlib.Path
        Chemin de la figure enregistrée.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    valeurs_m = np.asarray(resultats["valeurs_m"], dtype=float)
    ber = np.array([r.ber for r in resultats["zf"]])
    bornes = np.array([r.borne_observation for r in resultats["zf"]])

    fig, ax = plt.subplots(figsize=(9, 6))
    ber_aff = np.where(ber > 0, ber, np.where(bornes > 0, 0.5 * bornes, 1e-12))
    ax.semilogy(valeurs_m, ber_aff, "s-", color="#1f77b4", label="ZF")

    masque_nul = ber == 0
    if np.any(masque_nul):
        ax.scatter(valeurs_m[masque_nul], ber_aff[masque_nul], s=70, marker="v",
                   facecolors="none", edgecolors="crimson", zorder=5,
                   label="BER observé nul (limite d'observation)")
        ax.annotate(
            "▼ : aucun bit erroné observé ; le BER réel est inférieur à "
            "≈ 1 / N_bits, pas théoriquement nul.",
            xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8, color="crimson",
        )

    snr = resultats.get("snr_db")
    ax.set_xlabel("Nombre d'antennes d'émission M")
    ax.set_ylabel("BER (taux d'erreur binaire)")
    ax.set_title(
        f"BER du ZF en fonction du nombre d'antennes M — K = {k} utilisateurs fixes, "
        f"SNR = {snr} dB\n"
        "La contrainte M ≥ K est indispensable au ZF classique ; "
        "plus M augmente, plus le gain de diversité spatiale est favorable."
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 7 : BER en fonction du nombre d'utilisateurs
# ------------------------------------------------------------

def figure_ber_utilisateurs(resultats: dict, m: int, chemin: Path) -> Path:
    """Trace le BER en fonction du nombre d'utilisateurs ``K`` (``M`` fixé).

    Parameters
    ----------
    resultats : dict
        Sortie de :func:`simulations.courbe_ber_utilisateurs`.
    m : int
        Nombre d'antennes maintenu fixe.
    chemin : pathlib.Path
        Fichier de sortie (PNG).

    Returns
    -------
    pathlib.Path
        Chemin de la figure enregistrée.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    valeurs_k = np.asarray(resultats["valeurs_k"], dtype=float)
    ber = np.array([r.ber for r in resultats["zf"]])
    bornes = np.array([r.borne_observation for r in resultats["zf"]])

    fig, ax = plt.subplots(figsize=(9, 6))
    ber_aff = np.where(ber > 0, ber, np.where(bornes > 0, 0.5 * bornes, 1e-12))
    ax.semilogy(valeurs_k, ber_aff, "o-", color="#2ca02c", label="ZF")

    masque_nul = ber == 0
    if np.any(masque_nul):
        ax.scatter(valeurs_k[masque_nul], ber_aff[masque_nul], s=70, marker="v",
                   facecolors="none", edgecolors="crimson", zorder=5,
                   label="BER observé nul (limite d'observation)")
        ax.annotate(
            "▼ : aucun bit erroné observé ; le BER réel est inférieur à "
            "≈ 1 / N_bits, pas théoriquement nul.",
            xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8, color="crimson",
        )

    snr = resultats.get("snr_db")
    ax.set_xlabel("Nombre d'utilisateurs K")
    ax.set_ylabel("BER (taux d'erreur binaire)")
    ax.set_title(
        f"BER du ZF en fonction du nombre d'utilisateurs K — M = {m} antennes fixes, "
        f"SNR = {snr} dB\n"
        "La difficulté dépend du rang et du conditionnement du canal "
        "(bruit de recodage), pas seulement du rapport M/K."
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin


# ------------------------------------------------------------
# Figure 8 : canaux corrélés
# ------------------------------------------------------------

def figure_canaux_correles(resultats: dict, chemin: Path) -> Path:
    """Compare les performances ZF selon la corrélation inter-antennes.

    Parameters
    ----------
    resultats : dict
        Sortie de :func:`simulations.etude_canaux_correles`.
    chemin : pathlib.Path
        Fichier de sortie (PNG).

    Returns
    -------
    pathlib.Path
        Chemin de la figure enregistrée.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)

    snr = np.asarray(resultats["snr_db"], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))

    for indice_rho, rho in enumerate(resultats["rho"]):
        serie = resultats["ber"][indice_rho]
        valeurs, masque_nul, bornes = valeurs_ber_affichables(serie)
        ax.semilogy(snr, valeurs, "o-", color=COULEURS_UTILISATEURS[indice_rho % len(COULEURS_UTILISATEURS)],
                    label=f"$\\rho$ = {rho:g} ({'i.i.d.' if rho == 0 else 'corrélé'})")
        _annoter_zeros_ber(ax, snr, valeurs, masque_nul, bornes)

    ax.set_xlabel("SNR en dB")
    ax.set_ylabel("BER (taux d'erreur binaire)")
    ax.set_title(
        "Effet de la corrélation entre antennes d'émission sur le BER du ZF\n"
        "Modèle exponentiel $R_{m,n} = \\rho^{|m-n|}$ (Kronecker côté émission) : "
        "la corrélation dégrade le conditionnement de H et amplifie le bruit de recodage."
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(chemin, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return chemin
