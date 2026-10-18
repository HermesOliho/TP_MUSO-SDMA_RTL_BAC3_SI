"""Configuration centralisée du TP MU-MISO / SDMA / Zero-Forcing.

Tous les paramètres de simulation sont regroupés dans la dataclass
``ConfigurationTP``. Modifier un seul endroit suffit pour changer la
configuration complète du TP : nombre d'antennes ``M``, nombre
d'utilisateurs ``K``, puissance totale ``P_T``, valeurs de SNR, graine
aléatoire, taille des expériences Monte-Carlo, etc.

Les chemins des fichiers sont toujours calculés relativement à la
racine du projet, jamais à partir du répertoire courant supposé.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------
# Racine du projet et chemins de référence
# ------------------------------------------------------------

# Racine du projet : le dossier qui contient ``src/``, ``data/``...
RACINE_PROJET = Path(__file__).resolve().parent.parent

# Dossier contenant les quatre images JPEG sources.
DOSSIER_ORIGINALES = RACINE_PROJET / "data" / "originales"

# Dossier de destination des images reconstruites.
DOSSIER_RECONSTRUITES = RACINE_PROJET / "data" / "reconstruites"

# Dossier de destination des figures.
DOSSIER_FIGURES = RACINE_PROJET / "figures"

# Noms des quatre images sources (un utilisateur par image).
NOMS_IMAGES_DEFAUT = ("image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg")


# ------------------------------------------------------------
# Dataclass de configuration
# ------------------------------------------------------------

@dataclass
class ConfigurationTP:
    """Paramètres complets d'une simulation du TP.

    Attributes
    ----------
    m : int
        Nombre d'antennes d'émission de la station de base (défaut : 8).
    k : int
        Nombre d'utilisateurs, c'est-à-dire de flux spatiaux simultanés
        (défaut : 4). ``k`` ne peut pas dépasser le nombre d'images
        disponibles pour les expériences sur images.
    p_t : float
        Puissance moyenne totale d'émission cible (défaut : 1.0).
        C'est une puissance *totale*, répartie sur les ``K`` flux et les
        ``M`` antennes par le précodage, pas une puissance par antenne.
    snr_db : float
        SNR en dB utilisé pour la transmission « démonstrative » avec
        bruit (reconstruction des images). Convention :
        ``sigma2 = P_T / 10**(snr_db/10)``.
    graine : int
        Graine aléatoire de référence. Chaque expérience dérive sa
        propre graine à partir de cette valeur pour être reproductible
        sans réutilisation involontaire de réalisations entre points.
    modulation : str
        Modulation de la transmission démonstrative : ``"BPSK"``,
        ``"4-QAM"`` (alias ``"QPSK"``) ou ``"16-QAM"``. Toutes les
        constellations sont normalisées à une puissance moyenne
        unitaire.
    snr_db_courbe : tuple
        Valeurs de SNR en dB de la courbe BER en fonction du SNR
        (expérience « étude », mode complet). Le TP impose
        ``{0, 5, 10, 15, 20, 25, 30}``.
    snr_db_correles : tuple
        Valeurs de SNR en dB de l'étude des canaux corrélés.
    n_bits_courbe : int
        Nombre de bits simulé par utilisateur et par point de courbe
        Monte-Carlo, pour chaque réalisation de canal.
    n_realisations : int
        Nombre de réalisations indépendantes du canal par point de
        courbe. Le canal est renouvelé à chaque réalisation.
    valeurs_m : tuple
        Valeurs de ``M`` étudiées (le TP impose ``{4, 8, 16, 32, 64}`` ;
        il faut ``M >= K`` pour le ZF classique). Les valeurs
        incompatibles avec le ``K`` courant sont écartées (voir
        ``valeurs_m_etudiees``).
    valeurs_k : tuple
        Valeurs de ``K`` étudiées dans la courbe BER en fonction du
        nombre d'utilisateurs (le TP impose ``{2, 4, 8, 12, 16}`` avec
        ``M`` fixé à ``m_etude_k``). Les valeurs incompatibles avec
        ``m_etude_k`` sont écartées (voir ``valeurs_k_etudiees``). Ces
        expériences utilisent des bits aléatoires, pas les images.
    modulations_etude : tuple
        Modulations comparées dans les études (``BPSK``, ``4-QAM``,
        ``16-QAM``).
    rho_correles : tuple
        Coefficients de corrélation exponentielle entre antennes
        d'émission étudiés dans l'étude des canaux corrélés.
    valeurs_beta : tuple
        Valeurs du degré de similitude ``beta`` entre les deux premiers
        canaux utilisateurs (``0`` : canaux indépendants ; ``beta -> 1`` :
        canaux quasi identiques).
    m_etude_k : int
        Nombre d'antennes fixé dans l'étude « BER en fonction de K »
        (le TP impose ``M = 16``).
    snr_db_etude_m : float
        SNR (dB) fixé dans les études « BER en fonction de M » et « BER
        en fonction de K ».
    snr_reconstruction : tuple
        Valeurs de SNR (dB) auxquelles les images sont reconstruites
        pour les trois modulations (le TP impose ``{5, 15, 25}``).
    n_bits_conditionnement : int
        Nombre de bits par utilisateur et par réalisation dans l'étude
        du conditionnement.
    n_realisations_conditionnement : int
        Nombre de réalisations indépendantes du canal dans l'étude du
        conditionnement ``kappa(H H^H)`` contre le BER.
    noms_images : tuple
        Noms des fichiers images, dans l'ordre des utilisateurs.
    """

    m: int = 8
    k: int = 4
    p_t: float = 1.0
    snr_db: float = 10.0
    graine: int = 42

    # Modulation utilisée pour la transmission démonstrative et les études
    # par défaut ("BPSK", "4-QAM" ou "16-QAM").
    modulation: str = "BPSK"

    # Expériences du mode « complet » (études Monte-Carlo).
    snr_db_courbe: tuple = (0, 5, 10, 15, 20, 25, 30)
    snr_db_correles: tuple = (0, 5, 10, 15, 20)
    n_bits_courbe: int = 50_000
    n_realisations: int = 10
    valeurs_m: tuple = (4, 8, 16, 32, 64)
    valeurs_k: tuple = (2, 4, 8, 12, 16)
    modulations_etude: tuple = ("BPSK", "4-QAM", "16-QAM")
    rho_correles: tuple = (0.0, 0.5, 0.9)
    valeurs_beta: tuple = (0.0, 0.5, 0.9, 0.99)

    # Expérience « nombre d'utilisateurs » : M fixé à une valeur
    # supérieure à K (le TP impose M = 16), et SNR d'étude.
    m_etude_k: int = 16
    snr_db_etude_m: float = 15.0

    # Reconstruction des images à plusieurs SNR (questionnaire).
    snr_reconstruction: tuple = (5.0, 15.0, 25.0)

    # Étude du conditionnement (kappa) et du BER par réalisation.
    n_bits_conditionnement: int = 50_000
    n_realisations_conditionnement: int = 200

    # Images et chemins.
    noms_images: tuple = NOMS_IMAGES_DEFAUT
    dossier_originales: Path = field(default=DOSSIER_ORIGINALES)
    dossier_reconstruites: Path = field(default=DOSSIER_RECONSTRUITES)
    dossier_figures: Path = field(default=DOSSIER_FIGURES)

    # --------------------------------------------------------
    # Validation de la configuration
    # ------------------------------------------------------------

    def valider(self) -> None:
        """Vérifie la cohérence de la configuration.

        Raises
        ------
        ValueError
            Si une valeur est incohérente avec les hypothèses du ZF
            classique (``1 <= K <= M``, puissance positive, etc.).
        """
        if self.m < 1:
            raise ValueError(f"Le nombre d'antennes M doit être >= 1 (M={self.m}).")
        if self.k < 1:
            raise ValueError(f"Le nombre d'utilisateurs K doit être >= 1 (K={self.k}).")
        if self.k > self.m:
            raise ValueError(
                f"Le précodage ZF classique exige K <= M "
                f"(K={self.k}, M={self.m})."
            )
        if self.p_t <= 0:
            raise ValueError(f"La puissance totale P_T doit être positive (P_T={self.p_t}).")
        if self.n_bits_courbe < 1:
            raise ValueError("n_bits_courbe doit être strictement positif.")
        if self.n_realisations < 1:
            raise ValueError("n_realisations doit être strictement positif.")
        if not self.valeurs_m_etudiees:
            raise ValueError(
                "Aucune valeur de M étudiée n'est compatible avec K "
                f"(K={self.k}, valeurs_m={self.valeurs_m})."
            )
        if not self.valeurs_k_etudiees:
            raise ValueError(
                "Aucune valeur de K étudiée n'est compatible avec M "
                f"(M d'étude = {self.m_etude_k}, valeurs_k={self.valeurs_k})."
            )
        # Vérifie que la modulation demandée est supportée.
        from .modulation import normaliser_nom_modulation

        normaliser_nom_modulation(self.modulation)
        for nom in self.modulations_etude:
            normaliser_nom_modulation(nom)

    @property
    def valeurs_m_etudiees(self) -> tuple:
        """Valeurs de ``M`` compatibles avec le ``K`` courant (``M >= K``).

        Lorsque ``K`` est surchargé au-delà de certaines valeurs de la
        grille par défaut, ces valeurs sont simplement écartées de
        l'étude plutôt que de rendre la configuration invalide.
        """
        return tuple(m for m in self.valeurs_m if m >= self.k)

    @property
    def valeurs_k_etudiees(self) -> tuple:
        """Valeurs de ``K`` compatibles avec ``m_etude_k`` (``K <= M``)."""
        return tuple(k for k in self.valeurs_k if k <= self.m_etude_k)


def configuration_rapide(**surcharge) -> ConfigurationTP:
    """Retourne la configuration du mode rapide.

    Le mode rapide valide toute la chaîne (images -> bits -> BPSK ->
    canal -> ZF -> détection -> reconstruction) sans lancer les
    expériences Monte-Carlo coûteuses, qui ne sont exécutées qu'en mode
    complet.

    Parameters
    ----------
    **surcharge
        Paramètres à remplacer dans la configuration par défaut.
    """
    config = ConfigurationTP(**surcharge)
    config.valider()
    return config


def configuration_complete(**surcharge) -> ConfigurationTP:
    """Retourne la configuration du mode complet (études Monte-Carlo).

    Parameters
    ----------
    **surcharge
        Paramètres à remplacer dans la configuration par défaut.
    """
    config = ConfigurationTP(**surcharge)
    config.valider()
    return config
