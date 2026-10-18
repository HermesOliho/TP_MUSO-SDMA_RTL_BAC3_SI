"""Expériences numériques du TP : transmissions et courbes BER.

Toutes les simulations sont reproductibles à graine fixée : chaque
expérience utilise son propre générateur dérivé de la graine de la
configuration, afin de ne pas réutiliser involontairement les mêmes
données ou le même bruit entre deux points de courbe.

Conventions
-----------
- Modèle : ``Y = H @ W_norm @ S + N``, avec ``S`` de forme ``(K, T)``,
  ``W_norm`` de forme ``(M, K)``, ``H`` de forme ``(K, M)``.
- SNR : ``sigma2 = P_T / 10**(SNR_dB / 10)`` (voir ``canal.py``).
- Canal de Rayleigh ``CN(0, 1)`` par coefficient, renouvelé à chaque
  réalisation Monte-Carlo, connu parfaitement par l'émetteur (CSI
  parfait), constant pendant un bloc.
- Pour les expériences BER, le canal est donc renouvelé à chaque
  réalisation, les bits sont tirés aléatoirement (loi uniforme), et le
  BER agrège erreurs et bits sur l'ensemble des réalisations.
- Les flux d'images de longueurs différentes sont alignés par
  remplissage ; les symboles de remplissage sont masqués et ne sont
  jamais comptés dans le BER.

Comparaison ZF / sans précodage
-------------------------------
Le scénario « sans précodage » attribue une antenne d'émission à
chaque utilisateur (les ``K`` premières antennes) avec une part de
puissance ``P_T / K`` par flux, soit ``W = sqrt(P_T / K) * [I_K ; 0]``.
C'est la même puissance totale d'émission ``P_T``, le même canal, le
même bruit et le même détecteur (décision de signe) que pour le ZF :
la comparaison est équitable. Comme le canal est complexe, le
récepteur compense la phase de son propre coefficient de canal
``H[k, k]`` (CSI récepteur, hypothèse explicite) avant la décision —
sans quoi la décision sur la seule partie réelle serait erronée même
en l'absence de bruit. Sans précodage, chaque utilisateur reçoit les
interférences des ``K - 1`` autres flux, ce qui produit un plancher de
BER (interférence inter-utilisateurs, indépendant du SNR) que le ZF
supprime.
"""

from dataclasses import dataclass

import numpy as np

from .canal import (
    appliquer_correlation_emission,
    generer_canal,
    sigma2_depuis_snr_db,
)
from .detection import ResultatReception, evaluer_reception, transmettre
from .images import convertir_en_bits, reconstruire_image
from .modulation import moduler_bpsk
from .precodage_zf import diagnostics_canal, normaliser_puissance, precodage_zf


# ------------------------------------------------------------
# Alignement des flux de longueurs différentes
# ------------------------------------------------------------

def aligner_flux(symboles_par_utilisateur: list) -> tuple:
    """Aligne les flux de symboles de longueurs différentes en un bloc.

    Les flux sont padés à la longueur maximale par des symboles ``+1``
    (bit 1 arbitraire de remplissage). Un masque booléen indique les
    positions réellement transmises.

    Parameters
    ----------
    symboles_par_utilisateur : list of numpy.ndarray
        Symboles BPSK de chaque utilisateur, séquences de tailles
        possiblement différentes.

    Returns
    -------
    tuple
        ``(S, masque)`` : bloc de symboles ``(K, T_max)`` et masque de
        validité ``(K, T_max)``.

    Raises
    ------
    ValueError
        Si la liste est vide ou si un flux est vide.
    """
    if not symboles_par_utilisateur:
        raise ValueError("Aucun flux de symboles à aligner.")

    longueurs = [len(np.asarray(s).ravel()) for s in symboles_par_utilisateur]
    if min(longueurs) < 1:
        raise ValueError("Un flux de symboles est vide : transmission impossible.")

    k = len(symboles_par_utilisateur)
    t_max = max(longueurs)

    # Remplissage par défaut : symbole +1 (puissance unitaire comme les autres).
    s_bloc = np.ones((k, t_max), dtype=np.float64)
    masque = np.zeros((k, t_max), dtype=bool)

    for indice, symboles in enumerate(symboles_par_utilisateur):
        symboles = np.asarray(symboles, dtype=np.float64).ravel()
        longueur = symboles.size
        s_bloc[indice, :longueur] = symboles
        masque[indice, :longueur] = True

    return s_bloc, masque


# ------------------------------------------------------------
# Transmission complète des images
# ------------------------------------------------------------

@dataclass
class ResultatTransmission:
    """Bilan d'une transmission MU-MISO des images.

    Attributes
    ----------
    h, w_norm, alpha, g : numpy.ndarray
        Canal, précodage normalisé, facteur de normalisation et matrice
        effective correspondants.
    s_bloc, masque : numpy.ndarray
        Bloc de symboles alignés et masque de validité.
    y_sans_bruit, y : numpy.ndarray
        Signaux reçus sans bruit (``G @ S``) et avec bruit
        (``sigma2`` donné). ``y`` est égal à ``y_sans_bruit`` lorsque
        ``sigma2 == 0``.
    bits_emis : list of numpy.ndarray
        Bits transmis par utilisateur.
    reception : ResultatReception
        BER par utilisateur, BER global et bits reçus valides.
    images_reconstruites : list of numpy.ndarray
        Images reconstruites par utilisateur (uniquement bits valides).
    identiques : list of bool
        Comparaison pixel par pixel avec l'image source en niveaux de
        gris.
    diagnostics : DiagnosticsCanal
        Diagnostics du canal utilisé.
    """

    h: np.ndarray
    w_norm: np.ndarray
    alpha: float
    g: np.ndarray
    s_bloc: np.ndarray
    masque: np.ndarray
    y_sans_bruit: np.ndarray
    y: np.ndarray
    bits_emis: list
    reception: ResultatReception
    images_reconstruites: list
    identiques: list
    diagnostics: object


def transmettre_images(
    images_gris: list,
    h: np.ndarray,
    w_norm: np.ndarray,
    alpha: float,
    sigma2: float = 0.0,
    rng: np.random.Generator | None = None,
) -> ResultatTransmission:
    """Transmet les images de tous les utilisateurs à travers le modèle MU-MISO.

    Chaque image est convertie en bits puis en symboles BPSK ; les
    quatre flux sont alignés par remplissage, transmis simultanément à
    travers ``Y = H @ W_norm @ S + N``, détectés par décision de signe,
    puis reconstruits image par image à partir des seuls bits valides.

    Parameters
    ----------
    images_gris : list of numpy.ndarray
        Images sources en niveaux de gris (dimensions quelconques,
        possiblement différentes).
    h : numpy.ndarray
        Canal ``(K, M)``.
    w_norm : numpy.ndarray
        Précodage ZF normalisé ``(M, K)``.
    alpha : float
        Facteur de normalisation (gain utile diagonal de ``G``).
    sigma2 : float
        Variance du bruit complexe (0 pour une transmission sans
        bruit).
    rng : numpy.random.Generator, optional
        Générateur du bruit, requis si ``sigma2 > 0``.

    Returns
    -------
    ResultatTransmission
        Bilan complet de la transmission.
    """
    # 1. Images -> bits -> symboles BPSK, par utilisateur.
    bits_emis = [convertir_en_bits(image) for image in images_gris]
    symboles = [moduler_bpsk(bits) for bits in bits_emis]

    # 2. Alignement des flux de longueurs différentes.
    s_bloc, masque = aligner_flux(symboles)

    k = s_bloc.shape[0]
    if h.shape[0] != k:
        raise ValueError(
            f"Nombre d'utilisateurs incohérent : {k} flux d'images pour K = {h.shape[0]} "
            "lignes de canal."
        )

    # 3. Canal effectif puis transmission.
    g = h @ w_norm
    y_sans_bruit = g @ s_bloc
    y = transmettre(s_bloc, h, w_norm, sigma2=sigma2, rng=rng)

    # 4. Détection et BER (le remplissage est exclu par le masque).
    reception = evaluer_reception(bits_emis, y, masque)

    # 5. Reconstruction image par image, à partir des bits valides.
    images_reconstruites = []
    identiques = []
    for indice, image_source in enumerate(images_gris):
        hauteur, largeur = image_source.shape
        image_rec = reconstruire_image(
            reception.bits_recus[indice], hauteur, largeur
        )
        images_reconstruites.append(image_rec)
        identiques.append(bool(np.array_equal(image_source, image_rec)))

    return ResultatTransmission(
        h=h,
        w_norm=w_norm,
        alpha=alpha,
        g=g,
        s_bloc=s_bloc,
        masque=masque,
        y_sans_bruit=y_sans_bruit,
        y=y,
        bits_emis=bits_emis,
        reception=reception,
        images_reconstruites=images_reconstruites,
        identiques=identiques,
        diagnostics=diagnostics_canal(h),
    )


# ------------------------------------------------------------
# Points Monte-Carlo pour les courbes BER
# ------------------------------------------------------------

@dataclass
class ResultatPoint:
    """Résultat Monte-Carlo d'un point de courbe BER.

    Attributes
    ----------
    ber : float
        BER observé (erreurs effectives / bits comparés). Un BER observé
        nul ne prouve pas que le BER théorique est nul : consulter
        ``nb_bits`` pour la limite d'observation.
    erreurs : int
        Nombre total d'erreurs observées.
    nb_bits : int
        Nombre total de bits comparés sur l'ensemble des réalisations.
    """

    ber: float
    erreurs: int
    nb_bits: int

    @property
    def borne_observation(self) -> float:
        """Plus petit BER mesurable avec cette taille d'échantillon.

        En pratique, un BER observé nul situe le BER réel sous cet
        ordre de grandeur, avec une incertitude statistique.
        """
        if self.nb_bits == 0:
            return float("nan")
        return 1.0 / self.nb_bits


def simuler_ber_point(
    m: int,
    k: int,
    snr_db: float,
    p_t: float = 1.0,
    n_bits: int = 100_000,
    n_realisations: int = 10,
    rng: np.random.Generator | None = None,
    rho: float = 0.0,
    methode: str = "zf",
) -> ResultatPoint:
    """Simule un point Monte-Carlo du BER (par utilisateur moyen).

    À chaque réalisation : canal de Rayleigh renouvelé (optionnellement
    corrélé entre antennes d'émission), bits tirés aléatoirement,
    précodage recalculé (CSI parfait), transmission avec bruit,
    détection BPSK. Le BER agrège les erreurs et les bits de toutes les
    réalisations.

    Parameters
    ----------
    m : int
        Nombre d'antennes d'émission.
    k : int
        Nombre d'utilisateurs.
    snr_db : float
        SNR en dB (convention ``sigma2 = P_T / 10^(SNR_dB/10)``).
    p_t : float
        Puissance moyenne totale d'émission.
    n_bits : int
        Bits par utilisateur et par réalisation.
    n_realisations : int
        Nombre de réalisations du canal.
    rng : numpy.random.Generator
        Générateur reproductible.
    rho : float
        Coefficient de corrélation entre antennes d'émission (0 pour un
        canal i.i.d.).
    methode : str
        ``"zf"`` : précodage Zero-Forcing normalisé. ``"aucun"`` :
        sans précodage, une antenne par utilisateur, même puissance
        totale (comparaison équitable, voir l'en-tête du module) ; le
        récepteur compense la phase de son coefficient de canal
        ``H[k, k]`` (CSI récepteur) avant la décision BPSK.

    Returns
    -------
    ResultatPoint
        BER observé, erreurs et nombre total de bits comparés.
    """
    if rng is None:
        raise ValueError("Un générateur aléatoire (rng) est requis.")
    if m < k:
        raise ValueError(
            f"Configuration invalide pour cette expérience : M = {m} < K = {k}."
        )

    sigma2 = sigma2_depuis_snr_db(snr_db, p_t)
    erreurs_totales = 0
    bits_totaux = 0

    for _ in range(n_realisations):
        # Canal renouvelé à chaque réalisation, connu parfaitement.
        h = generer_canal(k, m, rng)
        if rho > 0.0:
            h = appliquer_correlation_emission(h, rho)

        # Bits tirés uniformément, modulés en BPSK.
        bits = rng.integers(0, 2, size=(k, n_bits)).astype(np.uint8)
        s = 2.0 * bits.astype(np.float64) - 1.0

        if methode == "zf":
            w_zf = precodage_zf(h)
            w_norm, _ = normaliser_puissance(w_zf, p_t)
        elif methode == "aucun":
            # Une antenne par utilisateur, puissance totale P_T répartie
            # à égalité : W = sqrt(P_T / K) [I_K ; 0].
            w_norm = np.zeros((m, k), dtype=np.float64)
            np.fill_diagonal(w_norm, np.sqrt(p_t / k))
        else:
            raise ValueError(
                f"Méthode inconnue : '{methode}'. Choisissez 'zf' ou 'aucun'."
            )

        y = transmettre(s, h, w_norm, sigma2=sigma2, rng=rng)

        if methode == "aucun":
            # Hypothèse explicite du scénario sans précodage : chaque
            # récepteur connaît son coefficient de canal H[k, k] et
            # compense sa phase avant la décision (sinon la décision
            # sur la seule partie réelle serait erronée même sans bruit,
            # le canal étant complexe). Cette compensation est l'équivalent
            # récepteur du travail de phase accompli par le ZF côté émission.
            phases = np.angle(np.diagonal(h))
            y = y * np.exp(-1j * phases)[:, np.newaxis]

        bits_decides = (np.real(y) >= 0.0).astype(np.uint8)

        erreurs_totales += int(np.count_nonzero(bits_decides != bits))
        bits_totaux += int(bits.size)

    ber = float(erreurs_totales) / bits_totaux if bits_totaux else 0.0
    return ResultatPoint(ber=ber, erreurs=erreurs_totales, nb_bits=bits_totaux)


# ------------------------------------------------------------
# Expériences du mode complet
# ------------------------------------------------------------

def courbe_ber_snr(config) -> dict:
    """Courbe BER en fonction du SNR : ZF vs sans précodage.

    Parameters
    ----------
    config : ConfigurationTP
        Configuration (``M``, ``K``, ``P_T``, valeurs de SNR, taille des
        expériences, graine).

    Returns
    -------
    dict
        ``{"snr_db": [...], "zf": [ResultatPoint...],
        "sans_precodage": [ResultatPoint...]}``
    """
    rng = np.random.default_rng(config.graine + 101)

    ber_zf = [
        simuler_ber_point(
            m=config.m, k=config.k, snr_db=snr, p_t=config.p_t,
            n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
            rng=rng, methode="zf",
        )
        for snr in config.snr_db_courbe
    ]
    ber_sans = [
        simuler_ber_point(
            m=config.m, k=config.k, snr_db=snr, p_t=config.p_t,
            n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
            rng=rng, methode="aucun",
        )
        for snr in config.snr_db_courbe
    ]

    return {
        "m": config.m,
        "k": config.k,
        "snr_db": list(config.snr_db_courbe),
        "zf": ber_zf,
        "sans_precodage": ber_sans,
    }


def courbe_ber_antennes(config) -> dict:
    """BER en fonction du nombre d'antennes ``M``, à SNR fixé.

    ``K`` et tous les autres paramètres sont maintenus constants. Seul
    ``M`` varie, avec la contrainte ``M >= K`` indispensable au ZF
    classique. Le SNR retenu est celui de la configuration.

    Parameters
    ----------
    config : ConfigurationTP
        Configuration du TP.

    Returns
    -------
    dict
        ``{"valeurs_m": [...], "zf": [ResultatPoint...], "snr_db": float}``
    """
    rng = np.random.default_rng(config.graine + 202)

    resultats = [
        simuler_ber_point(
            m=m, k=config.k, snr_db=config.snr_db, p_t=config.p_t,
            n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
            rng=rng, methode="zf",
        )
        for m in config.valeurs_m
    ]

    return {
        "valeurs_m": list(config.valeurs_m),
        "zf": resultats,
        "snr_db": config.snr_db,
    }


def courbe_ber_utilisateurs(config) -> dict:
    """BER en fonction du nombre d'utilisateurs ``K``, à SNR fixé.

    ``M`` est maintenu fixe. Cette expérience utilise des bits
    aléatoires (pas les images), donc ``K`` peut dépasser le nombre
    d'images disponibles. La difficulté du ZF dépend du rang et du
    conditionnement du canal, pas seulement du rapport ``M / K``.

    Parameters
    ----------
    config : ConfigurationTP
        Configuration du TP.

    Returns
    -------
    dict
        ``{"valeurs_k": [...], "zf": [ResultatPoint...], "snr_db": float}``
    """
    rng = np.random.default_rng(config.graine + 303)

    resultats = [
        simuler_ber_point(
            m=config.m, k=k, snr_db=config.snr_db, p_t=config.p_t,
            n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
            rng=rng, methode="zf",
        )
        for k in config.valeurs_k
    ]

    return {
        "valeurs_k": list(config.valeurs_k),
        "zf": resultats,
        "snr_db": config.snr_db,
    }


def etude_canaux_correles(config) -> dict:
    """Comparaison des performances ZF selon la corrélation du canal.

    Compare un canal i.i.d. (``rho = 0``) à des canaux corrélés entre
    antennes d'émission (modèle exponentiel ``R[i, j] = rho^|i-j|``,
    modèle de Kronecker côté émission). Plus la corrélation est forte,
    plus le conditionnement du canal se dégrade et plus le ZF amplifie
    le bruit (bruit de recodage).

    Parameters
    ----------
    config : ConfigurationTP
        Configuration du TP (listes ``rho_correles`` et
        ``snr_db_correles``).

    Returns
    -------
    dict
        ``{"rho": [...], "snr_db": [...], "ber": [[ResultatPoint...], ...]}``
    """
    rng = np.random.default_rng(config.graine + 404)

    ber_par_rho = []
    for rho in config.rho_correles:
        ber_par_rho.append([
            simuler_ber_point(
                m=config.m, k=config.k, snr_db=snr, p_t=config.p_t,
                n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
                rng=rng, rho=rho, methode="zf",
            )
            for snr in config.snr_db_correles
        ])

    return {
        "rho": list(config.rho_correles),
        "snr_db": list(config.snr_db_correles),
        "ber": ber_par_rho,
    }
