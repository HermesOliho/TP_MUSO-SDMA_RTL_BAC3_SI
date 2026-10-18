"""Expériences numériques du TP : transmissions, BER et études.

Toutes les simulations sont reproductibles à graine fixée : chaque
expérience utilise son propre générateur dérivé de la graine de la
configuration, afin de ne pas réutiliser involontairement les mêmes
données ou le même bruit entre deux points de courbe.

Conventions
-----------
- Modèle : ``Y = H @ W_norm @ S + N``, avec ``S`` de forme ``(K, T)``,
  ``W_norm`` de forme ``(M, K)``, ``H`` de forme ``(K, M)``.
- Égalisation avant démodulation : ``s_hat_k = y_k / g_k`` avec
  ``g_k = h_k^H w_k`` (diagonale de ``G``).
- SNR : ``sigma2 = P_T / 10**(SNR_dB / 10)`` (voir ``canal.py``).
- Canal de Rayleigh ``CN(0, 1)`` par coefficient, renouvelé à chaque
  réalisation Monte-Carlo, connu parfaitement par l'émetteur (CSI
  parfait), constant pendant un bloc.
- Pour les expériences BER, les bits sont tirés aléatoirement (loi
  uniforme) et le BER agrège erreurs et bits sur les réalisations.
- Les flux d'images de longueurs différentes sont alignés par
  remplissage ; les symboles de remplissage sont masqués et ne sont
  jamais comptés dans le BER.

Comparaison ZF / sans précodage
-------------------------------
Le scénario « sans précodage » attribue une antenne d'émission à
chaque utilisateur (les ``K`` premières antennes) avec une part de
puissance ``P_T / K`` par flux, soit ``W = sqrt(P_T / K) * [I_K ; 0]``.
C'est la même puissance totale d'émission ``P_T``, le même canal, le
même bruit et le même détecteur (après égalisation) que pour le ZF : la
comparaison est équitable. Sans précodage, chaque utilisateur subit les
interférences des ``K - 1`` autres flux, ce qui produit un plancher de
BER (interférence inter-utilisateurs, indépendant du SNR) que le ZF
supprime.
"""

from dataclasses import dataclass

import numpy as np

from .canal import (
    appliquer_correlation_emission,
    generer_canal,
    rendre_utilisateurs_similaires,
    sigma2_depuis_snr_db,
)
from .detection import ResultatReception, egaliser, evaluer_reception, gains_utiles, transmettre
from .images import convertir_en_bits, reconstruire_image
from .modulation import (
    bits_par_symbole,
    demoduler,
    moduler,
    normaliser_nom_modulation,
    table_constellation,
)
from .precodage_zf import (
    CanalSingularError,
    diagnostics_canal,
    normaliser_puissance,
    precodage_zf,
)


# ------------------------------------------------------------
# Alignement des flux de longueurs différentes
# ------------------------------------------------------------

def aligner_flux(symboles_par_utilisateur: list, valeur_remplissage: float = 1.0) -> tuple:
    """Aligne les flux de symboles de longueurs différentes en un bloc.

    Les flux sont padés à la longueur maximale par ``valeur_remplissage``
    (un symbole valide de la constellation utilisée). Un masque booléen
    indique les positions réellement transmises.

    Parameters
    ----------
    symboles_par_utilisateur : list of numpy.ndarray
        Symboles de chaque utilisateur, séquences de tailles
        possiblement différentes.
    valeur_remplissage : float or complex
        Symbole de remplissage (par défaut ``+1``).

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

    dtype = np.complex128 if any(np.iscomplexobj(s) for s in symboles_par_utilisateur) else np.float64
    s_bloc = np.full((k, t_max), complex(valeur_remplissage) if dtype == np.complex128
                     else float(np.real(valeur_remplissage)), dtype=dtype)
    masque = np.zeros((k, t_max), dtype=bool)

    for indice, symboles in enumerate(symboles_par_utilisateur):
        symboles = np.asarray(symboles).ravel()
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
    modulation : str
        Modulation utilisée.
    h, w_norm, alpha, g : numpy.ndarray
        Canal, précodage normalisé, facteur de normalisation et matrice
        effective correspondants.
    s_bloc, masque : numpy.ndarray
        Bloc de symboles alignés et masque de validité.
    y_sans_bruit, y, y_egalise : numpy.ndarray
        Signaux reçus sans bruit (``G @ S``), avec bruit, et après
        égalisation (``y / g_k``).
    bits_emis : list of numpy.ndarray
        Bits transmis par utilisateur.
    reception : ResultatReception
        BER par utilisateur, BER moyen, BER global et bits reçus.
    images_reconstruites : list of numpy.ndarray
        Images reconstruites par utilisateur (uniquement bits valides).
    identiques : list of bool
        Comparaison pixel par pixel avec l'image source en niveaux de
        gris.
    diagnostics : DiagnosticsCanal
        Diagnostics du canal utilisé.
    """

    modulation: str
    h: np.ndarray
    w_norm: np.ndarray
    alpha: float
    g: np.ndarray
    s_bloc: np.ndarray
    masque: np.ndarray
    y_sans_bruit: np.ndarray
    y: np.ndarray
    y_egalise: np.ndarray
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
    modulation: str = "BPSK",
) -> ResultatTransmission:
    """Transmet les images de tous les utilisateurs à travers le modèle MU-MISO.

    Chaque image est convertie en bits puis en symboles (BPSK, 4-QAM ou
    16-QAM) ; les flux sont alignés par remplissage, transmis
    simultanément à travers ``Y = H @ W_norm @ S + N``, égalisés
    (``y_k / g_k``), détectés par décision du plus proche voisin, puis
    reconstruits image par image à partir des seuls bits valides.

    Parameters
    ----------
    images_gris : list of numpy.ndarray
        Images sources en niveaux de gris (dimensions quelconques).
    h : numpy.ndarray
        Canal ``(K, M)``.
    w_norm : numpy.ndarray
        Précodage normalisé ``(M, K)``.
    alpha : float
        Facteur de normalisation (gain utile diagonal de ``G``).
    sigma2 : float
        Variance du bruit complexe (0 pour une transmission sans bruit).
    rng : numpy.random.Generator, optional
        Générateur du bruit, requis si ``sigma2 > 0``.
    modulation : str
        ``"BPSK"``, ``"4-QAM"`` ou ``"16-QAM"``.

    Returns
    -------
    ResultatTransmission
        Bilan complet de la transmission.
    """
    modulation = normaliser_nom_modulation(modulation)

    # 1. Images -> bits -> symboles, par utilisateur.
    bits_emis = [convertir_en_bits(image) for image in images_gris]
    symboles = [moduler(bits, modulation) for bits in bits_emis]

    # 2. Alignement des flux de longueurs différentes (remplissage valide).
    remplissage = table_constellation(modulation)[0]
    s_bloc, masque = aligner_flux(symboles, valeur_remplissage=remplissage)

    k = s_bloc.shape[0]
    if h.shape[0] != k:
        raise ValueError(
            f"Nombre d'utilisateurs incohérent : {k} flux d'images pour K = {h.shape[0]} "
            "lignes de canal."
        )

    # 3. Canal effectif, transmission et égalisation.
    g = h @ w_norm
    y_sans_bruit = g @ s_bloc
    y = transmettre(s_bloc, h, w_norm, sigma2=sigma2, rng=rng)
    y_egalise = egaliser(y, gains_utiles(h, w_norm))

    # 4. Détection et BER (le remplissage est exclu par le masque).
    reception = evaluer_reception(bits_emis, y_egalise, masque, modulation=modulation)

    # 5. Reconstruction image par image, à partir des bits valides.
    images_reconstruites = []
    identiques = []
    for indice, image_source in enumerate(images_gris):
        hauteur, largeur = image_source.shape
        image_rec = reconstruire_image(reception.bits_recus[indice], hauteur, largeur)
        images_reconstruites.append(image_rec)
        identiques.append(bool(np.array_equal(image_source, image_rec)))

    return ResultatTransmission(
        modulation=modulation,
        h=h,
        w_norm=w_norm,
        alpha=alpha,
        g=g,
        s_bloc=s_bloc,
        masque=masque,
        y_sans_bruit=y_sans_bruit,
        y=y,
        y_egalise=y_egalise,
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
        BER global observé (erreurs effectives / bits comparés). Un BER
        observé nul ne prouve pas que le BER théorique est nul :
        consulter ``nb_bits`` pour la limite d'observation.
    erreurs : int
        Nombre total d'erreurs observées.
    nb_bits : int
        Nombre total de bits comparés sur l'ensemble des réalisations.
    erreurs_par_utilisateur : numpy.ndarray, optional
        Erreurs par utilisateur (si le calcul par utilisateur a été
        demandé).
    nb_bits_par_utilisateur : numpy.ndarray, optional
        Bits comparés par utilisateur.
    """

    ber: float
    erreurs: int
    nb_bits: int
    erreurs_par_utilisateur: np.ndarray | None = None
    nb_bits_par_utilisateur: np.ndarray | None = None

    @property
    def borne_observation(self) -> float:
        """Plus petit BER non nul mesurable avec cet échantillon (1 / N)."""
        if self.nb_bits == 0:
            return float("nan")
        return 1.0 / self.nb_bits

    @property
    def ber_par_utilisateur(self) -> np.ndarray | None:
        """``BER_k`` par utilisateur, si disponible."""
        if self.erreurs_par_utilisateur is None or self.nb_bits_par_utilisateur is None:
            return None
        return self.erreurs_par_utilisateur / np.maximum(self.nb_bits_par_utilisateur, 1)

    @property
    def ber_moyen(self) -> float:
        """``BER_moy = (1/K) somme_k BER_k`` (formule du TP)."""
        par_utilisateur = self.ber_par_utilisateur
        if par_utilisateur is None:
            return self.ber
        return float(np.mean(par_utilisateur))


def _precodeur(h: np.ndarray, methode: str, p_t: float):
    """Retourne le précodeur normalisé selon la méthode demandée."""
    m, k = h.shape[1], h.shape[0]
    if methode == "zf":
        w_norm, _ = normaliser_puissance(precodage_zf(h), p_t)
        return w_norm
    if methode == "aucun":
        # Une antenne par utilisateur, puissance totale P_T à égalité.
        w_norm = np.zeros((m, k), dtype=np.complex128)
        np.fill_diagonal(w_norm, np.sqrt(p_t / k))
        return w_norm
    raise ValueError(f"Méthode inconnue : '{methode}'. Choisissez 'zf' ou 'aucun'.")


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
    modulation: str = "BPSK",
    beta_similarite: float = 0.0,
    par_utilisateur: bool = False,
) -> ResultatPoint:
    """Simule un point Monte-Carlo du BER.

    À chaque réalisation : canal de Rayleigh renouvelé (éventuellement
    corrélé entre antennes ou avec deux utilisateurs rendus similaires),
    bits tirés aléatoirement, modulation, précodage recalculé (CSI
    parfait), transmission avec bruit, égalisation ``y_k / g_k``,
    détection, comptage des erreurs.

    Parameters
    ----------
    m, k : int
        Nombre d'antennes et d'utilisateurs.
    snr_db : float
        SNR en dB (``sigma2 = P_T / 10^(SNR_dB/10)``).
    p_t : float
        Puissance moyenne totale d'émission.
    n_bits : int
        Bits par utilisateur et par réalisation.
    n_realisations : int
        Nombre de réalisations du canal.
    rng : numpy.random.Generator
        Générateur reproductible.
    rho : float
        Corrélation entre antennes d'émission (0 : i.i.d.).
    methode : str
        ``"zf"`` ou ``"aucun"`` (voir l'en-tête du module).
    modulation : str
        ``"BPSK"``, ``"4-QAM"`` ou ``"16-QAM"``.
    beta_similarite : float
        Degré de similitude entre les deux premiers utilisateurs (0 :
        indépendants).
    par_utilisateur : bool
        Si vrai, retourne aussi les erreurs par utilisateur.

    Returns
    -------
    ResultatPoint
        BER observé, erreurs et bits comparés (globaux et, sur demande,
        par utilisateur).
    """
    if rng is None:
        raise ValueError("Un générateur aléatoire (rng) est requis.")
    if m < k:
        raise ValueError(
            f"Configuration invalide pour cette expérience : M = {m} < K = {k}."
        )

    modulation = normaliser_nom_modulation(modulation)
    b = bits_par_symbole(modulation)
    n_bits_effectif = max(b, (int(n_bits) // b) * b)
    n_symboles = n_bits_effectif // b

    sigma2 = sigma2_depuis_snr_db(snr_db, p_t)
    erreurs_pu = np.zeros(k, dtype=np.int64)
    bits_pu = np.zeros(k, dtype=np.int64)

    for _ in range(n_realisations):
        # Canal renouvelé à chaque réalisation, connu parfaitement.
        h = generer_canal(k, m, rng)
        if rho > 0.0:
            h = appliquer_correlation_emission(h, rho)
        if beta_similarite > 0.0:
            h = rendre_utilisateurs_similaires(h, beta_similarite)

        bits = rng.integers(0, 2, size=(k, n_bits_effectif)).astype(np.uint8)
        s = moduler(bits, modulation)

        w_norm = _precodeur(h, methode, p_t)
        y = transmettre(s, h, w_norm, sigma2=sigma2, rng=rng)

        # Égalisation par le gain utile g_k = h_k^H w_k, puis détection.
        y_egalise = egaliser(y, gains_utiles(h, w_norm))
        decides = demoduler(y_egalise, modulation)

        erreurs_pu += np.count_nonzero(decides != bits, axis=1)
        bits_pu += n_bits_effectif

    erreurs_total = int(np.sum(erreurs_pu))
    bits_total = int(np.sum(bits_pu))
    ber = float(erreurs_total) / bits_total if bits_total else 0.0

    return ResultatPoint(
        ber=ber,
        erreurs=erreurs_total,
        nb_bits=bits_total,
        erreurs_par_utilisateur=erreurs_pu if par_utilisateur else None,
        nb_bits_par_utilisateur=bits_pu if par_utilisateur else None,
    )


# ------------------------------------------------------------
# Études du mode complet
# ------------------------------------------------------------

def courbe_ber_snr(config, modulation: str | None = None) -> dict:
    """Courbe BER en fonction du SNR : ZF vs sans précodage.

    Parameters
    ----------
    config : ConfigurationTP
    modulation : str, optional
        Modulation (par défaut celle de la configuration).

    Returns
    -------
    dict
        ``{"m", "k", "modulation", "snr_db", "zf", "sans_precodage"}``.
    """
    modulation = normaliser_nom_modulation(modulation or config.modulation)
    rng = np.random.default_rng(config.graine + 101)

    def _serie(methode):
        return [
            simuler_ber_point(
                m=config.m, k=config.k, snr_db=snr, p_t=config.p_t,
                n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
                rng=rng, methode=methode, modulation=modulation, par_utilisateur=True,
            )
            for snr in config.snr_db_courbe
        ]

    return {
        "m": config.m,
        "k": config.k,
        "modulation": modulation,
        "snr_db": list(config.snr_db_courbe),
        "zf": _serie("zf"),
        "sans_precodage": _serie("aucun"),
    }


def courbe_ber_snr_par_m(config, modulation: str | None = None) -> dict:
    """BER_k et BER moyen en fonction du SNR, pour chaque M (K fixé).

    Répond au point 8 du questionnaire : pour ``K`` fixé, on trace
    ``BER_1, ..., BER_K`` et le BER moyen en fonction du SNR pour chaque
    valeur de ``M``.

    Parameters
    ----------
    config : ConfigurationTP
    modulation : str, optional

    Returns
    -------
    dict
        ``{"valeurs_m", "k", "modulation", "snr_db", "resultats"}`` où
        ``resultats[M]`` est la liste des points sur ``snr_db``.
    """
    modulation = normaliser_nom_modulation(modulation or config.modulation)
    rng = np.random.default_rng(config.graine + 505)

    valeurs_m = list(config.valeurs_m_etudiees)
    resultats = {}
    for m in valeurs_m:
        resultats[m] = [
            simuler_ber_point(
                m=m, k=config.k, snr_db=snr, p_t=config.p_t,
                n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
                rng=rng, modulation=modulation, par_utilisateur=True,
            )
            for snr in config.snr_db_courbe
        ]

    return {
        "valeurs_m": valeurs_m,
        "k": config.k,
        "modulation": modulation,
        "snr_db": list(config.snr_db_courbe),
        "resultats": resultats,
    }


def courbe_ber_moyen_vs_m(config) -> dict:
    """BER moyen en fonction de M, à SNR fixé, pour les trois modulations.

    Répond au point 9 du questionnaire (SNR = 15 dB par défaut).
    """
    rng = np.random.default_rng(config.graine + 606)

    valeurs_m = list(config.valeurs_m_etudiees)
    resultats = {}
    for modulation in config.modulations_etude:
        modulation = normaliser_nom_modulation(modulation)
        resultats[modulation] = [
            simuler_ber_point(
                m=m, k=config.k, snr_db=config.snr_db_etude_m, p_t=config.p_t,
                n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
                rng=rng, modulation=modulation, par_utilisateur=True,
            )
            for m in valeurs_m
        ]

    return {
        "valeurs_m": valeurs_m,
        "modulations": [normaliser_nom_modulation(x) for x in config.modulations_etude],
        "snr_db": config.snr_db_etude_m,
        "k": config.k,
        "resultats": resultats,
    }


def courbe_ber_utilisateurs(config, modulation: str | None = None) -> dict:
    """BER moyen en fonction de K, à M fixé (M = 16 par défaut).

    Répond au point 10 du questionnaire : ``M`` est fixé à
    ``config.m_etude_k`` (16) et ``K`` varie jusqu'à ``M``.
    """
    modulation = normaliser_nom_modulation(modulation or config.modulation)
    rng = np.random.default_rng(config.graine + 303)

    resultats = [
        simuler_ber_point(
            m=config.m_etude_k, k=kv, snr_db=config.snr_db_etude_m, p_t=config.p_t,
            n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
            rng=rng, modulation=modulation, par_utilisateur=True,
        )
        for kv in config.valeurs_k_etudiees
    ]

    return {
        "valeurs_k": list(config.valeurs_k_etudiees),
        "m": config.m_etude_k,
        "snr_db": config.snr_db_etude_m,
        "modulation": modulation,
        "zf": resultats,
    }


def etude_conditionnement(config, modulation: str | None = None) -> dict:
    """Relie le conditionnement kappa(H H^H) au BER du ZF.

    Répond au point 12 du questionnaire : sur un ensemble de
    réalisations indépendantes du canal, on relève
    ``kappa(H H^H)`` et le BER observé, afin de comparer des
    réalisations bien et mal conditionnées.

    Returns
    -------
    dict
        ``{"kappa", "ber", "snr_db", "m", "k", "modulation",
        "n_realisations"}``.
    """
    modulation = normaliser_nom_modulation(modulation or config.modulation)
    rng = np.random.default_rng(config.graine + 707)

    b = bits_par_symbole(modulation)
    n_bits = max(b, (config.n_bits_conditionnement // b) * b)
    sigma2 = sigma2_depuis_snr_db(config.snr_db, config.p_t)

    kappas, bers = [], []
    for _ in range(config.n_realisations_conditionnement):
        h = generer_canal(config.k, config.m, rng)
        try:
            w_zf = precodage_zf(h)
        except CanalSingularError:
            # Réalisation singulière (probabilité négligeable) : ignorée.
            continue
        w_norm, _ = normaliser_puissance(w_zf, config.p_t)

        kappas.append(float(np.linalg.cond(h @ h.conj().T)))

        bits = rng.integers(0, 2, size=(config.k, n_bits)).astype(np.uint8)
        s = moduler(bits, modulation)
        y = transmettre(s, h, w_norm, sigma2=sigma2, rng=rng)
        y_egalise = egaliser(y, gains_utiles(h, w_norm))
        decides = demoduler(y_egalise, modulation)
        bers.append(float(np.count_nonzero(decides != bits)) / bits.size)

    return {
        "kappa": kappas,
        "ber": bers,
        "snr_db": config.snr_db,
        "m": config.m,
        "k": config.k,
        "modulation": modulation,
        "n_realisations": len(kappas),
    }


def etude_canaux_correles(config) -> dict:
    """Performances ZF selon la corrélation entre antennes d'émission.

    Modèle exponentiel ``R[i, j] = rho^|i-j|`` (Kronecker côté
    émission) : la corrélation dégrade le conditionnement de ``H`` et
    amplifie le bruit de recodage du ZF.

    Returns
    -------
    dict
        ``{"rho", "snr_db", "ber"}`` où ``ber[i]`` correspond à
        ``rho[i]``.
    """
    rng = np.random.default_rng(config.graine + 404)

    ber_par_rho = []
    for rho in config.rho_correles:
        ber_par_rho.append([
            simuler_ber_point(
                m=config.m, k=config.k, snr_db=snr, p_t=config.p_t,
                n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
                rng=rng, rho=rho, modulation=config.modulation, par_utilisateur=True,
            )
            for snr in config.snr_db_correles
        ])

    return {
        "rho": list(config.rho_correles),
        "snr_db": list(config.snr_db_correles),
        "ber": ber_par_rho,
        "modulation": config.modulation,
    }


def etude_canaux_similaires(config) -> dict:
    """Canaux de deux utilisateurs progressivement colinéaires.

    Répond au point 13 du questionnaire : deux canaux utilisateurs sont
    rendus similaires via ``h_2 <- sqrt(1-beta²) h_2 + beta h_1``. Plus
    ``beta`` augmente, plus ``H H^H`` devient mal conditionnée et plus
    le ZF amplifie le bruit (BER tendant vers 0,5).

    Returns
    -------
    dict
        ``{"beta", "snr_db", "ber"}`` où ``ber[i]`` correspond à
        ``beta[i]``.
    """
    rng = np.random.default_rng(config.graine + 808)

    ber_par_beta = []
    for beta in config.valeurs_beta:
        ber_par_beta.append([
            simuler_ber_point(
                m=config.m, k=config.k, snr_db=snr, p_t=config.p_t,
                n_bits=config.n_bits_courbe, n_realisations=config.n_realisations,
                rng=rng, beta_similarite=beta, modulation=config.modulation,
                par_utilisateur=True,
            )
            for snr in config.snr_db_courbe
        ])

    return {
        "beta": list(config.valeurs_beta),
        "snr_db": list(config.snr_db_courbe),
        "ber": ber_par_beta,
        "modulation": config.modulation,
    }
