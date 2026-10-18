"""Modèle de canal radio et de bruit.

Modèle de canal
---------------
Le canal entre la station de base (``M`` antennes d'émission) et les
``K`` utilisateurs (une seule antenne de réception chacun) est la
matrice complexe ``H`` de dimensions ``(K, M)`` :

- canal de Rayleigh à évanouissement plat, constant pendant un bloc de
  transmission ;
- coefficient ``H[k, m] ~ CN(0, 1)``, généré par
  ``(randn + 1j * randn) / sqrt(2)`` : les parties réelle et imaginaire
  sont indépendantes, gaussiennes, de moyenne nulle et de variance
  1/2, donc ``E[|H[k, m]|^2] = 1``.

Modèle de bruit
---------------
Le bruit AWGN ``N`` de dimensions ``(K, T)`` est défini par
``N ~ CN(0, sigma2)`` : chaque coefficient suit une loi complexe de
variance totale ``E[|N[k, t]|^2] = sigma2``, répartie à parts égales
entre les composantes réelle et imaginaire (variance ``sigma2/2``
chacune).

Convention de SNR (utilisée partout dans le projet)
---------------------------------------------------
::

    sigma2 = P_T / 10**(SNR_dB / 10)

où ``P_T`` est la puissance moyenne **totale** d'émission (après
normalisation du précodage) et ``sigma2`` la variance du bruit complexe
au récepteur. Comme chaque coefficient de canal a une puissance moyenne
unitaire, ce SNR correspond au rapport puissance de signal reçue moyenne
par utilisateur / puissance du bruit, dans le modèle sans précodage. Le
précodage ZF redistribue la même puissance totale ``P_T`` entre les
flux : il ne change pas la définition du SNR.

Canaux corrélés
---------------
Le modèle corrélé optionnel utilise le modèle de Kronecker côté
émission avec une matrice de corrélation exponentielle
``R[m, n] = rho ** |m - n|`` : ``H_corrélé = H_i.i.d. @ R**(1/2)``. Le
module des coefficients diagonaux reste 1 ; la corrélation dégrade le
conditionnement du canal et donc les performances du ZF.
"""

import numpy as np


def generer_canal(k: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Génère une réalisation du canal de Rayleigh complexe ``H``.

    Parameters
    ----------
    k : int
        Nombre d'utilisateurs (nombre de lignes).
    m : int
        Nombre d'antennes d'émission (nombre de colonnes).
    rng : numpy.random.Generator
        Générateur aléatoire reproductible.

    Returns
    -------
    numpy.ndarray
        Matrice complexe ``H`` de forme ``(k, m)`` dont chaque
        coefficient suit ``CN(0, 1)``.

    Raises
    ------
    ValueError
        Si ``k`` ou ``m`` est strictement nul.
    """
    if k < 1 or m < 1:
        raise ValueError(f"Dimensions invalides du canal : k={k}, m={m}.")

    partie_reelle = rng.standard_normal((k, m))
    partie_imaginaire = rng.standard_normal((k, m))
    return (partie_reelle + 1j * partie_imaginaire) / np.sqrt(2.0)


def generer_bruit(
    k: int,
    t: int,
    sigma2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Génère un bruit AWGN complexe ``N ~ CN(0, sigma2)``.

    La variance totale par coefficient est ``E[|N|^2] = sigma2``, avec
    une variance ``sigma2/2`` sur la composante réelle et ``sigma2/2``
    sur la composante imaginaire.

    Parameters
    ----------
    k : int
        Nombre de lignes (utilisateurs).
    t : int
        Nombre d'instants d'échantillonnage (colonnes).
    sigma2 : float
        Variance totale du bruit complexe. Si ``sigma2 == 0``, la
        fonction retourne directement une matrice nulle (sans consommer
        d'aléatoire).
    rng : numpy.random.Generator
        Générateur aléatoire reproductible.

    Returns
    -------
    numpy.ndarray
        Matrice complexe ``(k, t)``.
    """
    if k < 1 or t < 1:
        raise ValueError(f"Dimensions invalides du bruit : k={k}, t={t}.")
    if sigma2 < 0:
        raise ValueError(f"La variance du bruit doit être positive (sigma2={sigma2}).")

    if sigma2 == 0.0:
        return np.zeros((k, t), dtype=np.complex128)

    ecart_type = np.sqrt(sigma2 / 2.0)
    partie_reelle = rng.standard_normal((k, t))
    partie_imaginaire = rng.standard_normal((k, t))
    return ecart_type * (partie_reelle + 1j * partie_imaginaire)


def sigma2_depuis_snr_db(snr_db: float, p_t: float = 1.0) -> float:
    """Convertit un SNR en dB en variance de bruit.

    Convention du projet : ``sigma2 = P_T / 10**(SNR_dB / 10)``.

    Parameters
    ----------
    snr_db : float
        Rapport signal-bruit en dB.
    p_t : float
        Puissance moyenne totale d'émission (par défaut 1.0).

    Returns
    -------
    float
        Variance totale du bruit complexe.
    """
    if p_t <= 0:
        raise ValueError(f"La puissance totale doit être positive (P_T={p_t}).")
    return float(p_t) / (10.0 ** (float(snr_db) / 10.0))


def snr_db_depuis_sigma2(sigma2: float, p_t: float = 1.0) -> float:
    """Opération inverse de :func:`sigma2_depuis_snr_db`.

    Parameters
    ----------
    sigma2 : float
        Variance totale du bruit complexe (strictement positive).
    p_t : float
        Puissance moyenne totale d'émission.

    Returns
    -------
    float
        SNR correspondant, en dB.
    """
    if sigma2 <= 0:
        raise ValueError(f"sigma2 doit être strictement positive (sigma2={sigma2}).")
    return 10.0 * np.log10(float(p_t) / float(sigma2))


def matrice_correlation_exponentielle(n: int, rho: float) -> np.ndarray:
    """Construit une matrice de corrélation exponentielle réelle.

    Le modèle utilisé est ``R[i, j] = rho ** |i - j|`` : corrélation
    entre deux antennes d'émission décroît exponentiellement avec leur
    distance physique (indice).

    Parameters
    ----------
    n : int
        Taille de la matrice (nombre d'antennes).
    rho : float
        Coefficient de corrélation entre antennes voisines, compris
        entre 0 (canal i.i.d.) et 1 (corrélation parfaite, matrice
        singulière ; la valeur 1 est refusée).

    Returns
    -------
    numpy.ndarray
        Matrice symétrique définie positive ``(n, n)``.

    Raises
    ------
    ValueError
        Si ``n < 1`` ou si ``rho`` n'est pas dans ``[0, 1[``.
    """
    if n < 1:
        raise ValueError(f"Taille de matrice invalide : n={n}.")
    if not (0.0 <= rho < 1.0):
        raise ValueError(
            f"Le coefficient de corrélation rho doit être dans [0, 1[ (rho={rho}). "
            "rho = 1 produirait une matrice singulière."
        )

    indices = np.arange(n)
    distances = np.abs(np.subtract.outer(indices, indices))
    return rho ** distances


def racine_matricielle_symmetrique(matrice: np.ndarray) -> np.ndarray:
    """Calcule la racine carrée matricielle symétrique d'une matrice SPD.

    Parameters
    ----------
    matrice : numpy.ndarray
        Matrice symétrique définie positive ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        Matrice ``R**(1/2)`` telle que ``R**(1/2) @ R**(1/2) = R``.
    """
    valeurs, vecteurs = np.linalg.eigh(matrice)
    # Petites valeurs propres négatives (arrondis) : borne à zéro.
    valeurs = np.clip(valeurs, 0.0, None)
    return (vecteurs * np.sqrt(valeurs)) @ vecteurs.conj().T


def appliquer_correlation_emission(h: np.ndarray, rho: float) -> np.ndarray:
    """Applique une corrélation entre antennes d'émission au canal.

    Modèle de Kronecker côté émission :
    ``H_corrélé = H_i.i.d. @ R**(1/2)`` avec
    ``R[i, j] = rho ** |i - j|``.

    Parameters
    ----------
    h : numpy.ndarray
        Canal i.i.d. de forme ``(k, m)``.
    rho : float
        Coefficient de corrélation entre antennes voisines.

    Returns
    -------
    numpy.ndarray
        Canal corrélé de même forme ``(k, m)``.
    """
    h = np.asarray(h)
    if h.ndim != 2:
        raise ValueError(f"Le canal doit être une matrice 2D (forme obtenue : {h.shape}).")

    if rho == 0.0:
        return h.copy()

    m = h.shape[1]
    racine_r = racine_matricielle_symmetrique(matrice_correlation_exponentielle(m, rho))
    return h @ racine_r
