"""Modèle de transmission MU-MISO et détection BPSK.

Modèle de transmission
----------------------
À un instant d'échantillonnage, les ``K`` symboles forment le vecteur
``s`` de dimension ``(K,)``. La station de base émet ``x = W @ s``
(dimension ``(M,)``), le canal produit ``y = H @ x + n``, donc :

::

    y = H @ W @ s + n

Pour un bloc de ``T`` instants, on utilise la convention matricielle
(ici ``S``, ``X``, ``Y``, ``N``) :

- ``S`` de forme ``(K, T)`` : une ligne par utilisateur ;
- ``X = W @ S`` de forme ``(M, T)`` : signaux des ``M`` antennes ;
- ``Y = H @ X + N`` de forme ``(K, T)`` : signaux reçus.

Les flux de longueurs différentes sont alignés par remplissage
(``padding``) : les colonnes de remplissage sont marquées par un masque
de validité et ne sont jamais comptées dans le BER ni intégrées aux
images reconstruites.

Détection
---------
En BPSK, la décision se fait par signe de la partie réelle :
``Re{Y[k, t]} >= 0`` donne le bit 1. Après précodage ZF normalisé,
``Y = alpha * S + N`` avec ``alpha`` réel positif : la mise à l'échelle
``alpha`` ne change donc pas la décision, ce qui rend la détection BPSK
directement compatible avec le ZF normalisé.
"""

from dataclasses import dataclass

import numpy as np

from .canal import generer_bruit
from .modulation import ResultatBER, ber_depuis_comptage, demoduler_bpsk


def transmettre(
    s: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    sigma2: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Applique le modèle MU-MISO ``Y = H @ W @ S + N``.

    Parameters
    ----------
    s : numpy.ndarray
        Symboles émis, de forme ``(K, T)`` (une ligne par utilisateur).
    h : numpy.ndarray
        Canal complexe ``(K, M)``.
    w : numpy.ndarray
        Précodage ``(M, K)``.
    sigma2 : float
        Variance du bruit complexe (0 pour une transmission sans
        bruit).
    rng : numpy.random.Generator, optional
        Générateur du bruit ; requis si ``sigma2 > 0``.

    Returns
    -------
    numpy.ndarray
        Signaux reçus ``Y`` de forme ``(K, T)``.

    Raises
    ------
    ValueError
        Si les dimensions des tableaux sont incohérentes, ou si un
        générateur aléatoire est requis mais absent.
    """
    s = np.asarray(s)
    h = np.asarray(h)
    w = np.asarray(w)

    if s.ndim == 1:
        # Vecteur unique (T,) interprété comme un bloc d'un seul instant.
        s = s.reshape(-1, 1)
    if s.ndim != 2:
        raise ValueError(
            f"Les symboles doivent former une matrice (K, T) ; forme obtenue : {s.shape}."
        )

    k, m = h.shape if h.ndim == 2 else (None, None)
    if h.ndim != 2:
        raise ValueError(f"Le canal H doit être 2D (forme obtenue : {h.shape}).")
    if w.shape != (m, k):
        raise ValueError(
            f"Dimensions incohérentes : H est ({k}, {m}) et W est {w.shape} ; "
            f"attendu ({m}, {k})."
        )
    if s.shape[0] != k:
        raise ValueError(
            f"Nombre de flux incohérent : S possède {s.shape[0]} ligne(s) "
            f"alors que le canal possède K = {k} utilisateur(s)."
        )

    # Signal émis par les M antennes, puis canal + bruit.
    x = w @ s                      # (M, T)
    y = h @ x                      # (K, T)

    if sigma2 > 0.0:
        if rng is None:
            raise ValueError(
                "Un générateur aléatoire (rng) est requis pour ajouter du bruit."
            )
        t = y.shape[1]
        y = y + generer_bruit(k, t, sigma2, rng)

    return y


def detecter_bpsk(y: np.ndarray) -> np.ndarray:
    """Détecte les bits BPSK par décision de signe sur la partie réelle.

    Parameters
    ----------
    y : numpy.ndarray
        Symboles reçus ``(K, T)`` (réels ou complexes).

    Returns
    -------
    numpy.ndarray
        Bits décidés ``uint8`` de forme ``(K, T)``.
    """
    return demoduler_bpsk(y)


@dataclass
class ResultatReception:
    """Bilan de réception d'un bloc multi-utilisateurs.

    Attributes
    ----------
    erreurs : numpy.ndarray
        Nombre d'erreurs par utilisateur, forme ``(K,)``.
    nb_bits : numpy.ndarray
        Nombre de bits valides comparés par utilisateur, forme ``(K,)``.
    ber_utilisateurs : numpy.ndarray
        BER de chaque utilisateur, forme ``(K,)``.
    ber : ResultatBER
        Agrégat global (erreurs et bits totaux sur les utilisateurs).
    bits_recus : list of numpy.ndarray
        Bits décidés par utilisateur, **uniquement les bits valides**
        (le remplissage est exclu), prêts pour la reconstruction.
    """

    erreurs: np.ndarray
    nb_bits: np.ndarray
    ber_utilisateurs: np.ndarray
    ber: ResultatBER
    bits_recus: list


def evaluer_reception(
    bits_emis: list,
    y: np.ndarray,
    masque: np.ndarray | None = None,
) -> ResultatReception:
    """Évalue la réception : BER par utilisateur et bits reçus valides.

    Parameters
    ----------
    bits_emis : list of numpy.ndarray
        Bits transmis par utilisateur (séquences de longueurs
        possiblement différentes, avant alignement par remplissage).
    y : numpy.ndarray
        Symboles reçus ``(K, T)``.
    masque : numpy.ndarray, optional
        Masque booléen ``(K, T)`` indiquant les positions valides (les
        colonnes de remplissage sont à ``False``). Si absent, tous les
        bits sont considérés comme valides.

    Returns
    -------
    ResultatReception
        Bilan complet de la réception.

    Raises
    ------
    ValueError
        Si les dimensions ne correspondent pas.
    """
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError(f"Le signal reçu Y doit être 2D (forme obtenue : {y.shape}).")

    k, t = y.shape
    if len(bits_emis) != k:
        raise ValueError(
            f"Nombre d'utilisateurs incohérent : {len(bits_emis)} flux de bits "
            f"pour K = {k} lignes reçues."
        )

    if masque is None:
        masque = np.ones((k, t), dtype=bool)
    else:
        masque = np.asarray(masque, dtype=bool)
        if masque.shape != (k, t):
            raise ValueError(
                f"Le masque de validité doit être de forme ({k}, {t}) ; "
                f"forme obtenue : {masque.shape}."
            )

    bits_decides = detecter_bpsk(y)

    erreurs = np.zeros(k, dtype=np.int64)
    nb_bits = np.zeros(k, dtype=np.int64)
    bits_recus = []

    for indice in range(k):
        valide = masque[indice]
        comptes = int(np.count_nonzero(valide))
        nb_bits[indice] = comptes

        # Vérification de cohérence : le nombre de bits valides doit
        # correspondre au nombre de bits réellement émis par cet utilisateur.
        if comptes != len(bits_emis[indice]):
            raise ValueError(
                f"Utilisateur {indice + 1} : {comptes} position(s) valide(s) dans le "
                f"bloc contre {len(bits_emis[indice])} bits émis."
            )

        decided_valides = bits_decides[indice][valide]
        reference = np.asarray(bits_emis[indice], dtype=np.uint8)
        erreurs[indice] = int(np.count_nonzero(decided_valides != reference))
        bits_recus.append(decided_valides.copy())

    ber_utilisateurs = np.array(
        [ber_depuis_comptage(int(e), int(n)) for e, n in zip(erreurs, nb_bits)]
    )
    resultat_global = ResultatBER(
        erreurs=int(np.sum(erreurs)),
        nb_bits=int(np.sum(nb_bits)),
        ber=ber_depuis_comptage(int(np.sum(erreurs)), int(np.sum(nb_bits))),
    )

    return ResultatReception(
        erreurs=erreurs,
        nb_bits=nb_bits,
        ber_utilisateurs=ber_utilisateurs,
        ber=resultat_global,
        bits_recus=bits_recus,
    )
