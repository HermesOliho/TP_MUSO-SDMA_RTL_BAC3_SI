"""Modèle de transmission MU-MISO, égalisation et détection.

Modèle de transmission
----------------------
À un instant d'échantillonnage, les ``K`` symboles forment le vecteur
``s`` de dimension ``(K,)``. La station de base émet ``x = W @ s``
(dimension ``(M,)``), le canal produit ``y = H @ x + n``, donc :

::

    y = H @ W @ s + n

Pour un bloc de ``T`` instants, on utilise la convention matricielle :

- ``S`` de forme ``(K, T)`` : une ligne par utilisateur ;
- ``X = W @ S`` de forme ``(M, T)`` : signaux des ``M`` antennes ;
- ``Y = H @ X + N`` de forme ``(K, T)`` : signaux reçus.

Les flux de longueurs différentes sont alignés par remplissage
(``padding``) : les colonnes de remplissage sont marquées par un masque
de validité et ne sont jamais comptées dans le BER ni intégrées aux
images reconstruites.

Égalisation
-----------
Le canal effectif de l'utilisateur ``k`` est ``g_k = h_k^H w_k``,
c'est-à-dire l'élément diagonal de ``G = H W``. L'utilisateur égalise

::

    s_hat_k = y_k / g_k

avant démodulation, ce qui compense à la fois le gain et la phase de son
canal utile et ramène la constellation à sa position nominale.

- Avec le ZF idéal, ``G = alpha * I_K`` (``alpha`` réel positif) :
  l'égalisation divise par ``alpha`` et la décision reste inchangée.
- Dans le scénario sans précodage, ``g_k`` est complexe : l'égalisation
  compense aussi la phase du canal (hypothèse de CSI au récepteur).

Détection
---------
- BPSK : décision par signe de la partie réelle (``Re{y} >= 0`` -> 1).
- 4-QAM et 16-QAM : décision du plus proche voisin dans la
  constellation normalisée, après égalisation.
"""

from dataclasses import dataclass

import numpy as np

from .canal import generer_bruit
from .modulation import (
    ResultatBER,
    ber_depuis_comptage,
    bits_par_symbole,
    demoduler,
    demoduler_bpsk,
)


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
        s = s.reshape(-1, 1)
    if s.ndim != 2:
        raise ValueError(
            f"Les symboles doivent former une matrice (K, T) ; forme obtenue : {s.shape}."
        )
    if h.ndim != 2:
        raise ValueError(f"Le canal H doit être 2D (forme obtenue : {h.shape}).")

    k, m = h.shape
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

    x = w @ s                      # (M, T)
    y = h @ x                      # (K, T)

    if sigma2 > 0.0:
        if rng is None:
            raise ValueError(
                "Un générateur aléatoire (rng) est requis pour ajouter du bruit."
            )
        y = y + generer_bruit(k, y.shape[1], sigma2, rng)

    return y


def gains_utiles(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Retourne les gains utiles ``g_k = h_k^H w_k`` (diagonale de G).

    Parameters
    ----------
    h : numpy.ndarray
        Canal ``(K, M)``.
    w : numpy.ndarray
        Précodage ``(M, K)``.

    Returns
    -------
    numpy.ndarray
        Vecteur complexe ``(K,)`` des gains utiles.
    """
    h = np.asarray(h)
    w = np.asarray(w)
    return np.diagonal(h @ w).copy()


def egaliser(y: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Égalise le signal reçu : ``s_hat_k = y_k / g_k``.

    Parameters
    ----------
    y : numpy.ndarray
        Signal reçu ``(K, T)``.
    gains : numpy.ndarray
        Gains utiles ``(K,)`` (diagonale de ``G``).

    Returns
    -------
    numpy.ndarray
        Signal égalisé, de même forme que ``y``.

    Raises
    ------
    ValueError
        Si les dimensions sont incohérentes ou si un gain est nul.
    """
    y = np.asarray(y)
    gains = np.asarray(gains)

    if y.ndim != 2:
        raise ValueError(f"Le signal reçu Y doit être 2D (forme obtenue : {y.shape}).")
    if gains.ndim != 1 or gains.shape[0] != y.shape[0]:
        raise ValueError(
            f"Le vecteur des gains doit être de forme ({y.shape[0]},) ; "
            f"forme obtenue : {gains.shape}."
        )
    if np.any(np.abs(gains) == 0.0):
        raise ValueError(
            "Un gain utile est nul : l'égalisation est impossible "
            "(canal non inversible pour cet utilisateur)."
        )

    return y / gains[:, np.newaxis]


def detecter_bpsk(y: np.ndarray) -> np.ndarray:
    """Détecte les bits BPSK par décision de signe sur la partie réelle."""
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
        ``BER_k`` de chaque utilisateur, forme ``(K,)``.
    ber : ResultatBER
        Agrégat global (erreurs et bits totaux sur les utilisateurs).
    ber_moyen : float
        ``BER_moy = (1/K) * somme_k BER_k`` (moyenne arithmétique des
        BER individuels, formule du TP).
    bits_recus : list of numpy.ndarray
        Bits décidés par utilisateur, **uniquement les bits valides**
        (le remplissage est exclu), prêts pour la reconstruction.
    """

    erreurs: np.ndarray
    nb_bits: np.ndarray
    ber_utilisateurs: np.ndarray
    ber: ResultatBER
    ber_moyen: float
    bits_recus: list


def evaluer_reception(
    bits_emis: list,
    y: np.ndarray,
    masque: np.ndarray | None = None,
    modulation: str = "BPSK",
) -> ResultatReception:
    """Évalue la réception : BER par utilisateur et bits reçus valides.

    Parameters
    ----------
    bits_emis : list of numpy.ndarray
        Bits transmis par utilisateur (séquences de longueurs
        possiblement différentes, avant alignement par remplissage).
    y : numpy.ndarray
        Symboles reçus égalisés, de forme ``(K, T)`` (déjà divisés par
        les gains utiles ``g_k``).
    masque : numpy.ndarray, optional
        Masque booléen ``(K, T)`` indiquant les symboles valides (les
        colonnes de remplissage sont à ``False``). Si absent, tous les
        symboles sont considérés comme valides.
    modulation : str
        Modulation utilisée (``"BPSK"``, ``"4-QAM"``, ``"16-QAM"``).

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

    b = bits_par_symbole(modulation)
    bits_decides = demoduler(y, modulation)          # (K, T*b)

    # Masque au niveau des bits : chaque symbole valide porte b bits.
    if b == 1:
        masque_bits = masque
    else:
        masque_bits = np.repeat(masque, b, axis=1)

    if bits_decides.shape != masque_bits.shape:
        raise ValueError(
            f"Le signal décidé ({bits_decides.shape}) et le masque "
            f"({masque_bits.shape}) ne correspondent pas."
        )

    erreurs = np.zeros(k, dtype=np.int64)
    nb_bits = np.zeros(k, dtype=np.int64)
    bits_recus = []

    for indice in range(k):
        valide = masque_bits[indice]
        comptes = int(np.count_nonzero(valide))
        nb_bits[indice] = comptes

        if comptes != len(bits_emis[indice]):
            raise ValueError(
                f"Utilisateur {indice + 1} : {comptes} bit(s) valide(s) dans le "
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
    ber_moyen = float(np.mean(ber_utilisateurs)) if k > 0 else 0.0

    return ResultatReception(
        erreurs=erreurs,
        nb_bits=nb_bits,
        ber_utilisateurs=ber_utilisateurs,
        ber=resultat_global,
        ber_moyen=ber_moyen,
        bits_recus=bits_recus,
    )
