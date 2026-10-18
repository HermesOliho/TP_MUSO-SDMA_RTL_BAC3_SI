"""Modulations numériques (BPSK, 4-QAM/QPSK, 16-QAM) et calcul du BER.

Modulations supportées
----------------------
===========================  ================  ==========================
Nom                          Bits / symbole    Constellation
===========================  ================  ==========================
``BPSK``                     1                 ``{±1}`` (axe réel)
``4-QAM`` (ou ``QPSK``)      2                 carré 2x2 Gray
``16-QAM``                   4                 carré 4x4 Gray
===========================  ================  ==========================

Toutes les constellations sont normalisées à une **puissance moyenne
unitaire** : ``E[|s|²] = 1``, ce qui rend les comparaisons entre
modulations équitables à puissance d'émission donnée.

Convention BPSK (conservée dans tout le projet) : bit ``0`` -> ``-1``,
bit ``1`` -> ``+1``.

Codage de Gray et correspondance bits/symbole
---------------------------------------------
- Les bits d'un utilisateur sont regroupés par paquets de ``b`` bits
  (``b`` = 1, 2 ou 4), bit de poids fort en tête.
- La valeur entière d'un paquet (``MSB`` premier) sélectionne un point
  de la constellation.
- Sur chaque axe (I puis Q), le niveau est obtenu par décodage de Gray
  ``niveau(v) = 2 * (v XOR (v >> 1)) - (L - 1)`` avec ``L`` niveaux par
  axe : deux points voisins ne diffèrent que d'un seul bit.
- Pour ``4`` et ``16``-QAM, les ``b/2`` premiers bits du paquet
  indexent l'axe réel (I) et les ``b/2`` suivants l'axe imaginaire (Q).

Démodulation
------------
- ``demoduler`` : décision du **plus proche voisin** dans la
  constellation (distance euclidienne) ; elle est compatible avec un
  gain réel positif (mise à l'échelle par ``alpha``) et avec
  l'égalisation ``y / g``, qui ramène la constellation utile à sa
  position nominale.
- ``demoduler_bpsk`` : décision par signe de la partie réelle (cas
  particulier conservé explicitement pour la lisibilité).

Définition du BER
-----------------
``BER = erreurs / nb_bits comparés``. Les symboles de remplissage
(alignement des flux de longueurs différentes) ne sont jamais comptés.
"""

from dataclasses import dataclass

import numpy as np

# Nombre de bits portés par un symbole, par modulation.
NB_BITS_PAR_SYMBOLE = {
    "BPSK": 1,
    "4-QAM": 2,
    "16-QAM": 4,
}

# Modulations proposées dans le projet.
MODULATIONS_SUPPORTEES = ("BPSK", "4-QAM", "16-QAM")

# Alias acceptés (saisie utilisateur).
_ALIAS_MODULATIONS = {
    "BPSK": "BPSK",
    "QPSK": "4-QAM",
    "4QAM": "4-QAM",
    "4-QAM": "4-QAM",
    "16QAM": "16-QAM",
    "16-QAM": "16-QAM",
}


@dataclass
class ResultatBER:
    """Résultat du calcul d'un taux d'erreur binaire.

    Attributes
    ----------
    erreurs : int
        Nombre de bits erronés.
    nb_bits : int
        Nombre de bits réellement comparés.
    ber : float
        Taux d'erreur binaire, ``erreurs / nb_bits``. Vaut ``0.0`` par
        convention lorsque aucun bit n'est comparé (séquence vide) ;
        un BER observé nul ne prouve pas que le BER théorique est nul :
        il faut toujours consulter ``nb_bits`` pour connaître la limite
        d'observation.
    """

    erreurs: int
    nb_bits: int
    ber: float

    @property
    def borne_observation(self) -> float:
        """Limite d'observation du BER pour cet échantillon.

        Un BER observé nul ne prouve pas que le BER théorique est nul :
        il signifie seulement qu'aucune erreur n'a été observée sur
        ``nb_bits`` bits. La limite d'observation vaut ``1 / nb_bits``
        (borne ``nan`` par convention si aucun bit n'a été comparé).
        """
        if self.nb_bits == 0:
            return float("nan")
        return 1.0 / self.nb_bits


# ------------------------------------------------------------
# Constellations
# ------------------------------------------------------------

def normaliser_nom_modulation(nom: str) -> str:
    """Normalise le nom d'une modulation (alias ``QPSK`` -> ``4-QAM``).

    Parameters
    ----------
    nom : str
        Nom saisi (``"BPSK"``, ``"QPSK"``, ``"4-QAM"``, ``"16-QAM"``...).

    Returns
    -------
    str
        Nom canonique.

    Raises
    ------
    ValueError
        Si la modulation n'est pas supportée.
    """
    cle = str(nom).strip().upper().replace(" ", "")
    if cle not in _ALIAS_MODULATIONS:
        raise ValueError(
            f"Modulation inconnue : '{nom}'. "
            f"Modulations supportées : {MODULATIONS_SUPPORTEES}."
        )
    return _ALIAS_MODULATIONS[cle]


def bits_par_symbole(modulation: str) -> int:
    """Retourne le nombre de bits par symbole d'une modulation."""
    return NB_BITS_PAR_SYMBOLE[normaliser_nom_modulation(modulation)]


def table_constellation(modulation: str) -> np.ndarray:
    """Construit la table des symboles, normalisée en puissance moyenne.

    L'indice ``i`` de la table correspond à la valeur entière du paquet
    de ``b`` bits (bit de poids fort en tête). Le codage est de type
    Gray sur chaque axe.

    Parameters
    ----------
    modulation : str
        ``"BPSK"``, ``"4-QAM"`` ou ``"16-QAM"``.

    Returns
    -------
    numpy.ndarray
        Table de ``2**b`` points (réels pour BPSK, complexes sinon),
        de puissance moyenne ``E[|s|²] = 1``.
    """
    modulation = normaliser_nom_modulation(modulation)
    b = NB_BITS_PAR_SYMBOLE[modulation]

    if b == 1:
        # BPSK : convention 0 -> -1, 1 -> +1 (puissance moyenne unitaire).
        return np.array([-1.0, 1.0], dtype=np.float64)

    b_axe = b // 2
    niveaux_par_axe = 1 << b_axe
    indices = np.arange(1 << b)
    indice_i = indices >> b_axe
    indice_q = indices & (niveaux_par_axe - 1)

    def niveau(valeur):
        # Décodage de Gray puis mise en niveaux PAM symétriques.
        gris = valeur ^ (valeur >> 1)
        return 2.0 * gris - (niveaux_par_axe - 1)

    table = niveau(indice_i) + 1j * niveau(indice_q)
    # Normalisation à la puissance moyenne unitaire.
    return table / np.sqrt(np.mean(np.abs(table) ** 2))


# ------------------------------------------------------------
# Mapping bits <-> symboles
# ------------------------------------------------------------

def _bits_vers_indices(bits: np.ndarray, nb_bits_par_symbole: int) -> np.ndarray:
    """Regroupe des bits par paquets et retourne leur valeur entière."""
    bits = np.asarray(bits, dtype=np.int64)
    nombre_symboles = bits.shape[-1] // nb_bits_par_symbole
    paquets = bits.reshape(bits.shape[:-1] + (nombre_symboles, nb_bits_par_symbole))
    poids = 1 << np.arange(nb_bits_par_symbole - 1, -1, -1, dtype=np.int64)
    return paquets @ poids


def moduler(bits: np.ndarray, modulation: str = "BPSK") -> np.ndarray:
    """Module une séquence de bits selon la modulation choisie.

    Parameters
    ----------
    bits : numpy.ndarray
        Bits (0 ou 1). La dernière dimension doit être un multiple du
        nombre de bits par symbole.
    modulation : str
        ``"BPSK"``, ``"4-QAM"`` ou ``"16-QAM"``.

    Returns
    -------
    numpy.ndarray
        Symboles normalisés : forme ``bits.shape[:-1] + (n/b,)`` ; réels
        pour la BPSK, complexes sinon.

    Raises
    ------
    ValueError
        Si des valeurs autres que 0/1 sont présentes, ou si le nombre de
        bits n'est pas compatible avec la modulation.
    """
    modulation = normaliser_nom_modulation(modulation)
    b = NB_BITS_PAR_SYMBOLE[modulation]
    table = table_constellation(modulation)

    bits = np.asarray(bits)
    if bits.ndim == 0:
        bits = bits.reshape(1)

    if bits.size == 0:
        return np.zeros(bits.shape[:-1] + (0,), dtype=table.dtype)

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("La séquence à moduler doit ne contenir que des 0 et des 1.")
    if bits.shape[-1] % b != 0:
        raise ValueError(
            f"Le nombre de bits ({bits.shape[-1]}) n'est pas un multiple de "
            f"{b} pour la modulation {modulation}."
        )

    return table[_bits_vers_indices(bits, b)]


def demoduler(symboles: np.ndarray, modulation: str = "BPSK") -> np.ndarray:
    """Démodule des symboles par décision du plus proche voisin.

    Pour les constellations carrées (4-QAM, 16-QAM), construites comme
    un produit cartésien de deux grilles PAM, la décision du plus proche
    voisin 2D est **exactement** séparable en une décision par axe
    (réel et imaginaire). Cette implémentation est donc à la fois
    optimale (même résultat que l'argmin 2D) et efficace sur de longs
    blocs.

    Parameters
    ----------
    symboles : numpy.ndarray
        Symboles reçus (réels ou complexes), après canal et, le cas
        échéant, après égalisation.
    modulation : str
        ``"BPSK"``, ``"4-QAM"`` ou ``"16-QAM"``.

    Returns
    -------
    numpy.ndarray
        Bits décidés ``uint8``, de forme ``symboles.shape[:-1] + (n*b,)``
        (les ``b`` bits de chaque symbole sont rangés consécutivement,
        bit de poids fort en tête).
    """
    modulation = normaliser_nom_modulation(modulation)
    b = NB_BITS_PAR_SYMBOLE[modulation]

    symboles = np.asarray(symboles)
    if symboles.ndim == 0:
        symboles = symboles.reshape(1)

    if symboles.size == 0:
        return np.zeros(symboles.shape[:-1] + (0,), dtype=np.uint8)

    if b == 1:
        # BPSK : décision par signe de la partie réelle.
        return (np.real(symboles) >= 0.0).astype(np.uint8)

    # Grille PAM par axe (niveaux entiers impairs symétriques).
    b_axe = b // 2
    n = 1 << b_axe
    niveaux = 2.0 * np.arange(n) - (n - 1)
    gris_inv = np.empty(n, dtype=np.int64)
    gris_inv[np.arange(n) ^ (np.arange(n) >> 1)] = np.arange(n)
    milieux = niveaux[:-1] + 1.0          # points milieux entre niveaux voisins
    norme = np.sqrt(2.0 * np.mean(niveaux ** 2))  # facteur de normalisation

    indice_i = np.searchsorted(milieux, np.real(symboles) * norme)
    indice_q = np.searchsorted(milieux, np.imag(symboles) * norme)
    valeur_i = gris_inv[indice_i]
    valeur_q = gris_inv[indice_q]

    decalages = np.arange(b_axe - 1, -1, -1)
    bits_i = (valeur_i[..., np.newaxis] >> decalages) & 1
    bits_q = (valeur_q[..., np.newaxis] >> decalages) & 1
    bits = np.concatenate([bits_i, bits_q], axis=-1)
    return bits.reshape(symboles.shape[:-1] + (-1,)).astype(np.uint8)


def demoduler_bruteforce(symboles: np.ndarray, modulation: str = "BPSK") -> np.ndarray:
    """Démodulation de référence par recherche exhaustive (tests).

    Retourne les mêmes bits que :func:`demoduler`, mais par calcul
    explicite de la distance euclidienne à chaque point de la
    constellation. Utilisée uniquement pour vérifier l'implémentation
    séparable.
    """
    modulation = normaliser_nom_modulation(modulation)
    b = NB_BITS_PAR_SYMBOLE[modulation]
    table = table_constellation(modulation)

    symboles = np.asarray(symboles)
    if symboles.ndim == 0:
        symboles = symboles.reshape(1)
    if symboles.size == 0:
        return np.zeros(symboles.shape[:-1] + (0,), dtype=np.uint8)

    distances = np.abs(symboles[..., np.newaxis] - table[np.newaxis, ...])
    indices = np.argmin(distances, axis=-1)
    decalages = np.arange(b - 1, -1, -1)
    bits = ((indices[..., np.newaxis] >> decalages) & 1).astype(np.uint8)
    return bits.reshape(symboles.shape[:-1] + (-1,))


# ------------------------------------------------------------
# Cas particulier BPSK (compatibilité et lisibilité)
# ------------------------------------------------------------

def moduler_bpsk(bits: np.ndarray) -> np.ndarray:
    """Module une séquence de bits en symboles BPSK (``0 -> -1``, ``1 -> +1``)."""
    bits = np.asarray(bits)
    if bits.size == 0:
        return np.zeros(bits.shape, dtype=np.float64)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("La séquence BPSK doit ne contenir que des valeurs 0 et 1.")
    return 2.0 * bits.astype(np.float64) - 1.0


def demoduler_bpsk(symboles: np.ndarray) -> np.ndarray:
    """Démodule des symboles BPSK par décision de signe (partie réelle).

    ``Re{y} >= 0`` -> bit ``1``, sinon bit ``0``.
    """
    symboles = np.asarray(symboles)
    return (np.real(symboles) >= 0.0).astype(np.uint8)


# ------------------------------------------------------------
# Calcul du BER
# ------------------------------------------------------------

def calculer_ber(bits_emis: np.ndarray, bits_recus: np.ndarray) -> ResultatBER:
    """Calcule le BER entre une séquence de bits émis et reçus.

    Parameters
    ----------
    bits_emis : numpy.ndarray
        Bits transmis.
    bits_recus : numpy.ndarray
        Bits décidés par le récepteur, de même forme que ``bits_emis``.

    Returns
    -------
    ResultatBER
        Nombre d'erreurs, nombre de bits comparés et BER. Pour une
        séquence vide, les trois valeurs sont ``(0, 0, 0.0)``.

    Raises
    ------
    ValueError
        Si les deux séquences n'ont pas la même forme.
    """
    bits_emis = np.asarray(bits_emis).ravel()
    bits_recus = np.asarray(bits_recus).ravel()

    if bits_emis.shape != bits_recus.shape:
        raise ValueError(
            f"Séquences de tailles différentes : {bits_emis.shape} bits émis "
            f"contre {bits_recus.shape} bits reçus."
        )

    nb_bits = int(bits_emis.size)
    if nb_bits == 0:
        return ResultatBER(erreurs=0, nb_bits=0, ber=0.0)

    erreurs = int(np.count_nonzero(bits_emis != bits_recus))
    return ResultatBER(erreurs=erreurs, nb_bits=nb_bits, ber=erreurs / nb_bits)


def ber_depuis_comptage(erreurs: int, nb_bits: int) -> float:
    """Calcule un BER à partir d'un comptage d'erreurs et de bits.

    Parameters
    ----------
    erreurs : int
        Nombre d'erreurs observées.
    nb_bits : int
        Nombre de bits comparés.

    Returns
    -------
    float
        ``erreurs / nb_bits``, ou ``0.0`` par convention si
        ``nb_bits == 0`` (aucune information mesurée).
    """
    if nb_bits == 0:
        return 0.0
    return float(erreurs) / float(nb_bits)


def ber_global(resultats: list) -> ResultatBER:
    """Agrège les BER de plusieurs flux en un BER global.

    Le BER global est le rapport entre le nombre total d'erreurs et le
    nombre total de bits comparés (sur les quatre utilisateurs).

    Parameters
    ----------
    resultats : list of ResultatBER
        Résultats par utilisateur.

    Returns
    -------
    ResultatBER
        Agrégat global.
    """
    erreurs = sum(r.erreurs for r in resultats)
    nb_bits = sum(r.nb_bits for r in resultats)
    return ResultatBER(erreurs=erreurs, nb_bits=nb_bits, ber=ber_depuis_comptage(erreurs, nb_bits))


def bits_aleatoires(nombre: int, rng: np.random.Generator) -> np.ndarray:
    """Génère une séquence binaire aléatoire uniforme.

    Parameters
    ----------
    nombre : int
        Nombre de bits à générer.
    rng : numpy.random.Generator
        Générateur aléatoire (reproductible si sa graine est fixée).

    Returns
    -------
    numpy.ndarray
        Tableau ``uint8`` de bits uniformément distribués entre 0 et 1.
    """
    return rng.integers(0, 2, size=nombre).astype(np.uint8)
