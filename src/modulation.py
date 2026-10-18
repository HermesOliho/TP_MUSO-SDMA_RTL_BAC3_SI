"""Modulation et démodulation BPSK, calcul du taux d'erreur binaire.

Convention de modulation (identique dans tout le projet)
---------------------------------------------------------
- bit ``0`` -> symbole réel ``-1``
- bit ``1`` -> symbole réel ``+1``

Convention de démodulation
--------------------------
Décision par signe sur la **partie réelle** du symbole reçu :
``Re{y} >= 0`` -> bit ``1``, sinon bit ``0``. Utiliser la partie réelle
est cohérent avec le fait que la BPSK encode l'information sur l'axe
réel ; après précodage ZF, le gain utile ``alpha`` est réel positif, de
sorte que la décision sur la partie réelle est optimale. La composante
imaginaire du signal reçu ne contient que du bruit et des interférences
résiduelles.

Définition du BER
-----------------
Le taux d'erreur binaire (BER) est le rapport entre le nombre de bits
décidés erronés et le nombre de bits effectivement comparés. Les
symboles de remplissage (padding) des blocs incomplets ne sont jamais
comptés dans le BER.
"""

from dataclasses import dataclass

import numpy as np


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


def moduler_bpsk(bits: np.ndarray) -> np.ndarray:
    """Module une séquence de bits en symboles BPSK.

    Parameters
    ----------
    bits : numpy.ndarray
        Séquence de bits (valeurs 0 ou 1), de n'importe quelle forme.

    Returns
    -------
    numpy.ndarray
        Tableau ``float64`` de même forme : ``0 -> -1``, ``1 -> +1``.
        Les symboles sont réels (la BPSK n'utilise pas la composante
        imaginaire).
    """
    bits = np.asarray(bits)
    if bits.size == 0:
        return np.zeros(bits.shape, dtype=np.float64)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("La séquence BPSK doit ne contenir que des valeurs 0 et 1.")
    return 2.0 * bits.astype(np.float64) - 1.0


def demoduler_bpsk(symboles: np.ndarray) -> np.ndarray:
    """Démodule des symboles BPSK par décision de signe.

    La décision porte sur la partie réelle du symbole reçu :
    ``Re{y} >= 0`` donne le bit ``1``, sinon le bit ``0``.

    Parameters
    ----------
    symboles : numpy.ndarray
        Symboles reçus (réels ou complexes), de n'importe quelle forme.

    Returns
    -------
    numpy.ndarray
        Tableau ``uint8`` de bits (0 ou 1), de même forme que l'entrée.
    """
    symboles = np.asarray(symboles)
    return (np.real(symboles) >= 0.0).astype(np.uint8)


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
