"""Tests unitaires de ``src.modulation`` (BPSK et BER)."""

import numpy as np
import pytest

from src.modulation import (
    ResultatBER,
    ber_depuis_comptage,
    ber_global,
    bits_aleatoires,
    bits_par_symbole,
    calculer_ber,
    demoduler,
    demoduler_bpsk,
    demoduler_bruteforce,
    moduler,
    moduler_bpsk,
    normaliser_nom_modulation,
    table_constellation,
)


def test_convention_modulation():
    """Convention : 0 -> -1 et 1 -> +1."""
    np.testing.assert_array_equal(moduler_bpsk([0, 1, 1, 0]), [-1.0, 1.0, 1.0, -1.0])


def test_modulation_demodulation_sans_bruit():
    """La chaîne BPSK complète restitue les bits d'origine."""
    rng = np.random.default_rng(2)
    for taille in (0, 1, 100, 5000):
        bits = bits_aleatoires(taille, rng)
        symboles = moduler_bpsk(bits)
        bits_rec = demoduler_bpsk(symboles)
        np.testing.assert_array_equal(bits, bits_rec)


def test_modulation_symboles_reels_puissance_unitaire():
    """Les symboles BPSK sont réels et de puissance moyenne unitaire."""
    rng = np.random.default_rng(3)
    bits = bits_aleatoires(10_000, rng)
    symboles = moduler_bpsk(bits)
    assert np.isrealobj(symboles)
    assert np.all(np.abs(symboles) == 1.0)
    assert abs(np.mean(symboles**2) - 1.0) < 1e-12


def test_modulation_valeurs_invalides():
    """Des valeurs autres que 0 et 1 doivent lever ValueError."""
    with pytest.raises(ValueError):
        moduler_bpsk([0, 2])
    with pytest.raises(ValueError):
        moduler_bpsk([0.5])


def test_demodulation_partie_reelle():
    """La décision se fait sur la partie réelle, jamais sur le module."""
    # 1+2j : partie réelle positive -> bit 1 (un grand module imaginaire
    # ne doit pas faire basculer la décision).
    np.testing.assert_array_equal(demoduler_bpsk([1 + 2j, -0.1 + 5j, 0 + 0j]),
                                  [1, 0, 1])


def test_ber_ideal_nul():
    """Dans un canal idéal sans bruit, le BER est nul."""
    rng = np.random.default_rng(4)
    bits = bits_aleatoires(4096, rng)
    resultat = calculer_ber(bits, bits)
    assert resultat.erreurs == 0
    assert resultat.nb_bits == 4096
    assert resultat.ber == 0.0


def test_ber_erreurs_connues():
    """BER exact sur une séquence à erreurs injectées."""
    bits_emis = np.zeros(10, dtype=np.uint8)
    bits_rec = bits_emis.copy()
    bits_rec[[0, 3, 7]] = 1  # 3 erreurs sur 10 bits
    resultat = calculer_ber(bits_emis, bits_rec)
    assert resultat.erreurs == 3
    assert resultat.nb_bits == 10
    assert resultat.ber == pytest.approx(0.3)


def test_ber_sequence_vide():
    """Une séquence vide ne doit pas lever d'erreur de division."""
    resultat = calculer_ber(np.array([], dtype=np.uint8),
                             np.array([], dtype=np.uint8))
    assert resultat == ResultatBER(erreurs=0, nb_bits=0, ber=0.0)


def test_ber_tailles_differentes():
    """Des séquences de tailles différentes doivent lever ValueError."""
    with pytest.raises(ValueError):
        calculer_ber(np.zeros(5, dtype=np.uint8), np.zeros(6, dtype=np.uint8))


def test_ber_depuis_comptage():
    """Agrégat à partir d'un comptage brut, y compris vide."""
    assert ber_depuis_comptage(3, 10) == pytest.approx(0.3)
    assert ber_depuis_comptage(0, 0) == 0.0


def test_ber_global():
    """Le BER global pondère par le nombre de bits de chaque flux."""
    r1 = ResultatBER(erreurs=1, nb_bits=99, ber=1 / 99)
    r2 = ResultatBER(erreurs=0, nb_bits=1, ber=0.0)
    global_ = ber_global([r1, r2])
    assert global_.erreurs == 1
    assert global_.nb_bits == 100
    assert global_.ber == pytest.approx(0.01)


# ------------------------------------------------------------
# Modulations 4-QAM et 16-QAM
# ------------------------------------------------------------

@pytest.mark.parametrize("modulation", ["BPSK", "4-QAM", "16-QAM"])
def test_constellation_puissance_moyenne_unitaire(modulation):
    """Toutes les constellations sont normalisées à E[|s|^2] = 1."""
    table = table_constellation(modulation)
    b = bits_par_symbole(modulation)
    assert table.size == 2 ** b
    assert float(np.mean(np.abs(table) ** 2)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("modulation", ["BPSK", "4-QAM", "16-QAM"])
def test_aller_retour_modulation_demodulation(modulation):
    """Sans bruit, moduler puis démoduler restitue les bits d'origine."""
    rng = np.random.default_rng(5)
    b = bits_par_symbole(modulation)
    bits = bits_aleatoires(4000, rng)[: (4000 // b) * b]
    symboles = moduler(bits, modulation)
    assert symboles.size == bits.size // b
    np.testing.assert_array_equal(demoduler(symboles, modulation), bits)


def test_demoduler_separable_equivaut_au_plus_proche_voisin():
    """La décision séparable par axe égale l'argmin 2D sur tout point reçu."""
    rng = np.random.default_rng(6)
    for modulation in ("BPSK", "4-QAM", "16-QAM"):
        recus = rng.normal(size=(3, 500)) + 1j * rng.normal(size=(3, 500))
        np.testing.assert_array_equal(
            demoduler(recus, modulation), demoduler_bruteforce(recus, modulation)
        )


@pytest.mark.parametrize("modulation", ["4-QAM", "16-QAM"])
def test_constellations_codage_de_gray(modulation):
    """Deux points voisins de la constellation ne diffèrent que d'un bit."""
    table = table_constellation(modulation)
    n = table.size
    for indice in range(n):
        distances = np.abs(table - table[indice])
        distances[indice] = np.inf
        voisin = int(np.argmin(distances))
        assert bin(indice ^ voisin).count("1") == 1


def test_alias_nom_modulation():
    """Les alias usuels sont normalisés ; un nom inconnu lève ValueError."""
    assert normaliser_nom_modulation("qpsk") == "4-QAM"
    assert normaliser_nom_modulation("16qam") == "16-QAM"
    assert normaliser_nom_modulation("bpsk") == "BPSK"
    with pytest.raises(ValueError):
        normaliser_nom_modulation("8-QAM")


def test_moduler_taille_invalide():
    """Un nombre de bits non multiple du débit symbole lève ValueError."""
    with pytest.raises(ValueError):
        moduler([0, 1, 0], "16-QAM")     # 3 bits non multiple de 4
    with pytest.raises(ValueError):
        moduler([0, 2], "BPSK")          # valeur non binaire


def test_demoduler_sequence_vide():
    """Une séquence vide ne doit pas lever d'erreur."""
    assert demoduler(np.array([]), "16-QAM").size == 0
    assert moduler(np.array([], dtype=np.uint8), "16-QAM").size == 0
