"""Tests unitaires de ``src.modulation`` (BPSK et BER)."""

import numpy as np
import pytest

from src.modulation import (
    ResultatBER,
    ber_depuis_comptage,
    ber_global,
    bits_aleatoires,
    calculer_ber,
    demoduler_bpsk,
    moduler_bpsk,
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
