"""Tests unitaires de ``src.canal`` (Rayleigh, bruit AWGN, corrélation)."""

import numpy as np
import pytest

from src.canal import (
    appliquer_correlation_emission,
    generer_bruit,
    generer_canal,
    matrice_correlation_exponentielle,
    racine_matricielle_symmetrique,
    sigma2_depuis_snr_db,
    snr_db_depuis_sigma2,
)


def test_forme_du_canal():
    """Le canal a bien la forme (K, M) et un type complexe."""
    rng = np.random.default_rng(42)
    for k, m in [(4, 8), (1, 1), (6, 12)]:
        h = generer_canal(k, m, rng)
        assert h.shape == (k, m)
        assert np.iscomplexobj(h)


def test_canal_reproductible():
    """La même graine produit exactement le même canal."""
    h1 = generer_canal(4, 8, np.random.default_rng(7))
    h2 = generer_canal(4, 8, np.random.default_rng(7))
    np.testing.assert_array_equal(h1, h2)


def test_puissance_moyenne_canal():
    """E[|H[k, m]|^2] doit être proche de 1 (CN(0, 1))."""
    rng = np.random.default_rng(123)
    h = generer_canal(64, 64, rng)
    puissance_moyenne = float(np.mean(np.abs(h) ** 2))
    # Écart-type de la moyenne d'échantillon ≈ 1/sqrt(4096) ≈ 0.016.
    assert puissance_moyenne == pytest.approx(1.0, abs=0.06)


def test_moyenne_nulle():
    """Les parties réelle et imaginaire sont de moyenne nulle."""
    rng = np.random.default_rng(99)
    h = generer_canal(64, 64, rng)
    assert abs(np.mean(h.real)) < 0.05
    assert abs(np.mean(h.imag)) < 0.05


def test_dimensions_invalides():
    """Des dimensions nulles doivent lever ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generer_canal(0, 8, rng)
    with pytest.raises(ValueError):
        generer_canal(4, 0, rng)


def test_variance_du_bruit():
    """Le bruit CN(0, sigma2) a bien une variance totale sigma2."""
    rng = np.random.default_rng(5)
    sigma2 = 0.5
    n = generer_bruit(32, 32, sigma2, rng)
    variance = float(np.mean(np.abs(n) ** 2))
    assert n.shape == (32, 32)
    assert variance == pytest.approx(sigma2, rel=0.1)
    # Répartition égale entre réel et imaginaire.
    assert float(np.var(n.real)) == pytest.approx(sigma2 / 2, rel=0.15)
    assert float(np.var(n.imag)) == pytest.approx(sigma2 / 2, rel=0.15)


def test_bruit_nul():
    """sigma2 = 0 produit une matrice nulle exactement."""
    n = generer_bruit(4, 10, 0.0, np.random.default_rng(0))
    np.testing.assert_array_equal(n, np.zeros((4, 10)))


def test_conversion_snr_sigma2():
    """Convention sigma2 = P_T / 10^(SNR_dB/10)."""
    assert sigma2_depuis_snr_db(0.0, 1.0) == pytest.approx(1.0)
    assert sigma2_depuis_snr_db(10.0, 1.0) == pytest.approx(0.1)
    assert sigma2_depuis_snr_db(20.0, 1.0) == pytest.approx(0.01)
    assert sigma2_depuis_snr_db(3.0, 2.0) == pytest.approx(2.0 / 10 ** 0.3)
    # Aller-retour exact.
    for snr in (-5.0, 0.0, 7.5, 25.0):
        sigma2 = sigma2_depuis_snr_db(snr)
        assert snr_db_depuis_sigma2(sigma2) == pytest.approx(snr)


def test_matrice_correlation():
    """Propriétés du modèle de corrélation exponentiel."""
    r0 = matrice_correlation_exponentielle(4, 0.0)
    np.testing.assert_array_equal(r0, np.eye(4))

    r9 = matrice_correlation_exponentielle(4, 0.9)
    assert r9.shape == (4, 4)
    assert np.allclose(r9, r9.T)
    assert r9[0, 0] == 1.0
    assert r9[0, 1] == pytest.approx(0.9)
    assert r9[0, 2] == pytest.approx(0.81)
    # Symétrique définie positive.
    assert np.min(np.linalg.eigvalsh(r9)) > 0


def test_matrice_correlation_invalide():
    """rho hors de [0, 1[ ou taille nulle doivent lever ValueError."""
    with pytest.raises(ValueError):
        matrice_correlation_exponentielle(4, 1.0)
    with pytest.raises(ValueError):
        matrice_correlation_exponentielle(4, -0.2)
    with pytest.raises(ValueError):
        matrice_correlation_exponentielle(0, 0.5)


def test_racine_matricielle():
    """R**(1/2) @ R**(1/2) doit reproduire R."""
    r = matrice_correlation_exponentielle(5, 0.7)
    racine = racine_matricielle_symmetrique(r)
    np.testing.assert_allclose(racine @ racine, r, atol=1e-12)


def test_canal_correlation_appliquee():
    """Le canal corrélé conserve sa forme et sa puissance moyenne."""
    rng = np.random.default_rng(21)
    h = generer_canal(16, 8, rng)
    h_corr = appliquer_correlation_emission(h, 0.9)
    assert h_corr.shape == h.shape
    # rho = 0 : canal inchangé (copie).
    np.testing.assert_array_equal(appliquer_correlation_emission(h, 0.0), h)
    # rho > 0 : le canal change mais la puissance moyenne par coefficient
    # reste de l'ordre de 1 (diagonale de R égale à 1).
    assert not np.allclose(h_corr, h)
    assert float(np.mean(np.abs(h_corr) ** 2)) == pytest.approx(1.0, abs=0.3)
