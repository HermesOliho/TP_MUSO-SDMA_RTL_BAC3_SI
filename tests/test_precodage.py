"""Tests unitaires de ``src.precodage_zf`` (précodage et normalisation).

Le test de non-régression reproduit le canal de référence du TP
(graine historique 42, générateur MT19937 de ``numpy.random``) pour le
quel ``alpha ≈ 1.10307513367`` avait été validé à l'étape 3.
"""

import numpy as np
import pytest

from src.canal import generer_canal
from src.precodage_zf import (
    CanalSingularError,
    diagnostics_canal,
    matrice_effective,
    normaliser_puissance,
    precodage_ridge,
    precodage_zf,
    verifier_dimensions_canal,
    verifier_puissance_normee,
)


def canal_de_reference(graine: int = 42) -> np.ndarray:
    """Reproduit le canal de Rayleigh de référence du TP (K=4, M=8)."""
    return generer_canal(4, 8, np.random.default_rng(graine))


def test_shapes_canal_et_precodage():
    """H a la forme (K, M) et W_ZF la forme (M, K)."""
    h = canal_de_reference()
    assert h.shape == (4, 8)
    w = precodage_zf(h)
    assert w.shape == (8, 4)
    g = matrice_effective(h, w)
    assert g.shape == (4, 4)


def test_h_wzf_proche_identite():
    """Pour un canal de rang plein, H @ W_ZF ≈ I_K."""
    h = canal_de_reference()
    w = precodage_zf(h)
    g = h @ w
    np.testing.assert_allclose(g, np.eye(4), atol=1e-10)


def test_pseudo_inverse_equivalente_formule():
    """pinv(H) et H^H inv(H H^H) concordent pour un canal de rang plein."""
    h = canal_de_reference()
    w_pinv = precodage_zf(h, methode="pseudo_inverse")
    w_formule = precodage_zf(h, methode="formule")
    np.testing.assert_allclose(w_pinv, w_formule, atol=1e-12)


def test_normalisation_puissance():
    """La puissance normalisée est égale à P_T."""
    h = canal_de_reference()
    w = precodage_zf(h)
    w_norm, alpha = normaliser_puissance(w, p_t=1.0)
    assert alpha > 0
    puissance = float(np.linalg.norm(w_norm, "fro") ** 2)
    assert puissance == pytest.approx(1.0, abs=1e-9)
    assert verifier_puissance_normee(w_norm, 1.0)
    # Puissance cible quelconque.
    w_norm2, _ = normaliser_puissance(w, p_t=2.5)
    assert float(np.linalg.norm(w_norm2, "fro") ** 2) == pytest.approx(2.5, abs=1e-9)


def test_canal_effectif_apres_normalisation():
    """Après normalisation, G = alpha * I_K (diagonale = alpha, hors diagonale ≈ 0)."""
    h = canal_de_reference()
    w = precodage_zf(h)
    w_norm, alpha = normaliser_puissance(w, p_t=1.0)
    g = matrice_effective(h, w_norm)

    masque_diagonale = np.eye(4, dtype=bool)
    np.testing.assert_allclose(np.abs(g[masque_diagonale]), alpha, atol=1e-10)
    np.testing.assert_allclose(g[~masque_diagonale], 0.0, atol=1e-10)
    # La diagonale est réelle positive : pas de phase résiduelle.
    np.testing.assert_allclose(np.angle(g[masque_diagonale]), 0.0, atol=1e-10)


def test_non_regression_alpha_canal_historique():
    """Non-régression : alpha du canal de référence historique.

    Le canal est généré avec le générateur MT19937 historique
    (``np.random.RandomState(42)``) qui produisait le canal validé à
    l'étape 3, avec ``alpha ≈ 1.10307513367``.
    """
    rs = np.random.RandomState(42)
    h = (rs.randn(4, 8) + 1j * rs.randn(4, 8)) / np.sqrt(2)

    w = precodage_zf(h, methode="formule")
    _, alpha = normaliser_puissance(w, p_t=1.0)
    assert alpha == pytest.approx(1.10307513367, abs=1e-9)

    # La même valeur est obtenue avec la pseudo-inverse par défaut.
    w_pinv = precodage_zf(h)
    _, alpha_pinv = normaliser_puissance(w_pinv, p_t=1.0)
    assert alpha_pinv == pytest.approx(alpha, abs=1e-12)


def test_k_superieur_a_m_leve_erreur():
    """ZF impossible si K > M."""
    h = np.ones((5, 4), dtype=complex) + np.eye(5, 4, dtype=complex)
    with pytest.raises(ValueError, match="K <= M"):
        precodage_zf(h)


def test_canal_singular_leve_erreur():
    """Un canal de rang deficient doit lever CanalSingularError."""
    # Deux lignes identiques : rang 1 < K = 2.
    h = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]], dtype=complex)
    with pytest.raises(CanalSingularError, match="rang plein"):
        precodage_zf(h)


def test_methode_inconnue():
    """Une méthode de calcul inconnue doit lever ValueError."""
    with pytest.raises(ValueError, match="Méthode inconnue"):
        precodage_zf(canal_de_reference(), methode="toto")


def test_canal_2d_requis():
    """Une matrice non 2D doit lever ValueError."""
    with pytest.raises(ValueError):
        verifier_dimensions_canal(np.ones(4))
    with pytest.raises(ValueError):
        precodage_zf(np.ones(4))


def test_diagnostics_canal():
    """Les diagnostics reflètent le rang et le conditionnement."""
    h = canal_de_reference()
    diag = diagnostics_canal(h)
    assert diag.k == 4 and diag.m == 8
    assert diag.rang == 4
    assert diag.conditionnement >= 1.0
    assert diag.valeurs_singulieres.shape == (4,)
    assert diag.warning is None  # Canal de référence bien conditionné.

    # Un canal mal conditionné (lignes quasi parallèles) déclenche un warning.
    h_mal = np.array([[1.0, 0.0], [1.0, 1e-10]], dtype=complex)
    diag_mal = diagnostics_canal(h_mal)
    assert diag_mal.warning is not None


def test_precodage_ridge_option_distincte():
    """Le précodeur régularisé est une option distincte, normalisée."""
    h = canal_de_reference()
    w_ridge = precodage_ridge(h, mu=0.1, p_t=1.0)
    assert w_ridge.shape == (8, 4)
    assert float(np.linalg.norm(w_ridge, "fro") ** 2) == pytest.approx(1.0, abs=1e-9)
    # Le ridge n'annule pas exactement les interférences (ce n'est pas du ZF).
    g = h @ w_ridge
    interferences = np.abs(g[~np.eye(4, dtype=bool)])
    assert float(np.max(interferences)) > 1e-8
