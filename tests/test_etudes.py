"""Tests des études Monte-Carlo du mode complet et de leurs figures.

Ces tests utilisent une configuration réduite (peu de bits, peu de
réalisations, peu de points de courbe) afin de rester rapides tout en
couvrant le code des études « questionnaire » : BER vs SNR par M,
BER vs M pour les trois modulations, BER vs K, conditionnement
kappa(H H^H) contre BER, canaux corrélés et canaux utilisateurs
similaires, ainsi que la figure de reconstruction multi-SNR.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src import figures
from src.config import ConfigurationTP
from src.simulations import (
    courbe_ber_moyen_vs_m,
    courbe_ber_snr,
    courbe_ber_snr_par_m,
    courbe_ber_utilisateurs,
    etude_canaux_correles,
    etude_canaux_similaires,
    etude_conditionnement,
)


@pytest.fixture
def config_reduite():
    """Configuration allégée pour exercer toutes les études rapidement."""
    config = ConfigurationTP(
        m=8,
        k=4,
        snr_db=10.0,
        graine=1,
        modulation="BPSK",
        snr_db_courbe=(0.0, 10.0),
        snr_db_correles=(0.0, 10.0),
        n_bits_courbe=2000,
        n_realisations=2,
        valeurs_m=(4, 8),
        valeurs_k=(2, 4),
        modulations_etude=("BPSK", "4-QAM"),
        rho_correles=(0.0, 0.5),
        valeurs_beta=(0.0, 0.9),
        m_etude_k=8,
        snr_db_etude_m=10.0,
        snr_reconstruction=(5.0, 15.0),
        n_bits_conditionnement=5000,
        n_realisations_conditionnement=10,
    )
    config.valider()
    return config


def test_courbe_ber_snr_structure(config_reduite, tmp_path):
    """La courbe ZF vs sans précodage a la bonne structure et se trace."""
    resultats = courbe_ber_snr(config_reduite)
    assert len(resultats["zf"]) == len(config_reduite.snr_db_courbe)
    assert len(resultats["sans_precodage"]) == len(config_reduite.snr_db_courbe)
    chemin = figures.figure_ber_snr(resultats, tmp_path / "ber_snr.png")
    assert chemin.is_file()


def test_courbe_ber_snr_par_m(config_reduite, tmp_path):
    """BER_k et BER moyen sont disponibles pour chaque M."""
    resultats = courbe_ber_snr_par_m(config_reduite)
    assert set(resultats["resultats"]) == set(config_reduite.valeurs_m)
    premier_point = resultats["resultats"][config_reduite.valeurs_m[0]][0]
    assert premier_point.ber_par_utilisateur is not None
    assert premier_point.ber_par_utilisateur.size == config_reduite.k
    chemin = figures.figure_ber_snr_par_m(resultats, tmp_path / "pm.png")
    assert chemin.is_file()


def test_courbe_ber_moyen_vs_m(config_reduite, tmp_path):
    """BER moyen en fonction de M, pour chaque modulation étudiée."""
    resultats = courbe_ber_moyen_vs_m(config_reduite)
    assert set(resultats["resultats"]) == set(config_reduite.modulations_etude)
    chemin = figures.figure_ber_moyen_vs_m(resultats, tmp_path / "mm.png")
    assert chemin.is_file()


def test_courbe_ber_utilisateurs(config_reduite, tmp_path):
    """BER moyen en fonction de K, avec M fixé à m_etude_k."""
    resultats = courbe_ber_utilisateurs(config_reduite)
    assert resultats["m"] == config_reduite.m_etude_k
    assert len(resultats["zf"]) == len(config_reduite.valeurs_k)
    chemin = figures.figure_ber_utilisateurs(resultats, tmp_path / "k.png")
    assert chemin.is_file()


def test_etude_conditionnement(config_reduite, tmp_path):
    """Le conditionnement et le BER sont relevés par réalisation."""
    resultats = etude_conditionnement(config_reduite)
    assert len(resultats["kappa"]) == len(resultats["ber"])
    assert 1 <= len(resultats["kappa"]) <= config_reduite.n_realisations_conditionnement
    assert all(kappa >= 1.0 for kappa in resultats["kappa"])
    chemin = figures.figure_conditionnement_ber(resultats, tmp_path / "cond.png")
    assert chemin.is_file()


def test_etude_canaux_similaires(config_reduite, tmp_path):
    """Le BER est relevé pour chaque degré de similitude beta."""
    resultats = etude_canaux_similaires(config_reduite)
    assert len(resultats["ber"]) == len(config_reduite.valeurs_beta)
    chemin = figures.figure_canaux_similaires(resultats, tmp_path / "sim.png")
    assert chemin.is_file()


def test_etude_canaux_correles(config_reduite, tmp_path):
    """Le BER est relevé pour chaque coefficient de corrélation rho."""
    resultats = etude_canaux_correles(config_reduite)
    assert len(resultats["ber"]) == len(config_reduite.rho_correles)
    chemin = figures.figure_canaux_correles(resultats, tmp_path / "corr.png")
    assert chemin.is_file()


def test_figure_images_reconstruites(config_reduite, tmp_path):
    """La grille original/reconstruit se génère pour plusieurs SNR."""
    image = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
    reconstructions = {
        (modulation, snr): image
        for modulation in config_reduite.modulations_etude
        for snr in config_reduite.snr_reconstruction
    }
    chemin = figures.figure_images_reconstruites(
        image, reconstructions, config_reduite.snr_reconstruction,
        config_reduite.modulations_etude, tmp_path / "images.png", utilisateur=1,
    )
    assert chemin.is_file()
