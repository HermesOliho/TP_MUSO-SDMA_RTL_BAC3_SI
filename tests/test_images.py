"""Tests unitaires de ``src.images`` (conversions pixels <-> bits)."""

from pathlib import Path

import numpy as np
import pytest

from src.images import (
    NB_BITS_PAR_PIXEL,
    charger_image,
    charger_images,
    convertir_en_bits,
    informations_images,
    reconstruire_image,
    sauvegarder_image,
    verifier_identite,
)


def test_conversion_bits_msb_en_tete():
    """Le pixel 5 (0b00000101) doit donner les bits [0,0,0,0,0,1,0,1]."""
    image = np.array([[5]], dtype=np.uint8)
    bits = convertir_en_bits(image)
    np.testing.assert_array_equal(bits, [0, 0, 0, 0, 0, 1, 0, 1])


def test_conversion_pixels_bits_pixels_exacte():
    """La chaîne pixels -> bits -> pixels doit être exacte."""
    rng = np.random.default_rng(0)
    for forme in [(3, 5), (4, 4), (1, 17), (10, 13), (2, 2)]:
        image = rng.integers(0, 256, size=forme).astype(np.uint8)
        bits = convertir_en_bits(image)
        assert bits.size == image.size * NB_BITS_PAR_PIXEL
        image_rec = reconstruire_image(bits, *forme)
        assert np.array_equal(image, image_rec)


def test_dimensions_differentes_conservees(images_reelles):
    """Les quatre images réelles conservent leurs propres dimensions."""
    formes = [img.shape for img in images_reelles]
    assert len(formes) == 4
    for forme in formes:
        assert len(forme) == 2 and min(forme) > 0
    # Le projet contient des images de tailles différentes entre elles.
    assert len(set(formes)) > 1

    for info in informations_images(images_reelles):
        img = images_reelles[info["utilisateur"] - 1]
        assert info["hauteur"] == img.shape[0]
        assert info["largeur"] == img.shape[1]
        assert info["bits"] == img.size * NB_BITS_PAR_PIXEL


def test_reconstruire_image_longueur_invalide():
    """Un nombre de bits incohérent doit lever ValueError."""
    with pytest.raises(ValueError):
        reconstruire_image(np.zeros(10, dtype=np.uint8), 3, 5)
    with pytest.raises(ValueError):
        reconstruire_image(np.zeros(120, dtype=np.uint8), -1, 15)


def test_charger_image_introuvable():
    """Un fichier absent doit lever FileNotFoundError avec le chemin."""
    with pytest.raises(FileNotFoundError, match="Image introuvable"):
        charger_image(Path("data") / "originales" / "fichier_inexistant_xyz.jpg")


def test_charger_images_reelles(images_reelles):
    """Chargement des quatre images sources en niveaux de gris uint8."""
    assert len(images_reelles) == 4
    for img in images_reelles:
        assert img.ndim == 2
        assert img.dtype == np.uint8
        assert img.min() >= 0 and img.max() <= 255


def test_charger_images_liste_vide():
    """Une liste de noms vide doit lever ValueError."""
    with pytest.raises(ValueError):
        charger_images(noms=[])


def test_verifier_identite():
    """Comparaison pixel par pixel exacte et inexacte."""
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    b = a.copy()
    assert verifier_identite(a, b)
    b[1, 2] = 255
    assert not verifier_identite(a, b)


def test_sauvegarder_et_recharger(tmp_path):
    """Sauvegarde puis rechargement d'une image : fidélité exacte."""
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(6, 7)).astype(np.uint8)
    chemin = sauvegarder_image(tmp_path / "sortie.png", image)
    assert chemin.is_file()
    rechargement = charger_image(chemin)
    assert np.array_equal(image, rechargement)
