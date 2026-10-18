"""Configuration pytest pour les tests du TP MU-MISO.

Ce ``conftest.py`` garantit que la racine du projet figure dans
``sys.path``, ce qui permet d'importer le package ``src`` depuis les
tests, quel que soit le répertoire courant.
"""

import sys
from pathlib import Path

import pytest

# Racine du projet : le dossier parent de tests/, qui contient src/...
RACINE_PROJET = Path(__file__).resolve().parent.parent

if str(RACINE_PROJET) not in sys.path:
    sys.path.insert(0, str(RACINE_PROJET))


@pytest.fixture
def images_reelles():
    """Les quatre images sources du projet, ou saut du test si absentes."""
    from src.config import DOSSIER_ORIGINALES, NOMS_IMAGES_DEFAUT

    chemins = [DOSSIER_ORIGINALES / nom for nom in NOMS_IMAGES_DEFAUT]
    if not all(chemin.is_file() for chemin in chemins):
        pytest.skip("Images sources absentes de data/originales/ : test sauté.")

    from src.images import charger_images

    return charger_images()
