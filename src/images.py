"""Chargement des images et conversions entre pixels et bits.

Conventions
-----------
- Chaque image est convertie en niveaux de gris (mode ``L`` de Pillow) :
  chaque pixel est un entier ``uint8`` compris entre 0 et 255.
- Chaque pixel est codé sur **8 bits**, du bit de poids fort (MSB) au
  bit de poids faible (LSB), ce qui correspond au comportement par
  défaut de ``numpy.unpackbits``. Par exemple, le pixel de valeur 5
  devient ``[0, 0, 0, 0, 0, 1, 0, 1]``.
- Les dimensions (hauteur, largeur) de chaque image sont conservées
  telles quelles : aucune image n'est redimensionnée, même si les
  quatre images ont des tailles différentes.
- La reconstruction utilise ``numpy.packbits`` (opération inverse de
  ``unpackbits``) : dans le cas sans bruit, la reconstruction est
  exacte, pixel par pixel.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# Nombre de bits par pixel pour l'encodage des niveaux de gris.
NB_BITS_PAR_PIXEL = 8


def charger_image(chemin: Path) -> np.ndarray:
    """Charge une image JPEG et la convertit en niveaux de gris.

    Parameters
    ----------
    chemin : pathlib.Path
        Chemin complet du fichier image.

    Returns
    -------
    numpy.ndarray
        Tableau ``uint8`` de forme ``(hauteur, largeur)`` contenant les
        niveaux de gris, de 0 (noir) à 255 (blanc).

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si le fichier n'est pas une image lisible.
    """
    chemin = Path(chemin)

    if not chemin.is_file():
        raise FileNotFoundError(
            f"Image introuvable : {chemin}\n"
            "Vérifiez le nom du fichier et le chemin du dossier "
            "'data/originales' à la racine du projet."
        )

    try:
        with Image.open(chemin) as image:
            # Conversion en niveaux de gris : chaque pixel vaut 0..255.
            image_gris = image.convert("L")
            tableau = np.array(image_gris, dtype=np.uint8)
    except Exception as exc:  # pragma: no cover - dépend du fichier
        raise ValueError(
            f"Fichier image invalide ou illisible : {chemin}\n"
            f"Détail : {exc}"
        ) from exc

    if tableau.ndim != 2:
        raise ValueError(
            f"L'image '{chemin}' ne produit pas un tableau 2D après "
            "conversion en niveaux de gris."
        )

    return tableau


def charger_images(
    dossier=None,
    noms=None,
) -> list:
    """Charge plusieurs images JPEG en niveaux de gris.

    Parameters
    ----------
    dossier : pathlib.Path or str, optional
        Dossier contenant les images. Par défaut, le dossier
        ``data/originales`` du projet.
    noms : sequence of str, optional
        Noms des fichiers, dans l'ordre des utilisateurs. Par défaut :
        ``image1.jpg`` ... ``image4.jpg``.

    Returns
    -------
    list of numpy.ndarray
        Une image ``uint8`` 2D par utilisateur, avec ses propres
        dimensions.

    Raises
    ------
    FileNotFoundError
        Si un fichier demandé est absent ; le message indique
        explicitement le chemin recherché.
    """
    from .config import DOSSIER_ORIGINALES, NOMS_IMAGES_DEFAUT

    dossier = Path(dossier) if dossier is not None else DOSSIER_ORIGINALES
    noms = tuple(noms) if noms is not None else NOMS_IMAGES_DEFAUT

    if not noms:
        raise ValueError("Aucune image demandée : la liste de noms est vide.")

    return [charger_image(dossier / nom) for nom in noms]


def convertir_en_bits(image_gris: np.ndarray) -> np.ndarray:
    """Convertit une image en niveaux de gris en une séquence de bits.

    Les pixels sont parcourus ligne par ligne (ordre de balayage
    ``'C'`` de NumPy, c'est-à-dire gauche→droite puis haut→bas). Chaque
    pixel est décomposé en 8 bits, du bit de poids fort au bit de poids
    faible.

    Parameters
    ----------
    image_gris : numpy.ndarray
        Tableau ``uint8`` 2D de niveaux de gris.

    Returns
    -------
    numpy.ndarray
        Tableau 1D de bits (valeurs 0 ou 1), de longueur
        ``hauteur * largeur * 8``.
    """
    image_gris = np.asarray(image_gris)

    if image_gris.ndim != 2:
        raise ValueError(
            f"Attendu un tableau d'image 2D, obtenu {image_gris.ndim} dimension(s)."
        )
    if image_gris.size == 0:
        raise ValueError("L'image est vide : aucune donnée à convertir.")

    # Aplatir l'image puis coder chaque pixel uint8 sur 8 bits (MSB en tête).
    pixels = image_gris.astype(np.uint8).ravel()
    return np.unpackbits(pixels)


def reconstruire_image(bits: np.ndarray, hauteur: int, largeur: int) -> np.ndarray:
    """Reconstruit une image en niveaux de gris à partir de ses bits.

    Effectue l'opération inverse de :func:`convertir_en_bits` : les bits
    sont regroupés par 8 (MSB en tête) pour reconstituer des pixels
    ``uint8``, puis remis en forme selon les dimensions d'origine.

    Parameters
    ----------
    bits : numpy.ndarray
        Séquence de bits (0 ou 1), de longueur exactement
        ``hauteur * largeur * 8``.
    hauteur, largeur : int
        Dimensions d'origine de l'image.

    Returns
    -------
    numpy.ndarray
        Tableau ``uint8`` de forme ``(hauteur, largeur)``.

    Raises
    ------
    ValueError
        Si la longueur de la séquence binaire ne correspond pas
        exactement au nombre de pixels multiplié par 8.
    """
    bits = np.asarray(bits).astype(np.uint8).ravel()

    if hauteur < 1 or largeur < 1:
        raise ValueError(
            f"Dimensions invalides : {hauteur} x {largeur}."
        )

    attendu = hauteur * largeur * NB_BITS_PAR_PIXEL
    if bits.size != attendu:
        raise ValueError(
            f"Nombre de bits incorrect pour une image {hauteur}x{largeur} : "
            f"{bits.size} bits fournis, {attendu} bits attendus "
            f"({hauteur * largeur} pixels x {NB_BITS_PAR_PIXEL} bits)."
        )

    pixels = np.packbits(bits)
    return pixels.reshape(hauteur, largeur)


def sauvegarder_image(chemin: Path, image_gris: np.ndarray) -> Path:
    """Sauvegarde une image en niveaux de gris sur le disque.

    Le dossier de destination est créé s'il n'existe pas.

    Parameters
    ----------
    chemin : pathlib.Path
        Chemin du fichier de sortie (format déduit de l'extension).
    image_gris : numpy.ndarray
        Tableau ``uint8`` 2D de niveaux de gris.

    Returns
    -------
    pathlib.Path
        Chemin effectivement écrit.
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image_gris, dtype=np.uint8), mode="L").save(chemin)
    return chemin


def verifier_identite(image_a: np.ndarray, image_b: np.ndarray) -> bool:
    """Vérifie l'égalité pixel par pixel de deux images en niveaux de gris.

    La comparaison porte sur la représentation effectivement transmise,
    c'est-à-dire les niveaux de gris (et non l'image couleur d'origine).

    Parameters
    ----------
    image_a, image_b : numpy.ndarray
        Tableaux d'images à comparer.

    Returns
    -------
    bool
        ``True`` si les deux tableaux ont les mêmes dimensions et les
        mêmes valeurs de pixels.
    """
    return np.array_equal(np.asarray(image_a), np.asarray(image_b))


def informations_images(images: list) -> list:
    """Retourne un résumé des propriétés de chaque image.

    Parameters
    ----------
    images : list of numpy.ndarray
        Images en niveaux de gris, une par utilisateur.

    Returns
    -------
    list of dict
        Pour chaque image : dimensions, nombre de pixels, nombre de
        bits correspondant et chemin supposé.
    """
    return [
        {
            "utilisateur": k + 1,
            "hauteur": int(img.shape[0]),
            "largeur": int(img.shape[1]),
            "pixels": int(img.size),
            "bits": int(img.size) * NB_BITS_PAR_PIXEL,
        }
        for k, img in enumerate(images)
    ]
