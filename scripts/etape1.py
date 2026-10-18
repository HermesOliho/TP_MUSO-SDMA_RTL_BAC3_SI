
# ============================================================
# TP MU-MISO : ETAPE 1
# Chargement des images, conversion en bits, modulation BPSK,
# transmission sans bruit et reconstruction des images.
# A ce stade, le canal radio MU-MISO et le précodage ZF
# ne sont pas encore utilises.
# ============================================================

# Importer Path pour construire des chemins de fichiers fiables.
from pathlib import Path

# Importer NumPy pour manipuler les tableaux de pixels et les bits.
import numpy as np

# Importer Image pour lire et convertir les images JPEG.
from PIL import Image

# Importer pyplot pour afficher les images et les resultats.
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. DEFINIR LES PARAMETRES DU PROGRAMME
# ------------------------------------------------------------

# Recuperer le dossier dans lequel se trouve ce script Python.
DOSSIER_PROJET = Path(__file__).resolve().parent.parent

# Indiquer le dossier contenant les quatre images.
DOSSIER_IMAGES = DOSSIER_PROJET / "data" / "originales"

# Indiquer les noms des fichiers images.
# Remplacer ces noms par les noms reels de vos fichiers.
noms_images = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg",
    "image4.jpg"
]

# Le nombre d'utilisateurs correspond au nombre d'images.
K = len(noms_images)

# Creer des listes pour conserver les donnees de chaque utilisateur.
images_gris = []
bits_originaux = []
symboles_emis = []
bits_recus = []
images_reconstruites = []
ber_utilisateurs = []
dimensions_images = []


# ------------------------------------------------------------
# 2. CHARGER LES IMAGES ET LES CONVERTIR EN NIVEAUX DE GRIS
# ------------------------------------------------------------

# Parcourir les quatre fichiers, un utilisateur a la fois.
for k, nom in enumerate(noms_images):

    # Construire le chemin complet du fichier image.
    chemin_image = DOSSIER_IMAGES / nom

    # Verifier que le fichier existe avant de le charger.
    if not chemin_image.is_file():
        raise FileNotFoundError(
            f"Image introuvable : {chemin_image}\n"
            "Verifiez le nom du fichier et le dossier images."
        )

    # Ouvrir l'image JPEG.
    with Image.open(chemin_image) as image:

        # Convertir l'image en niveaux de gris.
        # Chaque pixel est alors un entier entre 0 et 255.
        image_gris = image.convert("L")

        # Convertir l'image en tableau NumPy de pixels uint8.
        tableau_pixels = np.array(image_gris, dtype=np.uint8)

    # Recuperer la hauteur et la largeur de l'image.
    hauteur, largeur = tableau_pixels.shape

    # Conserver l'image et ses dimensions.
    images_gris.append(tableau_pixels)
    dimensions_images.append((hauteur, largeur))

    # Afficher les informations utiles pour cet utilisateur.
    print(f"\nUtilisateur {k + 1}")
    print(f"Fichier : {nom}")
    print(f"Dimensions : {hauteur} x {largeur} pixels")
    print(f"Nombre de pixels : {hauteur * largeur}")


# ------------------------------------------------------------
# 3. CONVERTIR LES PIXELS EN BITS
# ------------------------------------------------------------

# Traiter chaque image indépendamment.
for k, image in enumerate(images_gris):

    # Transformer l'image 2D en une suite 1D de pixels.
    pixels = image.flatten()

    # Convertir chaque pixel de 8 bits en ses 8 bits individuels.
    # Par exemple, le pixel 5 devient 00000101.
    bits = np.unpackbits(pixels)

    # Conserver la sequence binaire de cet utilisateur.
    bits_originaux.append(bits)

    # Afficher le nombre de bits obtenus.
    print(
        f"Utilisateur {k + 1} : "
        f"{len(bits)} bits"
    )


# ------------------------------------------------------------
# 4. MODULER LES BITS EN BPSK
# ------------------------------------------------------------

# Convention choisie :
# bit 0 -> symbole -1
# bit 1 -> symbole +1

# Traiter les flux binaires des utilisateurs.
for k, bits in enumerate(bits_originaux):

    # Convertir les bits 0 et 1 en symboles BPSK -1 et +1.
    symboles = 2.0 * bits.astype(np.float64) - 1.0

    # Conserver les symboles emis.
    symboles_emis.append(symboles)

    # Afficher le nombre de symboles emis.
    # En BPSK, un bit correspond a un symbole.
    print(
        f"Utilisateur {k + 1} : "
        f"{len(symboles)} symboles BPSK"
    )


# ------------------------------------------------------------
# 5. SIMULER UNE TRANSMISSION SANS BRUIT
# ------------------------------------------------------------

# Dans ce premier test ideal, le canal est l’identité :
# le récepteur obtient exactement les symboles émis.
# Aucun bruit, aucune attenuation et aucune interférence.
for k, symboles in enumerate(symboles_emis):

    # Copier les symboles émis pour representer les symboles reçus.
    symboles_recus = symboles.copy()

    # Conserver les symboles reçus.
    # Le symbole n'est pas modifié durant cette transmission idéale.
    # Nous allons ensuite démoduler ces symboles.


    # --------------------------------------------------------
    # 6. DÉMODULER LES SYMBOLES BPSK
    # --------------------------------------------------------

    # Si le symbole est négatif, decider que le bit est 0.
    # Sinon, decider que le bit est 1.
    bits = (symboles_recus >= 0).astype(np.uint8)

    # Conserver les bits démodulés.
    bits_recus.append(bits)


# ------------------------------------------------------------
# 7. CALCULER LE TAUX D'ERREUR BINAIRE (BER)
# ------------------------------------------------------------

# Comparer les bits originaux et les bits démodulés.
for k in range(K):

    # Compter le nombre de bits différents.
    nombre_erreurs = np.count_nonzero(
        bits_originaux[k] != bits_recus[k]
    )

    # Compter le nombre total de bits transmis.
    nombre_bits = len(bits_originaux[k])

    # Calculer le BER : erreurs / nombre total de bits.
    ber = nombre_erreurs / nombre_bits

    # Conserver le BER de cet utilisateur.
    ber_utilisateurs.append(ber)

    # Afficher le résultat.
    print(f"\nUtilisateur {k + 1}")
    print(f"Nombre de bits : {nombre_bits}")
    print(f"Nombre d'erreurs : {nombre_erreurs}")
    print(f"BER = {ber:.8f}")


# ------------------------------------------------------------
# 8. RECONSTRUIRE LES IMAGES A PARTIR DES BITS RECUS
# ------------------------------------------------------------

# Traiter chaque utilisateur.
for k in range(K):

    # Récupérer les dimensions originales de l'image en niveaux
    # de gris pour cet utilisateur.
    hauteur, largeur = dimensions_images[k]

    # Regrouper chaque ensemble de 8 bits en un pixel uint8.
    pixels_recus = np.packbits(bits_recus[k])

    # Remettre les pixels dans leur forme d'image 2D.
    image_reconstruite = pixels_recus.reshape(
        hauteur, largeur
    )

    # Conserver l'image reconstruite.
    images_reconstruites.append(image_reconstruite)

    # Comparer pixel par pixel les images en niveaux de gris.
    identiques = np.array_equal(
        images_gris[k],
        image_reconstruite
    )

    # Afficher le résultat de la comparaison.
    print(f"\nUtilisateur {k + 1}")
    print(f"Images identiques : {identiques}")


# ------------------------------------------------------------
# 9. AFFICHER LES IMAGES ORIGINALES ET RECONSTRUITES
# ------------------------------------------------------------

# Créer une figure avec 4 lignes et 2 colonnes.
# Chaque ligne correspond a un utilisateur.
# Colonne gauche : image en niveaux de gris originale.
# Colonne droite : image reconstruite apres transmission.
fig, axes = plt.subplots(
    nrows=K,
    ncols=2,
    figsize=(10, 4 * K),
    constrained_layout=True
)

# Parcourir les utilisateurs pour remplir la figure.
for k in range(K):

    # Afficher l'image originale en niveaux de gris.
    axes[k, 0].imshow(images_gris[k], cmap="gray", vmin=0, vmax=255)
    axes[k, 0].set_title(f"Utilisateur {k + 1} - Originale")

    # Afficher l'image reconstruite.
    axes[k, 1].imshow(
        images_reconstruites[k],
        cmap="gray",
        vmin=0,
        vmax=255
    )
    axes[k, 1].set_title(
        f"Utilisateur {k + 1} - Reconstruite"
    )

    # Retirer les axes pour mieux voir les images.
    axes[k, 0].axis("off")
    axes[k, 1].axis("off")

# Afficher la figure.
plt.show()


# ------------------------------------------------------------
# 10. VERIFICATION FINALE
# ------------------------------------------------------------

# Verifier que tous les BER sont nuls.
tous_les_ber_sont_nuls = all(
    ber == 0.0 for ber in ber_utilisateurs
)

# Verifier que les quatre images ont ete reconstruites exactement.
toutes_les_images_sont_identiques = all(
    np.array_equal(images_gris[k], images_reconstruites[k])
    for k in range(K)
)

# Afficher le bilan final.
print("\n==========================================")
print("BILAN DE LA PREMIERE ETAPE")
print("==========================================")
print(f"Nombre d'utilisateurs : {K}")
print(f"Tous les BER sont nuls : {tous_les_ber_sont_nuls}")
print(
    "Toutes les images sont identiques : "
    f"{toutes_les_images_sont_identiques}"
)

# Signaler si les deux tests ont reussi.
if tous_les_ber_sont_nuls and toutes_les_images_sont_identiques:
    print("SUCCES : la chaine numerique ideale est validee.")
else:
    print("ECHEC : verifier les conversions et les dimensions.")
