"""Tests unitaires et d'intégration de ``src.simulations`` et ``src.main``.

Ces tests couvrent l'alignement des flux de longueurs différentes, le
modèle ``Y = H W S + N``, la transmission sans bruit exacte (y compris
sur les images réelles du projet), l'exclusion du remplissage du BER,
les points Monte-Carlo et le mode rapide du programme principal.
"""

import time

import numpy as np
import pytest

from src.canal import generer_canal, sigma2_depuis_snr_db
from src.detection import detecter_bpsk, egaliser, evaluer_reception, gains_utiles, transmettre
from src.images import convertir_en_bits, reconstruire_image
from src.modulation import bits_par_symbole, moduler_bpsk
from src.precodage_zf import normaliser_puissance, precodage_zf
from src.simulations import (
    ResultatPoint,
    aligner_flux,
    simuler_ber_point,
    transmettre_images,
)


# ------------------------------------------------------------
# Alignement des flux de longueurs différentes
# ------------------------------------------------------------

def test_aligner_flux_dimensions_et_masque():
    """Les flux sont padés à la longueur max et correctement masqués."""
    s1 = moduler_bpsk(np.array([0, 1, 1], dtype=np.uint8))   # 3 symboles
    s2 = moduler_bpsk(np.array([1, 0], dtype=np.uint8))      # 2 symboles
    s_bloc, masque = aligner_flux([s1, s2])

    assert s_bloc.shape == (2, 3)
    assert masque.shape == (2, 3)
    np.testing.assert_array_equal(masque, [[True, True, True], [True, True, False]])
    np.testing.assert_array_equal(s_bloc[0], s1)
    np.testing.assert_array_equal(s_bloc[1, :2], s2)
    assert s_bloc[1, 2] == 1.0  # Symbole de remplissage.


def test_aligner_flux_invalides():
    """Liste vide ou flux vide doivent lever ValueError."""
    with pytest.raises(ValueError):
        aligner_flux([])
    with pytest.raises(ValueError):
        aligner_flux([np.array([], dtype=np.float64)])


# ------------------------------------------------------------
# Modèle de transmission Y = H W S + N
# ------------------------------------------------------------

def test_transmettre_sans_bruit_conforme_modele():
    """Sans bruit, Y doit être exactement H @ W @ S."""
    rng = np.random.default_rng(11)
    k, m, t = 4, 8, 7
    h = generer_canal(k, m, rng)
    w = precodage_zf(h)
    w_norm, _ = normaliser_puissance(w, 1.0)
    s = 2.0 * rng.integers(0, 2, size=(k, t)) - 1.0

    y = transmettre(s, h, w_norm, sigma2=0.0)
    assert y.shape == (k, t)
    np.testing.assert_allclose(y, h @ w_norm @ s, atol=1e-12)


def test_transmettre_dimensions_incoherentes():
    """Des dimensions incohérentes doivent lever ValueError."""
    rng = np.random.default_rng(0)
    h = generer_canal(4, 8, rng)
    w = precodage_zf(h)
    with pytest.raises(ValueError):
        transmettre(np.ones((3, 5)), h, w)  # 3 lignes pour K = 4.
    with pytest.raises(ValueError):
        transmettre(np.ones((4, 5)), h, np.ones((7, 4)))  # W incohérent.
    with pytest.raises(ValueError):
        transmettre(np.ones((4, 5)), h, w, sigma2=0.1, rng=None)  # rng requis.


def test_transmettre_avec_bruit():
    """Avec bruit, Y diffère du modèle sans bruit mais reste de bonne forme."""
    rng = np.random.default_rng(1)
    k, m, t = 4, 8, 2000
    h = generer_canal(k, m, rng)
    w_norm, _ = normaliser_puissance(precodage_zf(h), 1.0)
    s = 2.0 * rng.integers(0, 2, size=(k, t)) - 1.0

    sigma2 = sigma2_depuis_snr_db(10.0)
    y = transmettre(s, h, w_norm, sigma2=sigma2, rng=rng)
    y_sans = h @ w_norm @ s
    assert y.shape == (k, t)
    assert not np.allclose(y, y_sans)
    # Le bruit ajouté reste du même ordre que sqrt(sigma2).
    bruit = y - y_sans
    assert float(np.mean(np.abs(bruit) ** 2)) == pytest.approx(sigma2, rel=0.2)


def test_detection_bpsk_dimensions():
    """La détection produit des bits de même forme que le signal reçu."""
    y = np.array([[0.5 + 1j, -2.0 + 0j], [1e-12, -1e-12 + 1j]])
    bits = detecter_bpsk(y)
    assert bits.shape == (2, 2)
    assert bits.dtype == np.uint8
    np.testing.assert_array_equal(bits, [[1, 0], [1, 0]])


def test_evaluer_reception_exclut_remplissage():
    """Le BER n'inclut jamais les positions de remplissage."""
    k, t = 2, 10
    bits_emis = [np.zeros(4, dtype=np.uint8), np.ones(9, dtype=np.uint8)]
    # Signal reçu sans bruit : decisions exactes attendues.
    y = np.full((k, t), -0.5 + 0j)  # tout décidé à 0
    y[1, :] = 0.5 + 0j              # utilisateur 1 tout décidé à 1
    masque = np.zeros((k, t), dtype=bool)
    masque[0, :4] = True
    masque[1, :9] = True

    reception = evaluer_reception(bits_emis, y, masque)
    np.testing.assert_array_equal(reception.nb_bits, [4, 9])
    np.testing.assert_array_equal(reception.erreurs, [0, 0])
    assert reception.ber.nb_bits == 13
    assert reception.ber.erreurs == 0
    assert len(reception.bits_recus[0]) == 4
    assert len(reception.bits_recus[1]) == 9


# ------------------------------------------------------------
# Transmission sans bruit exacte (petites images synthétiques)
# ------------------------------------------------------------

def _petites_images():
    """Deux images synthétiques de tailles différentes, déterministes."""
    rng = np.random.default_rng(3)
    return [
        rng.integers(0, 256, size=(3, 5)).astype(np.uint8),
        rng.integers(0, 256, size=(4, 4)).astype(np.uint8),
        rng.integers(0, 256, size=(1, 7)).astype(np.uint8),
    ]


def test_transmission_sans_bruit_exacte_images_synthetiques():
    """Canal + ZF sans bruit : BER nul et images reconstruites exactes."""
    rng = np.random.default_rng(42)
    images = _petites_images()
    k, m = len(images), 8

    h = generer_canal(k, m, rng)
    w_norm, alpha = normaliser_puissance(precodage_zf(h), 1.0)

    resultat = transmettre_images(images, h, w_norm, alpha, sigma2=0.0)

    assert resultat.reception.ber.erreurs == 0
    assert resultat.reception.ber.ber == 0.0
    assert all(resultat.identiques)
    for image_source, image_rec in zip(images, resultat.images_reconstruites):
        assert image_rec.shape == image_source.shape
        assert np.array_equal(image_source, image_rec)


def test_transmission_sans_bruit_images_reelles(images_reelles):
    """Sur les quatre images réelles du projet, la reconstruction est exacte."""
    rng = np.random.default_rng(42)
    k, m = len(images_reelles), 8

    h = generer_canal(k, m, rng)
    w_norm, alpha = normaliser_puissance(precodage_zf(h), 1.0)

    resultat = transmettre_images(images_reelles, h, w_norm, alpha, sigma2=0.0)

    assert resultat.reception.ber.erreurs == 0
    assert all(resultat.identiques)
    # Chaque utilisateur garde ses propres dimensions et bits.
    for indice, image_source in enumerate(images_reelles):
        bits_attendus = image_source.size * 8
        assert resultat.reception.nb_bits[indice] == bits_attendus
        assert len(resultat.bits_emis[indice]) == bits_attendus
        assert resultat.images_reconstruites[indice].shape == image_source.shape


def test_dernier_bloc_partiel_pas_derreur_artificielle():
    """Le remplissage du dernier bloc n'introduit aucune erreur artificielle."""
    rng = np.random.default_rng(8)
    # Images de 10 et 25 pixels, donc 80 et 200 bits : le premier flux
    # est rempli sur 15 positions de bloc.
    images = [
        np.arange(10, dtype=np.uint8).reshape(5, 2),
        np.arange(25, dtype=np.uint8).reshape(5, 5),
    ]
    k, m = 2, 4
    h = generer_canal(k, m, rng)
    w_norm, alpha = normaliser_puissance(precodage_zf(h), 1.0)

    resultat = transmettre_images(images, h, w_norm, alpha, sigma2=0.0)

    np.testing.assert_array_equal(resultat.reception.nb_bits, [80, 200])
    assert resultat.reception.ber.erreurs == 0
    assert all(resultat.identiques)


# ------------------------------------------------------------
# Points Monte-Carlo
# ------------------------------------------------------------

def test_resultat_point_borne_observation():
    """La borne d'observation vaut 1 / N_bits."""
    point = ResultatPoint(ber=0.0, erreurs=0, nb_bits=200_000)
    assert point.borne_observation == pytest.approx(5e-6)


def test_simuler_ber_point_compte_bits():
    """Le comptage de bits est exact : n_bits x n_realisations x K."""
    rng = np.random.default_rng(13)
    point = simuler_ber_point(
        m=8, k=4, snr_db=10.0, n_bits=1000, n_realisations=3, rng=rng
    )
    assert point.nb_bits == 1000 * 3 * 4
    assert 0.0 <= point.ber <= 1.0
    assert point.erreurs == int(point.ber * point.nb_bits)


def test_simuler_ber_point_reproductible():
    """Deux appels avec la même graine donnent le même résultat."""
    p1 = simuler_ber_point(m=8, k=4, snr_db=8.0, n_bits=2000,
                           n_realisations=2, rng=np.random.default_rng(77))
    p2 = simuler_ber_point(m=8, k=4, snr_db=8.0, n_bits=2000,
                           n_realisations=2, rng=np.random.default_rng(77))
    assert p1.erreurs == p2.erreurs
    assert p1.ber == p2.ber


def test_simuler_ber_bruit_vs_snr():
    """Le BER doit être bien plus faible à haut SNR qu'à bas SNR."""
    bas = simuler_ber_point(m=8, k=4, snr_db=-2.0, n_bits=5000,
                            n_realisations=4, rng=np.random.default_rng(21))
    haut = simuler_ber_point(m=8, k=4, snr_db=18.0, n_bits=5000,
                             n_realisations=4, rng=np.random.default_rng(21))
    assert bas.erreurs > 0            # à -2 dB, des erreurs sont attendues
    assert haut.ber < bas.ber / 10    # gain net de 20 dB


def test_simuler_ber_methode_invalide():
    """Méthode de précodage inconnue ou M < K doivent lever ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simuler_ber_point(m=8, k=4, snr_db=10.0, rng=rng, methode="toto")
    with pytest.raises(ValueError):
        simuler_ber_point(m=2, k=4, snr_db=10.0, rng=rng)


def test_simuler_ber_sans_precodage_plan_interferences():
    """Sans précodage, le BER plante à cause des interférences (même P_T)."""
    point = simuler_ber_point(m=8, k=4, snr_db=30.0, n_bits=5000,
                              n_realisations=4, rng=np.random.default_rng(31),
                              methode="aucun")
    # Interférence inter-utilisateurs persistante même à très haut SNR.
    assert point.ber > 0.05


# ------------------------------------------------------------
# Mode rapide du programme principal
# ------------------------------------------------------------

def test_mode_rapide_complet_et_rapide(tmp_path):
    """Le mode rapide valide toute la chaîne sans les courbes Monte-Carlo."""
    from src.main import principale

    dossier_figures = tmp_path / "figures"
    dossier_reconstruites = tmp_path / "reconstruites"

    debut = time.perf_counter()
    code = principale(
        ["--mode", "rapide", "--snr-db", "12", "--graine", "42"],
        dossier_figures=dossier_figures,
        dossier_reconstruites=dossier_reconstruites,
    )
    duree = time.perf_counter() - debut

    assert code == 0
    # Les figures du mode rapide sont produites...
    for nom in ("schema_systeme.png", "partage_sdma.png",
                "heatmap_G.png", "constellations.png"):
        assert (dossier_figures / nom).is_file(), f"figure manquante : {nom}"
    # ...mais pas les courbes Monte-Carlo du mode complet.
    for nom in ("ber_snr.png", "ber_snr_par_m.png", "ber_moyen_vs_m.png",
                "ber_utilisateurs.png", "conditionnement_ber.png",
                "canaux_correles.png", "canaux_similaires.png",
                "images_reconstruites.png"):
        assert not (dossier_figures / nom).is_file(), (
            f"le mode rapide ne doit pas produire {nom}"
        )
    # Les quatre images reconstruites sont enregistrées.
    for indice in range(1, 5):
        assert (dossier_reconstruites / f"utilisateur{indice}_recue.png").is_file()
    # Rapidité : le mode rapide ne doit pas lancer les longues études.
    assert duree < 120.0


def test_mode_rapide_config_surchargee(tmp_path):
    """M et K configurables sans modifier plusieurs fichiers."""
    from src.main import principale

    code = principale(
        ["--mode", "rapide", "--m", "6", "--k", "3", "--snr-db", "5"],
        dossier_figures=tmp_path / "figures",
        dossier_reconstruites=tmp_path / "reconstruites",
    )
    assert code == 0
    assert (tmp_path / "figures" / "heatmap_G.png").is_file()


def test_reconstruction_depuis_bits():
    """aller-retour bits -> image via la chaîne BPSK complète."""
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, size=(5, 6)).astype(np.uint8)
    bits = convertir_en_bits(image)
    symboles = moduler_bpsk(bits)
    bits_rec = detecter_bpsk(symboles)
    image_rec = reconstruire_image(bits_rec, 5, 6)
    assert np.array_equal(image, image_rec)


# ------------------------------------------------------------
# Égalisation ŝ_k = y_k / g_k
# ------------------------------------------------------------

def test_egaliser_compense_le_gain_utile():
    """Après égalisation par g_k, le signal reçu redonne les symboles émis."""
    rng = np.random.default_rng(11)
    h = generer_canal(4, 8, rng)
    w_norm, _ = normaliser_puissance(precodage_zf(h), 1.0)
    s = rng.normal(size=(4, 50)) + 1j * rng.normal(size=(4, 50))

    y = h @ w_norm @ s
    y_egalise = egaliser(y, gains_utiles(h, w_norm))
    np.testing.assert_allclose(y_egalise, s, atol=1e-10)


def test_egaliser_gain_nul_leve_erreur():
    """Un gain utile nul rend l'égalisation impossible."""
    with pytest.raises(ValueError):
        egaliser(np.ones((2, 3), dtype=complex), np.array([1.0, 0.0]))


# ------------------------------------------------------------
# Modulations 4-QAM et 16-QAM de bout en bout
# ------------------------------------------------------------

@pytest.mark.parametrize("modulation", ["BPSK", "4-QAM", "16-QAM"])
def test_transmission_sans_bruit_qam(modulation):
    """Canal + ZF sans bruit : reconstruction exacte pour les trois modulations."""
    rng = np.random.default_rng(42)
    images = _petites_images()
    h = generer_canal(len(images), 8, rng)
    w_norm, alpha = normaliser_puissance(precodage_zf(h), 1.0)

    resultat = transmettre_images(images, h, w_norm, alpha, sigma2=0.0,
                                  modulation=modulation)

    assert resultat.reception.ber.erreurs == 0
    assert all(resultat.identiques)
    for image_source, image_rec in zip(images, resultat.images_reconstruites):
        assert np.array_equal(image_source, image_rec)


@pytest.mark.parametrize("modulation", ["BPSK", "4-QAM", "16-QAM"])
def test_simuler_ber_point_qam_compte_bits(modulation):
    """Le comptage de bits tient compte du débit symbole de la modulation."""
    b = bits_par_symbole(modulation)
    n_effectif = (1000 // b) * b
    point = simuler_ber_point(
        m=8, k=4, snr_db=10.0, n_bits=1000, n_realisations=3,
        rng=np.random.default_rng(1), modulation=modulation,
    )
    assert point.nb_bits == n_effectif * 3 * 4


def test_ber_16qam_decroit_avec_le_snr():
    """Le BER 16-QAM décroît nettement lorsque le SNR augmente."""
    bas = simuler_ber_point(m=8, k=4, snr_db=0.0, n_bits=5000, n_realisations=4,
                            rng=np.random.default_rng(2), modulation="16-QAM")
    haut = simuler_ber_point(m=8, k=4, snr_db=20.0, n_bits=5000, n_realisations=4,
                             rng=np.random.default_rng(2), modulation="16-QAM")
    assert haut.ber < bas.ber
