"""Point d'entrée principal du TP MU-MISO / SDMA / Zero-Forcing.

Usage
-----
::

    python -m src.main --mode rapide          # validation complète en quelques secondes
    python -m src.main --mode complet         # + courbes BER Monte-Carlo et études
    python -m src.main --modulation 16-QAM    # modulation de la transmission (BPSK, 4-QAM, 16-QAM)
    python -m src.main --snr-db 15            # SNR de la transmission démonstrative (dB)
    python -m src.main --graine 7             # graine aléatoire
    python -m src.main --afficher             # affiche les figures en plus de les enregistrer

Le mode rapide exécute toute la chaîne (images -> bits -> modulation ->
canal -> ZF -> égalisation -> détection -> reconstruction) et produit
les figures structurales (schéma, SDMA, heatmap, constellations des
trois modulations). Le mode complet ajoute les expériences Monte-Carlo
coûteuses : BER vs SNR (ZF vs sans précodage), BER individuel et moyen
vs SNR pour chaque M, BER moyen vs M pour les trois modulations, BER
moyen vs K, étude du conditionnement, canaux corrélés, canaux
utilisateurs similaires, ainsi que la reconstruction des images à
plusieurs SNR (5, 15, 25 dB) pour les trois modulations.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .config import ConfigurationTP, configuration_complete, configuration_rapide
from .images import charger_images, informations_images, sauvegarder_image
from .modulation import ResultatBER, normaliser_nom_modulation
from .precodage_zf import (
    diagnostics_canal,
    matrice_effective,
    normaliser_puissance,
    precodage_zf,
    verifier_puissance_normee,
)


def analyseur_arguments(argv=None) -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (par défaut : ``sys.argv[1:]``).

    Returns
    -------
    argparse.Namespace
        Arguments validés.
    """
    analyseur = argparse.ArgumentParser(
        prog="python -m src.main",
        description=(
            "Simulation MU-MISO / SDMA avec précodage Zero-Forcing : "
            "transmission d'images à K utilisateurs via M antennes, "
            "modulation BPSK, 4-QAM ou 16-QAM."
        ),
    )
    analyseur.add_argument(
        "--mode",
        choices=["rapide", "complet", "quick", "full"],
        default="rapide",
        help=(
            "Mode d'exécution : 'rapide' (défaut) valide toute la chaîne sans "
            "les courbes Monte-Carlo ; 'complet' ajoute les études BER "
            "(les alias anglais quick/full sont acceptés)."
        ),
    )
    analyseur.add_argument(
        "--modulation",
        choices=["BPSK", "4-QAM", "QPSK", "16-QAM"],
        default="BPSK",
        help=(
            "Modulation utilisée pour la transmission démonstrative "
            "(défaut : BPSK). Les figures structurales et les études "
            "couvrent de toute façon les trois modulations."
        ),
    )
    analyseur.add_argument(
        "--snr-db",
        type=float,
        default=10.0,
        help="SNR en dB de la transmission démonstrative avec bruit (défaut : 10).",
    )
    analyseur.add_argument(
        "--graine",
        type=int,
        default=42,
        help="Graine aléatoire de référence, pour la reproductibilité (défaut : 42).",
    )
    analyseur.add_argument(
        "--m",
        type=int,
        default=8,
        help="Nombre d'antennes d'émission de la station de base (défaut : 8).",
    )
    analyseur.add_argument(
        "--k",
        type=int,
        default=4,
        help="Nombre d'utilisateurs, au plus le nombre d'images disponibles (défaut : 4).",
    )
    analyseur.add_argument(
        "--afficher",
        action="store_true",
        help="Affiche les figures dans des fenêtres interactives, en plus de les enregistrer.",
    )
    return analyseur.parse_args(argv)


def _fmt_ber(res: ResultatBER, indice: int | None = None) -> str:
    """Formate un résultat BER pour l'affichage console."""
    prefixe = f"Utilisateur {indice + 1}" if indice is not None else "Global    "
    return (
        f"{prefixe} : {res.erreurs} erreur(s) sur {res.nb_bits} bits  ->  "
        f"BER = {res.ber:.6e}"
    )


def _chemin_relatif(chemin: Path) -> Path:
    """Retourne un chemin relatif à la racine du projet si possible."""
    from .config import RACINE_PROJET

    try:
        return chemin.relative_to(RACINE_PROJET)
    except ValueError:
        return chemin


def principale(argv=None,
               dossier_figures: Path | None = None,
               dossier_reconstruites: Path | None = None) -> int:
    """Exécute la simulation complète du TP.

    Parameters
    ----------
    argv : list of str, optional
        Arguments de la ligne de commande.
    dossier_figures, dossier_reconstruites : pathlib.Path, optional
        Dossiers de sortie (destinés aux tests ; par défaut, ceux de la
        configuration).

    Returns
    -------
    int
        Code de retour du programme (0 si la chaîne est validée).
    """
    args = analyseur_arguments(argv)

    # Backends matplotlib : Agg (fichier) par défaut, interactif si --afficher.
    import matplotlib
    if not args.afficher and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")

    mode = "complet" if args.mode in ("complet", "full") else "rapide"

    # --------------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------------
    fabrique = configuration_complete if mode == "complet" else configuration_rapide
    config: ConfigurationTP = fabrique(
        m=args.m, k=args.k, snr_db=args.snr_db, graine=args.graine,
        modulation=args.modulation,
    )
    if dossier_figures is not None:
        config.dossier_figures = Path(dossier_figures)
    if dossier_reconstruites is not None:
        config.dossier_reconstruites = Path(dossier_reconstruites)

    duree_debut = time.perf_counter()

    print("=" * 68)
    print("TP MU-MISO / SDMA — PRÉCODAGE ZERO-FORCING")
    print("=" * 68)
    print(f"Mode d'exécution      : {mode}")
    print(f"Antennes d'émission M : {config.m}")
    print(f"Utilisateurs K        : {config.k}")
    print(f"Puissance totale P_T  : {config.p_t}")
    print(f"Modulation            : {config.modulation}")
    print(f"SNR démonstratif      : {config.snr_db} dB")
    print(f"Graine aléatoire      : {config.graine}")

    # --------------------------------------------------------
    # 2. Chargement des images et conversion en bits
    # --------------------------------------------------------
    print("\n--- 1. Images et conversion en bits ---")
    try:
        images_gris = charger_images(config.dossier_originales, config.noms_images)
    except FileNotFoundError as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 2

    if config.k > len(images_gris):
        print(
            f"ERREUR : K = {config.k} utilisateur(s) demandé(s) mais seulement "
            f"{len(images_gris)} image(s) disponible(s) dans "
            f"{_chemin_relatif(config.dossier_originales)}. "
            "Chaque utilisateur doit disposer de sa propre image.",
            file=sys.stderr,
        )
        return 2

    images_gris = images_gris[: config.k]
    for info in informations_images(images_gris):        print(
            f"Utilisateur {info['utilisateur']} : {info['hauteur']} x {info['largeur']} "
            f"pixels = {info['pixels']} pixels = {info['bits']} bits"
        )

    # --------------------------------------------------------
    # 3. Canal, précodage ZF et normalisation
    # --------------------------------------------------------
    print("\n--- 2. Canal de Rayleigh et précodage ZF ---")
    rng_demo = np.random.default_rng(config.graine)

    from .canal import generer_canal, sigma2_depuis_snr_db

    h = generer_canal(config.k, config.m, rng_demo)
    diag = diagnostics_canal(h)
    print(diag.afficher())
    if diag.warning:
        print(f"AVERTISSEMENT : {diag.warning}")

    w_zf = precodage_zf(h)                              # (M, K)
    w_norm, alpha = normaliser_puissance(w_zf, config.p_t)
    g = matrice_effective(h, w_norm)                    # (K, K)

    print(f"\nDimensions de H      : {h.shape}  (K x M attendu)")
    print(f"Dimensions de W_ZF   : {w_zf.shape}  (M x K attendu)")
    print(f"Facteur alpha        : {alpha:.11f}")
    print(f"Puissance ||W||_F^2  : {np.linalg.norm(w_norm, 'fro')**2:.12f} (cible P_T = {config.p_t})")

    puissance_ok = verifier_puissance_normee(w_norm, config.p_t)
    print(f"Puissance conforme à P_T : {puissance_ok}")
    if not puissance_ok:
        print("ECHEC : la puissance d'émission n'est pas conforme.", file=sys.stderr)
        return 1

    erreur_g = float(np.max(np.abs(g - alpha * np.eye(config.k))))
    print(f"Écart max |G - alpha*I_K| : {erreur_g:.3e} (interférences résiduelles)")

    # --------------------------------------------------------
    # 4. Transmission idéale sans bruit (validation de la chaîne)
    # --------------------------------------------------------
    print("\n--- 3. Transmission idéale sans bruit (canal + ZF) ---")
    from .simulations import transmettre_images

    res_ideal = transmettre_images(
        images_gris, h, w_norm, alpha, sigma2=0.0, modulation=config.modulation
    )
    for indice, (ber_u, identique) in enumerate(
        zip(res_ideal.reception.ber_utilisateurs, res_ideal.identiques)
    ):
        print(
            f"Utilisateur {indice + 1} : BER = {ber_u:.3e} | "
            f"image reconstruite identique : {identique}"
        )
    print(_fmt_ber(res_ideal.reception.ber))
    print(f"BER moyen (formule du TP) : {res_ideal.reception.ber_moyen:.3e}")

    chaine_validee = (
        res_ideal.reception.ber.erreurs == 0 and all(res_ideal.identiques)
    )
    print(f"Validation sans bruit : {'SUCCES' if chaine_validee else 'ECHEC'}")

    # --------------------------------------------------------
    # 5. Transmission avec bruit pour les trois modulations
    # --------------------------------------------------------
    print(f"\n--- 4. Transmission avec bruit (SNR = {config.snr_db} dB) ---")
    sigma2 = sigma2_depuis_snr_db(config.snr_db, config.p_t)
    print(f"Variance du bruit sigma2 = {sigma2:.6f}")

    res_bruit = transmettre_images(
        images_gris, h, w_norm, alpha, sigma2=sigma2, rng=rng_demo,
        modulation=config.modulation,
    )
    for indice in range(len(res_bruit.reception.ber_utilisateurs)):
        res_utilisateur = ResultatBER(
            erreurs=int(res_bruit.reception.erreurs[indice]),
            nb_bits=int(res_bruit.reception.nb_bits[indice]),
            ber=float(res_bruit.reception.ber_utilisateurs[indice]),
        )
        print(_fmt_ber(res_utilisateur, indice))
    print(_fmt_ber(res_bruit.reception.ber))
    print(f"BER moyen (moyenne des BER_k) : {res_bruit.reception.ber_moyen:.3e}")
    if res_bruit.reception.ber.nb_bits > 0 and res_bruit.reception.ber.erreurs == 0:
        print(
            "Note : aucun bit erroné observé à ce SNR sur "
            f"{res_bruit.reception.ber.nb_bits} bits ; cela ne prouve pas que le BER "
            "théorique soit nul (limite d'observation ≈ "
            f"{res_bruit.reception.ber.borne_observation:.2e})."
        )

    # Reconstruction et sauvegarde des images reçues (modulation de référence).
    config.dossier_reconstruites.mkdir(parents=True, exist_ok=True)
    chemins_reconstruites = []
    for indice, (image_rec, identique) in enumerate(
        zip(res_bruit.images_reconstruites, res_bruit.identiques)
    ):
        chemin = config.dossier_reconstruites / f"utilisateur{indice + 1}_recue.png"
        sauvegarder_image(chemin, image_rec)
        chemins_reconstruites.append(chemin)
        source = images_gris[indice]
        print(
            f"Utilisateur {indice + 1} : image enregistrée -> {_chemin_relatif(chemin)} "
            f"| identique à la source : {identique} "
            f"({int(np.count_nonzero(source != image_rec))} pixel(s) erroné(s))"
        )

    # --------------------------------------------------------
    # 6. Figures structurales (mode rapide et complet)
    # --------------------------------------------------------
    print("\n--- 5. Figures structurales ---")
    from . import figures

    figures_generees = []
    figures_generees.append(
        figures.figure_schema_systeme(
            config.dossier_figures / "schema_systeme.png", k=config.k, m=config.m
        )
    )
    figures_generees.append(
        figures.figure_partage_sdma(config.dossier_figures / "partage_sdma.png", k=config.k)
    )
    figures_generees.append(
        figures.figure_heatmap_g(g, alpha, config.dossier_figures / "heatmap_G.png")
    )

    # Démonstration des constellations : bits aléatoires (indépendants du
    # contenu des images) afin d'afficher la constellation complète de
    # chaque modulation, à moindre coût.
    from .detection import egaliser, gains_utiles
    from .detection import transmettre as transmettre_symboles
    from .modulation import moduler

    rng_constellations = np.random.default_rng(config.graine + 2)
    n_bits_constellations = 4096 * 4        # multiple de 1, 2 et 4

    donnees_constellations = []
    for modulation in config.modulations_etude:
        modulation = normaliser_nom_modulation(modulation)
        bits = rng_constellations.integers(
            0, 2, size=(config.k, n_bits_constellations)
        ).astype(np.uint8)
        s = moduler(bits, modulation)
        y = transmettre_symboles(s, h, w_norm, sigma2=sigma2, rng=rng_constellations)
        y_egalise = egaliser(y, gains_utiles(h, w_norm))
        donnees_constellations.append((modulation, s, y, y_egalise))
    figures_generees.append(
        figures.figure_constellations(
            donnees_constellations, config.dossier_figures / "constellations.png",
        )
    )
    for chemin in figures_generees:
        print(f"Figure enregistrée -> {chemin}")

    # --------------------------------------------------------
    # 7. Études Monte-Carlo (mode complet uniquement)
    # --------------------------------------------------------
    if mode == "complet":
        print("\n--- 6. Études de performances (Monte-Carlo) ---")
        print(
            f"Par point : {config.n_bits_courbe} bits/utilisateur x "
            f"{config.n_realisations} réalisation(s) de canal."
        )

        from .simulations import (
            courbe_ber_moyen_vs_m,
            courbe_ber_snr,
            courbe_ber_snr_par_m,
            courbe_ber_utilisateurs,
            etude_canaux_correles,
            etude_canaux_similaires,
            etude_conditionnement,
        )

        debut = time.perf_counter()

        print("Courbe BER en fonction du SNR (ZF vs sans précodage)...")
        resultats_snr = courbe_ber_snr(config)
        chemin = figures.figure_ber_snr(resultats_snr, config.dossier_figures / "ber_snr.png")
        figures_generees.append(chemin)
        print(f"  -> {chemin}")
        for snr_p, res_zf, res_sans in zip(
            resultats_snr["snr_db"], resultats_snr["zf"], resultats_snr["sans_precodage"]
        ):
            print(
                f"  SNR = {snr_p:>3} dB : BER ZF = {res_zf.ber:.3e} "
                f"({res_zf.erreurs}/{res_zf.nb_bits}) | "
                f"sans précodage = {res_sans.ber:.3e} "
                f"({res_sans.erreurs}/{res_sans.nb_bits})"
            )

        print("BER individuel et moyen en fonction du SNR, pour chaque M...")
        resultats_pm = courbe_ber_snr_par_m(config)
        chemin = figures.figure_ber_snr_par_m(resultats_pm, config.dossier_figures / "ber_snr_par_m.png")
        figures_generees.append(chemin)
        print(f"  -> {chemin}")

        print("BER moyen en fonction de M (BPSK, 4-QAM, 16-QAM)...")
        resultats_mm = courbe_ber_moyen_vs_m(config)
        chemin = figures.figure_ber_moyen_vs_m(resultats_mm, config.dossier_figures / "ber_moyen_vs_m.png")
        figures_generees.append(chemin)
        print(f"  -> {chemin}")

        print("BER moyen en fonction du nombre d'utilisateurs K (M = "
              f"{config.m_etude_k})...")
        resultats_k = courbe_ber_utilisateurs(config)
        chemin = figures.figure_ber_utilisateurs(
            resultats_k, config.dossier_figures / "ber_utilisateurs.png"
        )
        figures_generees.append(chemin)
        print(f"  -> {chemin}")

        print("Étude du conditionnement kappa(H H^H) et du BER...")
        resultats_cond = etude_conditionnement(config)
        chemin = figures.figure_conditionnement_ber(
            resultats_cond, config.dossier_figures / "conditionnement_ber.png"
        )
        figures_generees.append(chemin)
        print(f"  -> {chemin}")

        print("Étude des canaux corrélés (antennes d'émission)...")
        resultats_corr = etude_canaux_correles(config)
        chemin = figures.figure_canaux_correles(
            resultats_corr, config.dossier_figures / "canaux_correles.png"
        )
        figures_generees.append(chemin)
        print(f"  -> {chemin}")

        print("Étude des canaux utilisateurs progressivement similaires...")
        resultats_sim = etude_canaux_similaires(config)
        chemin = figures.figure_canaux_similaires(
            resultats_sim, config.dossier_figures / "canaux_similaires.png"
        )
        figures_generees.append(chemin)
        print(f"  -> {chemin}")

        # Reconstruction des images à plusieurs SNR, pour les trois modulations.
        print("Reconstruction des images à plusieurs SNR (5, 15, 25 dB)...")
        rng_images = np.random.default_rng(config.graine + 909)
        reconstructions = {}
        for modulation in config.modulations_etude:
            modulation = normaliser_nom_modulation(modulation)
            for snr in config.snr_reconstruction:
                sigma2_rec = sigma2_depuis_snr_db(snr, config.p_t)
                res = transmettre_images(
                    images_gris, h, w_norm, alpha, sigma2=sigma2_rec,
                    rng=rng_images, modulation=modulation,
                )
                reconstructions[(modulation, snr)] = res.images_reconstruites
                for indice, image_rec in enumerate(res.images_reconstruites):
                    nom = f"utilisateur{indice + 1}_{modulation}_{snr:g}dB.png"
                    sauvegarder_image(config.dossier_reconstruites / nom, image_rec)
                erreurs_pixels = [
                    int(np.count_nonzero(image_rec != images_gris[indice]))
                    for indice, image_rec in enumerate(res.images_reconstruites)
                ]
                print(
                    f"  {modulation:>6} @ {snr:>4.0f} dB : pixels erronés par "
                    f"utilisateur = {erreurs_pixels}"
                )

        # Figure de synthèse sur l'utilisateur 1.
        reconstructions_utilisateur = {
            cle: images[0] for cle, images in reconstructions.items()
        }
        chemin = figures.figure_images_reconstruites(
            images_gris[0], reconstructions_utilisateur,
            config.snr_reconstruction, config.modulations_etude,
            config.dossier_figures / "images_reconstruites.png", utilisateur=1,
        )
        figures_generees.append(chemin)
        print(f"  -> {chemin}")

        duree_etudes = time.perf_counter() - debut
        print(f"Durée des études : {duree_etudes:.1f} s")
    else:
        print(
            "\n(Expériences Monte-Carlo non exécutées en mode rapide : "
            "utilisez '--mode complet' pour les courbes BER, les études de "
            "conditionnement et la reconstruction multi-SNR.)"
        )

    # --------------------------------------------------------
    # 8. Affichage interactif éventuel
    # --------------------------------------------------------
    if args.afficher:
        import matplotlib.pyplot as plt
        plt.show()

    # --------------------------------------------------------
    # 9. Résumé final
    # --------------------------------------------------------
    duree_totale = time.perf_counter() - duree_debut
    print("\n" + "=" * 68)
    print("RÉSUMÉ")
    print("=" * 68)
    print(f"Mode                : {mode}")
    print(f"Canal H             : {h.shape}, rang {diag.rang}, conditionnement {diag.conditionnement:.3e}")
    print(f"Précodage ZF        : {w_norm.shape}, alpha = {alpha:.11f}")
    print(f"Puissance d'émission: {np.linalg.norm(w_norm, 'fro')**2:.12f} (cible {config.p_t})")
    print(f"Sans bruit          : BER global = {res_ideal.reception.ber.ber:.3e}, "
          f"images identiques = {all(res_ideal.identiques)}")
    print(f"Avec bruit ({config.snr_db} dB, {config.modulation}) : "
          f"BER global = {res_bruit.reception.ber.ber:.3e} "
          f"({res_bruit.reception.ber.erreurs}/{res_bruit.reception.ber.nb_bits} bits)")
    print(f"Figures générées    : {len(figures_generees)}")
    for chemin in figures_generees:
        print(f"  - {chemin}")
    print(f"Images reconstruites: {len(chemins_reconstruites)} (SNR de référence {config.snr_db} dB)")
    for chemin in chemins_reconstruites:
        print(f"  - {chemin}")
    print(f"Durée totale        : {duree_totale:.1f} s")

    if not chaine_validee:
        print("\nATTENTION : la transmission idéale sans bruit n'est pas parfaite.")
        return 1

    print("\nTerminé : la chaîne MU-MISO / SDMA / ZF est validée.")
    return 0


if __name__ == "__main__":
    sys.exit(principale())
