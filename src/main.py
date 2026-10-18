"""Point d'entrée principal du TP MU-MISO / SDMA / Zero-Forcing.

Usage
-----
::

    python -m src.main --mode rapide       # validation complète en quelques secondes
    python -m src.main --mode complet      # + courbes BER Monte-Carlo et études
    python -m src.main --snr-db 15         # SNR de la transmission démonstrative
    python -m src.main --graine 7          # graine aléatoire
    python -m src.main --afficher          # affiche les figures en plus de les enregistrer

Le mode rapide exécute toute la chaîne (images -> bits -> BPSK ->
canal -> ZF -> détection -> reconstruction) et produit les figures
structurales (schéma, SDMA, heatmap, constellations). Le mode complet
ajoute les expériences Monte-Carlo coûteuses (BER vs SNR, vs M, vs K,
canaux corrélés). Par défaut, les images reconstruites le sont au SNR
configuré (10 dB par défaut, voir ``--snr-db``).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .config import ConfigurationTP, configuration_complete, configuration_rapide
from .images import charger_images, informations_images, sauvegarder_image
from .modulation import ResultatBER
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
            "transmission d'images en BPSK à K utilisateurs via M antennes."
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
    config: ConfigurationTP = fabrique(m=args.m, k=args.k, snr_db=args.snr_db,
                                       graine=args.graine)
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

    images_gris = images_gris[: config.k]
    for info in informations_images(images_gris):
        print(
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

    res_ideal = transmettre_images(images_gris, h, w_norm, alpha, sigma2=0.0)
    for indice, (ber_u, identique) in enumerate(
        zip(res_ideal.reception.ber_utilisateurs, res_ideal.identiques)
    ):
        print(
            f"Utilisateur {indice + 1} : BER = {ber_u:.3e} | "
            f"image reconstruite identique : {identique}"
        )
    print(_fmt_ber(res_ideal.reception.ber))

    chaine_validee = (
        res_ideal.reception.ber.erreurs == 0 and all(res_ideal.identiques)
    )
    print(f"Validation sans bruit : {'SUCCES' if chaine_validee else 'ECHEC'}")

    # --------------------------------------------------------
    # 5. Transmission avec bruit au SNR configuré
    # --------------------------------------------------------
    print(f"\n--- 4. Transmission avec bruit (SNR = {config.snr_db} dB) ---")
    sigma2 = sigma2_depuis_snr_db(config.snr_db, config.p_t)
    print(f"Variance du bruit sigma2 = {sigma2:.6f}")

    res_bruit = transmettre_images(
        images_gris, h, w_norm, alpha, sigma2=sigma2, rng=rng_demo
    )
    for indice in range(len(res_bruit.reception.ber_utilisateurs)):
        res_utilisateur = ResultatBER(
            erreurs=int(res_bruit.reception.erreurs[indice]),
            nb_bits=int(res_bruit.reception.nb_bits[indice]),
            ber=float(res_bruit.reception.ber_utilisateurs[indice]),
        )
        print(_fmt_ber(res_utilisateur, indice))
    print(_fmt_ber(res_bruit.reception.ber))
    if res_bruit.reception.ber.nb_bits > 0 and res_bruit.reception.ber.erreurs == 0:
        print(
            "Note : aucun bit erroné observé à ce SNR sur "
            f"{res_bruit.reception.ber.nb_bits} bits ; cela ne prouve pas que le BER "
            "théorique soit nul (limite d'observation ≈ "
            f"{res_bruit.reception.ber.borne_observation:.2e})."
        )

    # Reconstruction et sauvegarde des images reçues.
    config.dossier_reconstruites.mkdir(parents=True, exist_ok=True)
    chemins_reconstruites = []
    for indice, (image_rec, identique) in enumerate(
        zip(res_bruit.images_reconstruites, res_bruit.identiques)
    ):
        chemin = config.dossier_reconstruites / f"utilisateur{indice + 1}_recue.png"
        sauvegarder_image(chemin, image_rec)
        chemins_reconstruites.append(chemin)
        source = images_gris[indice]
        from .config import RACINE_PROJET
        try:
            affichage = chemin.relative_to(RACINE_PROJET)
        except ValueError:
            affichage = chemin
        print(
            f"Utilisateur {indice + 1} : image enregistrée -> {affichage} "
            f"| identique à la source : {identique} "
            f"({int(np.count_nonzero(source != image_rec))} pixel(s) erroné(s))"
        )

    # --------------------------------------------------------
    # 6. Figures structurales (mode rapide et complet)
    # --------------------------------------------------------
    print("\n--- 5. Figures ---")
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
    figures_generees.append(figures.figure_heatmap_g(g, alpha, config.dossier_figures / "heatmap_G.png"))
    figures_generees.append(
        figures.figure_constellations(
            res_bruit.s_bloc, res_bruit.y_sans_bruit, res_bruit.y, alpha,
            config.dossier_figures / "constellations.png",
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

        debut = time.perf_counter()
        resultats_snr = None
        try:
            from .simulations import (
                courbe_ber_antennes,
                courbe_ber_snr,
                courbe_ber_utilisateurs,
                etude_canaux_correles,
            )

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

            print("Courbe BER en fonction du nombre d'antennes M...")
            resultats_m = courbe_ber_antennes(config)
            chemin = figures.figure_ber_antennes(
                resultats_m, k=config.k, chemin=config.dossier_figures / "ber_antennes.png"
            )
            figures_generees.append(chemin)
            print(f"  -> {chemin}")

            print("Courbe BER en fonction du nombre d'utilisateurs K...")
            resultats_k = courbe_ber_utilisateurs(config)
            chemin = figures.figure_ber_utilisateurs(
                resultats_k, m=config.m, chemin=config.dossier_figures / "ber_utilisateurs.png"
            )
            figures_generees.append(chemin)
            print(f"  -> {chemin}")

            print("Étude des canaux corrélés...")
            resultats_corr = etude_canaux_correles(config)
            chemin = figures.figure_canaux_correles(
                resultats_corr, config.dossier_figures / "canaux_correles.png"
            )
            figures_generees.append(chemin)
            print(f"  -> {chemin}")
        except Exception as exc:  # pragma: no cover - diagnostics
            print(f"ERREUR pendant les études Monte-Carlo : {exc}", file=sys.stderr)
            raise

        duree_etudes = time.perf_counter() - debut
        print(f"Durée des études : {duree_etudes:.1f} s")
    else:
        print(
            "\n(Expériences Monte-Carlo non exécutées en mode rapide : "
            "utilisez '--mode complet' pour les courbes BER, etc.)"
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
    print(f"Avec bruit ({config.snr_db} dB) : BER global = {res_bruit.reception.ber.ber:.3e} "
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
