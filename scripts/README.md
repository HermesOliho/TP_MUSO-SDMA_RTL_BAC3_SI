# Scripts pédagogiques du TP

Ce dossier regroupe les **quatre scripts historiques du TP**, exécutés
et validés au fil des étapes. Ce sont des programmes **autonomes**,
écrits de façon très commentée à destination de l'apprentissage : ils
reproduisent chaque étape sans dépendre du code applicatif de `src/`.

Le code applicatif complet et maintenu se trouve dans `src/` ; en cas
de divergence, **`src/` fait foi**. Les scripts restent volontairement
simples et n'utilisent pas les modules de `src/`.

## Exécution

Depuis la racine du projet :

```bash
python scripts/etape1.py
python scripts/etape2.py
python scripts/etape3.py
python scripts/etape4.py
```

Les chemins des images sont calculés relativement à la racine du projet
(deux dossiers au-dessus du script), indépendamment du répertoire
courant.

## Contenu de chaque script

| Script | Étape | Contenu |
|---|---|---|
| `etape1.py` | Images et BPSK | Charge les 4 images JPEG (`data/originales/`), les convertit en niveaux de gris puis en bits (8 bits/pixel), module en BPSK (`0 -> -1`, `1 -> +1`), simule une transmission idéale sans bruit, démodule, calcule le BER et reconstruit les images. |
| `etape2.py` | Canal et ZF | Génère le canal de Rayleigh `H` de dimensions `(K, M) = (4, 8)` (graine 42), vérifie le rang, calcule `W_ZF = H^H @ inv(H H^H)`, vérifie `H @ W_ZF ≈ I_K` et mesure les interférences résiduelles. |
| `etape3.py` | Normalisation de puissance | Recalcule le même canal et le même ZF, applique `alpha = sqrt(P_T / ||W_ZF||_F^2)`, vérifie `||W_norm||_F^2 = P_T = 1` et que `G_norm = alpha * I_K`. Pour le canal de référence (graine 42), `alpha ≈ 1.10307513367`. |
| `etape4.py` | Heatmap | Trace et enregistre la heatmap annotée de `|G| = |H @ W_norm|` dans `figures/heatmap_G_etape4.png` (diagonale ≈ 1.103075, hors diagonale ≈ 1e-16). |

## Remarques

- `etape4.py` enregistre sa figure dans `figures/heatmap_G_etape4.png`
  afin de ne pas écraser la figure homonyme produite par
  `python -m src.main` (qui utilise sa propre graine dérivée).
- Les scripts affichent les figures avec `plt.show()` : dans un
  environnement sans interface graphique, utilisez un backend
  non interactif (`MPLBACKEND=Agg`).
