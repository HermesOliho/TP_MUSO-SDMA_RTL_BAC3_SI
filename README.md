# TP MU-MISO / SDMA avec précodage Zero-Forcing

Simulation complète d'un système de communication sans fil
**MU-MISO** (*Multi-User Multiple Input Single Output*) utilisant le
**SDMA** (*Space Division Multiple Access*) et le **précodage
Zero-Forcing (ZF)**, en Python (NumPy / Matplotlib / Pillow).

Une station de base dotée de **M = 8 antennes** transmet
simultanément **K = 4 flux de données** (une image JPEG par utilisateur)
à **K utilisateurs** sur les mêmes ressources temps-fréquence. La
séparation spatiale est assurée par le précodage ZF, qui annule les
interférences entre utilisateurs. Chaque image est convertie en niveaux
de gris puis en bits, modulée en **BPSK, 4-QAM (QPSK) ou 16-QAM**,
transmise à travers un canal complexe de Rayleigh à évanouissement plat
avec bruit AWGN, **égalisée** (`ŝ_k = y_k / g_k`), détectée par décision
du plus proche voisin, puis reconstruite et comparée à l'originale.

---

## 1. Objectif du TP

- Modéliser et simuler une chaîne complète MU-MISO : images → bits →
  modulation (BPSK / 4-QAM / 16-QAM) → précodage ZF → canal de Rayleigh →
  égalisation → détection → reconstruction d'images.
- Valider numériquement l'annulation des interférences
  inter-utilisateurs par le ZF (matrice effective `G = H @ W_norm`).
- Étudier les performances (BER) en fonction du SNR, du nombre
  d'antennes `M`, du nombre d'utilisateurs `K`, de la **modulation**, du
  **conditionnement** `κ(H H^H)` et de la **corrélation / similitude**
  entre canaux.

## 2. Principe du MU-MISO et du SDMA

En MU-MISO, la station de base dispose de `M` antennes d'émission et
chaque utilisateur d'une seule antenne de réception. Le SDMA consiste à
adresser `K` utilisateurs **simultanément** sur les mêmes ressources
temps-fréquence, en exploitant les différences de positionnement
spatial : le précodage adapte les phases et amplitudes émises par les
`M` antennes pour que chaque utilisateur reçoive principalement **son
propre flux**. Le Zero-Forcing force l'orthogonalisation du canal
effectif : `H @ W = alpha × I_K`, c'est-à-dire aucune fuite du flux `j`
vers l'utilisateur `k ≠ j`.

## 3. Rôle de chaque module

| Module | Rôle |
|---|---|
| `src/config.py` | Configuration centralisée (`M`, `K`, `P_T`, modulation, listes de SNR / `M` / `K`, graine, tailles d'expériences, chemins). |
| `src/images.py` | Chargement des images JPEG en niveaux de gris, conversions pixels ↔ bits, reconstruction et sauvegarde, vérification pixel par pixel. |
| `src/modulation.py` | Modulations BPSK, 4-QAM et 16-QAM (codage de Gray, puissance moyenne unitaire), démodulation au plus proche voisin, calcul du BER. |
| `src/canal.py` | Canal de Rayleigh complexe `CN(0,1)`, bruit AWGN `CN(0, σ²)`, conversion SNR ↔ variance, modèles corrélé (Kronecker / exponentiel) et « canaux utilisateurs similaires ». |
| `src/precodage_zf.py` | Calcul du ZF (pseudo-inverse par défaut, formule explicite en option), normalisation de puissance, diagnostics de rang et de conditionnement, précodeur régularisé « ridge » (option distincte, **pas du ZF**). |
| `src/detection.py` | Modèle `Y = H @ W @ S + N`, **égalisation** `ŝ_k = y_k / g_k`, détection (BPSK par signe, QAM au plus proche voisin), BER par utilisateur, BER moyen et BER global, exclusion du remplissage. |
| `src/simulations.py` | Transmission des images, alignement des flux de longueurs différentes, expériences Monte-Carlo (BER vs SNR, BER par utilisateur vs SNR pour chaque `M`, BER moyen vs `M` pour les trois modulations, vs `K`, conditionnement, canaux corrélés, canaux similaires, comparaison sans précodage). |
| `src/figures.py` | Génération des 12 figures du TP. |
| `src/main.py` | Point d'entrée qui orchestre les modules (`python -m src.main`). |

Les scripts pédagogiques historiques (`etape1.py` … `etape4.py`) sont
conservés dans `scripts/` : voir `scripts/README.md`.

## 4. Structure des dossiers

```text
TP-RT-GROUPE-8/
│
├── rapport/
│   └── rapport_technique.pdf     # rapport (s'il est fourni)
│
├── src/
│   ├── __init__.py
│   ├── main.py                   # point d'entrée
│   ├── config.py
│   ├── images.py
│   ├── modulation.py
│   ├── canal.py
│   ├── precodage_zf.py
│   ├── detection.py
│   ├── simulations.py
│   └── figures.py
│
├── scripts/
│   ├── README.md
│   ├── etape1.py … etape4.py     # scripts pédagogiques autonomes
│
├── tests/
│   ├── conftest.py
│   ├── test_images.py
│   ├── test_modulation.py
│   ├── test_canal.py
│   ├── test_precodage.py
│   ├── test_simulations.py
│   └── test_etudes.py            # études Monte-Carlo et figures du mode complet
│
├── data/
│   ├── originales/               # les 4 images JPEG sources
│   └── reconstruites/            # images reçues (générées)
│
├── figures/                      # 12 figures (générées)
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. Installation

**Python 3.10 ou supérieur** (développé et testé sous Python 3.13).
Aucun GPU n'est nécessaire.

```bash
# Créer éventuellement un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt
```

Dépendances : `numpy`, `matplotlib`, `Pillow`, et `pytest` pour les
tests uniquement.

## 6. Placement des images

Les quatre images sources doivent se trouver dans `data/originales/` :

```text
data/originales/image1.jpg
data/originales/image2.jpg
data/originales/image3.jpg
data/originales/image4.jpg
```

- Une image distincte est attribuée à chaque utilisateur ; ses
  dimensions propres sont conservées (aucun redimensionnement), même si
  les quatre images ont des tailles différentes.
- Si un fichier est absent, le programme s'arrête avec un message
  explicite indiquant le chemin recherché (aucune image n'est inventée).
- Le nombre d'utilisateurs `K` (option `--k`) ne peut pas dépasser le
  nombre d'images disponibles.

## 7. Exécution du mode rapide

```bash
python -m src.main --mode rapide
```

Le mode rapide (par défaut) valide toute la chaîne en une vingtaine de
secondes : chargement des images, conversion en bits, modulation, canal
`H (4×8)`, ZF, vérification de la puissance d'émission, transmission
**sans bruit** (BER nul exigé, images reconstruites exactes),
transmission **avec bruit** au SNR configuré (10 dB par défaut),
égalisation, reconstruction et sauvegarde des 4 images. Il produit les
figures `schema_systeme.png`, `partage_sdma.png`, `heatmap_G.png` et
`constellations.png` (les trois modulations, avant et après
égalisation). Il ne lance **pas** les courbes Monte-Carlo.

## 8. Exécution du mode complet

```bash
python -m src.main --mode complet
```

Le mode complet ajoute les expériences Monte-Carlo (environ 2 à 3
minutes selon la machine) et produit en plus :
`ber_snr.png`, `ber_snr_par_m.png`, `ber_moyen_vs_m.png`,
`ber_utilisateurs.png`, `conditionnement_ber.png`,
`canaux_correles.png`, `canaux_similaires.png` et
`images_reconstruites.png`. Il reconstruit également les images aux SNR
`{5, 15, 25}` dB pour les trois modulations (fichiers
`utilisateurN_<modulation>_<SNR>dB.png` dans `data/reconstruites/`).

## 9. Options disponibles

```bash
python -m src.main --help
```

| Option | Défaut | Description |
|---|---|---|
| `--mode {rapide,complet}` | `rapide` | Mode d'exécution (alias anglais `quick`/`full` acceptés). |
| `--modulation {BPSK,4-QAM,QPSK,16-QAM}` | `BPSK` | Modulation de la transmission démonstrative (les études couvrent de toute façon les trois modulations). |
| `--snr-db VALEUR` | `10.0` | SNR en dB de la transmission démonstrative avec bruit (celle qui produit les images reconstruites). |
| `--graine VALEUR` | `42` | Graine aléatoire de référence (reproductibilité). |
| `--m VALEUR` | `8` | Nombre d'antennes d'émission de la station de base. |
| `--k VALEUR` | `4` | Nombre d'utilisateurs (au plus le nombre d'images). |
| `--afficher` | désactivé | Affiche les figures dans des fenêtres interactives, en plus de les enregistrer. |

Exemples :

```bash
python -m src.main --mode rapide --snr-db 15
python -m src.main --mode rapide --modulation 16-QAM
python -m src.main --mode complet --graine 7
python -m src.main --mode rapide --m 6 --k 3
```

Par défaut, les images reconstruites le sont au **SNR de référence
configuré** (`--snr-db`, 10 dB par défaut), et non au dernier point
d'une courbe.

## 10. Modulations et conventions

### BPSK
- bit `0` → symbole réel `-1` ; bit `1` → symbole réel `+1` ;
- décision du récepteur : signe de la partie réelle, `Re{y} ≥ 0` → `1`.

### 4-QAM (QPSK) et 16-QAM
- constellations carrées (2×2 et 4×4), **codage de Gray** (deux points
  voisins ne diffèrent que d'un seul bit) ;
- les bits sont regroupés par paquets de 2 (4-QAM) ou 4 (16-QAM), bit de
  poids fort en tête ; les `b/2` premiers bits indexent l'axe réel, les
  `b/2` suivants l'axe imaginaire ;
- **puissance moyenne normalisée** `E[|s|²] = 1` pour toutes les
  modulations (comparaison équitable à puissance d'émission donnée) ;
- démodulation par décision du **plus proche voisin** (implémentée par
  décision séparable par axe, strictement équivalente à l'argmin 2D pour
  ces constellations carrées).

### Bits et pixels
8 bits par pixel, du bit de poids fort (MSB) au bit de poids faible
(LSB) ; balayage des pixels ligne par ligne.

## 11. Modèle de canal

- Canal de Rayleigh complexe à **évanouissement plat**, matrice
  `H` de dimensions `(K, M)` :

  ```python
  H = (H_reel + 1j * H_imag) / sqrt(2)
  ```

  les parties réelle et imaginaire sont indépendantes, gaussiennes, de
  moyenne nulle et de variance `1/2`, donc `E[|H[k,m]|²] = 1`.

- Le canal est **constant pendant un bloc de transmission** et
  **renouvelé à chaque réalisation Monte-Carlo** (voir §18).
- Le canal est considéré comme **connu parfaitement** par la station de
  base (CSI parfait) pour calculer le précodage, et par le récepteur
  pour l'égalisation.

## 12. Formule du Zero-Forcing

```text
W_ZF = H^H @ inv(H @ H^H)          (dimensions M × K)
```

Hypothèses nécessaires : `K ≤ M` et `rank(H) = K` (rang plein en
lignes). Un canal singulier déclenche une erreur explicite
(`CanalSingularError`) : aucune correction silencieuse n'est appliquée.
Une régularisation « ridge » (`precodage_ridge`) existe mais est une
**option distincte, jamais présentée comme du ZF classique**.

*Choix numérique* : l'implémentation par défaut utilise la
pseudo-inverse de Moore-Penrose (`numpy.linalg.pinv`), égale à la
formule théorique pour un canal de rang plein, mais plus robuste (SVD
tronquée). La formule explicite reste disponible via
`methode="formule"` ; un test unitaire vérifie leur concordance.

## 13. Normalisation de puissance

Puissance moyenne totale d'émission cible `P_T = 1.0` :

```text
alpha = sqrt(P_T / ||W_ZF||_F²)
W_norm = alpha * W_ZF
```

Vérification : `||W_norm||_F² = P_T` (testée automatiquement).
Conséquence : le canal effectif normalisé vaut `G = H @ W_norm =
alpha × I_K` — la diagonale vaut `alpha` (gain utile) et **non 1** ;
les termes hors diagonale (interférences) restent nuls à la précision
numérique (~1e-16).

## 14. Égalisation et détection

Le gain utile de l'utilisateur `k` est `g_k = h_k^H w_k` (diagonale de
`G = H W`). Le récepteur **égalise** avant démodulation :

```text
ŝ_k = y_k / g_k
```

- Avec le ZF idéal, `G = alpha × I_K` (gain réel positif) : l'égalisation
  divise par `alpha` et ne change pas la décision.
- Dans le scénario sans précodage, `g_k` est complexe : l'égalisation
  compense aussi la phase du canal (hypothèse de CSI au récepteur).

## 15. Modèle de bruit et convention SNR

- Bruit AWGN complexe `N ~ CN(0, σ²)` : variance totale
  `E[|N|²] = σ²`, répartie également entre les composantes réelle et
  imaginaire (`σ²/2` chacune).
- **Convention SNR** (unique dans tout le projet) :

  ```text
  σ² = P_T / 10^(SNR_dB / 10)
  ```

  `P_T` est la puissance moyenne **totale** d'émission. Comme chaque
  coefficient de canal a une puissance moyenne unitaire, ce SNR
  correspond au rapport puissance reçue moyenne par utilisateur /
  puissance du bruit. Ce n'est ni un SNR par antenne ni un SNR
  post-précodage.

## 16. Définition du BER

Le taux d'erreur binaire est `BER = (nombre de bits erronés) / (nombre
de bits comparés)`. Deux agrégats sont rapportés :

- **BER moyen** `BER_moy = (1/K) Σ_k BER_k` (moyenne arithmétique des
  BER individuels, formule du TP) ;
- **BER global** = (total des erreurs) / (total des bits comparés), qui
  coïncide avec le BER moyen lorsque les flux ont la même longueur.

Les **symboles de remplissage** (alignement des flux de longueurs
différentes) sont exclus du BER et des images reconstruites. Un BER
observé nul est toujours rapporté au nombre de bits testés : il indique
que le BER réel est sous la limite d'observation (≈ 1 / N_bits), pas
qu'il est théoriquement nul.

## 17. Signification de chaque figure

| Figure | Contenu | Condition de génération |
|---|---|---|
| `figures/schema_systeme.png` | Schéma fonctionnel de la chaîne (images → bits → modulation → ZF + 8 antennes → canal → égalisation/détection → images). | Toujours (rapide et complet). |
| `figures/partage_sdma.png` | Illustration pédagogique du SDMA : 4 utilisateurs, mêmes ressources temps-fréquence, flux séparés spatialement. | Toujours. |
| `figures/heatmap_G.png` | Heatmap annotée de `\|H @ W_norm\|` : diagonale = gain `alpha`, hors diagonale ≈ 0 (interférences annulées). | Toujours. |
| `figures/constellations.png` | Constellations (BPSK, 4-QAM, 16-QAM) : émises, reçues **avant** égalisation, reçues **après** égalisation `y_k/g_k`. | Toujours. |
| `figures/ber_snr.png` | BER moyen vs SNR : ZF vs scénario sans précodage (même `P_T`, même canal, même égalisation). | Mode complet. |
| `figures/ber_snr_par_m.png` | Pour `K` fixé : `BER_1 … BER_K` et BER moyen en fonction du SNR, un sous-graphe par `M ∈ {4, 8, 16, 32, 64}`. | Mode complet. |
| `figures/ber_moyen_vs_m.png` | BER moyen en fonction de `M` (SNR = 15 dB) pour BPSK, 4-QAM et 16-QAM. | Mode complet. |
| `figures/ber_utilisateurs.png` | BER moyen en fonction de `K ∈ {2, 4, 8, 12, 16}` avec `M = 16` fixé (effet de `K → M`). | Mode complet. |
| `figures/conditionnement_ber.png` | Nuage de points : BER observé en fonction de `κ(H H^H)` sur 200 réalisations (bien/mal conditionné). | Mode complet. |
| `figures/canaux_correles.png` | BER vs SNR pour canaux i.i.d. et corrélés (`ρ = 0, 0.5, 0.9`, modèle exponentiel entre antennes). | Mode complet. |
| `figures/canaux_similaires.png` | BER vs SNR lorsque deux canaux utilisateurs deviennent progressivement colinéaires (`β = 0, 0.5, 0.9, 0.99`). | Mode complet. |
| `figures/images_reconstruites.png` | Image originale et reconstruite (utilisateur 1) à 5, 15 et 25 dB pour BPSK, 4-QAM et 16-QAM. | Mode complet. |

Sur les courbes BER, un point observé à zéro est affiché par un
marqueur ouvert ▼ positionné à la limite d'observation (convention de
tracé documentée), jamais présenté comme une mesure nulle.

## 18. Exécution des tests

```bash
python -m pytest tests/
```

92 tests couvrent notamment : l'aller-retour exact pixels → bits →
pixels (images de tailles différentes), les trois modulations (puissance
unitaire, codage de Gray, équivalence décision séparable / plus proche
voisin, aller-retour), la chaîne complète, l'égalisation `ŝ = y/g`, le
BER (y compris séquences vides et remplissage), les formes `(K, M)` et
`(M, K)`, `H @ W_ZF ≈ I_K`, la puissance normalisée `= P_T`, le
modèle `Y = H W S + N`, la reconstruction exacte sans bruit (images
réelles incluses, pour les trois modulations), la non-régression du
`alpha ≈ 1.10307513367` du canal de référence (graine 42 historique),
le levé d'erreur sur canal singulier, les études Monte-Carlo du mode
complet et leurs figures, et l'exécution complète du mode rapide sans
les courbes Monte-Carlo.

## 19. Limites du modèle et hypothèses retenues

- **CSI parfait** : la station de base connaît exactement `H` ; aucune
  estimation de canal, aucune erreur de quantification.
- **Canal plat, non sélectif en fréquence** : pas de multi-trajet
  sélectif, pas de mouvement (canal statique pendant un bloc).
- **Synchronisation parfaite** : aucun décalage temporel ni fréquentiel,
  pas de Doppler.
- **Modulations sans codage de canal** : BPSK, 4-QAM et 16-QAM
  uniquement ; aucun codage correcteur d'erreurs.
- **ZF non régularisé** : proche du seuil du bruit de recodage quand `K`
  s'approche de `M` (ou quand deux canaux utilisateurs deviennent
  colinéaires) ; le module propose un précodeur « ridge » distinct à
  titre comparatif.
- **Canal renouvelé à chaque réalisation Monte-Carlo** : les courbes BER
  moyennent sur l'évanouissement (plutôt qu'une seule réalisation par
  trame).
- **Comparaison sans précodage équitable mais simplifiée** : une antenne
  par utilisateur, même `P_T`, même égalisation ; d'autres structures
  (MRC côté récepteur, détecteurs ML multi-utilisateur) ne sont pas
  modélisées.
- Les statistiques Monte-Carlo reposent sur `n_bits_courbe = 50 000`
  bits/utilisateur × `n_realisations = 10` réalisations par point, ce
  qui borne la mesure à environ `5·10^-7` en BER observé.

## Reproductibilité

Toutes les simulations utilisent des générateurs aléatoires à graine
fixée, dérivés de la graine de configuration (`--graine`, défaut 42).
Deux exécutions avec les mêmes paramètres produisent exactement les
mêmes résultats, figures et images reconstruites.
