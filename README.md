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
de gris puis en bits, modulée en BPSK, transmise à travers un canal
complexe de Rayleigh à évanouissement plat avec bruit AWGN, détectée en
BPSK, puis reconstruite et comparée à l'originale.

---

## 1. Objectif du TP

- Modéliser et simuler une chaîne complète MU-MISO : images → bits →
  BPSK → précodage ZF → canal de Rayleigh → détection BPSK →
  reconstruction d'images.
- Valider numériquement l'annulation des interférences
  inter-utilisateurs par le ZF (matrice effective `G = H @ W_norm`).
- Étudier les performances (BER) en fonction du SNR, du nombre
  d'antennes `M`, du nombre d'utilisateurs `K`, et de la corrélation
  entre antennes d'émission.

## 2. Principe du MU-MISO et du SDMA

En MU-MISO, la station de base dispose de `M` antennes d'émission et
chaque utilisateur d'une seule antenne de réception. Le SDMA consiste à
adresser `K` utilisateurs **simultanément** sur les mêmes ressources
temps-fréquence, en exploitant les différences de positionnement
spatial : le précodage adapte les phases et amplitudes émises par les
`M` antennes pour que chaque utilisateur reçoive principalement **son
propre flux**. Le Zero-Forcing Force l'orthogonalisation du canal
effectif : `H @ W = I_K`, c'est-à-dire aucune fuite du flux `j` vers
l'utilisateur `k ≠ j`.

## 3. Rôle de chaque module

| Module | Rôle |
|---|---|
| `src/config.py` | Configuration centralisée (`M`, `K`, `P_T`, SNR, graine, tailles d'expériences, chemins). |
| `src/images.py` | Chargement des images JPEG en niveaux de gris, conversions pixels ↔ bits, reconstruction et sauvegarde, vérification pixel par pixel. |
| `src/modulation.py` | Modulation / démodulation BPSK, calcul du BER (erreurs / bits comparés). |
| `src/canal.py` | Canal de Rayleigh complexe `CN(0,1)`, bruit AWGN `CN(0, σ²)`, conversion SNR ↔ variance, modèle corrélé (Kronecker / exponentiel). |
| `src/precodage_zf.py` | Calcul du ZF (pseudo-inverse par défaut, formule explicite en option), normalisation de puissance, diagnostics de rang et de conditionnement, précodeur régularisé « ridge » (option distincte, **pas du ZF**). |
| `src/detection.py` | Modèle `Y = H @ W @ S + N`, détection BPSK par signe, BER par utilisateur et global, exclusion du remplissage. |
| `src/simulations.py` | Transmission des images, alignement des flux de longueurs différentes, expériences Monte-Carlo (BER vs SNR, vs `M`, vs `K`, canaux corrélés, comparaison sans précodage). |
| `src/figures.py` | Génération des 8 figures du rapport. |
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
│   └── test_simulations.py
│
├── data/
│   ├── originales/               # les 4 images JPEG sources
│   └── reconstruites/            # images reçues (générées)
│
├── figures/                      # 8 figures (générées)
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

Le mode rapide (par défaut) valide toute la chaîne en quelques
secondes (~7 s) : chargement des images, conversion en bits, BPSK,
canal `H (4×8)`, ZF, vérification de la puissance d'émission,
transmission **sans bruit** (BER nul exigé, images reconstruites
exactes), transmission **avec bruit** au SNR configuré (10 dB par
défaut), reconstruction et sauvegarde des 4 images, et production des
figures `schema_systeme.png`, `partage_sdma.png`, `heatmap_G.png` et
`constellations.png`. Il ne lance **pas** les courbes Monte-Carlo.

## 8. Exécution du mode complet

```bash
python -m src.main --mode complet
```

Le mode complet ajoute les expériences Monte-Carlo (~40 s de plus) et
produit en plus `ber_snr.png`, `ber_antennes.png`,
`ber_utilisateurs.png` et `canaux_correles.png`.

## 9. Options disponibles

```bash
python -m src.main --help
```

| Option | Défaut | Description |
|---|---|---|
| `--mode {rapide,complet}` | `rapide` | Mode d'exécution (alias anglais `quick`/`full` acceptés). |
| `--snr-db VALEUR` | `10.0` | SNR en dB de la transmission démonstrative avec bruit (celle qui produit les images reconstruites). |
| `--graine VALEUR` | `42` | Graine aléatoire de référence (reproductibilité). |
| `--m VALEUR` | `8` | Nombre d'antennes d'émission de la station de base. |
| `--k VALEUR` | `4` | Nombre d'utilisateurs (au plus le nombre d'images). |
| `--afficher` | désactivé | Affiche les figures dans des fenêtres interactives, en plus de les enregistrer. |

Exemples :

```bash
python -m src.main --mode rapide --snr-db 15
python -m src.main --mode complet --graine 7
python -m src.main --mode rapide --m 6 --k 3
```

Par défaut, les images reconstruites le sont au **SNR de référence
configuré** (`--snr-db`, 10 dB par défaut), et non au dernier point
d'une courbe.

## 10. Convention BPSK

- bit `0` → symbole réel `-1`
- bit `1` → symbole réel `+1`
- Décision du récepteur : signe de la partie réelle du symbole reçu,
  `Re{y} ≥ 0` → bit `1`, sinon bit `0`.
- 8 bits par pixel, du bit de poids fort (MSB) au bit de poids faible
  (LSB) ; balayage des pixels ligne par ligne. Après précodage ZF, le
  gain utile `alpha` est réel strictement positif : la mise à l'échelle
  ne modifie donc pas la décision BPSK.

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
  base (CSI parfait) pour calculer le précodage.

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

## 14. Modèle de bruit et convention SNR

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

## 15. Définition du BER

Le taux d'erreur binaire est `BER = (nombre de bits erronés) / (nombre
de bits comparés)`, calculé par utilisateur puis agrégé globalement.
Les **symboles de remplissage** (alignement des flux de longueurs
différentes) sont exclus du BER et des images reconstruites.
Un BER observé nul est toujours rapporté au nombre de bits testés : il
indique que le BER réel est sous la limite d'observation (≈ 1 /
N_bits), pas qu'il est théoriquement nul.

## 16. Signification de chaque figure

| Figure | Contenu | Condition de génération |
|---|---|---|
| `figures/schema_systeme.png` | Schéma fonctionnel de la chaîne (images → bits → BPSK → ZF + 8 antennes → canal → détection → images). | Toujours (rapide et complet). |
| `figures/partage_sdma.png` | Illustration pédagogique du SDMA : 4 utilisateurs, mêmes ressources temps-fréquence, flux séparés spatialement. | Toujours. |
| `figures/heatmap_G.png` | Heatmap annotée de `\|H @ W_norm\|` : diagonale = gain `alpha`, hors diagonale ≈ 0 (interférences annulées). | Toujours. |
| `figures/constellations.png` | Constellations BPSK complexes : émises (±1), reçues sans bruit (`±alpha` sur l'axe réel), reçues avec bruit (nuage 2D). | Toujours. |
| `figures/ber_snr.png` | BER vs SNR : ZF vs scénario sans précodage (même `P_T`, même canal, compensation de phase récepteur). | Mode complet. |
| `figures/ber_antennes.png` | BER du ZF vs nombre d'antennes `M` (`K` fixé, contrainte `M ≥ K`). | Mode complet. |
| `figures/ber_utilisateurs.png` | BER du ZF vs nombre d'utilisateurs `K` (`M` fixé). | Mode complet. |
| `figures/canaux_correles.png` | BER vs SNR pour canaux i.i.d. et corrélés (`ρ = 0, 0.5, 0.9`, modèle exponentiel). | Mode complet. |

Sur les courbes BER, un point observé à zéro est affiché par un
marqueur ouvert ▼ positionné à la limite d'observation (convention de
tracé documentée), jamais présenté comme une mesure nulle.

## 17. Exécution des tests

```bash
python -m pytest tests/
```

63 tests couvrent notamment : l'aller-retour exact pixels → bits →
pixels (images de tailles différentes), la chaîne BPSK, le BER (y
compris séquences vides et remplissage), les formes `(K, M)` et
`(M, K)`, `H @ W_ZF ≈ I_K`, la puissance normalisée `= P_T`, le
modèle `Y = H W S + N`, la reconstruction exacte sans bruit (images
réelles incluses), le non-régression du `alpha ≈ 1.10307513367` du
canal de référence (graine 42 historique), le levé d'erreur sur canal
singulier, et l'exécution complète du mode rapide sans les courbes
Monte-Carlo.

## 18. Limites du modèle et hypothèses retenues

- **CSI parfait** : la station de base connaît exactement `H` ; aucune
  estimation de canal, aucune erreur de quantification.
- **Canal plat, non sélectif en fréquence** : pas de multi-trajet
  sélectif, pas de mouvement (canal statique pendant un bloc).
- **Synchronisation parfaite** : aucun décalage temporel ni fréquentiel,
  pas de Doppler.
- **BPSK uniquement** : une seule modulation simple, adaptée à la
  validation pédagogique ; le ZF s'appliquerait de la même façon à
  d'autres modulations.
- **ZR (Zero-Forcing) non régularisé** : proche du seuil du bruit de
  recodage quand `K` s'approche de `M` ; le module propose un précodeur
  « ridge » distinct à titre comparatif.
- **Canal renouvelé à chaque réalisation Monte-Carlo** : les courbes BER
  moyennent sur l'évanouissement (plutôt qu'une seule réalisation par
  trame).
- **Comparaison sans précodage équitable mais simplifiée** : une antenne
  par utilisateur, même `P_T`, compensation de phase par CSI récepteur ;
  d'autres structures (MRC côté récepteur, détecteurs ML multi-utilisateur)
  ne sont pas modélisées.
- Les statistiques Monte-Carlo reposent sur `n_bits_courbe = 100 000`
  bits/utilisateur × `n_realisations = 10` réalisations par point, ce
  qui borne la mesure à environ `10^-6` en BER observé.

## Reproductibilité

Toutes les simulations utilisent des générateurs aléatoires à graine
fixée, dérivés de la graine de configuration (`--graine`, défaut 42).
Deux exécutions avec les mêmes paramètres produisent exactement les
mêmes résultats, figures et images reconstruites.
