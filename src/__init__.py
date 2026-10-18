"""Package source du TP MU-MISO / SDMA / précodage Zero-Forcing.

Modules :
- ``config``        : configuration centralisée du TP ;
- ``images``        : chargement des images, conversions pixels <-> bits ;
- ``modulation``    : modulation et démodulation BPSK, calcul du BER ;
- ``canal``         : canal de Rayleigh complexe, bruit AWGN, corrélation ;
- ``precodage_zf``  : précodage Zero-Forcing et normalisation de puissance ;
- ``detection``     : modèle de transmission et détection BPSK ;
- ``simulations``   : expériences numériques ( BER, Monte-Carlo ) ;
- ``figures``       : génération des figures du rapport ;
- ``main``          : point d'entrée principal (``python -m src.main``).
"""
