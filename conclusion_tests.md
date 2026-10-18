**Validation du précodage Zero-Forcing et de la normalisation de puissance**

La simulation a été réalisée pour un système MU-MISO comportant huit antennes d’émission et quatre utilisateurs. La matrice du canal \(H\) est de dimension \(4\times8\), tandis que la matrice de précodage ZF normalisée \(W\_{\mathrm{norm}}\) est de dimension \(8\times4\).

Les résultats montrent que la puissance moyenne totale émise est égale à la puissance cible \(P*T=1\), à la précision numérique près. La matrice effective \(G=HW*{\mathrm{norm}}\) présente des coefficients diagonaux de module proche de \(1{,}103075\) et des coefficients hors diagonale pratiquement nuls.

La heatmap confirme ainsi l’annulation des interférences inter-utilisateurs par le précodage ZF dans les conditions idéales de simulation, avec une connaissance parfaite du canal et sans prise en compte du bruit de réception.
