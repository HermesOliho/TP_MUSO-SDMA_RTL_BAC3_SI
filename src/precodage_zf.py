"""Précodage Zero-Forcing (ZF) et normalisation de puissance.

Rappel du modèle
----------------
La station de base émet ``x = W @ s`` où ``s`` est le vecteur des
``K`` symboles des utilisateurs et ``W`` la matrice de précodage de
dimensions ``(M, K)``. Le signal reçu s'écrit ``y = H @ W @ s + n``,
la matrice effective étant ``G = H @ W``.

Précodage Zero-Forcing
----------------------
Le ZF classique annule exactement les interférences entre flux quand le
canal est de rang plein en lignes :

::

    W_ZF = H^H @ inv(H @ H^H)          (dimensions : M x K)

Hypothèses nécessaires :

- ``K <= M`` : on ne peut pas annuler ``K`` interférences avec moins
  de ``K`` antennes d'émission ;
- ``rank(H) == K`` (canal de rang plein en lignes) : sinon la matrice
  ``H @ H^H`` est singulière et le ZF classique n'existe pas. Ce cas
  lève une erreur explicite (``CanalSingularError``) : il n'est jamais
  masqué par une correction silencieuse.

Choix numérique
---------------
L'implémentation par défaut utilise la pseudo-inverse de Moore-Penrose
(``numpy.linalg.pinv``), mathématiquement égale à la formule
théorique ``H^H @ inv(H @ H^H)`` lorsque ``H`` est de rang plein, mais
plus robuste numériquement (décomposition SVD tronquée). La formule
explicite reste disponible avec ``methode="formule"`` à des fins
pédagogiques ; un test unitaire vérifie que les deux méthodes
concordent.

Normalisation de puissance
--------------------------
Les symboles émis étant de puissance moyenne unitaire et la matrice de
covariance des flux étant l'identité, la puissance moyenne totale
d'émission vaut ``E[||W s||^2] = ||W||_F^2``. Le facteur

::

    alpha = sqrt(P_T / ||W_ZF||_F^2)

ramène cette puissance à la cible ``P_T`` : ``W_norm = alpha * W_ZF``
vérifie ``||W_norm||_F^2 = P_T``. Après normalisation, le canal effectif
vaut ``G = alpha * I_K`` (et non plus ``I_K``) : la diagonale vaut
``alpha``, ce qui n'est pas une interférence mais le gain utile de
chaque flux.
"""

from dataclasses import dataclass

import numpy as np


class CanalSingularError(ValueError):
    """Erreur levée lorsque le canal n'est pas de rang plein pour le ZF.

    Le précodage Zero-Forcing classique exige ``rank(H) == K``. Plutôt
    que de masquer le problème par une inversion régularisée, ce cas
    produit une erreur explicite.
    """


@dataclass
class DiagnosticsCanal:
    """Diagnostics numériques sur une matrice de canal.

    Attributes
    ----------
    k, m : int
        Dimensions du canal ``(K, M)``.
    rang : int
        Rang numérique de ``H`` (tolérance par défaut de NumPy).
    valeurs_singulieres : numpy.ndarray
        Valeurs singulières de ``H`` (décroissantes).
    conditionnement : float
        Rapport entre la plus grande et la plus petite valeur singulière.
        Grand signale un canal mal conditionné : le ZF amplifie alors le
        bruit (bruit de recodage).
    warning : str or None
        Message d'avertissement explicite si le canal est mal
        conditionné, sinon ``None``.
    """

    k: int
    m: int
    rang: int
    valeurs_singulieres: np.ndarray
    conditionnement: float
    warning: str | None

    def afficher(self) -> str:
        """Retourne un résumé textuel des diagnostics."""
        lignes = [
            f"Dimensions de H : ({self.k}, {self.m})",
            f"Rang de H : {self.rang} (K = {self.k})",
            f"Conditionnement de H : {self.conditionnement:.3e}",
        ]
        if self.warning:
            lignes.append(f"AVERTISSEMENT : {self.warning}")
        return "\n".join(lignes)


def verifier_dimensions_canal(h: np.ndarray) -> tuple:
    """Vérifie qu'une matrice de canal est bien 2D et non vide.

    Parameters
    ----------
    h : numpy.ndarray
        Matrice supposée du canal.

    Returns
    -------
    tuple of int
        Dimensions ``(K, M)``.

    Raises
    ------
    ValueError
        Si la matrice n'est pas 2D ou si une dimension est nulle.
    """
    h = np.asarray(h)
    if h.ndim != 2:
        raise ValueError(
            f"La matrice de canal doit être 2D (forme attendue (K, M)) ; "
            f"forme obtenue : {h.shape}."
        )
    k, m = h.shape
    if k < 1 or m < 1:
        raise ValueError(f"Dimensions de canal invalides : ({k}, {m}).")
    return k, m


def precodage_zf(h: np.ndarray, methode: str = "pseudo_inverse") -> np.ndarray:
    """Calcule la matrice de précodage Zero-Forcing.

    Parameters
    ----------
    h : numpy.ndarray
        Canal complexe ``(K, M)`` supposé de rang plein en lignes.
    methode : str
        ``"pseudo_inverse"`` (par défaut) : utilise
        ``numpy.linalg.pinv`` (décomposition SVD), équivalent à la
        formule théorique pour un canal de rang plein, mais plus
        robuste numériquement. ``"formule"`` : applique littéralement
        ``H^H @ inv(H @ H^H)``, utile pour la pédagogie et la
        vérification.

    Returns
    -------
    numpy.ndarray
        Matrice de précodage ``(M, K)`` non normalisée.

    Raises
    ------
    ValueError
        Si ``K > M`` (impossible d'annuler ``K`` interférences avec
        moins de ``K`` antennes) ou si la méthode est inconnue.
    CanalSingularError
        Si ``rank(H) < K`` : le ZF classique n'existe pas et aucune
        correction silencieuse n'est appliquée.
    """
    k, m = verifier_dimensions_canal(h)

    if k > m:
        raise ValueError(
            f"Le précodage ZF classique exige K <= M (K={k}, M={m})."
        )

    rang = np.linalg.matrix_rank(h)
    if rang < k:
        raise CanalSingularError(
            f"Le canal n'est pas de rang plein en lignes : rang(H) = {rang} < K = {k}. "
            "Le précodage Zero-Forcing classique n'est pas applicable. "
            "Vérifiez le modèle de canal ou utilisez explicitement une "
            "méthode régularisée (voir precodage_ridge), qui n'est pas du ZF."
        )

    if methode == "pseudo_inverse":
        # Pour un canal de rang plein en lignes : pinv(H) = H^H (H H^H)^-1.
        w_zf = np.linalg.pinv(h)
    elif methode == "formule":
        h_dagger = h.conj().T
        w_zf = h_dagger @ np.linalg.inv(h @ h_dagger)
    else:
        raise ValueError(
            f"Méthode inconnue : '{methode}'. "
            "Choisissez 'pseudo_inverse' ou 'formule'."
        )

    if w_zf.shape != (m, k):
        raise RuntimeError(
            f"Dimensions inattendues pour W_ZF : {w_zf.shape}, attendu ({m}, {k})."
        )

    return w_zf


def normaliser_puissance(w: np.ndarray, p_t: float = 1.0) -> tuple:
    """Normalise une matrice de précodage à une puissance totale cible.

    La puissance moyenne totale d'émission vaut ``||W||_F^2`` pour des
    symboles de puissance unitaire. Le facteur retourné vérifie
    ``||alpha * W||_F^2 = P_T``.

    Parameters
    ----------
    w : numpy.ndarray
        Matrice de précodage ``(M, K)`` non normalisée.
    p_t : float
        Puissance moyenne totale d'émission cible (par défaut 1.0).

    Returns
    -------
    tuple
        ``(W_norm, alpha)`` : matrice normalisée et facteur de
        normalisation réel positif.

    Raises
    ------
    ValueError
        Si la puissance de ``W`` est nulle ou si ``P_T`` n'est pas
        strictement positif.
    """
    w = np.asarray(w)
    if w.ndim != 2:
        raise ValueError(f"La matrice de précodage doit être 2D (forme : {w.shape}).")
    if p_t <= 0:
        raise ValueError(f"La puissance cible doit être strictement positive (P_T={p_t}).")

    puissance = float(np.linalg.norm(w, "fro") ** 2)
    if puissance <= 0.0:
        raise ValueError(
            "La matrice de précodage est nulle : impossible de normaliser la puissance."
        )

    alpha = float(np.sqrt(p_t / puissance))
    return alpha * w, alpha


def matrice_effective(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Calcule la matrice effective ``G = H @ W`` du système précodé.

    Parameters
    ----------
    h : numpy.ndarray
        Canal ``(K, M)``.
    w : numpy.ndarray
        Précodage ``(M, K)``.

    Returns
    -------
    numpy.ndarray
        Matrice effective ``(K, K)``. Sa diagonale contient les gains
        utiles de chaque flux, ses termes hors diagonale les
        interférences résiduelles entre utilisateurs.

    Raises
    ------
    ValueError
        Si les dimensions ne sont pas cohérentes.
    """
    k, m = verifier_dimensions_canal(h)
    w = np.asarray(w)
    if w.shape != (m, k):
        raise ValueError(
            f"Dimensions incohérentes : H est ({k}, {m}) mais W est {w.shape} ; "
            f"attendu ({m}, {k})."
        )
    return h @ w


def verifier_puissance_normee(w_norm: np.ndarray, p_t: float = 1.0, tol: float = 1e-9) -> bool:
    """Vérifie que ``||W_norm||_F^2`` est égal à la puissance cible.

    Parameters
    ----------
    w_norm : numpy.ndarray
        Matrice de précodage normalisée.
    p_t : float
        Puissance cible.
    tol : float
        Tolérance d'arrondi numérique.

    Returns
    -------
    bool
        ``True`` si la puissance est conforme à la cible.
    """
    puissance = float(np.linalg.norm(np.asarray(w_norm), "fro") ** 2)
    return abs(puissance - p_t) <= tol


def diagnostics_canal(h: np.ndarray) -> DiagnosticsCanal:
    """Produit des diagnostics sur le canal (rang, conditionnement).

    Le conditionnement est calculé à partir des valeurs singulières de
    ``H``. Un conditionnement très élevé signale un canal mal
    conditionné : le ZF reste applicable mais amplifie le bruit de
    réception (bruit de recodage), au détriment du BER.

    Parameters
    ----------
    h : numpy.ndarray
        Canal ``(K, M)``.

    Returns
    -------
    DiagnosticsCanal
        Rang, valeurs singulières, conditionnement et avertissement
        éventuel.
    """
    k, m = verifier_dimensions_canal(h)

    valeurs_singulieres = np.linalg.svd(h, compute_uv=False)
    s_max = float(valeurs_singulieres[0])
    s_min = float(valeurs_singulieres[-1])
    conditionnement = s_max / s_min if s_min > 0 else float("inf")
    rang = int(np.linalg.matrix_rank(h))

    warning = None
    if rang < k:
        warning = (
            f"Canal de rang deficient : rang(H) = {rang} < K = {k} : "
            "le ZF classique n'est pas applicable."
        )
    elif conditionnement > 1e4:
        warning = (
            f"Canal mal conditionné (cond = {conditionnement:.3e}) : "
            "le ZF amplifie le bruit de réception (bruit de recodage)."
        )

    return DiagnosticsCanal(
        k=k,
        m=m,
        rang=rang,
        valeurs_singulieres=valeurs_singulieres,
        conditionnement=conditionnement,
        warning=warning,
    )


def precodage_ridge(h: np.ndarray, mu: float, p_t: float = 1.0) -> np.ndarray:
    """Précodage régularisé de type « ridge » (**ce n'est pas du ZF**).

    Ce précodeur minimise ``||H W - I||^2 + mu * ||W||^F^2`` ; il est
    fourni uniquement comme **option distincte** pour les canaux mal
    conditionnés ou singuliers. Il n'annule pas exactement les
    interférences et ne doit jamais être présenté comme du
    Zero-Forcing classique.

    Parameters
    ----------
    h : numpy.ndarray
        Canal ``(K, M)``.
    mu : float
        Coefficient de régularisation strictement positif.
    p_t : float
        Puissance totale cible (la sortie est normalisée).

    Returns
    -------
    numpy.ndarray
        Matrice de précodage normalisée ``(M, K)``.
    """
    k, m = verifier_dimensions_canal(h)
    if mu <= 0:
        raise ValueError(f"La régularisation doit être strictement positive (mu={mu}).")

    h_dagger = h.conj().T
    w_ridge = h_dagger @ np.linalg.inv(h @ h_dagger + mu * np.eye(k, dtype=complex))
    w_norm, _ = normaliser_puissance(w_ridge, p_t)
    return w_norm
