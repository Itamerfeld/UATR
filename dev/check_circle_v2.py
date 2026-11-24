"""
Direction-aware circularity score for transition matrices.

Idea:
    A row-stochastic transition matrix P of a directed Markov chain has
    spectrum on the closed unit disk |lambda| <= 1, with lambda = 1 always
    present. Cyclic structure of period d puts eigenvalues exactly at the
    d-th roots of unity. For a *fuzzy* cycle of length ~N (noisy ring),
    there is a leading complex eigenvalue lambda_c with
        |lambda_c| close to 1, arg(lambda_c) close to 2*pi/N.
    For mixing / uniform-random transitions, the entire non-trivial
    spectrum collapses toward 0.

    Therefore:
        circularity = |lambda_c|         in [0, 1]
        period      = 2*pi / |arg(lambda_c)|

This file exposes one main function `transition_matrix_circularity`,
plus a small visual helper and a battery of synthetic sanity checks
(run the file directly to see them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class CircularityResult:
    """Bundle of diagnostics returned by `transition_matrix_circularity`."""

    score: float                # |lambda_c|, in [0, 1]
    period: float               # estimated cycle length, NaN if undefined
    leading_eigenvalue: complex # the eigenvalue lambda_c used for the score
    eigenvalues: np.ndarray     # full spectrum of the row-stochastic P
    spectral_concentration: float  # mean |lambda_k| over the top few cyclic eigs


def _row_normalize(M: np.ndarray) -> np.ndarray:
    """Row-stochastic version of M. Empty rows become uniform (mixing)."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {M.shape}.")
    row_sums = M.sum(axis=1, keepdims=True)
    empty = (row_sums.squeeze() == 0)
    P = np.where(row_sums == 0, 1.0 / M.shape[0], M / np.where(row_sums == 0, 1.0, row_sums))
    if empty.any():
        # absorb-to-uniform avoids spurious extra lambda=1 from never-visited
        # rows; you could alternatively set a self-loop, but uniform keeps the
        # chain irreducible-ish.
        pass
    return P


def transition_matrix_circularity(
    M: np.ndarray,
    *,
    n_eigs: int = 4,
    eps_imag: float = 1e-9,
    trivial_tol: float = 1e-6,
) -> CircularityResult:
    """
    Compute a direction-aware circularity score for a transition-count
    or transition-probability matrix M.

    Parameters
    ----------
    M : (N, N) array
        Counts or probabilities of transitions i -> j. Need not be normalized.
    n_eigs : int
        How many leading non-trivial eigenvalues to use for the
        "spectral concentration" diagnostic. Score itself only uses the top one.
    eps_imag : float
        Eigenvalues with |Im(lambda)| < eps_imag are treated as real and
        skipped (they cannot encode rotation).
    trivial_tol : float
        Tolerance for skipping eigenvalues lambda ≈ 1 (these come from
        connected components and stationary distributions, not from cycles).

    Returns
    -------
    CircularityResult
        score in [0, 1], 1 = perfect cycle, 0 = no cyclic structure.
    """
    P = _row_normalize(M)
    eigs = np.linalg.eigvals(P)

    # sort by magnitude descending for downstream convenience
    eigs = eigs[np.argsort(-np.abs(eigs))]

    # candidates: complex (Im != 0) and not the trivial lambda=1
    is_nontrivial = np.abs(eigs - 1.0) > trivial_tol
    is_complex = np.abs(eigs.imag) > eps_imag
    cyclic = eigs[is_nontrivial & is_complex]

    if cyclic.size == 0:
        return CircularityResult(
            score=0.0,
            period=float("nan"),
            leading_eigenvalue=0 + 0j,
            eigenvalues=eigs,
            spectral_concentration=0.0,
        )

    # score = largest |lambda| among the cyclic eigenvalues
    score = float(np.max(np.abs(cyclic)))

    # period = read off from the *fundamental* harmonic, i.e. among the
    # eigenvalues at (near-)maximum modulus, the one with the smallest
    # positive |arg|. This breaks the tie correctly when many cyclic
    # eigenvalues sit on the unit circle (perfect cycle case).
    near_max = cyclic[np.abs(cyclic) >= score - 1e-6]
    lam_c = near_max[np.argmin(np.abs(np.angle(near_max)))]
    angle = float(np.abs(np.angle(lam_c)))
    period = float(2 * np.pi / angle) if angle > 0 else float("nan")

    # diagnostic: how concentrated is the rest of the cyclic spectrum?
    top = cyclic[: max(1, n_eigs)]
    spec_conc = float(np.mean(np.abs(top)))

    return CircularityResult(
        score=score,
        period=period,
        leading_eigenvalue=complex(lam_c),
        eigenvalues=eigs,
        spectral_concentration=spec_conc,
    )


# ----------------------------------------------------------------------
# Visual helper: plot eigenvalues of P on the complex plane
# ----------------------------------------------------------------------
def plot_spectrum(M: np.ndarray, ax: Optional[plt.Axes] = None, title: str = "") -> plt.Axes:
    res = transition_matrix_circularity(M)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3)
    ax.scatter(res.eigenvalues.real, res.eigenvalues.imag, s=20, alpha=0.7)
    ax.scatter([res.leading_eigenvalue.real], [res.leading_eigenvalue.imag],
               s=120, facecolors="none", edgecolors="red", linewidths=2,
               label=fr"$\lambda_c$, score={res.score:.3f}")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"{title}\nperiod≈{res.period:.1f}" if title else f"period≈{res.period:.1f}")
    ax.legend(loc="lower right", fontsize=8)
    return ax


# ----------------------------------------------------------------------
# Synthetic test cases
# ----------------------------------------------------------------------
def _perfect_cycle(N: int, shift: int = 1) -> np.ndarray:
    M = np.zeros((N, N))
    for i in range(N):
        M[i, (i + shift) % N] = 1.0
    return M


def _noisy_cycle(N: int, shift: int = 1, p_cycle: float = 0.85, seed: int = 0) -> np.ndarray:
    """Mixture: p_cycle of probability mass on the cycle, (1 - p_cycle) uniform."""
    rng = np.random.default_rng(seed)  # kept for API parity / future variants
    del rng
    cycle = _perfect_cycle(N, shift)        # rows already sum to 1
    uniform = np.ones((N, N)) / N           # rows already sum to 1
    return p_cycle * cycle + (1 - p_cycle) * uniform


def _uniform_noise(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((N, N))


def _path_random_walk(N: int) -> np.ndarray:
    """Reversible RW on a path: NOT a cycle (eigenvalues real)."""
    M = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            M[i, i - 1] = 1
        if i < N - 1:
            M[i, i + 1] = 1
    return M


def _two_disconnected_cycles(N1: int, N2: int) -> np.ndarray:
    M = np.zeros((N1 + N2, N1 + N2))
    M[:N1, :N1] = _perfect_cycle(N1)
    M[N1:, N1:] = _perfect_cycle(N2)
    return M


def _all_self_loops(N: int) -> np.ndarray:
    return np.eye(N)


def _gemini_example() -> np.ndarray:
    """The 10-node fuzzy directed ring used in check_circle.py."""
    return np.array([
        [ 0,  3, 19, 26, 12,  6,  2,  0,  0,  0],
        [ 1,  0,  5, 21, 19, 14,  7,  3,  1,  0],
        [ 0,  1,  0,  1, 22, 28,  8,  4,  1,  1],
        [ 0,  0,  1,  0,  0, 11, 26, 17,  3,  5],
        [ 4,  2,  0,  0,  0,  0, 11, 23, 15,  3],
        [ 6,  0,  0,  2,  0,  1,  1, 12, 25, 17],
        [13,  2,  1,  0,  0,  0,  0,  2, 16, 22],
        [29, 13,  1,  1,  0,  0,  0,  2,  2, 15],
        [13, 29, 16,  1,  0,  1,  1,  0,  0,  2],
        [ 2, 20, 23, 11,  5,  3,  1,  0,  0,  1],
    ], dtype=float)


def run_sanity_tests(verbose: bool = True) -> None:
    cases = {
        "perfect 10-cycle":          _perfect_cycle(10),
        "perfect 100-cycle":         _perfect_cycle(100),
        "perfect 10-cycle, shift 3": _perfect_cycle(10, shift=3),
        "noisy 10-cycle (85% mass)": _noisy_cycle(10, p_cycle=0.85),
        "noisy 10-cycle (50% mass)": _noisy_cycle(10, p_cycle=0.50),
        "noisy 10-cycle (10% mass)": _noisy_cycle(10, p_cycle=0.10),
        "uniform noise 10x10":       _uniform_noise(10),
        "uniform noise 100x100":     _uniform_noise(100),
        "path random walk N=10":     _path_random_walk(10),
        "two disconnected cycles":   _two_disconnected_cycles(7, 5),
        "all self-loops":            _all_self_loops(10),
        "gemini example (fuzzy ring N=10, shift~3)": _gemini_example(),
    }

    if verbose:
        print(f"{'case':<48s}  {'score':>7s}  {'period':>7s}  {'spec.conc':>9s}  arg(lam_c)")
        print("-" * 90)
    for name, M in cases.items():
        r = transition_matrix_circularity(M)
        if verbose:
            print(f"{name:<48s}  {r.score:7.3f}  {r.period:7.2f}  "
                  f"{r.spectral_concentration:9.3f}  "
                  f"{np.degrees(np.angle(r.leading_eigenvalue)):+7.2f} deg")

    # picture
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for ax, (name, M) in zip(axes.flat, cases.items()):
        plot_spectrum(M, ax=ax, title=name)
    fig.tight_layout()
    fig.savefig("/Users/itamarmerfeld/Projects/UATR/results/check_circle_v2_spectra.png",
                dpi=120)
    if verbose:
        print("\nSaved: results/check_circle_v2_spectra.png")


if __name__ == "__main__":
    run_sanity_tests()
