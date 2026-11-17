"""Per-pixel gating of the coherence matrix before phase linking.

Phase linking assumes one phase history that every entry of the coherence
matrix samples. Over ground that moves heterogeneously inside the window the
longer pairs read a different, slower population than the nearest-neighbour
pairs, so the joint solution is biased toward them. The gate here admits each
long entry per pixel only where it is coherent and consistent with the chained
nearest-neighbour entries, and reports a per-pixel bandwidth
(`max_consistent_lag`) that marks such ground without a mask.

Two tests, applied to entries beyond `keep_lag` (the nearest-neighbour band is
always kept so every acquisition stays connected):

- a coherence floor, set per window by the probability that a truly incoherent
  entry passes (`floor_threshold`);
- a consistency test: the lag bias of an entry is its phase minus the sum of
  the chained nearest-neighbour phases spanning it, and its noise is the
  Cramer-Rao phase standard deviation of the entry and of the chain at the
  realised look count. `entry` drops an entry whose bias exceeds `k` times its
  noise; `lag` pools the bias over every start of a lag and admits lags in
  order until the first whose pooled bias exceeds `k` times the noise of the
  pooled mean, which gives each pixel its own temporal bandwidth.

The pooled noise treats the entries of a lag as independent, which they are
not, so the lag gate is somewhat conservative toward dropping lags.

Compressed history. With `n_history` > 0 the first `n_history` elements of the
stack are compressed images of earlier batches, each carrying the base phase of
one real acquisition of the current stack, the anchor. They take no part in
the band or in the lags: those are defined on the real acquisitions alone. A
history image is treated as one more long entry per real acquisition, predicted
by the chained nearest-neighbour phases between the anchor and that
acquisition, and admitted per pixel by the same floor and the same consistency
test, pooled over the real acquisitions. History images are admitted from the
newest to the oldest until the first inconsistent one. Entries between two
history images are always dropped; they say nothing about the real
acquisitions. With `n_history` = 0 the function is unchanged.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from jax.typing import ArrayLike

# Fixed z threshold for the diagnostic layers, independent of the gate's `k`.
INCONSISTENT_Z = 3.0


class GateOutput(NamedTuple):
    """Per-pixel layers describing what the gate did."""

    max_consistent_lag: Array
    """Largest lag admitted in order by the pooled consistency test, 1 to nslc-1."""

    admitted_fraction: Array
    """Fraction of all upper-triangle entries left nonzero after gating."""

    inconsistent_admitted: Array
    """Fraction of admitted long entries whose |bias| exceeds 3 times its noise;
    NaN where no long entry is admitted."""

    admitted_bias: Array
    """Mean |lag bias| of the admitted long entries, radians; NaN where none."""

    signed_bias_lag2: Array | None = None
    """Pooled lag bias at lag 2, radians, signed: the phase of a lag-2 entry minus
    the sum of the two nearest-neighbour entries it spans, circular mean over
    every start. This is the mean closure phase of consecutive triplets. Computed
    before any entry is dropped; NaN where the stack is too short."""

    signed_bias_lag3: Array | None = None
    """As `signed_bias_lag2`, at lag 3."""

    signed_bias_lag4: Array | None = None
    """As `signed_bias_lag2`, at lag 4."""

    history_admitted: Array | None = None
    """Number of compressed history images with at least one admitted entry; 0
    without history."""

    history_bias: Array | None = None
    """Pooled bias of the newest history image against the chain, radians,
    signed, computed before any entry is dropped; NaN without history."""

    admitted_temporal_coherence: Array | None = None
    """Temporal coherence of the solution over the admitted entries only. Filled
    in by `run_cpl` once the phase is solved. It measures how well the solution
    fits what it was given, and is exactly 1 where only the nearest-neighbour
    band was admitted (a chain reproduces its own entries), so it is not a
    measure of data quality; `temp_coh`, over the whole matrix, is."""


def floor_threshold(false_admission: float, nlooks: int) -> float:
    """Sample-coherence magnitude that a truly incoherent entry exceeds with probability p.

    For L independent looks the magnitude of the sample coherence of an
    incoherent pair has P(|g| > t) = (1 - t^2)^(L - 1), so
    t = sqrt(1 - p^(1 / (L - 1))). The nominal window pixel count is the usual
    choice for L; where pixels are correlated the calibration is optimistic.
    """
    return float(np.sqrt(1.0 - false_admission ** (1.0 / (nlooks - 1))))


def phase_variance(coherence: ArrayLike, looks: ArrayLike) -> Array:
    """Cramer-Rao phase variance of a multilooked interferogram entry."""
    g = jnp.clip(coherence, 1e-3, 0.999)
    return (1 - g**2) / (2 * looks * g**2)


@partial(
    jit,
    static_argnames=("floor", "gate", "k", "keep_lag", "n_history", "anchor"),
)
def gate_coherence_matrices(
    C_arrays: ArrayLike,
    looks: ArrayLike,
    floor: float = 0.0,
    gate: str = "none",
    k: float = 3.0,
    keep_lag: int = 1,
    n_history: int = 0,
    anchor: int = 0,
    present: ArrayLike | None = None,
) -> tuple[Array, GateOutput]:
    """Zero the entries of each pixel's coherence matrix that fail the gate.

    Parameters
    ----------
    C_arrays : ArrayLike
        Coherence matrices, shape (rows, cols, nslc, nslc).
    looks : ArrayLike
        Realised look count per pixel, shape (rows, cols), e.g. the SHP count.
    floor : float
        Sample-coherence magnitude below which a long entry is dropped; 0 keeps all.
    gate : str
        "none" (floor only), "entry", "lag", or "chain" (drop every long entry).
    k : float
        Consistency threshold in units of the Cramer-Rao noise.
    keep_lag : int
        Lags up to this value are never dropped.
    n_history : int
        Number of leading elements that are compressed history images.
    anchor : int
        Index, among the real acquisitions, of the one whose base phase every
        history image carries.
    present : ArrayLike, optional
        Boolean, shape (rows, cols, number of real acquisitions): which real
        acquisitions have data at each pixel, where missing ones have been
        ordered last (`dolphin.phase_link._core.compact_missing_dates`).
        Entries of a missing acquisition are dropped and left out of every
        pooled statistic, so the gate sees the stack the pixel would have
        without them.

    Returns
    -------
    C_gated : Array
        The matrices with dropped entries set to zero, same shape as the input.
    GateOutput
        The per-pixel layers; computed before thresholding where they describe
        the data (`max_consistent_lag`) and after where they describe the gate.

    """
    if gate not in ("none", "entry", "lag", "chain"):
        msg = f"gate must be one of none, entry, lag, chain; got {gate!r}"
        raise ValueError(msg)
    if n_history:
        return _gate_with_history(
            C_arrays, looks, floor, gate, k, keep_lag, n_history, anchor, present
        )
    C_gated, out, _ = _gate_real(C_arrays, looks, floor, gate, k, keep_lag, present)
    return C_gated, out


def _gate_real(C_arrays, looks, floor, gate, k, keep_lag, present=None):
    """Gate a stack of real acquisitions; also return the chain it was tested against."""
    n = C_arrays.shape[-1]
    iu0, iu1 = np.triu_indices(n, 1)
    lag_of_pair = iu1 - iu0
    is_long = jnp.asarray(lag_of_pair > keep_lag)
    looks = jnp.maximum(looks, 1).astype(jnp.float32)[..., None]

    # The chained nearest-neighbour phases and their accumulated variance.
    band = jnp.stack([jnp.angle(C_arrays[..., i, i + 1]) for i in range(n - 1)], -1)
    band_var = jnp.stack(
        [
            phase_variance(jnp.abs(C_arrays[..., i, i + 1]), looks[..., 0])
            for i in range(n - 1)
        ],
        -1,
    )
    theta = jnp.concatenate([jnp.zeros_like(band[..., :1]), jnp.cumsum(band, -1)], -1)
    cum_var = jnp.concatenate(
        [jnp.zeros_like(band_var[..., :1]), jnp.cumsum(band_var, -1)], -1
    )
    entries = C_arrays[..., iu0, iu1]
    chain = theta[..., iu1] - theta[..., iu0]
    bias = jnp.angle(jnp.exp(1j * (jnp.angle(entries) - chain)))
    noise = jnp.sqrt(
        phase_variance(jnp.abs(entries), looks) + cum_var[..., iu1] - cum_var[..., iu0]
    )
    z = jnp.abs(bias) / jnp.maximum(noise, 1e-6)

    # Pooled per-lag consistency, admitting lags in order until the first failure.
    lag_passes = []
    signed = {}
    pair_present = None
    if present is not None:
        pair_present = present[..., iu0] & present[..., iu1]
    for lag in range(2, n):
        idx = lag_of_pair == lag
        if pair_present is None:
            pooled = jnp.angle(jnp.exp(1j * bias[..., idx]).mean(-1))
            pooled_noise = jnp.sqrt(
                (noise[..., idx] ** 2).mean(-1) / int(idx.sum())
            )
            passes = jnp.abs(pooled) / jnp.maximum(pooled_noise, 1e-6) <= k
        else:
            # Pool over the entries between acquisitions that have data; a lag
            # with none passes, and is capped out of the bandwidth below.
            w = pair_present[..., idx]
            count = w.sum(-1)
            pooled = jnp.angle((jnp.exp(1j * bias[..., idx]) * w).sum(-1))
            mean_square = (noise[..., idx] ** 2 * w).sum(-1) / jnp.maximum(count, 1)
            pooled_noise = jnp.sqrt(mean_square / jnp.maximum(count, 1))
            passes = jnp.where(
                count > 0, jnp.abs(pooled) / jnp.maximum(pooled_noise, 1e-6) <= k, True
            )
            pooled = jnp.where(count > 0, pooled, jnp.nan)
        if lag <= 4:
            signed[lag] = pooled.astype(jnp.float32)
        lag_passes.append(passes)
    ok_lag = jnp.cumprod(jnp.stack(lag_passes, -1).astype(jnp.int32), axis=-1) > 0
    max_consistent_lag = 1 + ok_lag.sum(-1)
    if present is not None:
        max_consistent_lag = jnp.minimum(
            max_consistent_lag, jnp.maximum(present.sum(-1) - 1, 1)
        )

    drop = jnp.zeros(bias.shape, dtype=bool)
    if floor > 0:
        drop = drop | (jnp.abs(entries) < floor)
    if gate == "entry":
        drop = drop | (z > k)
    elif gate == "lag":
        drop = drop | ~ok_lag[..., np.clip(lag_of_pair - 2, 0, n - 3)]
    elif gate == "chain":
        drop = jnp.ones_like(drop)
    drop = drop & is_long
    if pair_present is not None:
        drop = drop | ~pair_present

    upper = jnp.where(drop, 0, entries)
    C_gated = C_arrays.at[..., iu0, iu1].set(upper)
    C_gated = C_gated.at[..., iu1, iu0].set(jnp.conj(upper))

    keep = jnp.abs(upper) > 0
    flagged = z > INCONSISTENT_Z
    admitted_long = keep & is_long
    n_admitted = admitted_long.sum(-1)
    safe = jnp.maximum(n_admitted, 1)
    nan = jnp.float32(jnp.nan)
    missing = jnp.full(max_consistent_lag.shape, jnp.nan, dtype=jnp.float32)
    out = GateOutput(
        max_consistent_lag=max_consistent_lag.astype(jnp.float32),
        admitted_fraction=keep.mean(-1).astype(jnp.float32),
        inconsistent_admitted=jnp.where(
            n_admitted > 0, (flagged & admitted_long).sum(-1) / safe, nan
        ).astype(jnp.float32),
        admitted_bias=jnp.where(
            n_admitted > 0, (jnp.abs(bias) * admitted_long).sum(-1) / safe, nan
        ).astype(jnp.float32),
        signed_bias_lag2=signed.get(2, missing),
        signed_bias_lag3=signed.get(3, missing),
        signed_bias_lag4=signed.get(4, missing),
        history_admitted=jnp.zeros(max_consistent_lag.shape, dtype=jnp.float32),
        history_bias=missing,
    )
    return C_gated, out, (theta, cum_var)


def _gate_with_history(
    C_arrays, looks, floor, gate, k, keep_lag, n_history, anchor, present=None
):
    """Gate the real acquisitions, then test each history image against their chain."""
    h = n_history
    n_real = C_arrays.shape[-1] - h
    if not 0 <= anchor < n_real:
        msg = f"anchor {anchor} is not one of the {n_real} real acquisitions"
        raise ValueError(msg)
    real_gated, out, (theta, cum_var) = _gate_real(
        C_arrays[..., h:, h:], looks, floor, gate, k, keep_lag, present
    )
    looks_ = jnp.maximum(looks, 1).astype(jnp.float32)[..., None]

    # A history image stands at the anchor: its entry with real acquisition t is
    # predicted by the chain from the anchor to t.
    predicted = theta - theta[..., anchor : anchor + 1]
    chain_var = jnp.abs(cum_var - cum_var[..., anchor : anchor + 1])
    entries = C_arrays[..., :h, h:]  # (..., history, real)
    bias = jnp.angle(jnp.exp(1j * (jnp.angle(entries) - predicted[..., None, :])))
    noise = jnp.sqrt(
        phase_variance(jnp.abs(entries), looks_[..., None]) + chain_var[..., None, :]
    )
    z = jnp.abs(bias) / jnp.maximum(noise, 1e-6)
    if present is None:
        pooled = jnp.angle(jnp.exp(1j * bias).mean(-1))  # (..., history)
        pooled_noise = jnp.sqrt((noise**2).mean(-1) / n_real)
    else:
        w = present[..., None, :]
        count = jnp.maximum(w.sum(-1), 1)
        pooled = jnp.angle((jnp.exp(1j * bias) * w).sum(-1))
        pooled_noise = jnp.sqrt((noise**2 * w).sum(-1) / count / count)
    passes = jnp.abs(pooled) / jnp.maximum(pooled_noise, 1e-6) <= k
    # Newest first: admit history images until the first inconsistent one.
    ok = jnp.cumprod(passes[..., ::-1].astype(jnp.int32), axis=-1)[..., ::-1] > 0

    drop = jnp.zeros(bias.shape, dtype=bool)
    if floor > 0:
        drop = drop | (jnp.abs(entries) < floor)
    if gate == "entry":
        drop = drop | (z > k)
    elif gate == "lag":
        drop = drop | ~ok[..., None]
    elif gate == "chain":
        drop = jnp.ones_like(drop)
    if present is not None:
        drop = drop | ~present[..., None, :]
    kept = jnp.where(drop, 0, entries)

    C_gated = C_arrays.at[..., h:, h:].set(real_gated)
    C_gated = C_gated.at[..., :h, h:].set(kept)
    C_gated = C_gated.at[..., h:, :h].set(jnp.conj(jnp.swapaxes(kept, -1, -2)))
    diagonal = jnp.diagonal(C_arrays[..., :h, :h], axis1=-2, axis2=-1)
    between = jnp.zeros_like(C_arrays[..., :h, :h])
    idx = np.arange(h)
    between = between.at[..., idx, idx].set(diagonal)
    C_gated = C_gated.at[..., :h, :h].set(between)

    out = out._replace(
        history_admitted=(jnp.abs(kept) > 0).any(-1).sum(-1).astype(jnp.float32),
        history_bias=pooled[..., -1].astype(jnp.float32),
    )
    return C_gated, out
