import numpy as np
import numpy.testing as npt
import pytest

from dolphin._types import HalfWindow, Strides
from dolphin.phase_link import _core, simulate
from dolphin.phase_link._gate import (
    floor_threshold,
    gate_coherence_matrices,
)

N = 12
simulate._seed(7)


def consistent_stack(num_pixels=6, seed=0):
    """Noise-free coherence matrices with one consistent phase history per pixel."""
    rng = np.random.default_rng(seed)
    out = np.empty((1, num_pixels, N, N), dtype=np.complex64)
    truth = rng.uniform(-np.pi, np.pi, size=(num_pixels, N))
    for p in range(num_pixels):
        gamma = 0.9 ** np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
        phase = np.exp(1j * (truth[p][:, None] - truth[p][None, :]))
        out[0, p] = gamma * phase
    return out, truth


def biased_stack(lag_from=3, bias=1.0):
    """Entries of lag >= `lag_from` carry an extra phase the chain does not."""
    stack, truth = consistent_stack()
    lag = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    bump = np.where(lag >= lag_from, np.sign(np.arange(N)[None, :] - np.arange(N)[:, None]) * bias, 0.0)
    return (stack * np.exp(1j * bump)).astype(np.complex64), truth


def test_floor_threshold():
    # The two best absolute thresholds found on the Imja crop both sit near 3e-6.
    assert abs(floor_threshold(1e-6, 153) - 0.295) < 0.005
    assert abs(floor_threshold(1e-6, 45) - 0.52) < 0.01
    assert floor_threshold(1e-2, 153) < floor_threshold(1e-6, 153)


def test_consistent_matrix_is_untouched():
    stack, _ = consistent_stack()
    looks = np.full(stack.shape[:2], 100.0)
    C, out = gate_coherence_matrices(stack, looks, floor=0.0, gate="lag", k=3.0)
    npt.assert_array_equal(np.asarray(C), stack)
    assert np.all(np.asarray(out.max_consistent_lag) == N - 1)
    npt.assert_allclose(np.asarray(out.admitted_fraction), 1.0)
    npt.assert_allclose(np.asarray(out.inconsistent_admitted), 0.0)
    npt.assert_allclose(np.asarray(out.admitted_bias), 0.0, atol=1e-6)


def test_lag_gate_stops_at_first_inconsistent_lag():
    stack, _ = biased_stack(lag_from=3)
    looks = np.full(stack.shape[:2], 100.0)
    C, out = gate_coherence_matrices(stack, looks, gate="lag", k=3.0)
    C = np.asarray(C)
    lag = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    assert np.all(np.asarray(out.max_consistent_lag) == 2)
    assert np.all(C[..., lag >= 3] == 0)
    assert np.all(C[..., (lag > 0) & (lag < 3)] != 0)
    # Nothing admitted is inconsistent, and the admitted bias is at the noise level.
    npt.assert_allclose(np.asarray(out.inconsistent_admitted), 0.0)
    assert np.all(np.asarray(out.admitted_bias) < 1e-3)


def test_entry_gate_drops_biased_entries_only():
    stack, _ = biased_stack(lag_from=3)
    looks = np.full(stack.shape[:2], 100.0)
    C, _ = gate_coherence_matrices(stack, looks, gate="entry", k=3.0)
    lag = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    assert np.all(np.asarray(C)[..., lag >= 3] == 0)
    assert np.all(np.asarray(C)[..., lag < 3] != 0)


def test_chain_gate_and_keep_lag():
    stack, _ = consistent_stack()
    looks = np.full(stack.shape[:2], 100.0)
    lag = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    C, _ = gate_coherence_matrices(stack, looks, gate="chain", keep_lag=2)
    assert np.all(np.asarray(C)[..., lag > 2] == 0)
    assert np.all(np.asarray(C)[..., lag <= 2] != 0)


def test_floor_drops_weak_long_entries():
    stack, _ = consistent_stack()
    looks = np.full(stack.shape[:2], 100.0)
    C, out = gate_coherence_matrices(stack, looks, floor=0.5, gate="none")
    mag = np.abs(np.asarray(C))
    lag = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
    assert np.all(mag[..., (lag > 1) & (0.9**lag < 0.5)] == 0)
    assert np.all(mag[..., lag == 1] > 0)
    assert np.all(np.asarray(out.admitted_fraction) < 1)


def test_rejects_unknown_gate():
    stack, _ = consistent_stack()
    with pytest.raises(ValueError, match="gate"):
        gate_coherence_matrices(stack, np.full(stack.shape[:2], 10.0), gate="loose")


def chain_agreement(cpx, truth):
    est = np.angle(cpx[..., :-1] * np.conj(cpx[..., 1:]))
    expected = truth[..., :-1] - truth[..., 1:]
    return np.abs(np.mean(np.exp(1j * (est - expected)), axis=-1))


def test_weighted_evd_on_chain_reproduces_phases():
    stack, truth = consistent_stack()
    looks = np.full(stack.shape[:2], 100.0)
    C, _ = gate_coherence_matrices(stack, looks, gate="chain")
    cpx, _, estimator, _ = _core.process_coherence_matrices(
        C, use_weighted_evd=True, compute_crlb=False
    )
    assert np.all(np.asarray(estimator) == _core.EstimatorType.WEVD)
    assert np.all(chain_agreement(np.asarray(cpx), truth)[0] > 1 - 1e-5)


def test_weighted_evd_matches_evd_on_consistent_dense_matrix():
    stack, truth = consistent_stack()
    w, _, _, _ = _core.process_coherence_matrices(stack, use_weighted_evd=True, compute_crlb=False)
    e, _, _, _ = _core.process_coherence_matrices(stack, use_evd=True, eigen_solver="exact", compute_crlb=False)
    assert np.all(chain_agreement(np.asarray(w), truth)[0] > 1 - 1e-5)
    assert np.all(chain_agreement(np.asarray(e), truth)[0] > 1 - 1e-5)


def test_run_cpl_with_gate_returns_layers():
    C, _ = simulate.simulate_coh(num_acq=N, add_signal=True)
    samples = simulate.simulate_neighborhood_stack(C, 9 * 9).reshape(N, 9, 9)
    out = _core.run_cpl(
        samples, HalfWindow(x=4, y=4), Strides(x=1, y=1),
        use_weighted_evd=True, coherence_floor=0.2, consistency_gate="lag",
        consistency_k=3.0, keep_lag=1, eigen_solver="exact", compute_crlb=False,
    )
    assert out.gate is not None
    assert out.cpx_phase.shape == (N, 9, 9)
    lags = np.asarray(out.gate.max_consistent_lag)
    assert lags.shape == (9, 9) and lags.min() >= 1 and lags.max() <= N - 1
    assert np.all(np.asarray(out.estimator) == _core.EstimatorType.WEVD)
    assert np.all((np.asarray(out.temp_coh) >= 0) & (np.asarray(out.temp_coh) <= 1))
    plain = _core.run_cpl(samples, HalfWindow(x=4, y=4), Strides(x=1, y=1), compute_crlb=False)
    assert plain.gate is None
