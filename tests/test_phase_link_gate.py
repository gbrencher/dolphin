import jax.numpy as jnp
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


def test_run_phase_linking_keeps_gate_layers():
    C, _ = simulate.simulate_coh(num_acq=N, add_signal=True)
    samples = simulate.simulate_neighborhood_stack(C, 9 * 9).reshape(N, 9, 9)
    out = _core.run_phase_linking(
        samples, HalfWindow(x=4, y=4), Strides(x=1, y=1),
        use_weighted_evd=True, coherence_floor=0.2, consistency_gate="lag",
        compute_crlb=False,
    )
    assert out.gate is not None
    assert np.asarray(out.gate.max_consistent_lag).shape == (9, 9)


def test_temporal_coherence_is_over_the_whole_matrix_when_gated():
    """A chain gate fits its own entries exactly but not the matrix it came from."""
    C, _ = simulate.simulate_coh(num_acq=N, add_signal=True)
    samples = simulate.simulate_neighborhood_stack(C, 9 * 9).reshape(N, 9, 9)
    kwargs = dict(half_window=HalfWindow(x=4, y=4), strides=Strides(x=1, y=1),
                  use_weighted_evd=True, compute_crlb=False)
    chain = _core.run_cpl(samples, consistency_gate="chain", **kwargs)
    admitted = np.asarray(chain.gate.admitted_temporal_coherence)
    full = np.asarray(chain.temp_coh)
    # Over its admitted band the chain reproduces every entry.
    npt.assert_allclose(admitted, 1.0, atol=1e-4)
    # Against the whole matrix it cannot, on noisy data.
    assert np.all(full < admitted)
    assert np.all((full >= 0) & (full <= 1))
    # The delivered definition: the same number `estimate_temp_coh` gives for
    # this solution against the ungated matrix.
    from dolphin.phase_link import covariance, metrics
    C_hat = covariance.estimate_stack_covariance(
        samples, HalfWindow(x=4, y=4), Strides(x=1, y=1)
    )
    expected = metrics.estimate_temp_coh(np.moveaxis(np.asarray(chain.cpx_phase), 0, -1), C_hat)
    npt.assert_allclose(full, np.asarray(expected), atol=1e-5)
    plain = _core.run_cpl(samples, **kwargs)
    assert plain.gate is None


# ----- compressed history under the gate -----


def _stack_with_history(phases, anchor, history_offsets, coherence=0.6):
    """Coherence matrices for real dates with `phases`, plus history images.

    C[i, j] has phase phases[i] - phases[j]. Each history image stands at the
    anchor date, with its phase against every real date shifted by its offset.
    """
    n_real, h = len(phases), len(history_offsets)
    full = np.concatenate([np.full(h, phases[anchor]), phases])
    C = coherence * np.exp(1j * (full[:, None] - full[None, :]))
    for q, offset in enumerate(history_offsets):
        C[q, h:] *= np.exp(1j * offset)
        C[h:, q] *= np.exp(-1j * offset)
    C[np.arange(n_real + h), np.arange(n_real + h)] = 1
    return jnp.asarray(C[None, None].astype(np.complex64))


def _gate_history(C, n_history, anchor, gate="lag", floor=0.0):
    looks = jnp.full(C.shape[:2], 153.0)
    return gate_coherence_matrices(
        C, looks, floor=floor, gate=gate, k=3.0, n_history=n_history, anchor=anchor
    )


PHASES = np.array([0.0, 0.3, 0.5, 0.9, 1.4, 1.5, 1.9, 2.4])


def test_consistent_history_is_admitted():
    C = _stack_with_history(PHASES, anchor=2, history_offsets=[0.0])
    gated, out = _gate_history(C, 1, 2)
    assert float(out.history_admitted[0, 0]) == 1
    assert abs(float(out.history_bias[0, 0])) < 1e-5
    np.testing.assert_allclose(gated[0, 0, 0, 1:], C[0, 0, 0, 1:])


def test_inconsistent_history_is_dropped_and_its_bias_reported():
    C = _stack_with_history(PHASES, anchor=2, history_offsets=[1.0])
    gated, out = _gate_history(C, 1, 2)
    assert float(out.history_admitted[0, 0]) == 0
    np.testing.assert_allclose(float(out.history_bias[0, 0]), 1.0, atol=1e-5)
    assert np.all(np.asarray(gated[0, 0, 0, 1:]) == 0)
    assert np.all(np.asarray(gated[0, 0, 1:, 0]) == 0)
    assert gated[0, 0, 0, 0] == 1


def test_history_does_not_change_the_gating_of_real_dates():
    C = _stack_with_history(PHASES, anchor=2, history_offsets=[1.0, 0.0])
    gated, out = _gate_history(C, 2, 2)
    looks = jnp.full(C.shape[:2], 153.0)
    real, real_out = gate_coherence_matrices(C[..., 2:, 2:], looks, gate="lag", k=3.0)
    np.testing.assert_array_equal(np.asarray(gated[..., 2:, 2:]), np.asarray(real))
    np.testing.assert_array_equal(
        np.asarray(out.max_consistent_lag), np.asarray(real_out.max_consistent_lag)
    )


def test_history_is_admitted_from_the_newest_until_the_first_failure():
    # Index 0 is the older image, index 1 the newer.
    newer_bad = _stack_with_history(PHASES, anchor=0, history_offsets=[0.0, 1.0])
    _, out = _gate_history(newer_bad, 2, 0)
    assert float(out.history_admitted[0, 0]) == 0
    older_bad = _stack_with_history(PHASES, anchor=0, history_offsets=[1.0, 0.0])
    gated, out = _gate_history(older_bad, 2, 0)
    assert float(out.history_admitted[0, 0]) == 1
    assert np.all(np.asarray(gated[0, 0, 0, 2:]) == 0)
    assert np.all(np.asarray(gated[0, 0, 1, 2:]) != 0)


def test_entries_between_history_images_are_always_dropped():
    C = _stack_with_history(PHASES, anchor=0, history_offsets=[0.0, 0.0])
    gated, _ = _gate_history(C, 2, 0)
    assert gated[0, 0, 0, 1] == 0 and gated[0, 0, 1, 0] == 0
    assert gated[0, 0, 0, 0] == 1 and gated[0, 0, 1, 1] == 1


def test_history_respects_the_floor_and_the_chain_gate():
    weak = _stack_with_history(PHASES, anchor=2, history_offsets=[0.0], coherence=0.2)
    _, out = _gate_history(weak, 1, 2, gate="none", floor=0.3)
    assert float(out.history_admitted[0, 0]) == 0
    C = _stack_with_history(PHASES, anchor=2, history_offsets=[0.0])
    _, out = _gate_history(C, 1, 2, gate="chain")
    assert float(out.history_admitted[0, 0]) == 0


def test_anchor_must_be_a_real_date():
    C = _stack_with_history(PHASES, anchor=2, history_offsets=[0.0])
    with pytest.raises(ValueError, match="anchor"):
        _gate_history(C, 1, len(PHASES))


def test_signed_bias_is_the_mean_closure_of_a_lag():
    C = np.array(_stack_with_history(PHASES, anchor=0, history_offsets=[]))
    n = len(PHASES)
    for i in range(n - 2):
        C[0, 0, i, i + 2] *= np.exp(0.2j)
        C[0, 0, i + 2, i] *= np.exp(-0.2j)
    looks = jnp.full((1, 1), 153.0)
    _, out = gate_coherence_matrices(jnp.asarray(C), looks, gate="lag", k=3.0)
    np.testing.assert_allclose(float(out.signed_bias_lag2[0, 0]), 0.2, atol=1e-5)
    np.testing.assert_allclose(float(out.signed_bias_lag3[0, 0]), 0.0, atol=1e-5)
    assert float(out.history_admitted[0, 0]) == 0
    assert np.isnan(float(out.history_bias[0, 0]))


def test_gated_history_option_needs_the_gate_and_the_weighted_estimator():
    from dolphin.workflows.config import PhaseLinkingOptions

    with pytest.raises(ValueError, match="gate_compressed_history"):
        PhaseLinkingOptions(gate_compressed_history=True)
    with pytest.raises(ValueError, match="gate_compressed_history"):
        PhaseLinkingOptions(gate_compressed_history=True, consistency_gate="lag")
    options = PhaseLinkingOptions(
        gate_compressed_history=True, consistency_gate="lag", use_weighted_evd=True
    )
    assert options.history_anchor_idx == 0
