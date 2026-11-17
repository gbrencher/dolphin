import os

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from dolphin import shp
from dolphin._types import HalfWindow, Strides
from dolphin.phase_link import _core, covariance, simulate
from dolphin.shp._gaussian import estimate_neighbors as gaussian_estimate_neighbors
from dolphin.utils import gpu_is_available

GPU_AVAILABLE = gpu_is_available() and os.environ.get("NUMBA_DISABLE_JIT") != "1"
NUM_ACQ = 30
simulate._seed(1234)


@pytest.fixture(scope="module")
def slc_samples(C_truth):
    C, _ = C_truth
    ns = 11 * 11
    return simulate.simulate_neighborhood_stack(C, ns)


@pytest.mark.parametrize("eigen_solver", ["iterative", "exact"])
@pytest.mark.parametrize("baseline_lag", [None, 5])
@pytest.mark.parametrize("use_evd", [False, True])
def test_estimation(C_truth, slc_samples, use_evd, baseline_lag, eigen_solver):
    _, truth = C_truth

    C_hat = np.array(covariance.coh_mat_single(slc_samples))
    # Make the single-pixel comparisons with simple implementation
    est_mle_verify = np.angle(simulate.mle(C_hat))
    est_evd_verify = np.angle(simulate.evd(C_hat))

    # Check that the estimates are close to the truth
    err_deg = 10
    assert np.degrees(simulate.rmse(truth, est_evd_verify)) < err_deg
    assert np.degrees(simulate.rmse(truth, est_mle_verify)) < err_deg

    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)

    # cpx_phase, temp_coh, eigs, _ = _core.run_cpl(
    pl_out = _core.run_cpl(
        slc_stack,
        HalfWindow(x=5, y=5),
        Strides(x=1, y=1),
        use_evd=use_evd,
        baseline_lag=baseline_lag,
        eigen_solver=eigen_solver,
    )
    assert pl_out.cpx_phase.shape == (len(est_mle_verify), 11, 11)
    assert pl_out.temp_coh.shape == (11, 11)
    assert pl_out.eigenvalues.shape == (11, 11)
    if use_evd:
        expected_min_eig = baseline_lag if baseline_lag else NUM_ACQ / 3
        assert np.all(pl_out.eigenvalues > expected_min_eig)
        assert pl_out
    else:
        # should be 1, but floating point rounding sometimes drops
        assert np.all(pl_out.eigenvalues > 0.99)

    # The middle pixel should be the same, since it had the full window
    est_phase = np.angle(pl_out.cpx_phase[:, 5, 5])
    npt.assert_array_almost_equal(est_mle_verify, est_phase, decimal=1)


def test_masked(slc_samples, C_truth):
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
    mask = np.zeros((11, 11), dtype=bool)
    # Mask the top row
    mask[0, :] = True
    # slc_samples_masked = slc_stack[:, 1:, :].reshape(NUM_ACQ, -1)
    slc_samples_masked = slc_stack[:, ~mask]

    _, truth = C_truth
    C_hat1 = covariance.coh_mat_single(slc_samples_masked)
    est_mle = np.angle(simulate.mle(np.array(C_hat1)))

    err_deg = 10
    assert np.degrees(simulate.rmse(truth, est_mle)) < err_deg

    slc_stack_masked = slc_stack.copy()
    slc_stack_masked[:, mask] = np.nan

    C_hat2 = covariance.estimate_stack_covariance(
        slc_stack_masked, half_window=HalfWindow(5, 5)
    )
    est_full = _core.process_coherence_matrices(C_hat2)[0]
    # Middle pixel should be the same
    npt.assert_array_almost_equal(est_mle, np.angle(est_full[5, 5, :]), decimal=1)


def test_run_phase_linking(slc_samples):
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
    pl_out = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
    )

    C_hat = covariance.coh_mat_single(slc_samples)
    expected_phase = np.angle(simulate.mle(np.array(C_hat)))

    # Middle pixel should be the same
    npt.assert_array_almost_equal(
        expected_phase, np.angle(pl_out.cpx_phase[:, 5, 5]), decimal=1
    )


@pytest.mark.parametrize("use_slc_amp", [False, True])
@pytest.mark.parametrize("use_max_ps", [False, True])
def test_run_phase_linking_use_slc_amp(slc_samples, use_slc_amp, use_max_ps):
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
    ps_mask = np.zeros((11, 11), dtype=bool)
    # Specify at least 1 ps
    ps_mask[1, 1] = True
    pl_out = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
        ps_mask=ps_mask,
        use_slc_amp=use_slc_amp,
        use_max_ps=use_max_ps,
    )

    expected = np.abs(slc_stack) if use_slc_amp else 1
    assert np.allclose(np.abs(pl_out.cpx_phase), expected)


def test_run_phase_linking_with_shift(slc_samples):
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
    # Pretend there's a shift which should lead to nodata at the intersection
    # First row of the first image
    slc_stack[0, 0, :] = np.nan
    # last row of the second image
    slc_stack[1, -1, :] = np.nan
    pl_out = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
    )

    assert np.isnan(pl_out.cpx_phase[:, [0, -1], :]).all()
    assert np.isnan(pl_out.temp_coh[[0, -1], :]).all()

    assert np.all(~np.isnan(pl_out.cpx_phase[:, 1:-1, :]))
    assert np.all(~np.isnan(pl_out.temp_coh[1:-1, :]))


@pytest.mark.parametrize("with_ps", [False, True])
@pytest.mark.parametrize("use_max_ps", [False, True])
def test_run_phase_linking_no_shp_neighbors_at_edge(slc_samples, with_ps, use_max_ps):
    """Mark an unsupported burst edge as nodata unless a PS fills it."""
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
    strides = Strides(3, 3)
    # The first 3x3 looked cell keeps only its corner pixel: not all nodata, so the
    # looked nodata mask keeps it, but its center pixel (1, 1) is nodata
    nodata_mask = np.zeros((11, 11), dtype=bool)
    nodata_mask[:3, :3] = True
    nodata_mask[0, 0] = False
    slc_stack[:, nodata_mask] = np.nan
    ps_mask = np.zeros((11, 11), dtype=bool)
    ps_mask[0, 0] = with_ps

    amp = np.abs(slc_stack)
    neighbor_arrays = shp.estimate_neighbors(
        halfwin_rowcol=(5, 5),
        alpha=0.001,
        strides=strides,
        mean=np.mean(amp, axis=0),
        var=np.var(amp, axis=0),
        nslc=NUM_ACQ,
    )
    assert not neighbor_arrays[0, 0].any()

    pl_out = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
        strides=strides,
        ps_mask=ps_mask,
        use_max_ps=use_max_ps,
        neighbor_arrays=neighbor_arrays,
    )
    if with_ps:
        ps_phase = slc_stack[:, 0, 0] * slc_stack[0, 0, 0].conj()
        npt.assert_array_almost_equal(
            np.angle(ps_phase), np.angle(pl_out.cpx_phase[:, 0, 0])
        )
        npt.assert_allclose(np.abs(pl_out.cpx_phase[:, 0, 0]), 1, atol=1e-6)
        assert pl_out.temp_coh[0, 0] == 1
    else:
        assert np.isnan(pl_out.cpx_phase[:, 0, 0]).all()
        assert np.isnan(pl_out.temp_coh[0, 0])
        assert np.isnan(pl_out.crlb_std_dev[:, 0, 0]).all()
    # The fully covered cells are unaffected
    assert np.isfinite(pl_out.cpx_phase[:, 1:, 1:]).all()
    assert np.isfinite(pl_out.temp_coh[1:, 1:]).all()


def test_run_cpl_missing_acquisition_and_valid_zero_phase():
    """Keep supported zero phase while rejecting a zero-power acquisition."""
    slc_stack = np.ones((4, 5, 5), dtype=np.complex64)
    slc_stack[1, 2, 2] = 0
    neighbors = np.zeros((5, 5, 3, 3), dtype=bool)
    neighbors[:, :, 1, 1] = True

    result = _core.run_cpl(
        slc_stack,
        half_window=HalfWindow(1, 1),
        strides=Strides(1, 1),
        neighbor_arrays=neighbors,
        compute_crlb=False,
    )

    assert np.isnan(result.cpx_phase[:, 2, 2]).all()
    assert np.isnan(result.temp_coh[2, 2])
    assert np.isnan(result.crlb_std_dev[:, 2, 2]).all()
    npt.assert_allclose(result.cpx_phase[:, 1, 1], 1, atol=1e-6)
    npt.assert_allclose(result.temp_coh[1, 1], 1, atol=1e-6)


@pytest.mark.parametrize("strides", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("half_window", [5, 11])
@pytest.mark.parametrize("shape", [(10, 11), (15, 20), (50, 75)])
def test_strides_window_sizes(strides, half_window, shape):
    """Test that window and stride sizes don't cause errors for many input shapes."""
    data = np.random.normal(0, 1, size=(4, *shape)).astype(np.complex64)
    if 2 * half_window + 1 > shape[0]:
        # TODO: We should raise an appropriate error, then catch here
        pytest.skip("Window size is too large for input shape")
    _core.run_phase_linking(
        data,
        half_window=HalfWindow(half_window, half_window),
        strides=Strides(strides, strides),
    )


@pytest.mark.parametrize("use_max_ps", [True, False])
@pytest.mark.parametrize("strides", [1, 2, 3])
def test_run_phase_linking_ps_fill(slc_samples, use_max_ps, strides):
    slc_stack = slc_samples.copy().reshape(NUM_ACQ, 11, 11)
    ps_idx = 2
    ps_mask = np.zeros((11, 11), dtype=bool)
    ps_mask[ps_idx, ps_idx] = True
    # Ignore RuntimeWarning
    pl_out = _core.run_phase_linking(
        slc_stack,
        half_window=HalfWindow(5, 5),
        strides=Strides(strides, strides),
        ps_mask=ps_mask,
        use_max_ps=use_max_ps,
    )
    ps_phase = slc_stack[:, ps_idx, ps_idx]
    ps_phase *= ps_phase[0].conj()  # Reference to first acquisition

    out_idx = ps_idx // strides
    npt.assert_array_almost_equal(
        np.angle(ps_phase), np.angle(pl_out.cpx_phase[:, out_idx, out_idx])
    )

    assert pl_out.temp_coh[out_idx, out_idx] == 1


def test_run_phase_linking_gaussian_vs_rect(C_truth):
    """Test that Gaussian multilooking gives similar results to RECT for synthetic data.

    For homogeneous synthetic data (all from same distribution), Gaussian and RECT
    should give similar phase estimates since both methods use the full window
    (just with different weighting).
    """
    C, truth = C_truth
    ns = 11 * 11
    slc_samples = simulate.simulate_neighborhood_stack(C, ns)
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)

    half_window = HalfWindow(x=5, y=5)
    strides = Strides(x=1, y=1)

    # Run phase linking with RECT (no neighbor_arrays = uniform weighting)
    pl_out_rect = _core.run_phase_linking(
        slc_stack,
        half_window=half_window,
        strides=strides,
        neighbor_arrays=None,  # RECT method
    )

    # Run phase linking with Gaussian weights
    gaussian_weights = gaussian_estimate_neighbors(
        halfwin_rowcol=(half_window.y, half_window.x),
        input_shape=(11, 11),
        strides=(strides.y, strides.x),
    )
    pl_out_gaussian = _core.run_phase_linking(
        slc_stack,
        half_window=half_window,
        strides=strides,
        neighbor_arrays=np.array(gaussian_weights),
    )

    # Both methods should recover the truth reasonably well
    # Compare at the center pixel (full window available)
    rect_phase = np.angle(pl_out_rect.cpx_phase[:, 5, 5])
    gaussian_phase = np.angle(pl_out_gaussian.cpx_phase[:, 5, 5])

    # Check both methods are close to truth
    err_deg_threshold = 15  # degrees
    assert np.degrees(simulate.rmse(truth, rect_phase)) < err_deg_threshold
    assert np.degrees(simulate.rmse(truth, gaussian_phase)) < err_deg_threshold

    # Gaussian and RECT should give similar results for homogeneous data
    # Allow some tolerance since weighting differs
    phase_diff = np.angle(np.exp(1j * (rect_phase - gaussian_phase)))
    assert np.degrees(np.std(phase_diff)) < err_deg_threshold

    # Temporal coherence should be similar
    npt.assert_allclose(
        pl_out_rect.temp_coh[5, 5],
        pl_out_gaussian.temp_coh[5, 5],
        atol=0.1,
    )


def test_eigen_solver_rejects_unknown(slc_samples):
    slc_stack = slc_samples.reshape(NUM_ACQ, 11, 11)
    with pytest.raises(ValueError, match="eigen_solver"):
        _core.run_cpl(
            slc_stack, HalfWindow(x=5, y=5), Strides(x=1, y=1), eigen_solver="fast"
        )


def _history_matrices(history_offset):
    phases = np.array([0.0, 0.3, 0.5, 0.9, 1.4, 1.5, 1.9, 2.4])
    full = np.concatenate([[phases[2]], phases])
    C = 0.6 * np.exp(1j * (full[:, None] - full[None, :]))
    C[0, 1:] *= np.exp(1j * history_offset)
    C[1:, 0] *= np.exp(-1j * history_offset)
    C[np.arange(9), np.arange(9)] = 1
    return phases, jnp.asarray(C[None, None].astype(np.complex64))


@pytest.mark.parametrize("history_offset", [0.0, 1.0])
def test_gated_history_leaves_the_real_solution_referenced_to_a_real_date(
    history_offset,
):
    """Admitted or dropped, a history image must not disturb a consistent stack."""
    from dolphin.phase_link._core import process_coherence_matrices
    from dolphin.phase_link._gate import gate_coherence_matrices

    phases, C = _history_matrices(history_offset)
    looks = jnp.full((1, 1), 153.0)
    gated, out = gate_coherence_matrices(
        C, looks, gate="lag", k=3.0, n_history=1, anchor=2
    )
    assert float(out.history_admitted[0, 0]) == (history_offset == 0.0)
    estimate, *_ = process_coherence_matrices(
        gated,
        reference_idx=1,
        first_real_slc_idx=1,
        compute_crlb=False,
        use_weighted_evd=True,
        n_history=1,
    )
    got = np.angle(np.asarray(estimate[0, 0, 1:]) * np.exp(-1j * phases))
    np.testing.assert_allclose(got, 0.0, atol=1e-4)


def test_gated_history_with_only_the_band_uses_the_closed_form():
    from dolphin.phase_link._core import process_coherence_matrices
    from dolphin.phase_link._gate import gate_coherence_matrices

    phases, C = _history_matrices(0.0)
    looks = jnp.full((1, 1), 153.0)
    gated, _ = gate_coherence_matrices(
        C, looks, gate="chain", n_history=1, anchor=2
    )
    estimate, *_ = process_coherence_matrices(
        gated,
        reference_idx=1,
        first_real_slc_idx=1,
        compute_crlb=False,
        use_weighted_evd=True,
        n_history=1,
    )
    got = np.angle(np.asarray(estimate[0, 0, 1:]) * np.exp(-1j * phases))
    np.testing.assert_allclose(got, 0.0, atol=1e-5)


def _stack_with_a_blank_date(blank=3, n=7, shape=(40, 64), seed=0):
    """Coherent speckle with a known phase per date; one date blanked in the left half."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    phases = np.linspace(0, 1.5, n)
    stack = np.stack(
        [
            (0.95 * base + 0.3 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)))
            * np.exp(1j * phases[k])
            for k in range(n)
        ]
    ).astype(np.complex64)
    stack[blank, :, : shape[1] // 2] = 0
    return stack, phases


def test_compact_missing_dates_orders_dates_with_data_first():
    C = np.full((1, 2, 5, 5), 0.5, dtype=np.complex64)
    for k in range(5):
        C[..., k, k] = 1
    for d in (1, 3):  # dates 1 and 3 missing at the first pixel
        C[0, 0, d, :] = 0
        C[0, 0, :, d] = 0
    C[0, 0, 0, 2] = 0.5 * np.exp(0.3j)
    C[0, 0, 2, 0] = 0.5 * np.exp(-0.3j)
    ordered, present, order = _core.compact_missing_dates(jnp.asarray(C))
    assert np.asarray(order)[0, 0].tolist() == [0, 2, 4, 1, 3]
    assert np.asarray(present)[0, 0].tolist() == [True, True, True, False, False]
    ordered = np.asarray(ordered)
    np.testing.assert_allclose(ordered[0, 0, 0, 1], C[0, 0, 0, 2])  # 0 and 2 now adjacent
    assert not ordered[0, 0, 3:, :].any() and not ordered[0, 0, :, 3:].any()
    assert np.asarray(order)[0, 1].tolist() == [0, 1, 2, 3, 4]  # untouched where complete
    np.testing.assert_array_equal(ordered[0, 1], C[0, 1])


def test_compact_missing_dates_keeps_leading_history_in_place():
    C = np.full((1, 1, 4, 4), 0.5, dtype=np.complex64)
    for k in range(4):
        C[..., k, k] = 1
    C[0, 0, 0, :] = 0
    C[0, 0, :, 0] = 0  # a history image without data stays first
    C[0, 0, 2, :] = 0
    C[0, 0, :, 2] = 0
    _, present, order = _core.compact_missing_dates(jnp.asarray(C), n_lead=1)
    assert np.asarray(order)[0, 0].tolist() == [0, 1, 3, 2]
    assert np.asarray(present)[0, 0].tolist() == [False, True, True, False]


def test_bridging_a_blanked_date_keeps_the_pixel_and_matches_the_stack_without_it():
    stack, phases = _stack_with_a_blank_date()
    hw, strides = HalfWindow(4, 6), Strides(1, 1)
    kwargs = {
        "half_window": hw,
        "strides": strides,
        "use_weighted_evd": True,
        "consistency_gate": "lag",
        "coherence_floor": 0.1,
        "compute_crlb": False,
    }
    plain = _core.run_cpl(stack, **kwargs)
    bridged = _core.run_cpl(stack, bridge_missing_dates=True, **kwargs)
    without = _core.run_cpl(np.delete(stack, 3, axis=0), **kwargs)
    inside, outside = (slice(10, 30), slice(8, 20)), (slice(10, 30), slice(44, 56))
    # Unbridged, a pixel lacking one date has no estimate at all.
    assert np.isnan(np.asarray(plain.cpx_phase)[:, inside[0], inside[1]]).all()
    phase = np.asarray(bridged.cpx_phase)
    missing = np.asarray(bridged.missing_dates)
    assert missing[3][inside].all() and not missing[3][outside].any()
    assert not missing[[0, 1, 2, 4, 5, 6]].any()
    assert np.isnan(phase[3][inside]).all()
    keep = [0, 1, 2, 4, 5, 6]
    # Inside the blanked half: the same interferograms as the stack without the date.
    ref = np.asarray(without.cpx_phase)
    for a, b in ((0, 2), (2, 4), (4, 5)):
        got = phase[keep[b]] * np.conj(phase[keep[a]])
        want = ref[b] * np.conj(ref[a])
        npt.assert_allclose(np.angle(got[inside] * np.conj(want[inside])), 0, atol=1e-5)
    # Outside it: unchanged by the option.
    full = np.asarray(plain.cpx_phase)
    for a, b in ((2, 3), (3, 4), (0, 6)):
        got = phase[b] * np.conj(phase[a])
        want = full[b] * np.conj(full[a])
        npt.assert_allclose(np.angle(got[outside] * np.conj(want[outside])), 0, atol=1e-5)
