import numpy as np
import numpy.testing as npt
import pytest

from dolphin.phase_link._eigenvalues import (
    eigh_largest_stack,
    eigh_largest_stack_exact,
    eigh_smallest_stack,
    eigh_smallest_stack_exact,
)
from dolphin.phase_link.simulate import simulate_coh

# Used for matrix size
N = 20


def get_eigvec_phase_difference(a, b):
    """Compare the phase difference of two eigenvectors (or stacks).

    Using this since there may be a relative offset in the phase of
    each starting, depending on the algorithm.
    """
    a_0 = a * a[..., [0]].conj()
    b_0 = b * b[..., [0]].conj()
    return np.angle(a_0 * b_0.conj())


class TestEighStack:
    """Test the stack functionality to compare to scipy eigh."""

    @pytest.fixture
    def coh_stack(self):
        num_rows, num_cols = 6, 7
        out = np.empty((num_rows, num_cols, N, N), dtype=np.complex64)
        for row in range(num_rows):
            for col in range(num_cols):
                out[row, col] = simulate_coh(num_acq=N, add_signal=True)[0]
        return out

    @pytest.fixture
    def expected_largest(self, coh_stack):
        # Compare to numpy
        eig_vals, eig_vecs = np.linalg.eigh(coh_stack)
        assert np.all(eig_vals >= 0)
        expected_eig = eig_vals[:, :, -1]
        expected_evec = eig_vecs[:, :, :, -1]
        assert expected_eig.shape == (6, 7)
        assert expected_evec.shape == (6, 7, N)
        return expected_eig, expected_evec

    def test_eigh_largest_stack(self, coh_stack, expected_largest):
        expected_eig, expected_evec = expected_largest
        evalues, evecs = eigh_largest_stack(coh_stack)
        assert evalues.shape == (6, 7)
        assert evecs.shape == (6, 7, N)

        npt.assert_allclose(expected_eig, evalues, atol=2e-5)
        assert np.max(np.abs(get_eigvec_phase_difference(expected_evec, evecs))) < 5e-3

    @pytest.fixture
    def coh_gamma_inv_stack(self, coh_stack) -> np.ndarray:
        return coh_stack * np.abs(np.linalg.inv(coh_stack))

    @pytest.fixture
    def expected_smallest(self, coh_gamma_inv_stack):
        # Compare to numpy
        eig_vals, eig_vecs = np.linalg.eigh(coh_gamma_inv_stack)
        assert np.all(eig_vals >= 0)
        expected_eig = eig_vals[:, :, 0]
        expected_evec = eig_vecs[:, :, :, 0]
        assert expected_eig.shape == (6, 7)
        assert expected_evec.shape == (6, 7, N)
        return expected_eig, expected_evec

    def test_eigh_smallest_stack(self, coh_gamma_inv_stack, expected_smallest):
        expected_eig, expected_evec = expected_smallest
        mu = 0.99
        evalues, evecs = eigh_smallest_stack(coh_gamma_inv_stack, mu)
        assert evalues.shape == (6, 7)
        assert evecs.shape == (6, 7, N)

        # Compare the values:
        npt.assert_allclose(evalues, expected_eig, atol=2e-5)

        # Check the max phase difference
        assert np.max(np.abs(get_eigvec_phase_difference(expected_evec, evecs))) < 1e-4


class TestEighStackExact(TestEighStack):
    """The exact solvers must match numpy's eigh to rounding on dense matrices."""

    def test_eigh_largest_stack_exact(self, coh_stack, expected_largest):
        expected_eig, expected_evec = expected_largest
        evalues, evecs = eigh_largest_stack_exact(coh_stack)
        assert evalues.shape == (6, 7)
        assert evecs.shape == (6, 7, N)
        npt.assert_allclose(expected_eig, evalues, rtol=1e-5)
        assert np.max(np.abs(get_eigvec_phase_difference(expected_evec, evecs))) < 1e-4

    def test_eigh_smallest_stack_exact(self, coh_gamma_inv_stack, expected_smallest):
        expected_eig, expected_evec = expected_smallest
        evalues, evecs = eigh_smallest_stack_exact(coh_gamma_inv_stack)
        assert evalues.shape == (6, 7)
        assert evecs.shape == (6, 7, N)
        npt.assert_allclose(evalues, expected_eig, rtol=1e-4)
        assert np.max(np.abs(get_eigvec_phase_difference(expected_evec, evecs))) < 1e-4


def chain_only_stack(num_acq=15, coherence=0.6, num_pixels=8, seed=0):
    coherence = np.broadcast_to(coherence, (num_acq - 1,))
    """Coherence matrices keeping only the nearest-neighbour band.

    With only lag-1 entries the pair graph is a tree and nothing has to be
    reconciled, so every eigenvector of C * |C| carries the chained phases
    exactly: C = D T D^H with D the diagonal of chained phases and T real
    tridiagonal. The top eigenvalues of T are 1 + 2 g^2 cos(k pi / (N+1)) and
    sit in a ratio near 0.98 at N = 15, so power iteration cannot separate them
    in 50 iterations.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(-np.pi, np.pi, size=(num_pixels, num_acq))
    out = np.zeros((1, num_pixels, num_acq, num_acq), dtype=np.complex64)
    for p in range(num_pixels):
        C = np.eye(num_acq, dtype=np.complex64)
        for k in range(num_acq - 1):
            C[k, k + 1] = coherence[k] * np.exp(1j * (truth[p, k] - truth[p, k + 1]))
            C[k + 1, k] = np.conj(C[k, k + 1])
        out[0, p] = C
    return out, truth


def chain_agreement(evecs, truth):
    """Resultant length of the eigenvector's lag-1 phases against the chained truth."""
    est = np.angle(evecs[..., :-1] * np.conj(evecs[..., 1:]))
    expected = truth[..., :-1] - truth[..., 1:]
    return np.abs(np.mean(np.exp(1j * (est - expected)), axis=-1))


def test_chain_only_exact_reproduces_phases():
    stack, truth = chain_only_stack()
    _, evecs = eigh_largest_stack_exact(stack * np.abs(stack))
    assert np.all(chain_agreement(np.asarray(evecs), truth)[0] > 1 - 1e-5)


def test_chain_only_power_iteration_does_not_converge():
    """Documents why the exact solver exists; a solver fix would make this fail."""
    stack, truth = chain_only_stack()
    _, evecs = eigh_largest_stack(stack * np.abs(stack))
    assert np.any(chain_agreement(np.asarray(evecs), truth)[0] < 0.999)


def test_chain_with_weak_links_keeps_tiny_component_phases():
    """A weak link makes the principal eigenvector decay across it.

    Components beyond the link sit near single-precision resolution, and their
    phases are still the chained phases in exact arithmetic. The host solver
    must keep them; `jnp.linalg.eigh` in single precision does not.
    """
    stack, truth = weak_link_stack()
    _, evecs = eigh_largest_stack_exact(stack * np.abs(stack))
    evecs = np.asarray(evecs)
    assert np.abs(evecs).min() < 1e-6
    assert np.all(chain_agreement(evecs, truth)[0] > 1 - 1e-4)


def weak_link_stack():
    coherence = np.full(14, 0.6)
    coherence[[2, 3, 4, 9, 10, 11]] = 0.03
    return chain_only_stack(coherence=coherence, num_pixels=16)


def test_single_precision_jax_eigh_loses_tiny_component_phases():
    """Documents why the exact solver runs on the host; see `_eigh_host`."""
    import jax.numpy as jnp

    stack, truth = weak_link_stack()
    evecs = np.asarray(jnp.linalg.eigh(jnp.asarray(stack * np.abs(stack)))[1])
    assert np.any(chain_agreement(evecs[..., :, -1], truth)[0] < 0.9)


def test_exact_solver_nans_non_finite_pixels_only():
    """EMI relies on NaN where |Gamma| could not be inverted; NumPy must not raise."""
    stack = np.stack(
        [np.stack([simulate_coh(num_acq=10)[0] for _ in range(3)]) for _ in range(2)]
    ).astype(np.complex64)
    stack[1, 2] = np.nan
    evalues, evecs = eigh_smallest_stack_exact(stack)
    evalues, evecs = np.asarray(evalues), np.asarray(evecs)
    assert np.isnan(evalues[1, 2]) and np.isnan(evecs[1, 2]).all()
    finite = np.ones((2, 3), dtype=bool)
    finite[1, 2] = False
    assert np.isfinite(evecs[finite]).all() and np.isfinite(evalues[finite]).all()
