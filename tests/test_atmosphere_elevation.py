import numpy as np
import numpy.testing as npt
import pytest

from dolphin import io, timeseries
from dolphin.atmosphere.elevation import (
    ElevationCorrectionSettings,
    correct_elevation_dependence,
    correct_unwrapped_interferograms,
    fit_binned_polynomial,
)

SMALL_TILES = ElevationCorrectionSettings(tile_size=1800, step=600)


def scene(shape=(120, 160), seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    height = (1000 + 8 * cols + 3 * rows + rng.normal(0, 20, shape)).astype("float32")
    return height, np.full(shape, 0.99, "float32"), rng


def stratified(height):
    h = (height - 1500) / 1000
    return 0.5 + 2.0 * h - 0.6 * h**2  # radians


def test_fit_recovers_a_quadratic():
    rng = np.random.default_rng(0)
    elevation = rng.uniform(1000, 2500, 5000)
    values = stratified(elevation) + rng.normal(0, 0.05, elevation.size)
    coefficients, info = fit_binned_polynomial(elevation, values)
    assert info["status"] == "success" and info["order_used"] == 2
    npt.assert_allclose(
        np.polyval(coefficients, [1200, 1800, 2300]),
        stratified(np.array([1200, 1800, 2300])),
        atol=0.02,
    )


def test_fit_ignores_a_sparse_tight_bin():
    """Three tightly clustered samples must not outweigh thousands."""
    rng = np.random.default_rng(1)
    elevation = rng.uniform(1000, 2000, 3000)
    values = 0.002 * (elevation - 1500) + rng.normal(0, 0.05, elevation.size)
    elevation = np.r_[elevation, [2530.0, 2540.0, 2550.0]]
    values = np.r_[values, [9.0, 9.001, 9.002]]
    coefficients, _ = fit_binned_polynomial(elevation, values)
    assert abs(np.polyval(coefficients, 1900) - 0.8) < 0.05


def test_fit_fails_without_usable_bins():
    coefficients, info = fit_binned_polynomial(
        np.linspace(1000, 2000, 8), np.zeros(8), min_samples_per_bin=10
    )
    assert coefficients is None and info["reason"] == "no_valid_bins"
    coefficients, info = fit_binned_polynomial(np.full(50, 1500.0), np.zeros(50))
    assert coefficients is None and info["reason"] == "insufficient_elevation_range"


def test_removes_the_elevation_signal_and_the_offset():
    height, coherence, rng = scene()
    values = stratified(height) + rng.normal(0, 0.05, height.shape)
    corrected, records = correct_elevation_dependence(
        values, height, coherence, (30, 30), SMALL_TILES
    )
    assert all(r.status == "success" for r in records)
    # Tent weights are positive over the whole of a tile, image corners included.
    assert np.isfinite(corrected).all()
    assert abs(corrected.mean()) < 0.02 and corrected.std() < 0.07
    assert abs(np.corrcoef(height.ravel(), corrected.ravel())[0, 1]) < 0.1


def test_cone_blend_leaves_image_corners_empty():
    # The earlier scheme: zero weight at a tile corner, and an image corner is
    # only ever a tile corner.
    height, coherence, rng = scene()
    values = stratified(height) + rng.normal(0, 0.05, height.shape)
    settings = ElevationCorrectionSettings(**{**SMALL_TILES.__dict__, "blend": "cone"})
    corrected, _ = correct_elevation_dependence(values, height, coherence, (30, 30), settings)
    corners = np.zeros(height.shape, bool)
    corners[[0, 0, -1, -1], [0, -1, 0, -1]] = True
    assert np.isnan(corrected[corners]).all()
    assert np.isfinite(corrected[~corners]).all()


def test_incoherent_signal_survives_and_nodata_stays():
    height, coherence, rng = scene()
    target = np.zeros(height.shape, bool)
    target[50:60, 70:85] = True
    coherence[target] = 0.3  # not a reference
    values = stratified(height) + rng.normal(0, 0.05, height.shape)
    values[target] -= 3.0
    values[:5, :5] = np.nan
    corrected, _ = correct_elevation_dependence(
        values, height, coherence, (30, 30), SMALL_TILES
    )
    assert abs(np.median(corrected[target]) + 3.0) < 0.1
    assert np.isnan(corrected[:5, :5]).all()


def test_correction_is_held_constant_outside_the_reference_elevations():
    """High ground with no references gets the value at the top reference."""
    height, coherence, rng = scene()
    high = height > 2100
    coherence[high] = 0.2
    truth = stratified(height)
    values = truth + rng.normal(0, 0.02, height.shape)
    corrected, records = correct_elevation_dependence(
        values, height, coherence, (30, 30), SMALL_TILES
    )
    removed = values - corrected
    top = max(r.elevation_span[1] for r in records)
    # Bounded: never beyond what the relation reaches at the references' top.
    assert np.nanmax(np.abs(removed[high])) < np.abs(stratified(np.array(top))) + 0.3
    # An unclamped quadratic keeps bending; the clamped one stays flat with height.
    far = height > 2250
    assert np.nanstd(removed[far]) < 0.15


def test_no_references_no_correction():
    height, coherence, rng = scene()
    corrected, records = correct_elevation_dependence(
        rng.normal(0, 1, height.shape), height, coherence * 0.5, (30, 30), SMALL_TILES
    )
    assert np.isnan(corrected).all()
    assert all(r.status == "failed" for r in records)


def test_tiles_are_sized_in_metres_on_a_non_square_grid():
    height, coherence, rng = scene((60, 240))
    values = stratified(height) + rng.normal(0, 0.05, height.shape)
    _, records = correct_elevation_dependence(
        values, height, coherence, (10, 5), ElevationCorrectionSettings(400, 200)
    )
    # Tiles at the scene's edges are clipped; a whole one is 400 m both ways.
    assert max(r.row_stop - r.row_start for r in records) == 40
    assert max(r.col_stop - r.col_start for r in records) == 80


def _largest_step(field):
    return max(np.nanmax(np.abs(np.diff(field, axis=0))), np.nanmax(np.abs(np.diff(field, axis=1))))


def test_tent_blend_is_continuous_where_tiles_disagree():
    # A field that jumps by 10 across a line no polynomial in elevation can
    # follow: tiles on either side fit different constants. With tent weights
    # the correction crosses the jump over a tile width; with cone weights it
    # steps wherever a tile begins or ends.
    rows, cols = 240, 240
    elevation = np.tile(np.linspace(1000, 2000, rows, dtype="float32")[:, None], (1, cols))
    values = np.where(np.arange(cols)[None, :] < cols // 2, 0.0, 10.0).astype("float32")
    values = np.broadcast_to(values, (rows, cols)).copy()
    # A bin with no spread is not fitted, so the two levels carry a little noise.
    noise = np.random.default_rng(0).normal(0, 0.01, (rows, cols)).astype("float32")
    values += noise
    coherence = np.ones((rows, cols), dtype="float32")
    steps = {}
    for blend in ("tent", "cone"):
        settings = ElevationCorrectionSettings(tile_size=800.0, step=200.0, blend=blend)
        corrected, _ = correct_elevation_dependence(values, elevation, coherence, (20.0, 20.0), settings)
        assert blend == "cone" or np.isfinite(corrected).all()
        correction = values - corrected
        # Away from the jump itself, where the input is constant.
        steps[blend] = _largest_step(np.delete(correction, cols // 2, axis=1)[:, : cols - 2])
    # Crossing 10 over a tile of 40 pixels, the tent's steepest slope is under 1
    # per pixel; the cone adds its edge steps to that.
    assert steps["tent"] < 1.0
    assert steps["cone"] > 1.5 * steps["tent"]


def test_tent_blend_covers_the_scene_edges_evenly():
    rows, cols = 100, 100
    elevation = np.tile(np.linspace(1000, 2000, rows, dtype="float32")[:, None], (1, cols))
    values = (0.002 * elevation + 3).astype("float32")
    coherence = np.ones((rows, cols), dtype="float32")
    settings = ElevationCorrectionSettings(tile_size=800.0, step=200.0)
    corrected, records = correct_elevation_dependence(values, elevation, coherence, (20.0, 20.0), settings)
    assert np.isfinite(corrected).all()
    assert np.abs(corrected).max() < 0.05
    # Tiles start before the scene: more of them than a lattice from row 0.
    assert len(records) > (rows // 10) ** 2


def test_unknown_blend_rejected():
    field = np.ones((10, 10), dtype="float32")
    with pytest.raises(ValueError, match="blend"):
        correct_elevation_dependence(field, field, field, (20.0, 20.0), ElevationCorrectionSettings(blend="box"))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="Shapes differ"):
        correct_elevation_dependence(
            np.zeros((4, 4)), np.zeros((4, 5)), np.zeros((4, 4)), (30, 30)
        )


def test_files_and_inversion_without_a_reference_pixel(tmp_path, raster_100_by_200):
    """Corrected pairs invert to the signal; subtracting a pixel would shift it."""
    rng = np.random.default_rng(3)
    shape = (100, 200)
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    height = (1000 + 6 * cols + 2 * rows).astype("float32")
    dem_file = tmp_path / "dem.tif"
    io.write_arr(arr=height, output_name=dem_file, like_filename=raster_100_by_200)
    # Three dates, three pairs. A compact, incoherent signal on stable ground.
    target = np.zeros(shape, bool)
    target[40:50, 90:110] = True
    signal = {"20220101": 0.0, "20220113": -1.0, "20220125": -2.0}
    atmosphere = {d: rng.normal(0, 2) * (height - 1500) / 1000 for d in signal}
    unw_files, cor_files = [], []
    pairs = [("20220101", "20220113"), ("20220113", "20220125"), ("20220101", "20220125")]
    for a, b in pairs:
        phase = (signal[b] - signal[a]) * target + atmosphere[b] - atmosphere[a]
        phase = phase + rng.normal(0, 0.02, shape)
        coherence = np.where(target, 0.3, 0.95).astype("float32")
        unw = tmp_path / f"{a}_{b}.unw.tif"
        cor = tmp_path / f"{a}_{b}.int.cor.tif"
        io.write_arr(arr=phase.astype("float32"), output_name=unw,
                     like_filename=raster_100_by_200, nodata=0)
        io.write_arr(arr=coherence, output_name=cor, like_filename=raster_100_by_200)
        unw_files.append(unw)
        cor_files.append(cor)

    corrected = correct_unwrapped_interferograms(
        unw_files, cor_files, dem_file, tmp_path / "corrected", SMALL_TILES
    )
    assert [p.name for p in corrected] == [p.name for p in unw_files]
    assert (tmp_path / "corrected" / "elevation_correction.json").exists()
    first = io.load_gdal(corrected[0], masked=True)
    assert abs(np.ma.median(first[~target])) < 0.02
    assert abs(np.ma.median(first[target]) + 1.0) < 0.05

    # `invert_unw_network` reads its inputs through a date-sorted stack, so the
    # list must be sorted as `timeseries.run` does.
    corrected = sorted(corrected, key=str)
    # The reference pixel sits on the target, the worst place for one.
    reference = timeseries.ReferencePoint(45, 100)
    (tmp_path / "free").mkdir()
    (tmp_path / "pixel").mkdir()
    out_paths, _ = timeseries.invert_unw_network(
        unw_file_list=corrected, reference=reference, output_dir=tmp_path / "free",
        method="L1", num_threads=1, subtract_reference=False,
    )
    last = io.load_gdal(out_paths[-1], masked=True)
    # Phase units, displacement sign: -1 x phase.
    assert abs(np.ma.median(last[target]) - 2.0) < 0.1
    assert abs(np.ma.median(last[~target])) < 0.05
    out_paths, _ = timeseries.invert_unw_network(
        unw_file_list=corrected, reference=reference, output_dir=tmp_path / "pixel",
        method="L1", num_threads=1,
    )
    shifted = io.load_gdal(out_paths[-1], masked=True)
    assert abs(np.ma.median(shifted[~target]) + 2.0) < 0.2
