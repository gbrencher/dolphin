from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from dolphin.unwrap._post_looks import (
    labels_of_cells,
    look_after_unwrapping,
    looks_for_posting,
    mean_of_valid,
)

# 5 m columns and 10 m rows, like a geocoded Sentinel-1 burst.
TRANSFORM = Affine(5.0, 0.0, 500000.0, 0.0, -10.0, 3100000.0)


def test_looks_for_posting():
    assert looks_for_posting(20.0, TRANSFORM) == (2, 4)
    assert looks_for_posting(10.0, TRANSFORM) == (1, 2)
    with pytest.raises(ValueError, match="whole multiple"):
        looks_for_posting(15.0, TRANSFORM)
    with pytest.raises(ValueError, match="whole multiple"):
        looks_for_posting(5.0, TRANSFORM)


def test_mean_of_valid_ignores_nodata_and_nan():
    array = np.arange(16, dtype=np.float32).reshape(2, 8) + 1
    array[0, 0] = 0.0  # nodata
    array[1, 1] = np.nan
    mean, any_valid = mean_of_valid(array, 0.0, 2, 4)
    first = [2, 3, 4, 9, 11, 12]
    np.testing.assert_allclose(mean[0, 0], np.mean(first))
    np.testing.assert_allclose(mean[0, 1], array[:, 4:].mean())
    assert any_valid.all()
    array[:, :4] = 0.0
    mean, any_valid = mean_of_valid(array, 0.0, 2, 4)
    assert not any_valid[0, 0] and mean[0, 0] == 0 and any_valid[0, 1]


def test_edge_rows_and_columns_that_do_not_fill_a_cell_are_dropped():
    mean, _ = mean_of_valid(np.ones((5, 9), np.float32), None, 2, 4)
    assert mean.shape == (2, 2)


def test_a_cell_is_labelled_only_if_every_pixel_is():
    labels = np.ones((2, 12), dtype=np.uint16)
    labels[0, 5] = 0  # one unlabelled pixel in the second cell
    labels[:, 8:] = 65535  # third cell is all nodata
    labels[1, 2] = 3  # two labels in the first cell
    out = labels_of_cells(labels, 65535, 2, 4)
    assert out.tolist() == [[3, 0, 65535]] and out.dtype == np.uint16


def _write(path: Path, array, nodata):
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": array.dtype,
        "crs": "EPSG:32645",
        "transform": TRANSFORM,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def test_files_are_written_on_the_coarse_grid(tmp_path):
    rng = np.random.default_rng(0)
    unw = rng.normal(size=(8, 16)).astype(np.float32)
    unw[:2, :4] = 0.0
    cor = rng.uniform(0.1, 1, size=(8, 16)).astype(np.float32)
    labels = np.ones((8, 16), dtype=np.uint16)
    for name, array, nodata in (
        ("20200101_20200113.unw.tif", unw, 0.0),
        ("20200101_20200113.unw.conncomp.tif", labels, 65535),
        ("20200101_20200113.int.cor.tif", cor, 0.0),
        ("temporal_coherence.tif", cor, 0.0),
    ):
        _write(tmp_path / name, array, nodata)
    looked = look_after_unwrapping(
        [tmp_path / "20200101_20200113.unw.tif"],
        [tmp_path / "20200101_20200113.unw.conncomp.tif"],
        [tmp_path / "20200101_20200113.int.cor.tif"],
        posting=20.0,
        output_dir=tmp_path / "looked",
        quality_files=[tmp_path / "temporal_coherence.tif"],
    )
    assert looked.looks == (2, 4)
    assert looked.unwrapped_paths[0].name == "20200101_20200113.unw.tif"
    with rasterio.open(looked.unwrapped_paths[0]) as src:
        out = src.read(1)
        assert src.shape == (4, 4) and src.nodata == 0.0
        assert abs(src.transform.a) == 20.0 and abs(src.transform.e) == 20.0
        assert src.bounds == rasterio.open(tmp_path / "temporal_coherence.tif").bounds
    assert out[0, 0] == 0.0  # the cell with no valid pixel keeps the nodata value
    np.testing.assert_allclose(out[1, 1], unw[2:4, 4:8].mean(), rtol=1e-6)
    with rasterio.open(looked.conncomp_paths[0]) as src:
        assert src.read(1).tolist() == [[1] * 4] * 4 and src.dtypes[0] == "uint16"
    with rasterio.open(looked.quality_files[0]) as src:
        np.testing.assert_allclose(src.read(1)[3, 3], cor[6:, 12:].mean(), rtol=1e-6)


def test_the_option_is_off_by_default():
    from dolphin.workflows.config import PostUnwrapOptions

    assert PostUnwrapOptions().posting is None
    with pytest.raises(ValueError):
        PostUnwrapOptions(posting=0)
