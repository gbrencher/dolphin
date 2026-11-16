from pathlib import Path

import numpy as np
import pytest

from dolphin import io
from dolphin.unwrap import _utils

# Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
pytestmark = pytest.mark.filterwarnings(
    "ignore::rasterio.errors.NotGeoreferencedWarning",
    "ignore:.*io.FileIO.*:pytest.PytestUnraisableExceptionWarning",
)


@pytest.fixture()
def corr_raster(raster_100_by_200):
    # Make a correlation raster of all 1s in the same directory as the raster
    d = Path(raster_100_by_200).parent
    arr = np.ones((100, 200), dtype="float32")
    # The first 20 rows have nodata
    arr[:20, :] = np.nan
    filename = d / "corr_raster.cor.tif"
    io.write_arr(
        arr=arr,
        output_name=filename,
        like_filename=raster_100_by_200,
        nodata=np.nan,
        driver="GTiff",
    )
    return filename


@pytest.fixture()
def mask_raster(raster_100_by_200):
    # Make a correlation raster of all 1s in the same directory as the raster
    d = Path(raster_100_by_200).parent
    filename = d / "mask.tif"
    arr = np.ones((100, 200), dtype=np.uint8)
    # Mask the first 20 columns
    arr[:, :20] = 0
    io.write_arr(
        arr=arr,
        output_name=filename,
        like_filename=raster_100_by_200,
        driver="GTiff",
    )
    return filename


def test_create_combined_mask(corr_raster, mask_raster):
    out_raster = _utils.create_combined_mask(
        mask_filename=mask_raster, image_filename=corr_raster
    )
    assert Path(out_raster).name == "combined_mask.tif"
    mask = io.load_gdal(out_raster)
    assert (mask[:20] == 0).all()
    assert (mask[:, :20] == 0).all()
    assert (mask[20:, 20:] == 1).all()


def test_set_nodata_values(corr_raster):
    import rasterio

    corr_copy = corr_raster.parent / "corr_copy.tif"
    # The first 20 cols have nodata
    arr = np.ones((100, 200), dtype="float32")
    arr[:, :20] = -1
    io.write_arr(
        arr=arr,
        output_name=corr_copy,
        like_filename=corr_raster,
        nodata=-1,
        driver="GTiff",
    )
    with rasterio.open(corr_copy) as src:
        assert src.nodata == -1
        a = src.read(1)
        assert np.all(a[:, :20] == -1)
        assert np.all(a[:, 20:] == 1)

    _utils.set_nodata_values(filename=corr_copy, like_filename=corr_raster)
    # Without specifying `output_nodata`, should keep existing nodata of `filename`
    with rasterio.open(corr_copy) as src:
        assert src.nodata == -1
        a = src.read(1)
        assert np.all(a[:20] == -1)
        assert np.all(a[:, :20] == -1)

    _utils.set_nodata_values(
        filename=corr_copy, output_nodata=-2, like_filename=corr_raster
    )
    with rasterio.open(corr_copy) as src:
        assert src.nodata == -2
        a = src.read(1)
        assert np.all(a[:20] == -2)
        assert np.all(a[:, :20] == -2)


def test_nodata_of_complex_raster_compares_both_parts(tmp_path):
    """A purely imaginary pixel is data, not nodata, in a complex raster."""
    import rasterio

    ifg = np.ones((10, 10), dtype="complex64")
    ifg[0, :] = 0  # genuinely empty
    ifg[5, 5] = 1j  # phase of exactly pi/2
    ifg[6, 6] = -1j
    ifg_file = tmp_path / "ifg.int.tif"
    io.write_arr(arr=ifg, output_name=ifg_file, nodata=0, driver="GTiff")
    # The dataset mask itself gets this wrong, which is what the helper avoids.
    with rasterio.open(ifg_file) as src:
        assert not src.read_masks(1)[5, 5]
        valid = _utils._valid_mask(src)
    assert valid[5, 5] and valid[6, 6]
    assert not valid[0].any()

    unw_file = tmp_path / "ifg.unw.tif"
    io.write_arr(
        arr=np.full((10, 10), 3.0, dtype="float32"),
        output_name=unw_file,
        like_filename=ifg_file,
        nodata=0,
        driver="GTiff",
    )
    _utils.set_nodata_values(filename=unw_file, output_nodata=0, like_filename=ifg_file)
    unw = io.load_gdal(unw_file)
    assert unw[5, 5] == 3.0 and unw[6, 6] == 3.0
    assert (unw[0] == 0).all()

    mask_file = tmp_path / "mask.tif"
    io.write_arr(
        arr=np.ones((10, 10), dtype="uint8"), output_name=mask_file, like_filename=ifg_file
    )
    combined = io.load_gdal(
        _utils.create_combined_mask(mask_filename=mask_file, image_filename=ifg_file)
    )
    assert combined[5, 5] == 1 and combined[6, 6] == 1
    assert (combined[0] == 0).all()
