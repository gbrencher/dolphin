import pytest

from dolphin import stack
from dolphin.io import _readers
from dolphin.phase_link import simulate
from dolphin.utils import gpu_is_available
from dolphin.workflows import single

GPU_AVAILABLE = gpu_is_available()
simulate._seed(1234)


@pytest.mark.parametrize("write_extra", [False, True])
def test_sequential_gtiff(tmp_path, slc_file_list, write_extra: bool):
    """Run through the sequential estimation with a GeoTIFF stack."""
    vrt_file = tmp_path / "slc_stack.vrt"
    files = slc_file_list[:3]
    vrt_stack = _readers.VRTStack(files, outfile=vrt_file)
    is_compressed = [False] * len(files)
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=is_compressed,
    )

    hy, hx = 1, 2
    half_window = {"x": hx, "y": hy}
    strides = {"x": 1, "y": 1}
    output_folder = tmp_path / "single"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window=half_window,
        strides=strides,
        shp_method="rect",
        write_crlb=write_extra,
        write_closure_phase=write_extra,
    )

    assert output_folder.exists()
    # Check that all the expected outputs are there
    assert len(list(output_folder.glob("2*.slc.tif"))) == 3
    assert len(list(output_folder.glob("compressed_*tif"))) == 1
    assert len(list(output_folder.glob("temporal_coherence*tif"))) == 1


def test_single_writes_gate_layers(tmp_path, slc_file_list):
    """The gate layers reach the workflow's rasters and keep their own NaNs."""
    import numpy as np

    from dolphin import io

    vrt_stack = _readers.VRTStack(slc_file_list[:5], outfile=tmp_path / "slc_stack.vrt")
    ministack = stack.MiniStackInfo(
        file_list=vrt_stack.file_list,
        dates=vrt_stack.dates,
        is_compressed=[False] * 5,
    )
    output_folder = tmp_path / "single"
    single.run_wrapped_phase_single(
        vrt_stack=vrt_stack,
        ministack=ministack,
        output_folder=output_folder,
        half_window={"x": 2, "y": 1},
        strides={"x": 1, "y": 1},
        shp_method="rect",
        write_crlb=False,
        use_weighted_evd=True,
        # A floor near one drops every long entry, so no long entry is admitted
        # anywhere and `inconsistent_admitted` is NaN by definition.
        coherence_floor_false_admission=1e-300,
        consistency_gate="lag",
        write_consistency=True,
    )
    layers = {}
    for key in (
        "max_consistent_lag",
        "admitted_fraction",
        "inconsistent_admitted",
        "admitted_bias",
        "admitted_temporal_coherence",
    ):
        files = list(output_folder.glob(f"{key}_*.tif"))
        assert len(files) == 1
        layers[key] = io.load_gdal(files[0], masked=True).filled(np.nan)
    temp_coh = io.load_gdal(
        next(output_folder.glob("temporal_coherence_*.tif")), masked=True
    ).filled(np.nan)
    valid = np.isfinite(temp_coh)
    assert valid.any()
    # Layers that are defined everywhere share temporal coherence's footprint:
    # NaN-valued neighbours in another layer must not erode them.
    for key in ("max_consistent_lag", "admitted_fraction", "admitted_temporal_coherence"):
        np.testing.assert_array_equal(np.isfinite(layers[key]), valid)
    assert np.isnan(layers["inconsistent_admitted"]).all()
