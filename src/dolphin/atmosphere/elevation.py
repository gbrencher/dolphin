"""Tiled elevation-dependent correction of unwrapped interferograms.

In mountainous terrain the tropospheric delay between two acquisitions follows
elevation, with a relation that changes over a few kilometres. This estimates
that relation locally and removes it, one unwrapped interferogram at a time,
before the network is inverted:

1. The scene is cut into overlapping square tiles (`tile_size` metres, every
   `step` metres).
2. In each tile the reference pixels are those with a value, an elevation and
   an interferometric correlation above `coherence_threshold`, between the
   0.1 and 99.9 percentiles of such pixels' elevations. A tile with fewer
   than `min_refs` grows by one step on every side, up to `max_expansions`
   times.
3. The references are binned by elevation (`bin_width` metres). Bins of at
   least `min_samples_per_bin` pixels contribute their median, weighted by
   sqrt(n) / NMAD, the inverse of the standard error of a median up to a
   constant. A polynomial of order `order` is fitted to them, of lower order
   when there are too few bins.
4. The polynomial, evaluated at each pixel's elevation, is subtracted from the
   tile. Outside the elevations of the tile's references the correction is
   held at its value at the nearest referenced elevation: an extrapolated
   quadratic reaches tens of centimetres in high relief.
5. Overlapping tiles are blended with weights that fall linearly from each
   tile's centre to zero at its corners.

Every tile's polynomial has a constant term, so the corrected interferogram is
referenced locally: its zero is the median of coherent ground near each
elevation within a tile. It must not be referenced again to a single pixel
before inversion (`timeseries_options.subtract_reference`), since that would
add one pixel's leftover noise to the whole interferogram. Structure finer
than a tile is left in, and nothing here distinguishes delay from ground
motion that follows elevation.

The method is G. Brencher's (rg_uavsar project). Three points differ from that
implementation and were adopted on measurement: every reference is used where
it sampled 500, bins are weighted by their sample count, and the correction is
held constant outside the reference elevations.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from dolphin._types import Filename

logger = logging.getLogger("dolphin")

__all__ = [
    "ElevationCorrectionSettings",
    "correct_elevation_dependence",
    "correct_unwrapped_interferograms",
    "fit_binned_polynomial",
]


@dataclass(frozen=True)
class ElevationCorrectionSettings:
    """Parameters of the tiled elevation-dependent correction."""

    tile_size: float = 4000.0
    """Tile edge length, metres."""
    step: float = 1000.0
    """Spacing between tile origins, metres."""
    coherence_threshold: float = 0.8
    """Reference pixels have interferometric correlation above this."""
    min_refs: int = 25
    """A tile with fewer reference pixels grows before it is fitted."""
    max_expansions: int = 8
    """How many times a tile may grow by one step on every side."""
    order: int = 2
    """Polynomial order of the phase-elevation relation."""
    bin_width: float = 100.0
    """Elevation bin width, metres."""
    min_samples_per_bin: int = 10
    """Bins with fewer reference pixels are not used in the fit."""


@dataclass
class TileRecord:
    """What happened in one tile."""

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int
    expansions: int
    n_refs: int
    status: str
    reason: Optional[str] = None
    order_used: Optional[int] = None
    n_bins: int = 0
    elevation_span: Optional[tuple[float, float]] = None
    coefficients: Optional[list[float]] = field(default=None)


def fit_binned_polynomial(
    elevation: ArrayLike,
    values: ArrayLike,
    order: int = 2,
    bin_width: float = 100.0,
    min_samples_per_bin: int = 10,
) -> tuple[Optional[np.ndarray], dict]:
    """Fit a polynomial of `values` against `elevation` through binned medians.

    Parameters
    ----------
    elevation, values : ArrayLike
        1D arrays over the reference pixels.
    order : int
        Requested polynomial order; reduced to one less than the number of
        usable bins when there are too few.
    bin_width : float
        Width of the elevation bins, in the units of `elevation`.
    min_samples_per_bin : int
        Bins with fewer samples, or with zero spread, are not used.

    Returns
    -------
    coefficients : np.ndarray or None
        Highest power first, as `numpy.polyfit` returns; None if no fit.
    info : dict
        `status` ("success" or "failed"), `reason`, `n_bins`, `order_used`.

    """
    elevation = np.asarray(elevation, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(elevation) & np.isfinite(values)
    elevation, values = elevation[keep], values[keep]
    info = {"status": "failed", "reason": None, "n_bins": 0, "order_used": None}
    if elevation.size < 3:
        info["reason"] = "too_few_reference_points"
        return None, info
    lowest, highest = elevation.min(), elevation.max()
    if highest <= lowest:
        info["reason"] = "insufficient_elevation_range"
        return None, info
    edges = np.arange(lowest, highest + bin_width, bin_width)
    if len(edges) < 2:
        info["reason"] = "too_few_bin_edges"
        return None, info

    bins = np.digitize(elevation, edges)
    by_bin = np.argsort(bins, kind="stable")
    sorted_bins = bins[by_bin]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_bins)) + 1]
    stops = np.r_[starts[1:], sorted_bins.size]
    centres, medians, weights = [], [], []
    for a, b in zip(starts, stops, strict=True):
        i = sorted_bins[a]
        if b - a < min_samples_per_bin or i < 1 or i >= len(edges):
            continue
        data = values[by_bin[a:b]]
        spread = stats.median_abs_deviation(data, scale="normal")
        if not np.isfinite(spread) or spread <= 0:
            continue
        centres.append(edges[i - 1] + bin_width / 2)
        medians.append(np.median(data))
        # Standard error of a median is about 1.25 NMAD / sqrt(n).
        weights.append(np.sqrt(b - a) / spread)

    info["n_bins"] = len(centres)
    if not centres:
        info["reason"] = "no_valid_bins"
        return None, info
    order_used = min(order, len(centres) - 1)
    info["order_used"] = order_used
    info["status"] = "success"
    if order_used == 0:
        info["reason"] = "one_valid_bin_using_bin_median"
        return np.array([medians[0]]), info
    return np.polyfit(centres, medians, order_used, w=weights), info


def _blend_weights(n_rows: int, n_cols: int, row_spacing: float, col_spacing: float):
    """Weights falling linearly with ground distance from the centre to the corners."""
    rows, cols = np.ogrid[:n_rows, :n_cols]
    centre_row, centre_col = (n_rows - 1) / 2, (n_cols - 1) / 2
    distance = np.sqrt(
        ((rows - centre_row) * row_spacing) ** 2
        + ((cols - centre_col) * col_spacing) ** 2
    )
    farthest = np.sqrt(
        (centre_row * row_spacing) ** 2 + (centre_col * col_spacing) ** 2
    )
    if farthest == 0:
        return np.ones((n_rows, n_cols), dtype=np.float32)
    return (1 - distance / farthest).astype(np.float32)


def correct_elevation_dependence(
    values: ArrayLike,
    elevation: ArrayLike,
    coherence: ArrayLike,
    pixel_spacing: tuple[float, float],
    settings: ElevationCorrectionSettings = ElevationCorrectionSettings(),
    exclude: Optional[ArrayLike] = None,
) -> tuple[np.ndarray, list[TileRecord]]:
    """Remove the locally fitted elevation dependence from one 2D field.

    Parameters
    ----------
    values : ArrayLike
        The field to correct, e.g. unwrapped phase in radians. NaN where there
        is no data.
    elevation : ArrayLike
        Elevation on the same grid, metres. NaN where unknown.
    coherence : ArrayLike
        Quality on the same grid; reference pixels exceed
        `settings.coherence_threshold`.
    pixel_spacing : tuple[float, float]
        Ground size of a pixel along rows and along columns, metres. Tiles are
        sized in metres, so a grid with unequal spacing gets tiles that are
        square on the ground.
    settings : ElevationCorrectionSettings
        Parameters of the correction.
    exclude : ArrayLike, optional
        Boolean; True where a pixel must not be a reference.

    Returns
    -------
    corrected : np.ndarray
        float32, NaN where `values` is NaN or no tile could be fitted.
    records : list[TileRecord]
        One per tile that held data.

    """
    values = np.asarray(values, dtype=np.float32)
    elevation = np.asarray(elevation, dtype=np.float32)
    coherence = np.asarray(coherence, dtype=np.float32)
    if not (values.shape == elevation.shape == coherence.shape):
        msg = f"Shapes differ: {values.shape}, {elevation.shape}, {coherence.shape}"
        raise ValueError(msg)
    row_spacing, col_spacing = (abs(float(s)) for s in pixel_spacing)
    tile_rows = max(1, int(round(settings.tile_size / row_spacing)))
    tile_cols = max(1, int(round(settings.tile_size / col_spacing)))
    step_rows = max(1, int(round(settings.step / row_spacing)))
    step_cols = max(1, int(round(settings.step / col_spacing)))

    n_rows, n_cols = values.shape
    has_value = np.isfinite(values) & np.isfinite(elevation)
    candidate = (
        has_value & np.isfinite(coherence) & (coherence > settings.coherence_threshold)
    )
    if exclude is not None:
        candidate &= ~np.asarray(exclude, dtype=bool)

    corrected_sum = np.zeros((n_rows, n_cols), dtype=np.float32)
    weight_sum = np.zeros((n_rows, n_cols), dtype=np.float32)
    records: list[TileRecord] = []

    def references(r0, r1, c0, c1):
        tile = candidate[r0:r1, c0:c1]
        if not tile.any():
            return None
        heights = elevation[r0:r1, c0:c1][tile]
        lowest, highest = np.nanquantile(heights, [0.001, 0.999])
        if not np.isfinite(lowest) or not np.isfinite(highest) or highest <= lowest:
            return None
        inside = (heights >= lowest) & (heights <= highest)
        return heights[inside], values[r0:r1, c0:c1][tile][inside]

    for col0 in range(0, n_cols, step_cols):
        for row0 in range(0, n_rows, step_rows):
            r0, r1 = row0, min(row0 + tile_rows, n_rows)
            c0, c1 = col0, min(col0 + tile_cols, n_cols)
            if not has_value[r0:r1, c0:c1].any():
                continue
            refs = references(r0, r1, c0, c1)
            expansions = 0
            while (
                refs is None or refs[0].size < settings.min_refs
            ) and expansions < settings.max_expansions:
                expansions += 1
                grown = (
                    max(r0 - step_rows, 0),
                    min(r1 + step_rows, n_rows),
                    max(c0 - step_cols, 0),
                    min(c1 + step_cols, n_cols),
                )
                if grown == (r0, r1, c0, c1):
                    break
                r0, r1, c0, c1 = grown
                refs = references(r0, r1, c0, c1)

            n_refs = 0 if refs is None else int(refs[0].size)
            record = TileRecord(r0, r1, c0, c1, expansions, n_refs, "failed")
            records.append(record)
            if n_refs < settings.min_refs:
                record.reason = "too_few_reference_points"
                continue
            coefficients, info = fit_binned_polynomial(
                refs[0],
                refs[1],
                settings.order,
                settings.bin_width,
                settings.min_samples_per_bin,
            )
            record.status, record.reason = info["status"], info["reason"]
            record.order_used, record.n_bins = info["order_used"], info["n_bins"]
            if coefficients is None:
                continue
            lowest, highest = float(refs[0].min()), float(refs[0].max())
            record.elevation_span = (lowest, highest)
            record.coefficients = [float(c) for c in coefficients]

            tile_elevation = np.clip(
                elevation[r0:r1, c0:c1].astype(np.float64), lowest, highest
            )
            tile = values[r0:r1, c0:c1] - np.polyval(coefficients, tile_elevation)
            weights = _blend_weights(r1 - r0, c1 - c0, row_spacing, col_spacing)
            finite = np.isfinite(tile)
            weights[~finite] = 0
            corrected_sum[r0:r1, c0:c1] += (
                np.where(finite, tile, 0.0).astype(np.float32) * weights
            )
            weight_sum[r0:r1, c0:c1] += weights

    with np.errstate(invalid="ignore", divide="ignore"):
        corrected = np.where(weight_sum > 0, corrected_sum / weight_sum, np.nan)
    corrected[~np.isfinite(values)] = np.nan
    return corrected.astype(np.float32), records


def _elevation_on_grid(dem_file: Filename, like_file: Filename) -> np.ndarray:
    """Read `dem_file`, bilinearly resampled onto the grid of `like_file`."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(like_file) as like:
        out = np.full((like.height, like.width), np.nan, dtype=np.float32)
        with rasterio.open(dem_file) as dem:
            same_grid = (
                dem.crs == like.crs
                and dem.transform.almost_equals(like.transform)
                and (dem.height, dem.width) == (like.height, like.width)
            )
            if same_grid:
                return dem.read(1, masked=True).astype(np.float32).filled(np.nan)
            reproject(
                rasterio.band(dem, 1),
                out,
                src_transform=dem.transform,
                src_crs=dem.crs,
                dst_transform=like.transform,
                dst_crs=like.crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
    return out


def _correct_one(job) -> dict:
    import rasterio

    unw_file, cor_file, out_file, elevation, settings, unw_nodata = job
    with rasterio.open(unw_file) as src:
        phase = src.read(1).astype(np.float32)
        profile = src.profile
        nodata = src.nodata
        transform = src.transform
    missing = ~np.isfinite(phase)
    if nodata is not None and not np.isnan(nodata):
        missing |= phase == nodata
    phase[missing] = np.nan
    with rasterio.open(cor_file) as src:
        coherence = src.read(1).astype(np.float32)
    corrected, records = correct_elevation_dependence(
        phase, elevation, coherence, (transform.e, transform.a), settings
    )
    uncorrected = int((~missing & ~np.isfinite(corrected)).sum())
    profile.update(dtype="float32", nodata=unw_nodata, count=1)
    with rasterio.open(out_file, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(corrected), corrected, unw_nodata), 1)
        dst.update_tags(units="radians")
    statuses = [r.status for r in records]
    return {
        "file": str(out_file),
        "tiles": len(records),
        "tiles_failed": statuses.count("failed"),
        "tiles_expanded": sum(r.expansions > 0 for r in records),
        "pixels_without_correction": uncorrected,
    }


def correct_unwrapped_interferograms(
    unw_file_list: Sequence[Filename],
    cor_file_list: Sequence[Filename],
    dem_file: Filename,
    output_dir: Filename,
    settings: ElevationCorrectionSettings = ElevationCorrectionSettings(),
    num_workers: int = 1,
    unw_nodata: float = 0.0,
) -> list[Path]:
    """Correct each unwrapped interferogram for its elevation-dependent signal.

    Parameters
    ----------
    unw_file_list : Sequence[Filename]
        Unwrapped interferograms, radians.
    cor_file_list : Sequence[Filename]
        Interferometric correlation, one per interferogram, same order.
    dem_file : Filename
        Elevation raster in any projection covering the interferograms. It is
        resampled bilinearly to their grid.
    output_dir : Filename
        Directory for the corrected rasters, which keep the input file names.
    settings : ElevationCorrectionSettings
        Parameters of the correction.
    num_workers : int
        Interferograms corrected in parallel.
    unw_nodata : float
        Value written where there is no corrected phase.

    Returns
    -------
    list[Path]
        The corrected interferograms, in the order of `unw_file_list`.

    """
    import json
    import multiprocessing

    if len(unw_file_list) != len(cor_file_list):
        msg = f"{len(unw_file_list)} interferograms, {len(cor_file_list)} correlations"
        raise ValueError(msg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_files = [output_dir / Path(f).name for f in unw_file_list]
    todo = [
        (u, c, o)
        for u, c, o in zip(unw_file_list, cor_file_list, out_files, strict=True)
        if not o.exists()
    ]
    if not todo:
        logger.info("All elevation-corrected interferograms exist, skipping")
        return out_files
    logger.info(
        "Correcting %s unwrapped interferograms for elevation-dependent signal",
        len(todo),
    )
    elevation = _elevation_on_grid(dem_file, unw_file_list[0])
    jobs = [(u, c, o, elevation, settings, unw_nodata) for u, c, o in todo]
    if num_workers > 1:
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            summaries = list(pool.map(_correct_one, jobs))
    else:
        summaries = [_correct_one(job) for job in jobs]
    report = {"settings": asdict(settings), "interferograms": summaries}
    (output_dir / "elevation_correction.json").write_text(json.dumps(report, indent=2))
    failed = sum(s["tiles_failed"] for s in summaries)
    missing = sum(s["pixels_without_correction"] for s in summaries)
    logger.info(
        "Elevation correction: %s tile fits failed, %s pixels left without a value",
        failed,
        missing,
    )
    return out_files
