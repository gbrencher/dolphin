"""Average unwrapped interferograms onto a coarser product grid.

Phase linking and unwrapping can run on a finer grid than the products need:
finer posting gives the unwrapper more samples per fringe, while nothing after
unwrapping needs it. This takes the unwrapped interferograms and the rasters
that travel with them to a coarser posting before any correction or inversion.

A cell of the coarse grid covers a whole number of rows and columns of the
fine one. Its unwrapped phase is the unweighted mean of the valid pixels it
covers. Pixels of one cell can sit on different cycles where the unwrapper
left an error between them; their mean is then a fraction of a cycle, so
inversion residuals on the coarse grid are less often whole cycles than on the
fine one. Rows and columns that do not fill a cell at the far edges are dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
from rasterio.transform import Affine

from dolphin._types import Filename

logger = logging.getLogger("dolphin")

__all__ = ["LookedPaths", "look_after_unwrapping", "look_files", "looks_for_posting"]


@dataclass
class LookedPaths:
    """Rasters on the coarse grid, in the order they were given."""

    unwrapped_paths: list[Path]
    conncomp_paths: list[Path]
    cor_paths: list[Path]
    quality_files: list[Path]
    looks: tuple[int, int]


def looks_for_posting(posting: float, transform: Affine) -> tuple[int, int]:
    """Rows and columns of the fine grid in one cell of `posting` metres.

    Raises if `posting` is not a whole multiple of both pixel sizes.
    """
    rows, cols = posting / abs(transform.e), posting / abs(transform.a)
    if (
        abs(rows - round(rows)) > 1e-6
        or abs(cols - round(cols)) > 1e-6
        or round(rows) < 1
        or round(cols) < 1
    ):
        msg = (
            f"posting {posting} is not a whole multiple of the"
            f" {abs(transform.a)} x {abs(transform.e)} grid"
        )
        raise ValueError(msg)
    return int(round(rows)), int(round(cols))


def _blocks(array: np.ndarray, rows: int, cols: int) -> np.ndarray:
    ny, nx = array.shape[0] // rows, array.shape[1] // cols
    return array[: ny * rows, : nx * cols].reshape(ny, rows, nx, cols)


def mean_of_valid(array: np.ndarray, nodata: float | None, rows: int, cols: int):
    """Mean over each cell of the pixels that are finite and not `nodata`.

    Returns (mean, any_valid); the mean is 0 where no pixel of a cell is valid.
    """
    values = array.astype(np.complex128 if np.iscomplexobj(array) else np.float64)
    valid = np.isfinite(values)
    if nodata is not None and np.isfinite(nodata):
        valid &= values != nodata
    count = _blocks(valid, rows, cols).sum(axis=(1, 3))
    total = _blocks(np.where(valid, values, 0.0), rows, cols).sum(axis=(1, 3))
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0), count > 0


def labels_of_cells(array: np.ndarray, nodata: float | None, rows: int, cols: int):
    """Connected-component label of each cell.

    A cell keeps a label only if every pixel it covers is labelled (nonzero and
    not `nodata`); it then takes the largest of them. Otherwise it is 0, or
    `nodata` where every pixel is `nodata`.
    """
    blocks = _blocks(array, rows, cols)
    missing = np.zeros(array.shape, dtype=bool) if nodata is None else array == nodata
    missing_blocks = _blocks(missing, rows, cols)
    labelled = (blocks > 0) & ~missing_blocks
    out = np.where(labelled.all(axis=(1, 3)), blocks.max(axis=(1, 3)), 0)
    if nodata is not None:
        out = np.where(missing_blocks.all(axis=(1, 3)), nodata, out)
    return out.astype(array.dtype)


def any_in_cells(array: np.ndarray, nodata: float | None, rows: int, cols: int):
    """Whether any pixel of each cell is set, for a mask of zeros and ones.

    A cell is 1 if any pixel it covers is nonzero and not `nodata`, `nodata`
    if every pixel is `nodata`, and 0 otherwise.
    """
    missing = np.zeros(array.shape, dtype=bool) if nodata is None else array == nodata
    missing_blocks = _blocks(missing, rows, cols)
    out = (_blocks((array != 0) & ~missing, rows, cols)).any(axis=(1, 3))
    out = out.astype(array.dtype)
    if nodata is not None:
        out = np.where(missing_blocks.all(axis=(1, 3)), nodata, out)
    return out.astype(array.dtype)


def look_files(
    paths: Sequence[Filename], output_dir: Filename, posting: float, kind: str = "mean"
) -> list[Path]:
    """Write rasters at `posting` metres into `output_dir`, keeping their names.

    `kind` is "mean" (the mean of valid pixels; real or complex), "label"
    (connected components; see `labels_of_cells`) or "any" (a mask; see
    `any_in_cells`).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [_look_file(p, out_dir, posting, kind) for p in paths]


def _look_file(path: Filename, out_dir: Path, posting: float, kind: str) -> Path:
    out = out_dir / Path(path).name
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile
        nodata = src.nodata
        tags = src.tags()
    rows, cols = looks_for_posting(posting, profile["transform"])
    if kind == "label":
        looked = labels_of_cells(array, nodata, rows, cols)
    elif kind == "any":
        looked = any_in_cells(array, nodata, rows, cols)
    elif kind == "mean":
        mean, any_valid = mean_of_valid(array, nodata, rows, cols)
        fill = 0.0 if nodata is None else nodata
        if np.issubdtype(array.dtype, np.integer):
            mean = np.rint(mean)
        looked = np.where(any_valid, mean, fill).astype(array.dtype)
    else:
        msg = f"kind must be mean, label or any; got {kind!r}"
        raise ValueError(msg)
    t = profile["transform"]
    profile.update(
        height=looked.shape[0],
        width=looked.shape[1],
        transform=Affine(t.a * cols, t.b, t.c, t.d, t.e * rows, t.f),
    )
    for key in ("blockxsize", "blockysize", "tiled"):
        profile.pop(key, None)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(looked, 1)
        dst.update_tags(**tags)
    return out


def look_after_unwrapping(
    unw_file_list: Sequence[Filename],
    conncomp_file_list: Sequence[Filename],
    cor_file_list: Sequence[Filename],
    posting: float,
    output_dir: Filename,
    quality_files: Optional[Sequence[Filename]] = None,
) -> LookedPaths:
    """Write the unwrapped interferograms and their companions at `posting` metres.

    Parameters
    ----------
    unw_file_list : Sequence[Filename]
        Unwrapped interferograms. Averaged as the mean of valid pixels.
    conncomp_file_list : Sequence[Filename]
        Connected-component labels. A cell is labelled only if all its pixels are.
    cor_file_list : Sequence[Filename]
        Interferometric correlation. Averaged as the mean of valid pixels.
    posting : float
        Side of a square output cell, in the units of the rasters' projection.
    output_dir : Filename
        Directory for the output, which keeps the input file names.
    quality_files : Sequence[Filename], optional
        Further rasters to average the same way (e.g. temporal coherence).

    Returns
    -------
    LookedPaths
        The output paths, and the (rows, columns) averaged into each cell.

    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(unw_file_list[0]) as src:
        looks = looks_for_posting(posting, src.transform)
    logger.info(
        f"Averaging {len(unw_file_list)} unwrapped interferograms to {posting} m:"
        f" {looks[0]} rows x {looks[1]} columns per cell"
    )
    return LookedPaths(
        unwrapped_paths=[_look_file(p, out_dir, posting, "mean") for p in unw_file_list],
        conncomp_paths=[
            _look_file(p, out_dir, posting, "label") for p in conncomp_file_list
        ],
        cor_paths=[_look_file(p, out_dir, posting, "mean") for p in cor_file_list],
        quality_files=[
            _look_file(p, out_dir, posting, "mean") for p in (quality_files or [])
        ],
        looks=looks,
    )
