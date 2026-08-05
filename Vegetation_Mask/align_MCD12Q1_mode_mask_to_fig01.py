"""Align modal MCD12Q1 categorical masks to the Fig. 1 analysis grid.

The annual MCD12Q1 rasters and the Fig. 1 sensitivity/AI rasters have the
same shape and nominal resolution but different geotransforms. This script uses
nearest-neighbour reprojection so categorical IGBP and vegetation-group labels
are spatially aligned before they are used for vegetation-stratified analyses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


MASK_DIR = Path(__file__).resolve().parent
REFERENCE_GRID = Path(
    r"C:\Users\dell\Desktop\EIS\TS_TA_Sensitivity_FullData2\TS_TA_Sensitivity.tif"
)
SOURCES = {
    "MCD12Q1_2001_2023_mode_IGBP.tif": "MCD12Q1_2001_2023_mode_IGBP_aligned_to_Fig1.tif",
    "MCD12Q1_2001_2023_mode_veg5.tif": "MCD12Q1_2001_2023_mode_veg5_aligned_to_Fig1.tif",
}
METADATA_PATH = MASK_DIR / "mask_alignment_metadata.json"


def align_categorical(source_path: Path, output_path: Path, reference_path: Path) -> dict:
    with rasterio.open(reference_path) as ref, rasterio.open(source_path) as src:
        destination = np.zeros((ref.height, ref.width), dtype=np.uint8)
        reproject(
            source=src.read(1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        profile = ref.profile.copy()
        profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(destination, 1)
        return {
            "source": str(source_path),
            "output": str(output_path),
            "source_shape": [src.height, src.width],
            "source_transform": list(src.transform)[:6],
            "target_shape": [ref.height, ref.width],
            "target_transform": list(ref.transform)[:6],
            "crs": str(ref.crs),
            "resampling": "nearest",
            "nonzero_pixels": int(np.count_nonzero(destination)),
            "classes": [int(value) for value in np.unique(destination)],
        }


def main() -> None:
    if not REFERENCE_GRID.exists():
        raise FileNotFoundError(f"Missing Fig. 1 reference grid: {REFERENCE_GRID}")
    records = []
    for source_name, output_name in SOURCES.items():
        source_path = MASK_DIR / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing modal mask: {source_path}")
        record = align_categorical(source_path, MASK_DIR / output_name, REFERENCE_GRID)
        records.append(record)
        print(f"Aligned: {source_name} -> {output_name}")
        print(f"  nonzero pixels: {record['nonzero_pixels']:,}; classes: {record['classes']}")
    METADATA_PATH.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
