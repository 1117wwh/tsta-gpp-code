"""Build a multi-year modal MCD12Q1 vegetation mask.

Workflow
--------
1. Read annual 0.1-degree MCD12Q1 IGBP rasters for 2001--2023.
2. For each pixel, select the most frequent original IGBP class.
   Ties are resolved deterministically by the smallest IGBP code.
3. Reclassify the modal IGBP class into the five vegetation groups used
   in the manuscript: forest, shrubland, savanna, grassland and cropland.

The original script at 5.MCD12Q1-VEG.py is not modified. This standalone
script writes all outputs and statistics to the 20260810 Vegetation_Mask folder.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


INPUT_DIR = Path(
    r"C:\Users\dell\Downloads\GEE_MCD12Q1-20260304T024357Z-3-001\GEE_MCD12Q1"
)
OUTPUT_DIR = Path(__file__).resolve().parent
YEARS = range(2001, 2024)
FILE_TEMPLATE = "MCD12Q1_{year}_0p1deg.tif"

# Original MCD12Q1 IGBP classes retained as vegetation.
VEGETATION_CLASSES = {
    1: "Forest",
    2: "Shrublands",
    3: "Savannas",
    4: "Grasslands",
    5: "Croplands",
}

# IGBP -> manuscript five-class vegetation code.
MERGE_MAP = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,       # Forest
    6: 2, 7: 2,                            # Shrublands
    8: 3, 9: 3,                            # Savannas
    10: 4,                                 # Grasslands
    12: 5, 14: 5,                          # Croplands and mosaic
}

RAW_MODE_NAME = "MCD12Q1_2001_2023_mode_IGBP.tif"
VEG_MASK_NAME = "MCD12Q1_2001_2023_mode_veg5.tif"
STATS_NAME = "vegetation_class_statistics.csv"
README_NAME = "README_MCD12Q1_mode_mask.md"


def find_inputs() -> list[Path]:
    paths = [INPUT_DIR / FILE_TEMPLATE.format(year=year) for year in YEARS]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing annual MCD12Q1 files:\n" + "\n".join(map(str, missing))
        )
    return paths


def check_alignment(paths: list[Path]) -> tuple[int, int, dict]:
    with rasterio.open(paths[0]) as src:
        shape = (src.height, src.width)
        profile = src.profile.copy()
        reference = (src.height, src.width, src.transform, src.crs)
    for path in paths[1:]:
        with rasterio.open(path) as src:
            current = (src.height, src.width, src.transform, src.crs)
            if current != reference:
                raise ValueError(f"Grid mismatch: {path}")
    return shape[0], shape[1], profile


def modal_igbp(paths: list[Path], profile: dict) -> tuple[np.ndarray, dict[int, int]]:
    """Calculate the modal original IGBP class block by block.

    The smallest class code wins deterministic ties because argmax returns the
    first class in ascending code order.
    """
    height, width = profile["height"], profile["width"]
    modal = np.zeros((height, width), dtype=np.uint8)
    counts = {code: 0 for code in range(18)}
    block_h = 256
    block_w = 512
    classes = np.arange(18, dtype=np.uint8)

    sources = [rasterio.open(path) for path in paths]
    try:
        for row in range(0, height, block_h):
            for col in range(0, width, block_w):
                h = min(block_h, height - row)
                w = min(block_w, width - col)
                values = np.stack(
                    [src.read(1, window=Window(col, row, w, h)) for src in sources],
                    axis=0,
                )
                frequency = np.stack(
                    [(values == code).sum(axis=0) for code in classes], axis=0
                )
                block_mode = classes[np.argmax(frequency, axis=0)]
                modal[row : row + h, col : col + w] = block_mode
                for code in classes:
                    counts[int(code)] += int(np.count_nonzero(block_mode == code))
    finally:
        for src in sources:
            src.close()
    return modal, counts


def merge_five_classes(modal: np.ndarray) -> np.ndarray:
    merged = np.zeros_like(modal, dtype=np.uint8)
    for original, target in MERGE_MAP.items():
        merged[modal == original] = target
    return merged


def write_raster(path: Path, array: np.ndarray, profile: dict) -> None:
    out_profile = profile.copy()
    out_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array, 1)


def write_statistics(path: Path, modal: np.ndarray, merged: np.ndarray) -> None:
    total = int(merged.size)
    rows = []
    for code, name in [(1, "Forest"), (2, "Shrublands"), (3, "Savannas"),
                       (4, "Grasslands"), (5, "Croplands")]:
        count = int(np.count_nonzero(merged == code))
        rows.append({"level": "merged_vegetation", "code": code,
                     "name": name, "pixel_count": count,
                     "fraction_of_grid": count / total})
    for code in range(18):
        count = int(np.count_nonzero(modal == code))
        rows.append({"level": "modal_IGBP", "code": code,
                     "name": f"IGBP_{code}", "pixel_count": count,
                     "fraction_of_grid": count / total})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, inputs: list[Path]) -> None:
    text = f"""# MCD12Q1 multi-year modal vegetation mask

Generated by `build_MCD12Q1_mode_veg_mask.py`.

- Input product: MCD12Q1 annual IGBP land-cover rasters at 0.1 degree.
- Years: 2001–2023 ({len(inputs)} annual rasters).
- Pixel classification: the most frequent original IGBP code across all years.
- Tie rule: the smallest IGBP code is selected deterministically.
- Output grid: the first annual raster's CRS, transform, extent and resolution.

## Five-class merge used in the manuscript

| Output code | Group | Original IGBP codes |
|---:|---|---|
| 1 | Forest | 1–5 |
| 2 | Shrublands | 6–7 |
| 3 | Savannas | 8–9 |
| 4 | Grasslands | 10 |
| 5 | Croplands | 12 and 14 |

Original codes 0, 11, 13, 15, 16 and 17 are assigned to 0 (excluded/non-vegetated).

## Outputs

- `{RAW_MODE_NAME}`: modal original IGBP class (0–17; nodata=0).
- `{VEG_MASK_NAME}`: manuscript-consistent five-class vegetation mask (1–5; nodata=0).
- `{STATS_NAME}`: pixel counts and grid fractions for modal and merged classes.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    paths = find_inputs()
    height, width, profile = check_alignment(paths)
    modal, _ = modal_igbp(paths, profile)
    merged = merge_five_classes(modal)
    write_raster(OUTPUT_DIR / RAW_MODE_NAME, modal, profile)
    write_raster(OUTPUT_DIR / VEG_MASK_NAME, merged, profile)
    write_statistics(OUTPUT_DIR / STATS_NAME, modal, merged)
    write_readme(OUTPUT_DIR / README_NAME, paths)
    first_year = paths[0].stem.split("_")[1]
    last_year = paths[-1].stem.split("_")[1]
    print(f"Processed years: {first_year}–{last_year}")
    print(f"Grid: {height} x {width}")
    print(f"Vegetated pixels: {int(np.count_nonzero(merged)):,}")
    for code, name in [(1, "Forest"), (2, "Shrublands"), (3, "Savannas"),
                       (4, "Grasslands"), (5, "Croplands")]:
        print(f"{code} {name}: {int(np.count_nonzero(merged == code)):,}")


if __name__ == "__main__":
    main()
