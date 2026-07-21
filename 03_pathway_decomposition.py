# -*- coding: utf-8 -*-
r"""
Pathway decomposition of GPP sensitivity to Ts-Ta (Fig. 3 and Fig. 4).

The observed sensitivity of GPP to Ts-Ta is partitioned into three pathways
associated with atmospheric water demand (VPD), soil water supply (SM) and
background thermal conditions (T2M). All variables are standardized anomalies
from 01a. For each pixel:

  1. alpha_E : covariation of each mediator with Ts-Ta, from simple regressions
               E ~ Ts-Ta   (E in {VPD, SM, T2M}).
  2. beta_E  : partial effect of each mediator on GPP, from one multiple
               regression   GPP ~ VPD + SM + T2M.
  3. C_E = alpha_E * beta_E : the contribution of pathway E.
     The reconstructed sensitivity is C_VPD + C_SM + C_T2M (used in Fig. 4).

Multicollinearity among VPD, SM and T2M is screened with variance inflation
factors (VIF). Implementation note: after reading, each variable stack is written
to a memory-mapped file so that every parallel worker maps only the one image row
it needs, keeping memory use flat regardless of the number of workers.

Inputs : standardized anomalies from 01a ({PREPROC_ROOT}\{VAR}_ZSCORE for
         GPP, TSTA, VPD, SM, T2M).
Outputs: per-pixel alpha/beta maps, pathway contributions C_VPD/C_SM/C_T2M,
         dominant pathway, relative contributions, R^2 and VIF diagnostics,
         written to {OUT_DIR}.
"""

import os
import gc
import tempfile
import warnings
import numpy as np
import rasterio
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==================== Configuration ====================
PREPROC_ROOT = r"D:\DATA\FDI_PREPROC2"
OUTPUT_ROOT = r"C:\Users\dell\Desktop\EIS"

# Ts-Ta temporal window. NOON is the main analysis; DAYTIME / ALLDAY are optional
# robustness variants. The names map to the folders written by 01a.
TSTA_WINDOW = "NOON"
TSTA_VAR_BY_WINDOW = {
    "NOON": "TSTA",
    "DAYTIME": "TSTA_DAYTIME",
    "ALLDAY": "TSTA_ALLDAY",
}
TSTA_PREFIX = TSTA_VAR_BY_WINDOW[TSTA_WINDOW]
TSTA_DIR = os.path.join(PREPROC_ROOT, f"{TSTA_PREFIX}_ZSCORE")

# GPP (GOSIF GPP) is the response variable.
GPP_PREFIX = "GPP"
GPP_DIR = os.path.join(PREPROC_ROOT, "GPP_ZSCORE")

VPD_DIR = os.path.join(PREPROC_ROOT, "VPD_ZSCORE")
SM_DIR = os.path.join(PREPROC_ROOT, "SM_ZSCORE")
T2M_DIR = os.path.join(PREPROC_ROOT, "T2M_ZSCORE")

OUT_SUFFIX = "" if TSTA_WINDOW == "NOON" else f"_{TSTA_WINDOW}"
OUT_DIR = os.path.join(OUTPUT_ROOT, "Pathway_Decomposition_Final2" + OUT_SUFFIX)
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = range(2001, 2024)
MIN_SAMPLES = 150
VIF_THRESHOLD = 10
NUM_JOBS = 40
TEMP_DIR = r"D:\DATA\TEMP_MMAP"
os.makedirs(TEMP_DIR, exist_ok=True)


# ==================== Utilities ====================

def get_path(var_dir, prefix, year, doy):
    return os.path.join(var_dir, f"{prefix}_ZSCORE_{year}_{doy:03d}.tif")


def read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None and np.isfinite(src.nodata):
            arr[arr == src.nodata] = np.nan
        arr[arr == 65534] = np.nan
        arr[arr == 65535] = np.nan
    return arr


def save_tif(arr, filename, profile):
    path = os.path.join(OUT_DIR, filename)
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.float32), 1)
    print(f"  Saved: {filename}")


def calculate_vif(X):
    n, k = X.shape
    vif = []
    for i in range(k):
        mask = np.ones(k, dtype=bool)
        mask[i] = False
        X_reg = np.column_stack([np.ones(n), X[:, mask]])
        coefs, _, _, _ = np.linalg.lstsq(X_reg, X[:, i], rcond=None)
        y_pred = X_reg @ coefs
        ss_res = np.sum((X[:, i] - y_pred) ** 2)
        ss_tot = np.sum((X[:, i] - np.mean(X[:, i])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0
        vif.append(1.0 / (1.0 - r2) if r2 < 0.999 else np.inf)
    return np.array(vif)


# ==================== Per-row worker ====================

def _row_worker(row_idx,
                tsta_path, gpp_path, vpd_path, sm_path, t2m_path,
                shape, min_samples, vif_threshold):
    """Each worker maps only the single image row it needs from each memmap."""
    T, H, W = shape

    def load_row(path):
        mm = np.memmap(path, dtype="float32", mode="r", shape=shape)
        row = mm[:, row_idx, :].copy()
        del mm
        return row

    tsta_row = load_row(tsta_path)
    gpp_row = load_row(gpp_path)
    vpd_row = load_row(vpd_path)
    sm_row = load_row(sm_path)
    t2m_row = load_row(t2m_path)

    def empty():
        return np.full(W, np.nan, dtype=np.float32)

    alpha1, alpha2, alpha3 = empty(), empty(), empty()
    beta1, beta2, beta3 = empty(), empty(), empty()
    c_vpd, c_sm, c_t2m = empty(), empty(), empty()
    dominant = empty()
    c_dominant = empty()
    rel_vpd, rel_sm, rel_t2m = empty(), empty(), empty()
    r_squared = empty()
    vif_max = empty()
    valid_flag = empty()

    for col in range(W):
        ts = tsta_row[:, col]
        gpp = gpp_row[:, col]
        vpd = vpd_row[:, col]
        sm = sm_row[:, col]
        t2m = t2m_row[:, col]

        valid = (np.isfinite(ts) & np.isfinite(gpp) &
                 np.isfinite(vpd) & np.isfinite(sm) & np.isfinite(t2m))
        n = valid.sum()
        if n < min_samples:
            continue

        ts_v = ts[valid]; gpp_v = gpp[valid]
        vpd_v = vpd[valid]; sm_v = sm[valid]; t2m_v = t2m[valid]

        # Step 1: simple regressions -> alpha (Ts-Ta -> mediator covariation)
        a1, _, _, _, _ = stats.linregress(ts_v, vpd_v)
        a2, _, _, _, _ = stats.linregress(ts_v, sm_v)
        a3, _, _, _, _ = stats.linregress(ts_v, t2m_v)
        alpha1[col], alpha2[col], alpha3[col] = a1, a2, a3

        # Step 2: VIF screening
        X_vars = np.column_stack([vpd_v, sm_v, t2m_v])
        vif_vals = calculate_vif(X_vars)
        vmax = float(np.max(vif_vals))
        vif_max[col] = vmax
        if vmax > vif_threshold:
            valid_flag[col] = 0
            continue
        valid_flag[col] = 1

        # Step 3: multiple regression -> beta (partial mediator -> GPP effects)
        X = np.column_stack([np.ones(n), vpd_v, sm_v, t2m_v])
        try:
            coefs, _, _, _ = np.linalg.lstsq(X, gpp_v, rcond=None)
            b1, b2, b3 = float(coefs[1]), float(coefs[2]), float(coefs[3])
            y_pred = X @ coefs
            ss_res = np.sum((gpp_v - y_pred) ** 2)
            ss_tot = np.sum((gpp_v - np.mean(gpp_v)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        except Exception:
            continue

        beta1[col], beta2[col], beta3[col] = b1, b2, b3
        r_squared[col] = r2

        # Step 4: pathway contributions (sign preserved)
        cv, cs, ct = a1 * b1, a2 * b2, a3 * b3
        c_vpd[col], c_sm[col], c_t2m[col] = cv, cs, ct

        # Step 5: dominant pathway
        abs_c = np.array([abs(cv), abs(cs), abs(ct)])
        total_abs = abs_c.sum()
        if total_abs > 1e-10:
            dom_idx = int(np.argmax(abs_c))
            dominant[col] = dom_idx + 1
            c_dominant[col] = [cv, cs, ct][dom_idx]
            rel_vpd[col], rel_sm[col], rel_t2m[col] = abs_c / total_abs

    return (row_idx,
            alpha1, alpha2, alpha3,
            beta1, beta2, beta3,
            c_vpd, c_sm, c_t2m,
            dominant, c_dominant,
            rel_vpd, rel_sm, rel_t2m,
            r_squared, vif_max, valid_flag)


# ==================== Main ====================

def main():
    print("=" * 60)
    print("Pathway decomposition: Ts-Ta -> [VPD / SM / T2M] -> GPP")
    print(f"Ts-Ta window : {TSTA_WINDOW}")
    print(f"Ts-Ta dir    : {TSTA_DIR}")
    print(f"GPP dir      : {GPP_DIR}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Years        : {YEARS.start}-{YEARS.stop - 1}")
    print(f"Min samples  : {MIN_SAMPLES}  VIF threshold: {VIF_THRESHOLD}  jobs: {NUM_JOBS}")
    print("=" * 60)

    # Build the list of time steps for which all five variables exist.
    time_list = []
    for year in YEARS:
        for doy in range(1, 367, 8):
            files = [
                get_path(TSTA_DIR, TSTA_PREFIX, year, doy),
                get_path(GPP_DIR, GPP_PREFIX, year, doy),
                get_path(VPD_DIR, "VPD", year, doy),
                get_path(SM_DIR, "SM", year, doy),
                get_path(T2M_DIR, "T2M", year, doy),
            ]
            if all(os.path.exists(f) for f in files):
                time_list.append(tuple(files))

    print(f"Valid time steps: {len(time_list)}")
    if not time_list:
        raise RuntimeError("No valid files found; check the path configuration.")

    with rasterio.open(time_list[0][0]) as src:
        profile = src.profile.copy()
        H, W = src.height, src.width
    T = len(time_list)
    shape = (T, H, W)
    print(f"Grid size: {H}x{W}  time steps: {T}")
    print(f"Per-variable memmap size: {T * H * W * 4 / 1024 ** 3:.2f} GB")
    print(f"Per-worker memory increment: ~{T * W * 5 * 4 / 1024 ** 2:.0f} MB")

    # Read each variable into a memory-mapped file (one large array at a time).
    print("\nReading data into memory-mapped files...")
    var_info = [
        ("tsta", TSTA_DIR, TSTA_PREFIX, 0),
        ("gpp", GPP_DIR, GPP_PREFIX, 1),
        ("vpd", VPD_DIR, "VPD", 2),
        ("sm", SM_DIR, "SM", 3),
        ("t2m", T2M_DIR, "T2M", 4),
    ]
    mmap_paths = {}

    for vname, vdir, prefix, fidx in var_info:
        mmap_tag = f"{TSTA_WINDOW.lower()}_gpp"
        fp_mmap = os.path.join(TEMP_DIR, f"pathway_{mmap_tag}_{vname}.mmap")
        mmap_paths[vname] = fp_mmap

        if os.path.exists(fp_mmap):
            fsize = os.path.getsize(fp_mmap)
            expected = T * H * W * 4
            if fsize == expected:
                print(f"  {vname.upper()}: memmap exists, skip reading")
                continue
            else:
                print(f"  {vname.upper()}: memmap size mismatch, re-reading")
                os.remove(fp_mmap)

        print(f"  Reading {vname.upper()}...")
        stack = np.memmap(fp_mmap, dtype="float32", mode="w+", shape=shape)
        for t, files in enumerate(tqdm(time_list, desc=f"    {vname}")):
            try:
                stack[t] = read_tif(files[fidx])
            except Exception as e:
                print(f"    Warning: {files[fidx]} -> {e}")
                stack[t] = np.nan
        stack.flush()
        del stack
        gc.collect()

    # Parallel decomposition (each worker maps only the row it needs).
    print("\nParallel pathway decomposition...")
    results = Parallel(n_jobs=NUM_JOBS, backend="loky")(
        delayed(_row_worker)(
            row,
            mmap_paths["tsta"], mmap_paths["gpp"],
            mmap_paths["vpd"], mmap_paths["sm"],
            mmap_paths["t2m"],
            shape, MIN_SAMPLES, VIF_THRESHOLD
        )
        for row in tqdm(range(H), desc="Computing pathway decomposition")
    )

    print("\nAssembling results...")
    keys = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3',
            'cv', 'cs', 'ct', 'dom', 'c_dom',
            'rv', 'rs', 'rt', 'r2', 'vif', 'valid']
    maps = {k: np.full((H, W), np.nan, dtype=np.float32) for k in keys}

    for (row, a1, a2, a3, b1, b2, b3, cv, cs, ct,
         dom, c_dom, rv, rs, rt, r2, vif, valid) in results:
        maps['a1'][row] = a1;    maps['a2'][row] = a2
        maps['a3'][row] = a3;    maps['b1'][row] = b1
        maps['b2'][row] = b2;    maps['b3'][row] = b3
        maps['cv'][row] = cv;    maps['cs'][row] = cs
        maps['ct'][row] = ct;    maps['dom'][row] = dom
        maps['c_dom'][row] = c_dom; maps['rv'][row] = rv
        maps['rs'][row] = rs;    maps['rt'][row] = rt
        maps['r2'][row] = r2;    maps['vif'][row] = vif
        maps['valid'][row] = valid

    del results
    gc.collect()

    print("\nSaving results...")
    save_tif(maps['a1'], "alpha1_TSTA_to_VPD.tif", profile)
    save_tif(maps['a2'], "alpha2_TSTA_to_SM.tif", profile)
    save_tif(maps['a3'], "alpha3_TSTA_to_T2M.tif", profile)
    save_tif(maps['b1'], "beta1_VPD_to_GPP.tif", profile)
    save_tif(maps['b2'], "beta2_SM_to_GPP.tif", profile)
    save_tif(maps['b3'], "beta3_T2M_to_GPP.tif", profile)
    save_tif(maps['cv'], "C_VPD_pathway.tif", profile)
    save_tif(maps['cs'], "C_SM_pathway.tif", profile)
    save_tif(maps['ct'], "C_T2M_pathway.tif", profile)
    save_tif(maps['dom'], "Dominant_pathway_code.tif", profile)
    save_tif(maps['c_dom'], "C_dominant_signed.tif", profile)
    save_tif(maps['rv'], "RelContrib_VPD.tif", profile)
    save_tif(maps['rs'], "RelContrib_SM.tif", profile)
    save_tif(maps['rt'], "RelContrib_T2M.tif", profile)
    save_tif(maps['r2'], "R_squared.tif", profile)
    save_tif(maps['vif'], "VIF_max.tif", profile)
    save_tif(maps['valid'], "VIF_valid_flag.tif", profile)

    print("\n" + "=" * 60)
    print("Result summary")
    print("=" * 60)
    valid_mask = maps['valid'] == 1
    total_valid = np.isfinite(maps['valid']).sum()
    print(f"Passed VIF check: {valid_mask.sum():,} pixels "
          f"({valid_mask.sum() / total_valid * 100:.1f}%)")

    for name, key in [("C_VPD", 'cv'), ("C_SM", 'cs'), ("C_T2M", 'ct')]:
        v = maps[key][valid_mask & np.isfinite(maps[key])]
        print(f"\n{name}: mean={v.mean():.4f}  median={np.median(v):.4f}  "
              f"pos={(v > 0).mean() * 100:.1f}%  neg={(v < 0).mean() * 100:.1f}%")

    dom = maps['dom'][valid_mask & np.isfinite(maps['dom'])]
    c_d = maps['c_dom'][valid_mask & np.isfinite(maps['c_dom'])]
    print("\nDominant pathway:")
    for i, name in enumerate(["VPD", "SM", "T2M"]):
        m = dom == i + 1
        cd = c_d[m]
        print(f"  {name}: {m.sum():,} pixels ({m.mean() * 100:.1f}%)  "
              f"pos={(cd > 0).mean() * 100:.1f}%  neg={(cd < 0).mean() * 100:.1f}%")

    r2v = maps['r2'][valid_mask & np.isfinite(maps['r2'])]
    print(f"\nR^2: mean={r2v.mean():.3f}  median={np.median(r2v):.3f}")
    print(f"\nOutput dir: {OUT_DIR}")
    print("=" * 60)

    # Memmap files are kept for restartability; uncomment to clean up.
    # for p in mmap_paths.values():
    #     if os.path.exists(p):
    #         os.remove(p)


if __name__ == "__main__":
    main()
