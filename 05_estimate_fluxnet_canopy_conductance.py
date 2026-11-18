"""Calculate daytime FLUXNET canopy conductance for the selected 154 sites.

Read half-hourly flux and temperature records, station measurement heights
and daily GPP. Use Penman–Monteith inversion, a 24-hour dry window and
06:00–18:00 local mean solar time. Apply the site-year activity criterion
(daily GPP above 25% of its annual 95th percentile) before daily averaging.
Daily Gc is in m/s; each daily value requires at least four eligible intervals.
Use --site AT-Neu to run one site. Outputs feed scripts 02 and 06.

Inputs
------
The source and archive directories provide measurement heights, site identifiers and
inventories pointing to the half-hourly records and daily GPP. Requires pandas,
NumPy and PyArrow.

Run from the directory containing this script
--------------------------------------------
    python 05_estimate_fluxnet_canopy_conductance.py --site AT-Neu
    python 05_estimate_fluxnet_canopy_conductance.py

Outputs in results/05
---------------------
One daily CSV per processed site. Omit --site to process the full cohort before
using station mode in 02 or rebuilding annual inputs in 06.
"""

from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc

CP = 1004.67

RD = 287.05

RGAS = 8.314462618

EPSILON = 0.622

LAMBDA_V = 2.45e6

KARMAN = 0.40

COMMON_COLUMNS = [
    "TIMESTAMP_START",
    "TIMESTAMP_END",
    "TA_F_MDS",
    "VPD_F_MDS",
    "PA",
    "PA_F",
    "WS",
    "WS_F",
    "USTAR",
    "SW_IN_F_MDS",
    "P",
    "P_F",
    "RH",
    "NETRAD",
    "G_F_MDS",
    "LE_CORR",
    "LE_F_MDS",
    "H_CORR",
    "H_F_MDS",
    "G_F_MDS_QC",
    "LE_F_MDS_QC",
    "H_F_MDS_QC",
    "TA_F_MDS_QC",
    "VPD_F_MDS_QC",
    "SW_IN_F_MDS_QC",
]


def numeric(series: pd.Series) -> pd.Series:
    """Convert a FLUXNET field to numbers and mask missing-value codes at or below -9990."""
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values <= -9990)


def field_numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric field or an all-missing series when a column is absent."""
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return numeric(frame[name])


def safe_fallback(
    frame: pd.DataFrame, preferred: str, fallback: str
) -> tuple[pd.Series, pd.Series]:
    """Fill missing values from a second field and return the source field used for each
    observation.
    """
    a = numeric(frame[preferred]) if preferred in frame else pd.Series(np.nan, index=frame.index)
    b = numeric(frame[fallback]) if fallback in frame else pd.Series(np.nan, index=frame.index)
    use_a = a.notna()
    return a.fillna(b), pd.Series(
        np.where(use_a, preferred, np.where(b.notna(), fallback, "missing")), index=frame.index
    )


def pm_components(
    ta_c: pd.Series,
    vpd_hpa: pd.Series,
    pa_kpa: pd.Series,
    le: pd.Series,
    energy: pd.Series,
    ga_m_s: pd.Series,
) -> pd.DataFrame:
    """Invert Penman-Monteith and return its denominator and conductance in m/s and mol
    per square metre per second.
    """
    ta = ta_c.to_numpy(dtype=float)
    vpd_pa = vpd_hpa.to_numpy(dtype=float) * 100.0
    pressure_pa = pa_kpa.to_numpy(dtype=float) * 1000.0
    latent = le.to_numpy(dtype=float)
    available = energy.to_numpy(dtype=float)
    ga = ga_m_s.to_numpy(dtype=float)
    temp_k = ta + 273.15
    svp_kpa = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))
    delta = 4098.0 * svp_kpa / ((ta + 237.3) ** 2) * 1000.0
    gamma = CP * pressure_pa / (EPSILON * LAMBDA_V)
    rho = pressure_pa / (RD * temp_k)
    denominator = delta * available + rho * CP * vpd_pa * ga - latent * (delta + gamma)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gs_m_s = gamma * ga * latent / denominator
        gs_mol = gs_m_s * pressure_pa / (RGAS * temp_k)
    return pd.DataFrame(
        {
            "pm_denominator": denominator,
            "gs_m_s": gs_m_s,
            "gs_mol_m2_s": gs_mol,
        },
        index=ta_c.index,
    )


def fu_aerodynamic(
    ws: pd.Series,
    ustar: pd.Series,
    zm_m: float,
    h_meta_m: float,
    h_source_label: str = "",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, str]:
    """Return conductance, resistance, canopy height, geometry validity and the height
    source for the near-neutral estimate.
    """
    wind = ws.to_numpy(dtype=float)
    # h_meta_m is a single, site-level HEIGHTC or near-neutral Eq. 5 estimate.
    # It must not be recalculated row by row: doing so algebraically collapses
    # the Fu resistance toward the USTAR^2/WS approximation.
    h = np.full_like(
        wind, h_meta_m if np.isfinite(h_meta_m) and h_meta_m > 0 else np.nan, dtype=float
    )
    h_source = h_source_label or "site_level_resolved_canopy_height"
    zd = 0.67 * h
    z0 = 0.10 * h
    geometry = np.isfinite(zm_m) & (zm_m > 0) & np.isfinite(h) & (h > 0) & (zm_m > zd + z0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ra = np.log((zm_m - zd) / z0) ** 2 / (wind * KARMAN**2)
        ga = 1.0 / ra
    geometry &= np.isfinite(ra) & (ra > 0) & np.isfinite(ga) & (ga > 0)
    return (
        pd.Series(ga, index=ws.index),
        pd.Series(ra, index=ws.index),
        pd.Series(h, index=ws.index),
        pd.Series(geometry, index=ws.index),
        h_source,
    )


def parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse FLUXNET YYYYMMDDHHMM timestamps, marking invalid entries as missing."""
    text = series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)
    return pd.to_datetime(text, format="%Y%m%d%H%M", errors="coerce")


def rain_24_mask(frame: pd.DataFrame) -> pd.Series:
    """Flag dry intervals with 49 consecutive valid half-hourly readings from 24 hours
    ago through the current reading.
    """
    rain, _ = safe_fallback(frame, "P_F", "P")
    rain = rain.mask(rain < 0)
    timestamp = parse_timestamp(frame["TIMESTAMP_START"])
    step = timestamp.diff().eq(pd.Timedelta(minutes=30))
    count = rain.notna().astype(int).rolling(49, min_periods=49).sum()
    total = rain.fillna(0).rolling(49, min_periods=49).sum()
    continuous = step.astype(int).rolling(48, min_periods=48).sum().eq(48)
    return count.eq(49) & total.eq(0) & continuous


def choose_energy_tier(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return LE, energy and tier; fallback occurs only for missing inputs."""
    idx = frame.index
    le_corr = field_numeric(frame, "LE_CORR")
    le_raw = field_numeric(frame, "LE_F_MDS")
    h_corr = field_numeric(frame, "H_CORR")
    h_raw = field_numeric(frame, "H_F_MDS")
    rn_minus_g = field_numeric(frame, "NETRAD") - field_numeric(frame, "G_F_MDS")

    le = pd.Series(np.nan, index=idx, dtype=float)
    energy = pd.Series(np.nan, index=idx, dtype=float)
    tier = pd.Series("none", index=idx, dtype=object)

    candidates = [
        ("T1_corrLE_RnG", le_corr, rn_minus_g),
        ("T2_rawLE_RnG", le_raw, rn_minus_g),
        ("T3_corrLE_corrLEplusH", le_corr, le_corr + h_corr),
        ("T4_rawLE_rawLEplusH", le_raw, le_raw + h_raw),
    ]
    unassigned = pd.Series(True, index=idx)
    for label, candidate_le, candidate_energy in candidates:
        # Fallback is triggered by missing/non-finite inputs only. A present
        # but non-positive LE, low energy, failed PM denominator, or outlying
        # Gc is screened later and is never rescued by changing methods.
        available = (
            unassigned
            & candidate_le.notna()
            & np.isfinite(candidate_le)
            & candidate_energy.notna()
            & np.isfinite(candidate_energy)
        )
        le.loc[available] = candidate_le.loc[available]
        energy.loc[available] = candidate_energy.loc[available]
        tier.loc[available] = label
        unassigned.loc[available] = False
    return le, energy, tier


def read_columns(path, columns):
    """Read requested fields present in a station CSV and preserve integer timestamps."""
    header = pd.read_csv(path, nrows=0).columns
    use = [x for x in columns if x in header]
    types = {x: pa.int64() if x.startswith("TIMESTAMP") else pa.float64() for x in use}
    return pc.read_csv(
        path,
        read_options=pc.ReadOptions(use_threads=False),
        convert_options=pc.ConvertOptions(
            include_columns=use, column_types=types, null_values=["", "NA", "NaN", "nan"]
        ),
    ).to_pandas()


def active_dates(daily_file):
    """Return dates in 2001-2023 when GPP exceeds one quarter of the positive site-year
    95th percentile.
    """
    data = pd.read_csv(daily_file, usecols=["TIMESTAMP", "GPP_NT_VUT_REF"])
    dates = pd.to_datetime(data.TIMESTAMP.astype(str), format="%Y%m%d")
    gpp = data.GPP_NT_VUT_REF.where(data.GPP_NT_VUT_REF.abs() < 9000)
    p95 = gpp.groupby(dates.dt.year).transform("quantile", q=0.95)
    return set(
        dates[(p95 > 0) & (gpp > 0.25 * p95) & dates.dt.year.between(2001, 2023)].dt.strftime(
            "%Y-%m-%d"
        )
    )


def calculate_site(sid, height, daily_file, metadata):
    """Align and screen half-hourly observations and return daily conductance for active
    days with at least four eligible intervals.
    """
    hh = Path(metadata["hh_source"])
    ls = Path(metadata["lst_source"])
    shift = metadata["clock_to_LMST_hours"]
    header = pd.read_csv(hh, nrows=0).columns
    extra_flags = [
        c for c in header if c.endswith("_QC") and c.startswith(("LW_", "PA", "WS", "P_", "USTAR"))
    ]
    fields = list(dict.fromkeys(COMMON_COLUMNS + ["GPP_NT_VUT_REF", "TA_F"] + extra_flags))
    d = read_columns(hh, fields).replace(-9999, np.nan)
    assert d.TIMESTAMP_START.is_unique
    assert (
        (parse_timestamp(d.TIMESTAMP_END) - parse_timestamp(d.TIMESTAMP_START))
        .eq(pd.Timedelta(minutes=30))
        .all()
    )
    d["GPP_NT_VUT_REF"] = d.GPP_NT_VUT_REF.where(d.GPP_NT_VUT_REF.between(0, 30000))
    for name in ["P_F", "P"]:
        if name in d:
            invalid = d[name].lt(0)
            if name + "_QC" in d:
                invalid |= d[name + "_QC"].eq(3)
            d[name] = d[name].mask(invalid)
    extra_qc = pd.Series(True, index=d.index)
    for a, b in [("PA_F", "PA"), ("WS_F", "WS")]:
        _, source = safe_fallback(d, a, b)
        for name in [a, b]:
            if name + "_QC" in d:
                extra_qc &= ~(source.eq(name) & d[name + "_QC"].eq(3))
    if "USTAR_QC" in d:
        extra_qc &= ~d.USTAR_QC.eq(3)
    lst = pd.read_csv(ls)
    assert lst.TIMESTAMP_START.is_unique
    good = (
        np.isfinite(lst.LST)
        & lst.LST.gt(-273.15)
        & lst.BBE.gt(0)
        & lst.BBE.le(1)
        & lst.LW_IN.gt(0)
        & lst.LW_OUT.gt(0)
    )
    quality = d.set_index("TIMESTAMP_START")
    for field in ["LW_IN_COL", "LW_OUT_COL"]:
        for name in lst[field].dropna().unique():
            if name + "_QC" in quality:
                good &= ~(
                    lst[field].eq(name) & lst.TIMESTAMP_START.map(quality[name + "_QC"]).eq(3)
                )
    spikes = {"CA-Cbo": 201410301100, "FR-LBr": 200412101100, "US-EML": 202107091030}
    good &= lst.TIMESTAMP_START.ne(spikes.get(sid, -1))
    lst_c = d.TIMESTAMP_START.map(lst.set_index("TIMESTAMP_START").LST.where(good.to_numpy()))
    frame = d
    idx = frame.index
    timestamp = parse_timestamp(frame["TIMESTAMP_START"])
    timestamp = (
        timestamp
        + (parse_timestamp(frame["TIMESTAMP_END"]) - timestamp) / 2
        + pd.Timedelta(hours=shift)
    )
    minute = (
        timestamp.dt.hour * 60
        + timestamp.dt.minute
        + timestamp.dt.second / 60
        + timestamp.dt.microsecond / 60000000
    )
    ta = field_numeric(frame, "TA_F_MDS")
    vpd = field_numeric(frame, "VPD_F_MDS")
    pa, _ = safe_fallback(frame, "PA_F", "PA")
    wind, _ = safe_fallback(frame, "WS_F", "WS")
    ustar = field_numeric(frame, "USTAR")
    gpp = field_numeric(frame, "GPP_NT_VUT_REF")
    le, energy, tier = choose_energy_tier(frame)

    qc_base = pd.Series(True, index=idx)
    for col in ["TA_F_MDS_QC", "VPD_F_MDS_QC"]:
        if col in frame.columns:
            q = field_numeric(frame, col)
            if q.notna().any():
                qc_base &= ~q.eq(3)
    # G QC is relevant only when Rn-G is used. Raw turbulent fallback uses
    # the available LE/H QC fields when those flags exist.
    uses_rng = tier.isin(["T1_corrLE_RnG", "T2_rawLE_RnG"])
    if "G_F_MDS_QC" in frame.columns:
        q = field_numeric(frame, "G_F_MDS_QC")
        if q.notna().any():
            qc_base &= (~uses_rng) | (~q.eq(3))
    uses_raw = tier.eq("T4_rawLE_rawLEplusH")
    for col in ["LE_F_MDS_QC", "H_F_MDS_QC"]:
        if col in frame.columns:
            q = field_numeric(frame, col)
            if q.notna().any():
                uses_field = (
                    tier.isin(["T2_rawLE_RnG", "T4_rawLE_rawLEplusH"])
                    if col == "LE_F_MDS_QC"
                    else uses_raw
                )
                qc_base &= (~uses_field) | (~q.eq(3))

    qc_base &= extra_qc
    ga, _, _, geometry, _ = fu_aerodynamic(
        wind,
        ustar,
        height.wind_measurement_height_m,
        height.canopy_height_used_m,
        height.previous_canopy_height_source,
    )
    pm = pm_components(ta, vpd, pa, le, energy, ga)
    tts = lst_c - ta
    mask = (
        timestamp.notna()
        & minute.ge(360)
        & minute.le(1080)
        & ta.gt(-273.15)
        & vpd.gt(0)
        & pa.gt(0)
        & wind.gt(0.1)
        & ustar.gt(0.1)
        & rain_24_mask(frame)
        & qc_base
        & geometry
        & tier.ne("none")
        & le.gt(0)
        & energy.gt(50)
        & np.isfinite(pm["pm_denominator"])
        & pm["pm_denominator"].gt(0)
        & np.isfinite(pm["gs_m_s"])
        & pm["gs_m_s"].gt(0)
        & pm["gs_m_s"].le(0.05)
        & tts.notna()
        & gpp.notna()
        & np.isfinite(gpp)
    )
    # Apply the site-year GPP activity threshold before daily aggregation.
    mask &= timestamp.dt.strftime("%Y-%m-%d").isin(active_dates(daily_file))
    x = pd.DataFrame(
        {
            "timestamp": timestamp,
            "date": timestamp.dt.strftime("%Y-%m-%d"),
            "tts_c": tts,
            "gc_m_s": pm["gs_m_s"],
            "gpp_umol_m2_s": gpp,
            "tier": tier,
        }
    ).loc[mask]

    daily = x.groupby("date").agg(
        tts_c=("tts_c", "mean"),
        gc_m_s=("gc_m_s", "mean"),
        gpp_umol_m2_s=("gpp_umol_m2_s", "mean"),
        n_halfhour=("tier", "size"),
        dominant_tier=("tier", lambda values: values.value_counts().index[0]),
    )
    return daily[daily.n_halfhour >= 4]


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    archive = folder.parent / "Figure_releases_20261029"
    source = Path("D:/DATA/DeltaT_GPP_Analysis/Results/Temporary")
    heights = pd.read_csv(
        source / "20261023_S02_station_SEM_sensitivity_sign/Gc_H_LE_source_inventory.csv"
    ).set_index("site")
    sites = pd.read_csv(
        archive / "Figure_01/20261029_154sites/station_analysis/station_GPP_Gc_AI.csv"
    ).site
    output = folder / "results/05"
    output.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=list(sites))
    args = parser.parse_args()
    for site in [args.site] if args.site else sites:
        metadata = json.loads(
            (source / f"20261023_S01_station_8day_daytime_ridge_VPD/qc/{site}.json").read_text()
        )
        daily_file = json.loads(
            (archive / f"Shared_Gc_activity_20261029/qc/{site}.json").read_text()
        )["source_DD"]
        daily = calculate_site(site, heights.loc[site], daily_file, metadata)
        daily.to_csv(output / f"{site}.csv")
