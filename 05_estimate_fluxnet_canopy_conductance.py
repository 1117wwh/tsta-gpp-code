"""Estimate daytime canopy conductance (Gc, m/s) from FLUXNET half-hourly records and average eligible intervals by site and day."""

from pathlib import Path, PureWindowsPath
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc


PATHS_FILE = Path(__file__).resolve().parent / "05_data_paths.txt"

CP = 1004.67
RD = 287.05
EPSILON = 0.622
LAMBDA_V = 2.45e6
KARMAN = 0.40

COMMON_COLUMNS = [
    "TIMESTAMP_START",
    "TIMESTAMP_END",
    "GPP_NT_VUT_REF",
    "TA_F_MDS",
    "VPD_F_MDS",
    "PA",
    "PA_F",
    "WS",
    "WS_F",
    "USTAR",
    "P",
    "P_F",
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
    """Use the fallback only where the preferred field is missing."""
    a = field_numeric(frame, preferred)
    b = field_numeric(frame, fallback)
    use_a = a.notna()
    return a.fillna(b), pd.Series(
        np.where(use_a, preferred, np.where(b.notna(), fallback, "missing")), index=frame.index
    )


def invert_penman_monteith(
    ta_c: pd.Series,
    vpd_hpa: pd.Series,
    pa_kpa: pd.Series,
    le: pd.Series,
    energy: pd.Series,
    ga_m_s: pd.Series,
) -> pd.DataFrame:
    """Invert Penman–Monteith for canopy conductance in m/s."""
    ta = ta_c.to_numpy(dtype=float)
    vpd_pa = vpd_hpa.to_numpy(dtype=float) * 100.0
    pressure_pa = pa_kpa.to_numpy(dtype=float) * 1000.0
    latent = le.to_numpy(dtype=float)
    available = energy.to_numpy(dtype=float)
    ga = ga_m_s.to_numpy(dtype=float)
    temp_k = ta + 273.15
    svp_kpa = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))
    svp_slope_pa_per_k = 4098.0 * svp_kpa / ((ta + 237.3) ** 2) * 1000.0
    gamma = CP * pressure_pa / (EPSILON * LAMBDA_V)
    rho = pressure_pa / (RD * temp_k)
    denominator = (
        svp_slope_pa_per_k * available
        + rho * CP * vpd_pa * ga
        - latent * (svp_slope_pa_per_k + gamma)
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gc_m_s = gamma * ga * latent / denominator
    return pd.DataFrame(
        {
            "pm_denominator": denominator,
            "gc_m_s": gc_m_s,
        },
        index=ta_c.index,
    )


def aerodynamic_conductance(wind_speed, measurement_height_m, canopy_height_m):
    """Use site canopy height and wind speed in the near-neutral logarithmic profile."""
    wind = wind_speed.to_numpy(dtype=float)
    h = canopy_height_m
    zm_m = measurement_height_m
    zd = 0.67 * h
    z0 = 0.10 * h
    geometry = np.isfinite(zm_m) & (zm_m > 0) & np.isfinite(h) & (h > 0) & (zm_m > zd + z0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ra = np.log((zm_m - zd) / z0) ** 2 / (wind * KARMAN**2)
        ga = 1.0 / ra
    geometry = geometry & np.isfinite(ra) & (ra > 0) & np.isfinite(ga) & (ga > 0)
    return pd.Series(ga, index=wind_speed.index), pd.Series(geometry, index=wind_speed.index)


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
        # Select the first tier with complete energy inputs before applying physical limits.
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
    use = [name for name in columns if name in header]
    types = {name: pa.int64() if name.startswith("TIMESTAMP") else pa.float64() for name in use}
    return pc.read_csv(
        path,
        read_options=pc.ReadOptions(use_threads=False),
        convert_options=pc.ConvertOptions(
            include_columns=use, column_types=types, null_values=["", "NA", "NaN", "nan"]
        ),
    ).to_pandas()


def screen_flux_records(site, flux_file):
    """Apply the quality flags for the pressure, wind and rain measurements used."""
    header = pd.read_csv(flux_file, nrows=0).columns
    extra_flags = [
        c for c in header if c.endswith("_QC") and c.startswith(("LW_", "PA", "WS", "P_", "USTAR"))
    ]
    fields = COMMON_COLUMNS + extra_flags
    frame = read_columns(flux_file, fields).replace(-9999, np.nan)
    if not frame.TIMESTAMP_START.is_unique:
        raise ValueError(f"Duplicate flux timestamps at {site}")
    if not (
        (parse_timestamp(frame.TIMESTAMP_END) - parse_timestamp(frame.TIMESTAMP_START))
        .eq(pd.Timedelta(minutes=30))
        .all()
    ):
        raise ValueError(f"Expected half-hourly flux intervals at {site}")
    frame["GPP_NT_VUT_REF"] = frame.GPP_NT_VUT_REF.where(frame.GPP_NT_VUT_REF.between(0, 30000))
    for name in ["P_F", "P"]:
        if name in frame:
            invalid = frame[name].lt(0)
            if name + "_QC" in frame:
                invalid |= frame[name + "_QC"].eq(3)
            frame[name] = frame[name].mask(invalid)
    extra_qc = pd.Series(True, index=frame.index)
    for a, b in [("PA_F", "PA"), ("WS_F", "WS")]:
        _, source = safe_fallback(frame, a, b)
        for name in [a, b]:
            if name + "_QC" in frame:
                extra_qc &= ~(source.eq(name) & frame[name + "_QC"].eq(3))
    if "USTAR_QC" in frame:
        extra_qc &= ~frame.USTAR_QC.eq(3)
    return frame, extra_qc


def surface_temperature(site, temperature_file, frame):
    """Align temperatures with valid emissivity and longwave fluxes to flux timestamps."""
    lst = pd.read_csv(temperature_file)
    if not lst.TIMESTAMP_START.is_unique:
        raise ValueError(f"Duplicate surface temperature timestamps at {site}")
    good = (
        np.isfinite(lst.LST)
        & lst.LST.gt(-273.15)
        & lst.BBE.gt(0)
        & lst.BBE.le(1)
        & lst.LW_IN.gt(0)
        & lst.LW_OUT.gt(0)
    )
    quality = frame.set_index("TIMESTAMP_START")
    for field in ["LW_IN_COL", "LW_OUT_COL"]:
        for name in lst[field].dropna().unique():
            if name + "_QC" in quality:
                good &= ~(
                    lst[field].eq(name) & lst.TIMESTAMP_START.map(quality[name + "_QC"]).eq(3)
                )
    spikes = {"CA-Cbo": 201410301100, "FR-LBr": 200412101100, "US-EML": 202107091030}
    good &= lst.TIMESTAMP_START.ne(spikes.get(site, -1))
    return frame.TIMESTAMP_START.map(lst.set_index("TIMESTAMP_START").LST.where(good.to_numpy()))


def flux_quality(frame, tier):
    """Apply quality flags only to fields used by the chosen energy tier."""
    valid = pd.Series(True, index=frame.index)
    for name in ("TA_F_MDS_QC", "VPD_F_MDS_QC"):
        valid &= ~field_numeric(frame, name).eq(3)
    uses_ground_heat = tier.isin(["T1_corrLE_RnG", "T2_rawLE_RnG"])
    uses_raw_le = tier.isin(["T2_rawLE_RnG", "T4_rawLE_rawLEplusH"])
    uses_raw_h = tier.eq("T4_rawLE_rawLEplusH")
    valid &= ~(uses_ground_heat & field_numeric(frame, "G_F_MDS_QC").eq(3))
    valid &= ~(uses_raw_le & field_numeric(frame, "LE_F_MDS_QC").eq(3))
    valid &= ~(uses_raw_h & field_numeric(frame, "H_F_MDS_QC").eq(3))
    return valid


def calculate_half_hours(site, height, activity_file, flux_file, temperature_file, shift):
    """Invert Penman–Monteith for each eligible half-hour on active growing-season days."""
    frame, meteorology_qc = screen_flux_records(site, flux_file)
    lst_c = surface_temperature(site, temperature_file, frame)
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

    quality = meteorology_qc & flux_quality(frame, tier)
    ga, geometry = aerodynamic_conductance(
        wind, height.wind_measurement_height_m, height.canopy_height_used_m
    )
    pm = invert_penman_monteith(ta, vpd, pa, le, energy, ga)
    delta_t_c = lst_c - ta
    # A daytime midpoint must follow 24 rain-free hours and satisfy the inversion limits.
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
        & quality
        & geometry
        & tier.ne("none")
        & le.gt(0)
        & energy.gt(50)
        & np.isfinite(pm["pm_denominator"])
        & pm["pm_denominator"].gt(0)
        & np.isfinite(pm["gc_m_s"])
        & pm["gc_m_s"].gt(0)
        & pm["gc_m_s"].le(0.05)
        & delta_t_c.notna()
        & gpp.notna()
        & np.isfinite(gpp)
    )
    # The archived activity flag uses daily GPP > 25% of the positive site-year GPP P95.
    activity = pd.read_csv(activity_file, usecols=["date", "activity_day"], dtype={"activity_day": bool})
    active_dates = activity.loc[activity["activity_day"], "date"]
    date = timestamp.dt.strftime("%Y-%m-%d")
    mask &= date.isin(active_dates)
    return pd.DataFrame(
        {
            "timestamp_start": frame["TIMESTAMP_START"],
            "date": date,
            "delta_t_c": delta_t_c,
            "gc_m_s": pm["gc_m_s"],
            "gpp_umol_m2_s": gpp,
            "tier": tier,
        }
    ).loc[mask].set_index("timestamp_start")


def daily_means(half_hours):
    """Keep days with at least four eligible local daytime intervals."""
    daily = half_hours.groupby("date").agg(
        delta_t_c=("delta_t_c", "mean"),
        gc_m_s=("gc_m_s", "mean"),
        gpp_umol_m2_s=("gpp_umol_m2_s", "mean"),
        n_halfhour=("tier", "size"),
        dominant_tier=("tier", lambda values: values.value_counts().index[0]),
    )
    return daily[daily.n_halfhour >= 4]


if __name__ == "__main__":
    paths = json.loads(PATHS_FILE.read_text(encoding="utf-8-sig"))
    STATION_COHORT_PATH = PATHS_FILE.parent / paths["STATION_COHORT_PATH"]
    STATION_HEIGHTS_PATH = PATHS_FILE.parent / paths["STATION_HEIGHTS_PATH"]
    HALF_HOURLY_METADATA_DIR = PATHS_FILE.parent / paths["HALF_HOURLY_METADATA_DIR"]
    STATION_ACTIVITY_DIR = PATHS_FILE.parent / paths["STATION_ACTIVITY_DIR"]
    HALF_HOURLY_STATIONS_DIR = PATHS_FILE.parent / paths["HALF_HOURLY_STATIONS_DIR"]
    OUTPUT_DIR = PATHS_FILE.parent / paths["OUTPUT_DIR"]

    heights = pd.read_csv(STATION_HEIGHTS_PATH).set_index("site")
    sites = pd.read_csv(STATION_COHORT_PATH, usecols=["site"]).site
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for site in sites:
        flux_metadata = json.loads((HALF_HOURLY_METADATA_DIR / f"{site}.json").read_text(encoding="utf-8-sig"))
        site_dir = HALF_HOURLY_STATIONS_DIR / site
        flux_file = site_dir / PureWindowsPath(flux_metadata["hh_source"]).name
        temperature_file = site_dir / PureWindowsPath(flux_metadata["lst_source"]).name
        half_hours = calculate_half_hours(
            site, heights.loc[site], STATION_ACTIVITY_DIR / f"{site}.csv", flux_file, temperature_file,
            flux_metadata["clock_to_LMST_hours"],
        )
        daily_means(half_hours).to_csv(OUTPUT_DIR / f"{site}.csv")
