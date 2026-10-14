"""Calculate canopy conductance from half-hourly flux-tower observations."""

from pathlib import Path
import json
import numpy as np
import pandas as pd


def inverse_penman_monteith(Ta, vpd_hpa, pressure_kpa, LE, A, wind, wind_height, canopy_height):
    """Invert Penman-Monteith; conductance is returned in m/s and mol/m2/s."""
    displacement = 0.67 * canopy_height
    roughness = 0.10 * canopy_height
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ra = np.log((wind_height - displacement) / roughness) ** 2 / (wind * KARMAN**2)
        ga = 1 / ra
    # Convert temperature to K and both vapour deficit and pressure to Pa.
    temperature_k = Ta + 273.15
    pressure_pa = pressure_kpa * 1000
    vpd_pa = vpd_hpa * 100
    saturation_kpa = 0.6108 * np.exp(17.27 * Ta / (Ta + 237.3))
    # delta: saturation-pressure slope; gamma: psychrometric constant (Pa/K).
    delta = 4098 * saturation_kpa / (Ta + 237.3) ** 2 * 1000
    gamma = CP * pressure_pa / (EPSILON * LATENT_HEAT)
    rho_a = pressure_pa / (RD * temperature_k)
    denominator = delta * A + rho_a * CP * vpd_pa * ga - LE * (delta + gamma)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Gc = gamma * ga * LE / denominator
        Gc_molar = Gc * pressure_pa / (GAS_CONSTANT * temperature_k)
    return Gc, Gc_molar, ra, denominator


def resolve_canopy_height(frame, timestamp, dry_conditions, wind_height, bif_height):
    Ta = numeric_field(frame, "TA_F_MDS")
    pressure = preferred_field(frame, "PA_F", "PA")
    wind = preferred_field(frame, "WS_F", "WS")
    u_star = numeric_field(frame, "USTAR")
    sensible_heat_flux = preferred_field(frame, "H_CORR", "H_F_MDS")
    temperature_k = Ta + 273.15
    rho_a = pressure * 1000 / (RD * temperature_k)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        obukhov_length = -(rho_a * CP * temperature_k * u_star**3) / (KARMAN * GRAVITY * sensible_heat_flux)
        stability = wind_height / obukhov_length
        candidate_height = wind_height / (0.6 + 0.1 * np.exp(np.clip(KARMAN * wind / u_star, -700, 700)))
    neutral = (
        timestamp.notna()
        & Ta.notna()
        & pressure.gt(0)
        & wind.gt(0.1)
        & u_star.gt(0.1)
        & sensible_heat_flux.notna()
        & dry_conditions
        & np.isfinite(obukhov_length)
        & np.isfinite(stability)
        & stability.abs().le(0.1)
        & np.isfinite(candidate_height)
        & candidate_height.gt(0.01)
        & candidate_height.lt(wind_height / 0.77)
        & quality_mask(frame, ["TA_F_MDS_QC", "H_F_MDS_QC"])
    )
    candidates = candidate_height[neutral].to_numpy()
    if len(candidates):
        lower, upper = np.quantile(candidates, [0.01, 0.99])
        candidates = candidates[(candidates >= lower) & (candidates <= upper)]
    if len(candidates) >= MINIMUM_NEUTRAL_RECORDS:
        height = float(np.median(candidates))
        if wind_height > 0.77 * height:
            return height, "Fu_neutral_site_median_Eq5", len(candidates)
    if np.isfinite(bif_height) and bif_height > 0 and wind_height > 0.77 * bif_height:
        return bif_height, "BIF_HEIGHTC_reasonable_fallback", len(candidates)
    return np.nan, "missing_or_invalid_height", len(candidates)


def select_energy(frame):
    corrected_le = numeric_field(frame, "LE_CORR")
    raw_le = numeric_field(frame, "LE_F_MDS")
    corrected_h = numeric_field(frame, "H_CORR")
    raw_h = numeric_field(frame, "H_F_MDS")
    net_energy = numeric_field(frame, "NETRAD") - numeric_field(frame, "G_F_MDS")
    candidates = [
        ("T1_corrLE_RnG", corrected_le, net_energy),
        ("T2_rawLE_RnG", raw_le, net_energy),
        ("T3_corrLE_corrLEplusH", corrected_le, corrected_le + corrected_h),
        ("T4_rawLE_rawLEplusH", raw_le, raw_le + raw_h),
    ]
    LE = pd.Series(np.nan, index=frame.index)
    available_energy = LE.copy()
    energy_source = pd.Series("none", index=frame.index)
    for label, candidate_le, candidate_energy in candidates:
        use = energy_source.eq("none") & candidate_le.notna() & candidate_energy.notna()
        LE[use] = candidate_le[use]
        available_energy[use] = candidate_energy[use]
        energy_source[use] = label
    return LE, available_energy, energy_source


def dry_window(frame, timestamp):
    rain = preferred_field(frame, "P_F", "P")
    rain[rain < 0] = np.nan
    count = rain.notna().rolling(49, min_periods=49).sum()
    total = rain.rolling(49, min_periods=49).sum()
    # Use 49 consecutive half-hour records, including the current record.
    continuous = timestamp.diff().eq(pd.Timedelta(minutes=30)).rolling(48, min_periods=48).sum()
    return count.eq(49) & total.eq(0) & continuous.eq(48)


def quality_mask(frame, columns, selected=None):
    valid = pd.Series(True, index=frame.index)
    if selected is None:
        selected = pd.Series(True, index=frame.index)
    for column in columns:
        if column in frame:
            flag = numeric_field(frame, column)
            if flag.notna().any():
                valid &= ~selected | (flag.notna() & flag.le(2))
    return valid


def numeric_field(frame, name):
    if name not in frame:
        return pd.Series(np.nan, index=frame.index)
    values = pd.to_numeric(frame[name], errors="coerce")
    return values.mask((values <= -9990) | ~np.isfinite(values))


def preferred_field(frame, preferred, fallback):
    return numeric_field(frame, preferred).fillna(numeric_field(frame, fallback))


def parse_timestamp(values):
    text = values.astype(str)
    text = text.str.removesuffix(".0")
    text = text.str.zfill(12)
    return pd.to_datetime(text, format="%Y%m%d%H%M", errors="coerce")


def most_common_value(values):
    counts = values.value_counts()
    return counts.index[0]


def aggregate_daily(halfhour, site):
    daily = halfhour.groupby("date", sort=True).agg(
            TSS_c=("TSS_c", "mean"),
            Gc_m_s=("Gc_m_s", "mean"),
            Gc_mol_m2_s=("Gc_mol_m2_s", "mean"),
            GPP_umol_m2_s=("GPP_umol_m2_s", "mean"),
            n_halfhour=("energy_tier", "size"),
            dominant_tier=("energy_tier", most_common_value),
    )
    daily = daily.reset_index()
    daily = daily.loc[daily.n_halfhour.ge(MINIMUM_DAILY_RECORDS)].copy()
    daily.insert(0, "site_id", site.site_id)
    daily["AI"] = site.AI
    daily["vegetation_class"] = site.vegetation_class
    if len(daily):
        thresholds = np.quantile(daily.TSS_c, [0.10, 0.25, 0.50, 0.75, 0.90])
        daily["TSS_percentile_bin"] = np.asarray(BIN_LABELS)[
            np.searchsorted(thresholds, daily.TSS_c, side="right")
        ]
    else:
        daily["TSS_percentile_bin"] = pd.Series(dtype=str)
    return daily



def process_site(site):
    print(f"Reading complete half-hour record: {site.site_id}", flush=True)
    available_columns = pd.read_csv(site.fluxmet_file, nrows=0).columns
    selected_columns = []
    for column in available_columns:
        if column in FLUX_COLUMNS:
            selected_columns.append(column)
    frame = pd.read_csv(
        site.fluxmet_file,
        usecols=selected_columns,
        dtype={"TIMESTAMP_START": str},
        low_memory=False,
    )
    timestamp = parse_timestamp(frame["TIMESTAMP_START"])
    if timestamp.isna().any() or timestamp.duplicated().any() or not timestamp.is_monotonic_increasing:
        raise ValueError(f"Invalid, duplicate or unordered timestamps: {site.site_id}")
    Ta = numeric_field(frame, "TA_F_MDS")
    vpd = numeric_field(frame, "VPD_F_MDS")
    pressure = preferred_field(frame, "PA_F", "PA")
    wind = preferred_field(frame, "WS_F", "WS")
    u_star = numeric_field(frame, "USTAR")
    gpp = numeric_field(frame, "GPP_NT_VUT_REF")
    dry_conditions = dry_window(frame, timestamp)
    canopy_height, height_source, neutral_count = resolve_canopy_height(
        frame, timestamp, dry_conditions, site.wind_height_m, site.canopy_height_bif_m
    )
    LE, A, energy_source = select_energy(frame)
    valid_qc = quality_mask(frame, ["TA_F_MDS_QC", "VPD_F_MDS_QC"])
    valid_qc &= quality_mask(frame, ["G_F_MDS_QC"], energy_source.isin(["T1_corrLE_RnG", "T2_rawLE_RnG"]))
    valid_qc &= quality_mask(frame, ["LE_F_MDS_QC", "H_F_MDS_QC"], energy_source.eq("T4_rawLE_rawLEplusH"))
    Gc, Gc_molar, ra, denominator = inverse_penman_monteith(
        Ta, vpd, pressure, LE, A, wind, site.wind_height_m, canopy_height
    )
    lst = pd.read_csv(site.lst_file, usecols=["TIMESTAMP_START", "LST"], dtype={"TIMESTAMP_START": str})
    lst["timestamp"] = parse_timestamp(lst["TIMESTAMP_START"])
    lst["lst_c"] = numeric_field(lst, "LST")
    lookup = lst.drop_duplicates("timestamp", keep="last").set_index("timestamp")["lst_c"]
    tss = timestamp.map(lookup) - Ta
    minute = timestamp.dt.hour * 60 + timestamp.dt.minute
    valid_geometry = np.isfinite(canopy_height) and canopy_height > 0 and site.wind_height_m > 0.77 * canopy_height
    midday = (
        minute.ge(TIME_START_MINUTE)
        & minute.lt(TIME_END_MINUTE)
    )
    valid_meteorology = (
        Ta.gt(MINIMUM_AIR_TEMPERATURE)
        & vpd.gt(0)
        & pressure.gt(0)
        & wind.gt(0.1)
        & u_star.gt(0.1)
    )
    valid_energy = (
        energy_source.ne("none")
        & LE.gt(0)
        & A.gt(50)
    )
    valid_conductance = (
        np.isfinite(ra)
        & ra.gt(0)
        & np.isfinite(denominator)
        & denominator.gt(0)
        & np.isfinite(Gc)
        & Gc.gt(0)
        & Gc.le(MAXIMUM_CONDUCTANCE)
    )
    valid = (
        midday & valid_meteorology & dry_conditions & valid_qc
        & valid_geometry & valid_energy & valid_conductance
        & np.isfinite(tss) & np.isfinite(gpp)
    )
    halfhour = pd.DataFrame(
        {
            "timestamp": timestamp,
            "date": timestamp.dt.strftime("%Y-%m-%d"),
            "TSS_c": tss,
            "Gc_m_s": Gc,
            "Gc_mol_m2_s": Gc_molar,
            "GPP_umol_m2_s": gpp,
            "Ta_c": Ta,
            "VPD_hpa": vpd,
            "LE_W_m2": LE,
            "energy_W_m2": A,
            "ra_s_m": ra,
            "energy_tier": energy_source,
        }
    )
    halfhour = halfhour.loc[valid].copy()
    halfhour.insert(0, "site_id", site.site_id)
    daily = aggregate_daily(halfhour, site)
    coverage = {
        "site_id": site.site_id,
        "n_input_halfhours": len(frame),
        "n_valid_halfhours": len(halfhour),
        "n_valid_days": len(daily),
        "wind_height_m": site.wind_height_m,
        "canopy_height_m": canopy_height,
        "height_source": height_source,
        "neutral_records": neutral_count,
        "n_low_days": int(daily.TSS_percentile_bin.eq("P0-10").sum()),
        "n_high_days": int(daily.TSS_percentile_bin.eq("P90-100").sum()),
    }
    tiers = [
        {"site_id": site.site_id, "energy_tier": label, "n_halfhour": int(count)}
        for label, count in halfhour.energy_tier.value_counts().items()
    ]
    return halfhour, daily, coverage, tiers


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not OVERWRITE:
        raise FileExistsError(f"Results already exist: {OUTPUT_DIR}")
    sites = pd.read_csv(SITE_FILE)
    required = [
        "site_id",
        "fluxmet_file",
        "lst_file",
        "wind_height_m",
        "canopy_height_bif_m",
        "AI",
        "vegetation_class",
    ]
    if not set(required).issubset(sites.columns) or sites.site_id.duplicated().any():
        raise ValueError("Site metadata must contain the documented columns and unique site IDs")
    for site in sites.itertuples(index=False):
        for path in [site.fluxmet_file, site.lst_file]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    (OUTPUT_DIR / "halfhour").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "daily").mkdir(exist_ok=True)
    daily_tables, coverage_rows, tier_rows = [], [], []
    for index, site in enumerate(sites.itertuples(index=False), 1):
        halfhour, daily, coverage, tiers = process_site(site)
        halfhour.to_csv(
            OUTPUT_DIR / "halfhour" / f"{site.site_id}_Gc_halfhour.csv", index=False, float_format="%.17g"
        )
        daily.to_csv(OUTPUT_DIR / "daily" / f"{site.site_id}_Gc_daily.csv", index=False, float_format="%.17g")
        daily_tables.append(daily)
        coverage_rows.append(coverage)
        tier_rows.extend(tiers)
        print(
            f"{index}/{len(sites)}: {site.site_id}, {len(halfhour)} half-hours, {len(daily)} days", flush=True
        )
    daily = pd.concat(daily_tables, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    coverage["extreme_eligible"] = coverage.n_low_days.ge(3) & coverage.n_high_days.ge(3)
    tiers = pd.DataFrame(tier_rows)
    daily.to_csv(OUTPUT_DIR / "daily_Gc_all_sites.csv", index=False, float_format="%.17g")
    coverage.to_csv(OUTPUT_DIR / "site_coverage_and_heights.csv", index=False, float_format="%.17g")
    tiers.to_csv(OUTPUT_DIR / "energy_tier_counts.csv", index=False)
    summary = {
        "candidate_sites": len(sites),
        "sites_with_halfhours": int(coverage.n_valid_halfhours.gt(0).sum()),
        "sites_with_daily_values": int(coverage.n_valid_days.gt(0).sum()),
        "valid_halfhours": int(coverage.n_valid_halfhours.sum()),
        "site_days": len(daily),
        "extreme_eligible_sites": int(coverage.extreme_eligible.sum()),
        "halfhours_by_energy_tier": tiers.groupby("energy_tier").n_halfhour.sum().astype(int).to_dict(),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

# Input paths, physical constants and run settings
CODE_DIR = Path(__file__).resolve().parent
SITE_FILE = CODE_DIR / "inputs/05_canopy_conductance/site_inputs.csv"
OUTPUT_DIR = CODE_DIR / "outputs/05_canopy_conductance"
OVERWRITE = False
TIME_START_MINUTE = 600
TIME_END_MINUTE = 840
MINIMUM_DAILY_RECORDS = 4
MINIMUM_NEUTRAL_RECORDS = 20
MINIMUM_AIR_TEMPERATURE = 15
MAXIMUM_CONDUCTANCE = 0.05
CP = 1004.67
RD = 287.05
GAS_CONSTANT = 8.314462618
EPSILON = 0.622
LATENT_HEAT = 2.45e6
KARMAN = 0.40
GRAVITY = 9.80665
FLUX_COLUMNS = [
    "TIMESTAMP_START",
    "TA_F_MDS",
    "VPD_F_MDS",
    "PA_F",
    "PA",
    "WS_F",
    "WS",
    "USTAR",
    "P_F",
    "P",
    "NETRAD",
    "G_F_MDS",
    "LE_CORR",
    "LE_F_MDS",
    "H_CORR",
    "H_F_MDS",
    "GPP_NT_VUT_REF",
    "TA_F_MDS_QC",
    "VPD_F_MDS_QC",
    "G_F_MDS_QC",
    "LE_F_MDS_QC",
    "H_F_MDS_QC",
]
BIN_LABELS = ["P0-10", "P10-25", "P25-50", "P50-75", "P75-90", "P90-100"]

if __name__ == "__main__":
    main()
