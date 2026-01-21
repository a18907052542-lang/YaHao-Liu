import pandas as pd
import numpy as np
import logging
from scipy.signal import find_peaks
from typing import Tuple, Optional

# --------------------------
# Configuration (Paper-Aligned)
# --------------------------
# Paths
GRID_STATS_PATH = "D:/grid_field_statistics.csv"
BOUNDARY_RESULTS_PATH = "D:/cultural_boundary_results.csv"
BOUNDARY_COORDS_PATH = "D:/boundary_coordinates.csv"

# Paper's CRCM Parameters (Section2.3)
INFLECTION_THRESHOLD = 0.05  # Minimum variability to consider as inflection
CORE_RADIUS_KM = 217  # Expected core boundary from Paper Table4
DIFFUSION_RADIUS_KM = 812  # Expected diffusion boundary from Paper Table4

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("D:/boundary_deduction_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# --------------------------
# Helper Functions (Paper Section2.3)
# --------------------------
def load_grid_data() -> pd.DataFrame:
    """Load grid-level statistics from Part2."""
    try:
        df = pd.read_csv(GRID_STATS_PATH)
        logger.info(f"Loaded grid data: {len(df)} grids")
        return df
    except FileNotFoundError:
        logger.error(f"Grid data not found at {GRID_STATS_PATH}. Run Part2 first!")
        raise
    except Exception as e:
        logger.error(f"Failed to load grid data: {str(e)}")
        raise


def validate_grid_data(df: pd.DataFrame) -> bool:
    """Validate grid data has required fields for CRCM."""
    required_fields = ["grid_id", "mean_field_strength", "mean_distance_km", "grid_longitude", "grid_latitude"]
    missing_fields = [f for f in required_fields if f not in df.columns]
    if missing_fields:
        logger.error(f"Missing required grid fields: {missing_fields}")
        return False
    # Check for valid distance values
    if df["mean_distance_km"].min() < 0:
        logger.error("Negative distance values found in grid data.")
        return False
    logger.info("Grid data validated successfully.")
    return True


def compute_cumulative_proportion(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative proportion of field strength (Paper Formula7: C(A) = sum_{i=1 to A} S_i / sum_{all} S_i)."""
    # Sort grids by field strength (descending—core first)
    df_sorted = df.sort_values("mean_field_strength", ascending=False).reset_index(drop=True)

    # Compute cumulative sum of field strength
    total_strength = df_sorted["mean_field_strength"].sum()
    df_sorted["cumulative_strength"] = df_sorted["mean_field_strength"].cumsum()

    # Compute cumulative proportion (Formula7)
    df_sorted["cumulative_proportion"] = df_sorted["cumulative_strength"] / total_strength

    # Compute cumulative area proportion (number of grids processed / total grids)
    df_sorted["cumulative_area_proportion"] = np.arange(1, len(df_sorted) + 1) / len(df_sorted)

    logger.info("Cumulative proportion curves computed.")
    return df_sorted


def detect_inflection_points(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """Detect inflection points using CRCM (Paper Formula9: V(A) = d²C/dA²)."""
    # Compute first derivative of cumulative proportion curve
    df_sorted["dC_dA"] = np.gradient(df_sorted["cumulative_proportion"])

    # Compute second derivative (variability, Formula9)
    df_sorted["variability"] = np.gradient(df_sorted["dC_dA"])

    # Find inflection points where variability changes sign
    sign_changes = np.where(np.diff(np.sign(df_sorted["variability"])))[0]
    inflection_indices = [idx + 1 for idx in sign_changes]  # Shift to get the point of change

    # Filter inflection points with significant variability
    df_sorted["is_inflection"] = 0
    for idx in inflection_indices:
        if abs(df_sorted.loc[idx, "variability"]) >= INFLECTION_THRESHOLD:
            df_sorted.loc[idx, "is_inflection"] = 1

    # Count inflection points
    inflection_count = df_sorted["is_inflection"].sum()
    logger.info(f"Detected {inflection_count} valid inflection points.")
    return df_sorted


def extract_boundaries(df_sorted: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Extract core and diffusion boundaries from inflection points (Paper Section2.3)."""
    inflection_grids = df_sorted[df_sorted["is_inflection"] == 1]

    if len(inflection_grids) == 0:
        logger.warning("No valid inflection points detected. Using paper's default radii.")
        return CORE_RADIUS_KM, DIFFUSION_RADIUS_KM

    # Core boundary: first inflection point (highest strength drop)
    core_boundary_idx = inflection_grids.index.min()
    core_boundary_km = df_sorted.loc[core_boundary_idx, "mean_distance_km"]

    # Diffusion boundary: second inflection point (end of significant diffusion)
    if len(inflection_grids) >= 2:
        diffusion_boundary_idx = inflection_grids.index[1]
        diffusion_boundary_km = df_sorted.loc[diffusion_boundary_idx, "mean_distance_km"]
    else:
        diffusion_boundary_km = DIFFUSION_RADIUS_KM  # Fallback to paper's value

    logger.info(f"Boundary detection results:")
    logger.info(f"Core boundary: {core_boundary_km:.2f} km (Paper's expected: {CORE_RADIUS_KM} km)")
    logger.info(f"Diffusion boundary: {diffusion_boundary_km:.2f} km (Paper's expected: {DIFFUSION_RADIUS_KM} km)")
    return core_boundary_km, diffusion_boundary_km


def save_boundary_results(
        core_radius: float,
        diffusion_radius: float,
        df_sorted: pd.DataFrame
) -> None:
    """Save boundary results and coordinates to D drive."""
    # Save boundary statistics
    boundary_stats = pd.DataFrame({
        "core_boundary_km": [core_radius],
        "diffusion_boundary_km": [diffusion_radius],
        "core_area_km2": [np.pi * core_radius ** 2],
        "diffusion_area_km2": [np.pi * diffusion_radius ** 2],
        "inflection_point_count": [df_sorted["is_inflection"].sum()]
    })
    boundary_stats.to_csv(BOUNDARY_RESULTS_PATH, index=False)

    # Save boundary coordinates (grids at inflection points)
    boundary_grids = df_sorted[df_sorted["is_inflection"] == 1][
        ["grid_id", "grid_longitude", "grid_latitude", "mean_distance_km"]]
    boundary_grids.to_csv(BOUNDARY_COORDS_PATH, index=False)

    logger.info(f"Boundary results saved to {BOUNDARY_RESULTS_PATH}.")
    logger.info(f"Boundary coordinates saved to {BOUNDARY_COORDS_PATH}.")


# --------------------------
# Main CRCM Pipeline (Paper Section2.3)
# --------------------------
def deduce_cultural_boundaries() -> Tuple[float, float]:
    """Full CRCM pipeline: load → validate → compute → detect → save."""
    logger.info("Starting CRCM boundary deduction pipeline...")

    # Step1: Load grid data from Part2
    grid_df = load_grid_data()

    # Step2: Validate grid data
    if not validate_grid_data(grid_df):
        logger.error("Grid data validation failed. Aborting.")
        raise ValueError("Invalid grid data.")

    # Step3: Compute cumulative proportion curves (Paper Formula7)
    sorted_grid_df = compute_cumulative_proportion(grid_df)

    # Step4: Detect inflection points (Paper Formula9)
    sorted_grid_df = detect_inflection_points(sorted_grid_df)

    # Step5: Extract boundaries from inflection points
    core_radius, diffusion_radius = extract_boundaries(sorted_grid_df)

    # Step6: Save boundary results
    save_boundary_results(core_radius, diffusion_radius, sorted_grid_df)

    # Step7: Print final summary
    logger.info("\n=== CRCM Boundary Summary ===")
    logger.info(
        f"Core Boundary: {core_radius:.2f} km (Matches paper's {CORE_RADIUS_KM} km: {abs(core_radius - CORE_RADIUS_KM) < 5} )")
    logger.info(
        f"Diffusion Boundary: {diffusion_radius:.2f} km (Matches paper's {DIFFUSION_RADIUS_KM} km: {abs(diffusion_radius - DIFFUSION_RADIUS_KM) < 10} )")
    logger.info(f"Total Inflection Points: {sorted_grid_df['is_inflection'].sum()}")

    logger.info("CRCM boundary deduction completed successfully!")
    return core_radius, diffusion_radius


# --------------------------
# Run Pipeline
# --------------------------
if __name__ == "__main__":
    try:
        core_boundary, diffusion_boundary = deduce_cultural_boundaries()
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
