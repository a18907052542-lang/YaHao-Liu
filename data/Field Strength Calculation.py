import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

# --------------------------
# Configuration (Paper-Aligned)
# --------------------------
# Paths
CLEANED_DATA_PATH = "D:/cleaned_cultural_field_data.csv"
FIELD_STRENGTH_PATH = "D:/cultural_field_strength.csv"
GRID_STATISTICS_PATH = "D:/grid_field_statistics.csv"

# Paper's Field Strength Parameters (Section2.2)
SUBFIELD_WEIGHTS = {
    "inheritance_value_normalized": 0.3,  # Highest weight (core cultural gene)
    "media_value_normalized": 0.2,
    "marketing_value_normalized": 0.25,
    "education_value_normalized": 0.1,
    "academic_value_normalized": 0.1,
    "exhibition_value_normalized": 0.05
}
DECAY_HALF_LIFE = 217  # km (core radius, Paper Formula5)
MAX_FIELD_STRENGTH = 10.0  # Normalized to 0-10 scale

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("D:/field_strength_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# --------------------------
# Helper Functions (Paper Section2.2)
# --------------------------
def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned data from Part1 with validation."""
    try:
        df = pd.read_csv(CLEANED_DATA_PATH)
        logger.info(f"Loaded cleaned data: {len(df)} records")
        return df
    except FileNotFoundError:
        logger.error(f"Cleaned data not found at {CLEANED_DATA_PATH}. Run Part1 first!")
        raise
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise


def validate_subfields(df: pd.DataFrame) -> bool:
    """Validate all required subfields are present (Paper Data Quality Control)."""
    required_subfields = list(SUBFIELD_WEIGHTS.keys())
    missing_subfields = [sf for sf in required_subfields if sf not in df.columns]
    if missing_subfields:
        logger.error(f"Missing required subfields: {missing_subfields}")
        return False
    # Check for invalid values (0-1 range)
    for sf in required_subfields:
        if not (df[sf].between(0, 1).all()):
            logger.error(f"Invalid values in {sf} (must be 0-1)")
            return False
    logger.info("All subfields validated successfully.")
    return True


def compute_decay_kernel(distance_km: float, half_life: float = DECAY_HALF_LIFE) -> float:
    """Compute dynamic decay kernel (Paper Formula5: K(d) = exp(-d/λ))."""
    if distance_km < 0:
        return 1.0  # Core area (distance=0) has no decay
    return np.exp(-distance_km / half_life)


def compute_weighted_subfields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted subfield values with decay (Paper Formula6)."""
    # Add decay kernel to each record
    df["decay_kernel"] = df["distance_from_core_km"].apply(compute_decay_kernel)

    # Compute weighted subfields (subfield * weight * decay)
    for subfield, weight in SUBFIELD_WEIGHTS.items():
        df[f"{subfield}_weighted"] = df[subfield] * weight * df["decay_kernel"]

    logger.info("Weighted subfields computed with decay.")
    return df


def aggregate_field_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weighted subfields to get overall field strength (Paper Formula3-4)."""
    # Sum weighted subfields to get raw field strength
    weighted_cols = [f"{sf}_weighted" for sf in SUBFIELD_WEIGHTS.keys()]
    df["raw_field_strength"] = df[weighted_cols].sum(axis=1)

    # Normalize to 0-10 scale (Paper's standard)
    df["field_strength"] = df["raw_field_strength"] / df["raw_field_strength"].max() * MAX_FIELD_STRENGTH
    df["field_strength"] = df["field_strength"].clip(0, MAX_FIELD_STRENGTH)  # Ensure no values exceed 10

    # Add field strength category (Paper's classification)
    df["strength_category"] = pd.cut(
        df["field_strength"],
        bins=[0, 2, 5, 8, 10],
        labels=["Very Low", "Low", "Medium", "High"]
    )

    logger.info("Overall field strength aggregated successfully.")
    return df


def compute_grid_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute grid-level field strength statistics (Paper Grid Analysis)."""
    grid_stats = df.groupby("grid_id").agg({
        "field_strength": ["mean", "max", "min", "std"],
        "distance_from_core_km": "mean",
        "is_core_area": "sum",
        "record_id": "count"
    }).reset_index()

    # Flatten column names
    grid_stats.columns = ["grid_id", "mean_field_strength", "max_field_strength",
                          "min_field_strength", "std_field_strength", "mean_distance_km",
                          "core_area_records", "total_records"]

    # Add grid center coordinates
    grid_stats[["grid_longitude", "grid_latitude"]] = grid_stats["grid_id"].str.split("_", expand=True).iloc[
        :, 1:3].astype(float)

    logger.info(f"Grid statistics computed for {len(grid_stats)} grids.")
    return grid_stats


def save_field_strength_data(df: pd.DataFrame, grid_stats: pd.DataFrame) -> None:
    """Save field strength data to D drive."""
    # Save individual records
    df.to_csv(FIELD_STRENGTH_PATH, index=False)
    # Save grid statistics
    grid_stats.to_csv(GRID_STATISTICS_PATH, index=False)
    logger.info(f"Field strength data saved to {FIELD_STRENGTH_PATH}")
    logger.info(f"Grid statistics saved to {GRID_STATISTICS_PATH}")


# --------------------------
# Main Field Strength Calculation Pipeline (Paper Section2.2)
# --------------------------
def calculate_cultural_field_strength() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: load → validate → compute → aggregate → save."""
    logger.info("Starting field strength calculation pipeline...")

    # Step1: Load cleaned data
    df = load_cleaned_data()

    # Step2: Validate subfields
    if not validate_subfields(df):
        logger.error("Subfield validation failed. Aborting.")
        raise ValueError("Invalid subfield data.")

    # Step3: Compute weighted subfields with decay
    df = compute_weighted_subfields(df)

    # Step4: Aggregate to get overall field strength
    df = aggregate_field_strength(df)

    # Step5: Compute grid statistics
    grid_stats = compute_grid_statistics(df)

    # Step6: Save results
    save_field_strength_data(df, grid_stats)

    # Step7: Print summary
    logger.info("\n=== Field Strength Summary ===")
    logger.info(f"Total records processed: {len(df)}")
    logger.info(f"Average field strength: {df['field_strength'].mean():.2f}")
    logger.info(f"Max field strength: {df['field_strength'].max():.2f} (Core Area)")
    logger.info(f"Min field strength: {df['field_strength'].min():.2f} (Peripheral Area)")
    logger.info(f"Grid count: {len(grid_stats)}")

    logger.info("Field strength calculation completed successfully!")
    return df, grid_stats


# --------------------------
# Run Pipeline
# --------------------------
if __name__ == "__main__":
    try:
        field_strength_data, grid_statistics = calculate_cultural_field_strength()
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
