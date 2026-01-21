import pandas as pd
import numpy as np
from geopy.distance import geodesic
import logging
from typing import Tuple, Optional

# --------------------------
# Configuration (Paper-Aligned)
# --------------------------
# Paths
RAW_DATA_PATH = "D:/weixian_paper_cuttings_raw_data.csv"
CLEANED_DATA_PATH = "D:/cleaned_cultural_field_data.csv"
CORE_DATA_PATH = "D:/core_area_data.csv"
PERIPHERAL_DATA_PATH = "D:/peripheral_area_data.csv"

# Core area definition (Weixian Paper Cuttings Origin: Section 3.1)
CORE_COORDS = (39.8365, 114.5732)  # (Latitude, Longitude)
CORE_RADIUS_KM = 217  # From Paper Table 4 (core boundary radius)

# Valid coordinate ranges (China's geographic bounds)
MIN_LONGITUDE = 73.4047
MAX_LONGITUDE = 135.0853
MIN_LATITUDE = 3.8667
MAX_LATITUDE = 53.5500

# Subfield normalization parameters
SUBFIELD_MAX_VALUES = {
    "inheritance_value": 10.0,
    "media_value": 8.0,
    "marketing_value": 9.0,
    "education_value": 7.0,
    "academic_value": 6.0,
    "exhibition_value": 5.0
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("D:/preprocessing_log.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# --------------------------
# Helper Functions (Paper Section 2.1)
# --------------------------
def validate_coordinates(longitude: float, latitude: float) -> bool:
    """Validate if coordinates are within China's geographic bounds (Paper Data Quality Control)."""
    if not (MIN_LONGITUDE <= longitude <= MAX_LONGITUDE):
        logger.warning(f"Invalid longitude: {longitude}")
        return False
    if not (MIN_LATITUDE <= latitude <= MAX_LATITUDE):
        logger.warning(f"Invalid latitude: {latitude}")
        return False
    return True


def calculate_distance_from_core(latitude: float, longitude: float) -> Optional[float]:
    """Calculate distance from Weixian core using geodesic distance (Paper Section 2.2)."""
    try:
        point_coords = (latitude, longitude)
        distance_km = geodesic(CORE_COORDS, point_coords).km
        return round(distance_km, 2)
    except Exception as e:
        logger.error(f"Failed to calculate distance: {e}")
        return None


def clean_subfield_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize subfield values (Paper Data Cleaning Rules)."""
    # Fill missing values with 0 (valid for non-existent subfield data)
    df[["inheritance_value", "media_value", "marketing_value",
        "education_value", "academic_value", "exhibition_value"]] = df[
        ["inheritance_value", "media_value", "marketing_value",
         "education_value", "academic_value", "exhibition_value"]].fillna(0)

    # Normalize each subfield to 0-1 scale (Paper Feature Engineering)
    for subfield, max_val in SUBFIELD_MAX_VALUES.items():
        df[f"{subfield}_normalized"] = df[subfield] / max_val
        df[f"{subfield}_normalized"] = df[f"{subfield}_normalized"].clip(0, 1)  # Ensure no values exceed 1

    logger.info("Subfield values cleaned and normalized.")
    return df


def generate_grid_id(latitude: float, longitude: float, grid_size_deg: float = 0.5) -> str:
    """Generate grid ID using degree-based grid system (Paper Grid Division)."""
    # Align coordinates to grid centers
    grid_longitude = round(longitude / grid_size_deg) * grid_size_deg
    grid_latitude = round(latitude / grid_size_deg) * grid_size_deg
    return f"G_{grid_longitude:.2f}_{grid_latitude:.2f}"


# --------------------------
# Main Preprocessing Pipeline (Paper Section 2.1)
# --------------------------
def preprocess_cultural_field_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full preprocessing pipeline: load → validate → clean → split (Paper Data Engineering System)."""
    # Step 1: Load raw data
    logger.info("Loading raw dataset...")
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH)
        logger.info(f"Loaded raw data: {len(raw_df)} records")
    except FileNotFoundError:
        logger.error(f"Raw data not found at {RAW_DATA_PATH}")
        raise
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        raise

    # Step 2: Validate coordinates
    logger.info("Validating coordinates...")
    valid_coords_mask = raw_df.apply(
        lambda row: validate_coordinates(row["longitude"], row["latitude"]), axis=1
    )
    df = raw_df[valid_coords_mask].copy()
    logger.info(f"Valid coordinates: {len(df)} records (removed {len(raw_df) - len(df)} invalid)")

    # Step3: Calculate distance from core
    logger.info("Calculating distance from Weixian core...")
    df["distance_from_core_km"] = df.apply(
        lambda row: calculate_distance_from_core(row["latitude"], row["longitude"]), axis=1
    )
    df = df.dropna(subset=["distance_from_core_km"])
    logger.info(f"Distance calculated: {len(df)} records (removed {len(raw_df) - len(df)} invalid)")

    # Step4: Clean subfield values
    logger.info("Cleaning subfield values...")
    df = clean_subfield_values(df)

    # Step5: Split into core and peripheral areas
    logger.info("Splitting into core and peripheral areas...")
    core_mask = df["distance_from_core_km"] <= CORE_RADIUS_KM
    core_df = df[core_mask].copy()
    peripheral_df = df[~core_mask].copy()
    logger.info(f"Core area: {len(core_df)} records | Peripheral area: {len(peripheral_df)} records")

    # Step6: Generate grid IDs for boundary analysis
    logger.info("Generating grid IDs...")
    df["grid_id"] = df.apply(
        lambda row: generate_grid_id(row["latitude"], row["longitude"]), axis=1
    )

    # Step7: Add core area flag
    df["is_core_area"] = core_mask.astype(int)

    # Step8: Save cleaned data
    logger.info(f"Saving cleaned data to {CLEANED_DATA_PATH}...")
    df.to_csv(CLEANED_DATA_PATH, index=False)
    core_df.to_csv(CORE_DATA_PATH, index=False)
    peripheral_df.to_csv(PERIPHERAL_DATA_PATH, index=False)

    logger.info("Preprocessing completed successfully!")
    return core_df, peripheral_df


# --------------------------
# Run Preprocessing
# --------------------------
if __name__ == "__main__":
    logger.info("Starting data preprocessing pipeline...")
    try:
        core_data, peripheral_data = preprocess_cultural_field_data()

        # Print summary statistics
        logger.info("\n=== Preprocessing Summary ===")
        logger.info(f"Total cleaned records: {len(pd.read_csv(CLEANED_DATA_PATH))}")
        logger.info(f"Core area records: {len(core_data)}")
        logger.info(f"Peripheral area records: {len(peripheral_data)}")
        logger.info(
            f"Core area subfield mean: {core_data[['inheritance_value', 'media_value', 'marketing_value']].mean().to_dict()}")
        logger.info(
            f"Peripheral area subfield mean: {peripheral_data[['inheritance_value', 'media_value', 'marketing_value']].mean().to_dict()}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
