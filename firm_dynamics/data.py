from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = "ACRA_w_SO.csv"
STRUCK_OFF_COLUMNS = [f"Struck Off Date {i}" for i in range(1, 6)]


def prepare_df(
    path=DEFAULT_DATA_PATH,
    *,
    drop_invalid_coordinates=True,
    include_region=False,
    region_path="region_boundary.geojson",
):
    """Load ACRA-derived firm data and return columns used by survival analysis."""
    df = pd.read_csv(path)

    if drop_invalid_coordinates:
        df = df[(df["Coordinate_X"] != -1) & (df["Coordinate_Y"] != -1)]

    df_analysis = pd.DataFrame(
        {
            "uen": df["uen"],
            "Sector": df["Sector"],
            "status": df["status"],
        }
    )

    struck_off_dates = df[STRUCK_OFF_COLUMNS].apply(pd.to_datetime, errors="coerce")
    df_analysis["Exit Date"] = struck_off_dates.mean(axis=1)
    df_analysis["Entry Date"] = pd.to_datetime(
        df["registration_incorporation_date"],
        errors="coerce",
    )

    if include_region:
        df_analysis["Region"] = _assign_regions(df, region_path)

    return df_analysis


def _assign_regions(df, region_path):
    """Assign Singapore planning regions from coordinates when geospatial deps exist."""
    import geopandas as gpd
    from shapely.geometry import Point

    regions = gpd.read_file(Path(region_path))
    regions = regions.to_crs("EPSG:3414")
    regions["Name"] = regions["Name"].map(
        {
            "kml_1": "WR",
            "kml_2": "NR",
            "kml_3": "NER",
            "kml_4": "ER",
            "kml_5": "CR",
        }
    )

    geometry = [Point(xy) for xy in zip(df["Coordinate_X"], df["Coordinate_Y"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:3414")
    gdf_with_region = gpd.sjoin(gdf, regions, how="left", predicate="within")
    return gdf_with_region["Name"].values
