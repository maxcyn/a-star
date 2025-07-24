import numpy as np
import surpyval as surv
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import datetime as dt

def prepare_df():
    '''
    Creates a df for survival analysis
    '''
    df = pd.read_csv('ACRA_w_SO.csv')

    # Drop rows where Coordinate_X or Coordinate_Y is -1
    df = df[(df['Coordinate_X'] != -1) & (df['Coordinate_Y'] != -1)]

    # Initialise new dataframe for data we want to analyse
    df_analysis = pd.DataFrame()
    df_analysis['uen'] = df['uen']
    df_analysis['Sector'] = df['Sector']
    df_analysis['status'] = df['status']

    # Convert the 5 struck off dates to datetime
    date_cols = [f'Struck Off Date {i}' for i in range(1, 6)]
    df[date_cols] = df[date_cols].apply(pd.to_datetime)

    # Calculate the average date for each firm
    df_analysis['Exit Date'] = df[date_cols].mean(axis=1)

    # Convert Entry Date to datetime
    df_analysis['Entry Date'] = df['registration_incorporation_date']
    df_analysis['Entry Date'] = pd.to_datetime(df_analysis['Entry Date'])

    # Read the geojson file
    regions = gpd.read_file('region_boundary.geojson')
    regions = regions.to_crs("EPSG:3414") # Convert to Singapore CRS

    # Map the region names to the corresponding codes
    region_map = {
        'kml_1': 'WR',
        'kml_2': 'NR',
        'kml_3': 'NER',
        'kml_4': 'ER',
        'kml_5': 'CR'
    }
    regions['Name'] = regions['Name'].map(region_map)

    # Create a GeoDataFrame from the df DataFrame
    geometry = [Point(xy) for xy in zip(df['Coordinate_X'], df['Coordinate_Y'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:3414")

    # Spatial join: assign each firm the region it falls into
    gdf_with_region = gpd.sjoin(gdf, regions, how='left', predicate='within')

    # Update df_analysis with the mapped region assignments
    df_analysis['Region'] = gdf_with_region['Name'].values

    # Add age column
    df_analysis['Age'] = (max(df_analysis['Entry Date']) - df_analysis['Entry Date']).dt.days / 365.25

    return df_analysis