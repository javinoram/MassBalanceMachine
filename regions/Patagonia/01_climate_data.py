import pandas as pd
import os
import warnings
from tqdm.notebook import tqdm
import zipfile
import cdsapi
import zipfile
import numpy as np
import glob
import xarray as xr

warnings.filterwarnings('ignore')
path_ERA5_raw = './era5land/'

os.makedirs(path_ERA5_raw, exist_ok=True)
c = cdsapi.Client()
c.retrieve(
    'reanalysis-era5-land-monthly-means', {
        'product_type': ['monthly_averaged_reanalysis'],
        "variable": [
            "2m_temperature",
            "forecast_albedo",
            "surface_latent_heat_flux",
            "surface_net_thermal_radiation",
            "surface_sensible_heat_flux",
            "surface_solar_radiation_downwards",
            "total_precipitation"
        ],
        "year": [ 
            "2000", "2001", "2002", "2003",
            "2004", "2005", "2006", "2007",
            "2008", "2009", "2010", "2011",
            "2012", "2013", "2014",
            "2015", "2016", "2017",
            "2018", "2019", "2020",
            "2021", "2022", "2023", 
            "2024",
        ],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        'time': ["00:00"],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [-45, -74, -55, -69]
    }, path_ERA5_raw+'download.netcdf.zip')
with zipfile.ZipFile(path_ERA5_raw+'download.netcdf.zip', 'r') as zip:
    zip.extractall(path_ERA5_raw)


c.retrieve("reanalysis-era5-single-levels", {
        "product_type": ["reanalysis"],
        "variable": ["geopotential"],
        "year": ["2024"],
        "month": ["06"],
        "day": ["01"],
        "time": ["12:00"],
        "data_format": "netcdf"
    }, path_ERA5_raw+'era5_geopotential_pressure.nc')


ds  = xr.open_dataset(path_ERA5_raw+'data_stream-moda.nc').drop_vars(("number", "expver"))
ds.rename({'valid_time': 'time'}).to_netcdf(path_ERA5_raw+"era5_monthly_averaged_data.nc")
