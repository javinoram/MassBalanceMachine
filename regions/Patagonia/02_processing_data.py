import pandas as pd
import geopandas as gpd
import sys
sys.path.append('../../../')
import massbalancemachine as mbm
cfg = mbm.Config()

def main():
    # Specify the filename of the input file with the raw data
    target_data_fname = './data/datos-patagonia.csv'

    # Load the target data
    data = pd.read_csv(target_data_fname, sep = ',')
    data['YEAR'] = pd.to_datetime(data["TO_DATE"], errors="coerce", dayfirst=False).dt.year
    data["TO_DATE"] = pd.to_datetime(data["TO_DATE"], errors="coerce", dayfirst=False).dt.strftime("%Y%m%d")
    data["FROM_DATE"] = pd.to_datetime(data["FROM_DATE"], errors="coerce", dayfirst=False).dt.strftime("%Y%m%d")


    # Especifique el nombre del archivo de forma del contorno de los glaciares obtenido de RGIv6
    glacier_outline_fname = './rgiv6/17_rgi60_SouthernAndes/17_rgi60_SouthernAndes.shp'

    # Cargar contornos de glaciares
    glacier_outline = gpd.read_file(glacier_outline_fname)

    # Obtenga el ID de RGI para cada medición de estaca para la región de interés
    data = mbm.data_processing.utils.get_rgi(data=data, glacier_outlines=glacier_outline)
    dataset = mbm.data_processing.Dataset(cfg=cfg, data=data, region_name='SouthernAndesPatagonia', data_path='./data/', region_id=17)

    #print("funciona")
    voi_topographical = ['aspect', 'slope']
    dataset.get_topo_features(vois=voi_topographical)
    #dataset.clean_nonvalid_glaciers()

    # Especifique los archivos de datos climáticos que se corresponderán con las coordenadas de los datos de estaca
    era5_climate_data = './era5land/era5_monthly_averaged_data.nc'
    geopotential_data = './era5land/era5_geopotential_pressure.nc'

    # Haga coincidir las características climáticas, del archivo netCDF de ERA5Land, para cada conjunto de datos de medición de estaca
    dataset.get_climate_features(climate_data=era5_climate_data, geopotential_data=geopotential_data)

    # Especifique los nombres cortos de las variables climáticas disponibles en el conjunto de datos
    vois_climate = ['t2m', 'tp', 'slhf', 'sshf', 'ssrd', 'fal', 'str']
    dataset.convert_to_monthly(vois_climate=vois_climate, vois_topographical=voi_topographical)


if __name__ == "__main__":
    main()
