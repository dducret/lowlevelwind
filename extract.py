from datetime import datetime, timedelta, timezone
from collections.abc import Sequence
from meteodatalab import ogd_api
import regrid
from meteodatalab import grib_decoder, data_source
from meteodatalab.operators.destagger import destagger

import time
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
import numpy as np
import json
import os
import shutil
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

successful_api_calls = 0
failed_api_calls = 0

cache_path = 'sma-cache'
data_path = 'data'
data_copy_path = '/data/sites/station/application/lowlevelwind/data-copy/'

z_values = range(35, 81)  # According to MeteoSwiss, the published range is 1–80. Since Z35 is above 5,500 metres, no higher range is required

def get_collection(model):
    return f'ogd-forecasting-icon-{model}'

def get_horizons(model):
    #                                            34
    return [timedelta(hours=h) for h in range(0, 34 if model == 'ch1' else 121)]

def get_latest_reference_datetime(model, variable, perturbed, horizon):
    global successful_api_calls
    global failed_api_calls
    
    r = ogd_api.Request(
        collection=get_collection(model),
        variable=variable,
        reference_datetime="latest",
        perturbed=perturbed,
        horizon=horizon,
    )
    try:
        urls = ogd_api.get_asset_urls(r)
    except (requests.exceptions.JSONDecodeError, requests.exceptions.ConnectionError, requests.exceptions.HTTPError) as e:
        failed_api_calls += 1
        return None
    successful_api_calls += 1
    if len(urls) < 1:
        return None
    parts = urls[0].split('-')
    return datetime.strptime(parts[3], '%Y%m%d%H%M').replace(tzinfo=timezone.utc)

def get_latest_completed_reference_datetime(model):
    reference_datetimes = []
    horizons = get_horizons(model)
    for variable in ['U', 'V']:
        for perturbed in [False]:
            for horizon in horizons:
                print('Checking for latest data: variable', variable, 'perturbed', perturbed, 'horizon', horizon)
                reference_datetime = get_latest_reference_datetime(model, variable, perturbed, horizon)
                print(reference_datetime)
                if reference_datetime is not None:
                    reference_datetimes.append(reference_datetime)
    return min(reference_datetimes)

def save_png(f_U, f_V, filename):
    shift = 128
    ms_to_kmh = 3.6
    nan_value = 0

    def _scale_wind_component(values):
        return np.nan_to_num(
            values * ms_to_kmh + shift,
            nan=nan_value,
            posinf=nan_value,
            neginf=nan_value,
        )
        
    alpha = np.ones(f_U.values.shape)
    alpha[np.isnan(f_U.values)] = 0
    alpha *= 255

    f_U_vals = _scale_wind_component(f_U.values)
    f_V_vals = _scale_wind_component(f_V.values)

    rgba = np.stack(
        [
            f_U_vals.astype(np.uint8),
            f_V_vals.astype(np.uint8),
            np.zeros_like(f_U_vals, dtype=np.uint8),
            alpha.astype(np.uint8)
        ],
        axis=0
    )

    rgba_flipped = rgba[:, ::-1, :]

    with rasterio.open(
        filename,
        "w",
        driver="PNG",
        height=f_U.shape[0],
        width=f_U.shape[1],
        count=4,
        dtype=rgba.dtype,
    ) as dst:
        dst.write(rgba_flipped)
        dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]

def save_geotiff(da, filename):
    print(f'Writing {filename}...')
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=da.shape[0],
        width=da.shape[1],
        count=1,
        dtype=np.float32,
        crs=da.crs if hasattr(da, 'crs') else None,
        transform=da.rio.transform() if hasattr(da, 'rio') else None,
    ) as dst:
        dst.write(da.values[::-1, :].astype(np.float32), 1)

def get_delauny(da):
    xmin = 5.379264
    xmax = 11.024297
    ymin = 45.497280
    ymax = 48.105836

    nx = 429
    ny = 195

    destination = regrid.RegularGrid(
        CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax
    )
    return (destination,) + regrid.iconremap_delauny(da, destination)

def reproject_with_delauny(da, destination, indices, weights, lon, lat):
    return regrid.icon2regular(da, destination, indices, weights).assign_coords(
        lon=(("y", "x"), lon), lat=(("y", "x"), lat)
    ).squeeze()

def get_filename(model, variable, reference_datetime, perturbed, horizon, eps):
    member = "ctrl"
    if perturbed:
        if eps is None:
            raise ValueError("eps must be provided for perturbed members")
        member = str(eps)
    return f'icon-{model}-eps-{get_timestring(reference_datetime)}-{get_horizon_hours(horizon)}-{variable.lower()}-{member}.grib2'

def download(model, variable, reference_datetime, perturbed, horizon, eps):
    print(f'Download {get_filename(model, variable, reference_datetime, perturbed, horizon, eps)}...')
    req = ogd_api.Request(
        collection=get_collection(model),
        variable=variable,
        reference_datetime=reference_datetime,
        perturbed=perturbed,
        horizon=horizon,
    )
    ogd_api.download_from_ogd(req, Path(cache_path))

_GEO_COORDS_CACHE = None

def geo_coords(uuid):
    global _GEO_COORDS_CACHE

    if _GEO_COORDS_CACHE is None:
        ds = grib_decoder.load(
            source=data_source.FileDataSource(datafiles=[f"{cache_path}/horizontal_constants_icon-ch1-eps.grib2"]),
            request={"param": ["CLON", "CLAT"]},
            geo_coords=lambda uuid: {}
        )
        _GEO_COORDS_CACHE = {"lat": ds["CLAT"].squeeze(), "lon": ds["CLON"].squeeze()}

    return _GEO_COORDS_CACHE

def get_timestring(reference_datetime):
    return reference_datetime.strftime('%Y%m%d%H%M')

def get_horizon_hours(horizon):
    return int(horizon.total_seconds() / 3600)

def read(model, variable, reference_datetime, perturbed, horizon, eps, z):
    if isinstance(variable, str):
        variables = [variable]
    else:
        variables = list(variable)

    datafiles = [
        f"{cache_path}/{get_filename(model, var, reference_datetime, perturbed, horizon, eps)}"
        for var in variables
    ]

    if isinstance(z, Sequence) and not isinstance(z, (str, bytes)):
        levelist = list(z)
    else:
        levelist = z

    request = {
        "param": variables if len(variables) > 1 else variables[0],
    }

    if levelist is not None:
        request["levelist"] = levelist
    data = grib_decoder.load(
        source=data_source.FileDataSource(datafiles=datafiles),
        request=request,
        geo_coords=geo_coords
    )
    if len(variables) == 1:
        return data[variables[0]]

    return data

def make_horizon(reference_datetime, horizon, model, perturbed, eps):
    os.makedirs(data_path, exist_ok=True)

    destination = None
    indices = None
    weights = None
    lon = None
    lat = None

    levels = list(z_values)
    wind_data = read(model, ["U", "V"], reference_datetime, perturbed, horizon, eps, levels)

    da_U_levels = wind_data["U"].squeeze(drop=True)
    da_V_levels = wind_data["V"].squeeze(drop=True)

    ignored_dims = {"y", "x", "time", "step", "eps"}
    level_dim_U = next((dim for dim in da_U_levels.dims if dim not in ignored_dims), None)
    level_dim_V = next((dim for dim in da_V_levels.dims if dim not in ignored_dims), None)

    total_levels = len(levels)
    if level_dim_U:
        total_levels = min(total_levels, da_U_levels.sizes[level_dim_U])
    if level_dim_V:
        total_levels = min(total_levels, da_V_levels.sizes[level_dim_V])

    for idx, z in enumerate(levels[:total_levels]):
        print(f'Working on horizon={get_horizon_hours(horizon)}, z={z} of 80')
        
        if level_dim_U:
            da_U = da_U_levels.isel({level_dim_U: idx}).squeeze(drop=True)
        else:
            da_U = da_U_levels

        if level_dim_V:
            da_V = da_V_levels.isel({level_dim_V: idx}).squeeze(drop=True)
        else:
            da_V = da_V_levels
        if destination is None:
            destination, indices, weights, lon, lat = get_delauny(da_U)
            
        f_U = reproject_with_delauny(da_U, destination, indices, weights, lon, lat)
        f_V = reproject_with_delauny(da_V, destination, indices, weights, lon, lat)
        
        member_filename = f'EPS{eps}' if perturbed else 'CTRL'
        model_filename = model.upper()
        z_filename = f'Z{z}'
        time_filename = int((reference_datetime + horizon).timestamp())
        filename = f'{data_path}/{model_filename}-{member_filename}-{z_filename}-{time_filename}-wind.png'
        save_png(f_U, f_V, filename)
    return get_horizon_hours(horizon)

def make_height_fields():
    ds = grib_decoder.load(
        source=data_source.FileDataSource(datafiles=[f"{cache_path}/vertical_constants_icon-ch1-eps.grib2"]), 
        request={"param": "HHL"}, 
        geo_coords=geo_coords
    )
    hfl = destagger(ds["HHL"].squeeze(drop=True), "z")
    
    os.makedirs(data_path, exist_ok=True)
    
    destination = None
    indices = None
    weights = None
    lon = None
    lat = None
    for z in z_values:
        if destination is None:
            destination, indices, weights, lon, lat = get_delauny(hfl.sel(z=z))
        projected = reproject_with_delauny(hfl.sel(z=z), destination, indices, weights, lon, lat)
        save_geotiff(projected, f'{data_path}/hfl-Z{z}.tif')

def delete_all_files_in_folder(folder):
    if not os.path.exists(folder):
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Could not remove {file_path}: {e}")

def copy_all_files(src_folder, dst_folder):
    print(f'Copy files from {src_folder} to {dst_folder}...')
    os.makedirs(dst_folder, exist_ok=True)
    
    if not os.path.exists(src_folder):
        print(f"Source folder {src_folder} does not exist")
        return
    
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)
        
        if os.path.isfile(src_file):
            try:
                shutil.copy2(src_file, dst_file)
                print(f'Copied {src_file} to {dst_file}.')
            except IOError as e:
                print(f"Could not copy {src_file} to {dst_file}: {e}")

if __name__ == "__main__":
    while True:
        tic = time.time()

        print('Successful API calls', successful_api_calls)
        print('Failed API calls', failed_api_calls)

        last_run = ''
        try:
            with open('data/last_run.json') as f:
                last_run = json.load(f)['last_run']
        except FileNotFoundError:
            pass

        model = 'ch1'
        perturbed = False
        eps = 0

        reference_datetime = get_latest_completed_reference_datetime(model) # 2025-06-28 09:00:00+00:00
        
        latest_available_run = int(reference_datetime.timestamp())

        if last_run == latest_available_run:
            sleep_min = 15
            print(f'No new run available. Sleep for {sleep_min} min...')
            time.sleep(sleep_min * 60)
            continue
        
        print(f'Found new run {reference_datetime}...')
        delete_all_files_in_folder(data_path)
        
        horizons = get_horizons(model)

        for horizon in horizons:
            download(model, 'U', reference_datetime, perturbed, horizon, eps)
            download(model, 'V', reference_datetime, perturbed, horizon, eps)
            
        print('Make height fields...')
        make_height_fields()

        num_threads = 2
        print(f"Starting parallel tasks with {num_threads} threads...")
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            future_to_horizon = {
                executor.submit(make_horizon, reference_datetime, horizon, model, perturbed, eps): horizon 
                for horizon in horizons
            }
            
            for future in as_completed(future_to_horizon):
                horizon = future_to_horizon[future]
                result = future.result()
                print(f"Horizon Completed: {result} of {len(horizons)-1}")
                
        with open('data/last_run.json', 'w') as f:
            json.dump({"last_run": latest_available_run}, f)
        copy_all_files(data_path, data_copy_path)
        delete_all_files_in_folder(cache_path)

        print(f'Finished in {(time.time() - tic) / 60:.2f} min')
