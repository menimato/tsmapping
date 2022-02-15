# TODO: - implement check if mosaicing is really necessary before proceeding, and adapt everything for it.
#       - add functions help and description.
#       - missing dates are being filled with the previous available image
#           - the error is happening in the mosaicing phase.
#       - make sure if the problem wiht tqdm starting at the second path row persists in other runs of the code.
#       - substitute everywhere with os.path.join() instead of simple string operations. 
#       - implement creation of cube with wkt.
#       - in create stack function, make it sort the files by the appropriate key.


import subprocess
from tqdm import tqdm
import os
import shutil
import geopandas as gpd
import pandas as pd
import glob
import numpy as np
import datetime
import stac
import time
import rasterio as r

# absolute path
package_directory = os.path.dirname(os.path.abspath(__file__))

# to be used when downloading from google cloud
predefined_tiles = {'semiarido':{'key tiles':['23LMJ', '23LQJ', '24LUP', '24LXQ', '25LBL'],
                                'tiles':['23KLB','23KMA','23KMB','23KNA','23KNB','23KPB','23KQA','23KQB','23KRA','23KRB','23LLD',
                                         '23LLG','23LLH','23LLJ','23LMC','23LMD','23LME','23LMG','23LMH','23LMJ','23LMK','23LML',
                                         '23LNC','23LND','23LNE','23LNF','23LNG','23LNH','23LNJ','23LNK','23LNL','23LPC','23LPD',
                                         '23LPE','23LPF','23LPG','23LPH','23LPJ','23LPK','23LPL','23LQC','23LQD','23LQE','23LQF',
                                         '23LQG','23LQH','23LQJ','23LQK','23LQL','23LRC','23LRD','23LRE','23LRF','23LRG','23LRH',
                                         '23LRJ','23LRK','23LRL','23MNM','23MNN','23MPM','23MPN','23MPQ','23MQM','23MQN','23MQP',
                                         '23MQQ','23MQR','23MQS','23MQT','23MRM','23MRN','23MRP','23MRQ','23MRR','23MRS','23MRT',
                                         '24KTF','24KTG','24KUG','24KVG','24LTH','24LTJ','24LTK','24LTL','24LTM','24LTN','24LTP',
                                         '24LTQ','24LTR','24LUH','24LUJ','24LUK','24LUL','24LUM','24LUN','24LUP','24LUQ','24LUR',
                                         '24LVH','24LVJ','24LVK','24LVL','24LVM','24LVN','24LVP','24LVQ','24LVR','24LWM','24LWN',
                                         '24LWP','24LWQ','24LWR','24LXN','24LXP','24LXQ','24LXR','24LYP','24LYQ','24LYR','24LZQ',
                                         '24LZR','24MTA','24MTB','24MTS','24MTT','24MTU','24MTV','24MUA','24MUB','24MUC','24MUS',
                                         '24MUT','24MUU','24MUV','24MVA','24MVB','24MVS','24MVT','24MVU','24MVV','24MWA','24MWB',
                                         '24MWS','24MWT','24MWU','24MWV','24MXA','24MXS','24MXT','24MXU','24MXV','24MYS','24MYT',
                                         '24MYU','24MYV','24MZS','24MZT','24MZU','24MZV','25LBL','25MBM','25MBN','25MBP','25MBQ']}}

# to be used when downloading from Brazil Data Cube
predefined_paths_rows = {'caatinga': [[214, 64], [214, 65], [214, 66], [214, 67], [219, 62], [219, 63], [219, 64], [219, 65], [219, 66], [219, 67], [219, 68],
                                      [219, 69], [219, 70], [219, 71], [217, 62], [217, 63], [217, 64], [217, 65], [217, 66], [217, 67], [217, 68], [217, 69],
                                      [217, 70], [217, 71], [215, 63], [215, 64], [215, 65], [215, 66], [215, 67], [215, 68], [220, 66], [220, 67], [220, 68],
                                      [218, 62], [218, 63], [218, 64], [218, 65], [218, 66], [218, 67], [218, 68], [218, 69], [218, 70], [218, 71], [218, 72],
                                      [216, 63], [216, 64], [216, 65], [216, 66], [216, 67], [216, 68], [216, 69], [216, 70]]}


# create sentinel cubes with images from gcloud: all in one function
def create_cubes_gcloudSentinel(save_folder, bands, start_date, end_date, metadata_path, tiles, delete_auxiliary=True, interval=5, proj4 = '"+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +units=m +no_defs +type=crs"', clip_shapefile_path=None):
    """
    Creates data cube stacks for each band, downloading images from Google
    Cloud. More information about images in https://cloud.google.com/storage/docs/public-datasets/sentinel-2.
    Before using this function, please use tsmapping.cube.update_metadata() to
    retrieve Sentinel-2 metadata.

    Parameters
    ----------
    save_folder: string
        Path to the forder to save the data cube stacks.
    bands: list of strings
        Bands used to create the data cube stacks. Each band in the list will
        have a stack in the final process, inside save_folder. 
        Ex.: ['B02', 'B03', 'B04', 'B08']
    start_date: string
        Start date for the time series to be retrieved for the data cube. The
        date must be inserted in the following format: 'YYYY-MM-DD'.
    end_date: string
        End date for the time series to be retrieved for the data cube. The
        date must be inserted in the following format: 'YYYY-MM-DD'.
    metadata_path: string
        Path to the metadata file downloaded with tsmapping.cube.update_metadata().
        Filename included. 
    tiles: dict, optional
        This variable must follow the model presented in 
        tsmapping.cube.predefined_tile['semiarido']. 'tiles' is is a list of
        tiles that interset the study area. 'key tiles' is a list of tiles,
        where one tile exist for each Sentinel-2 acquisition path. The 'key
        tiles' are used to acquire the date of each Sentinel-2 acquisition date
        and must be at the center of the path, not overlapping with other paths.
        Key tiles can be outside of the 'tiles' list.
        Ex.: {'key tiles': ['24MVT', '24MYT'],
              'tiles': ['24MWT', '24MWS', '24MXT', '24MXS']}
        Obs.: To find 'key tiles' and 'tiles' the files under tsmapping.aux can
        be used.
    delete_auxiliary: bool, optional
        Wheter to delete intermediate files used to create the cube, like 
        bands without reprojecting or mosaics without clipping to shapefile.
    interval: int, optional
        The interval used to create the time series, in days. In case there are
        multiple images within the interval, the one with the lowest cloud
        percentage will be chosen.
    proj4: string, optional
        The projection used in the reprojection phase. This is the final
        projection of the data cubes. It is recommended to insert it in
        the proj4 format.
    clip_shapefile_path: string, optional
        The path to the shapefile used to clip the mosaics before creating the
        data cube. If it is not given, the clipping step is completely skipped.
    
    Returns
    -------
    None
    """
    # change current folder
    os.chdir(save_folder)
    
    ######################
    # downloading images
    print('----------------------------------------------------------', flush=True)
    if not os.path.exists('bands'):
        os.makedirs('bands')
    
    download_images_gcloud('./bands',
                           metadata_path,
                           bands,
                           start_date,
                           end_date,
                           tiles,
                           interval)
    
    ######################
    # reproject images
    print('----------------------------------------------------------', flush=True)
    if not os.path.exists('reprojected'):
        os.makedirs('reprojected')
    
    reproject_bands(files = glob.glob('./bands/*.jp2'), 
                    save_folder = './reprojected', 
                    proj4 = proj4,
                    nodata = 0)
    
    # delete auxiliary files if needed
    if delete_auxiliary:
        shutil.rmtree('./bands', ignore_errors=True)
    
    ######################
    # mosaic bands
    print('----------------------------------------------------------', flush=True)
    if not os.path.exists('mosaics'):
        os.makedirs('mosaics')
    
    mosaic_bands(bands_folder = './reprojected', 
                 save_folder = './mosaics', 
                 bands = bands, 
                 start_date = start_date, 
                 end_date = end_date, 
                 interval = interval,
                 date_charsinterval_in_bandnames = [7,15],
                 nodata = 0)
    
    # delete auxiliary files if needed
    if delete_auxiliary:
        shutil.rmtree('./reprojected', ignore_errors=True)
        
    # prepare for cube creation
    past_folder = './mosaics'
    
    ######################
    # crop shapefiles if it is needed
    if clip_shapefile_path is not None:
        print('----------------------------------------------------------', flush=True)
        if not os.path.exists('clipped_mosaics'):
            os.makedirs('clipped_mosaics')
        
        clip_shapefile(files = glob.glob('./mosaics/*.tif'), 
                       shapefile = clip_shapefile_path, 
                       save_folder = './clipped_mosaics')
        
        # delete auxiliary files if needed
        if delete_auxiliary:
            shutil.rmtree('./mosaics', ignore_errors=True)
            
        past_folder = './clipped_mosaics'
    
    ######################
    print('----------------------------------------------------------', flush=True)
    create_stacks(bands_folder = past_folder, 
                  save_folder = '.',
                  bands = bands)
    
    # delete auxiliary files if needed
    if delete_auxiliary:
        shutil.rmtree(past_folder, ignore_errors=True)
        
    print('----------------------------------------------------------\nCubes finished.')


# create cubes with images from BDC: all in one function
def create_cubes_BDC(save_folder, bands, access_token, start_date, end_date, delete_auxiliary=True, collection='LC8_SR-1', grid_images=predefined_paths_rows['caatinga'], proj4 = '"+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +units=m +no_defs +type=crs"', clip_shapefile_path=None):
    """
    Creates data cube stacks for each band, downloading images from Brazil
    Data Cube. More information in https://brazildatacube.dpi.inpe.br/portal/explore.
    
    Parameters
    ----------
    save_folder: string
        Path to the forder to save the data cube stacks. 
    bands: list of strings
        Bands used to create the data cube stacks. Each band in the list will
        have a stack in the final process, inside save_folder. 
        Ex.: ['sr_band2', 'sr_band3', 'sr_band4', 'sr_band5'] for Landsat-8
        surface reflectance, ['B02', 'B03', 'B04', 'B08'] for Sentinel-2 
        surface reflectance, and ['B2', 'B3', 'B4', 'B5'] for Landsat-8 digital
        number.
    access_token: string
        The token required to download from the Brazil Data Cube servers.
    start_date: string
        Start date for the time series to be retrieved for the data cube. The
        date must be inserted in the following format: 'YYYY-MM-DD'.
    end_date: string
        End date for the time series to be retrieved for the data cube. The
        date must be inserted in the following format: 'YYYY-MM-DD'.
    delete_auxiliary: bool, optional
        Wheter to delete intermediate files used to create the cube, like 
        bands without reprojecting or mosaics without clipping to shapefile.
    collection: string, optional
        The collection used to create the data cube stacks. Currrent supported
        options are:
        - 'LC8_DN-1' (Landsat-8 digital number)
            https://brazildatacube.dpi.inpe.br/stac/collections/LC8_DN-1?access_token=nZ97OpESX5DTWuFhkh3aZ3o4g4vBGocxz7Gzju3twv
        - 'LC8_SR-1' (Landsat-8 surface reflectance)
            https://brazildatacube.dpi.inpe.br/stac/collections/LC8_SR-1?access_token=nZ97OpESX5DTWuFhkh3aZ3o4g4vBGocxz7Gzju3twv
        - 'S2_L2A-1' (Sentinel-2 surface reflectance).
            https://brazildatacube.dpi.inpe.br/stac/collections/S2_L2A-1?access_token=nZ97OpESX5DTWuFhkh3aZ3o4g4vBGocxz7Gzju3twv
        The images' availability can vary drastically depending on the
        collection.
    grid_images: list of strings or list of lists of numbers, optional
        The list of Sentinel-2 tiles (string) or Landsat-8 path-rows that
        are within the study area.
        Ex.: ['23LQH', '23LQJ', '23LQK'] or
             [[214, 64], [214, 65], [214, 66]].
    proj4: string, optional
        The projection used in the reprojection phase. This is the final
        projection of the data cubes. It is recommended to insert it in
        the proj4 format.
    clip_shapefile_path: string, optional
        The path to the shapefile used to clip the mosaics before creating the
        data cube. If it is not given, the clipping step is completely skipped.

    Returns
    -------
    None
    """
    # change current folder
    os.chdir(save_folder)

    # define additional  parameters according to collection
    if collection == 'LC8_SR-1':
        nodata = -9999
        interval = 16
        date_charsinterval_in_bandnames = [17,25]
    elif collection == 'LC8_DN-1':
        nodata = 0
        interval = 16
        date_charsinterval_in_bandnames = [17,25]
    elif collection == 'S2_L2A-1':
        nodata = 0
        interval = 5
        date_charsinterval_in_bandnames = [7,15]
    
    ######################
    # downloading images
    print('----------------------------------------------------------', flush=True)
    if not os.path.exists('bands'):
        os.makedirs('bands')
    
    download_images_BDC(save_folder = './bands',
                        bands = bands,
                        access_token = access_token,
                        start_date = start_date,
                        end_date = end_date,
                        grid_images = grid_images,
                        collection = collection)
    
    ######################
    # reproject images
    print('----------------------------------------------------------', flush=True)
    if not os.path.exists('reprojected'):
        os.makedirs('reprojected')
    
    for band in bands:
        print(band)
        reproject_bands(files = glob.glob(f'./bands/*{band}*'),
                        save_folder = './reprojected',
                        proj4 = proj4,
                        nodata = nodata)
    
    # delete auxiliary files if needed
    if delete_auxiliary:
        shutil.rmtree('./bands', ignore_errors=True)
    
    ######################
    # mosaic bands
    print('----------------------------------------------------------', flush=True)
    if not os.path.exists('mosaics'):
        os.makedirs('mosaics')
    
    mosaic_bands(bands_folder = './reprojected', 
                 save_folder = './mosaics', 
                 bands = bands, 
                 start_date = start_date, 
                 end_date = end_date, 
                 interval = interval,
                 date_charsinterval_in_bandnames = date_charsinterval_in_bandnames,
                 nodata = nodata)
    
    # delete auxiliary files if needed
    if delete_auxiliary:
        shutil.rmtree('./reprojected', ignore_errors=True)
        
    # prepare for cube creation
    past_folder = './mosaics'
    
    ######################
    # crop shapefiles if it is needed
    if clip_shapefile_path is not None:
        print('----------------------------------------------------------', flush=True)
        if not os.path.exists('clipped_mosaics'):
            os.makedirs('clipped_mosaics')
        
        clip_shapefile(files = glob.glob('./mosaics/*.tif'), 
                       shapefile = clip_shapefile_path, 
                       save_folder = './clipped_mosaics')
        
        # delete auxiliary files if needed
        if delete_auxiliary:
            shutil.rmtree('./mosaics', ignore_errors=True)
            
        past_folder = './clipped_mosaics'
    
    ######################
    print('----------------------------------------------------------', flush=True)
    create_stacks(bands_folder = past_folder, 
                  save_folder = '.',
                  bands = bands)
    
    # delete auxiliary files if needed
    if delete_auxiliary:
        shutil.rmtree(past_folder, ignore_errors=True)
        
    print('----------------------------------------------------------\nCubes finished.')
    

# download images from google cloud
def download_images_gcloud(save_folder, metadata_path, bands, start_date, end_date, tiles=predefined_tiles['semiarido'], interval=5):
    """
    Downloads images directly from Google Cloud Storage.

    Parameters
    ----------
    save_folder: string
        Path to the folder where the images must be saved.
    metadata_path: string
        Path to the metadata file for the Google Cloud available images. 
        Retrieved with tsmapping.cube.update_metadata().
    bands: list of strings
        Bands to be downloaded. Ex.: ['B02', 'B03', 'B04', 'B08']
    start_date: string
        Start date for the images to be downloaded. The date must be inserted
        in the following format: 'YYYY-MM-DD'.
    end_date: string
        End date for the images to be downloaded. The date must be inserted in
        the following format: 'YYYY-MM-DD'.
    tiles: dict, optional
        This variable must follow the model presented in 
        tsmapping.cube.predefined_tile['semiarido']. 'tiles' is is a list of
        tiles that interset the study area. 'key tiles' is a list of tiles,
        where one tile exist for each Sentinel-2 acquisition path. The 'key
        tiles' are used to acquire the date of each Sentinel-2 acquisition date
        and must be at the center of the path, not overlapping with other paths.
        Key tiles can be outside of the 'tiles' list.
        Ex.: {'key tiles': ['24MVT', '24MYT'],
              'tiles': ['24MWT', '24MWS', '24MXT', '24MXS']}
        Obs.: To find 'key tiles' and 'tiles' the files under tsmapping.aux can
        be used.
    interval: int
        Time interval in days to search for bands. In case more than one image
        is found within the time interval, the entry with the lower percentage
        of clouds and cloud shadows will be downloaded.

    Returns
    -------
    None
    """

    print('- Download Images -', flush=True)
    print('Loading metadata...', flush=True)
    
    # creates list with the tiles of interest to download
    tiles_filter = []
    tiles_filter.extend(tiles['tiles'])
    tiles_filter.extend(tiles['key tiles'])
    tiles_filter = np.unique(tiles_filter)
    
    # read the metadata in chunks
    chunksize = 10 ** 6
    for chunk in pd.read_csv(metadata_path, chunksize=1):
        df = chunk[chunk['MGRS_TILE'].isin(tiles_filter)]
        break
    
    for chunk in pd.read_csv(metadata_path, chunksize=chunksize):
        df = df.append(chunk[chunk['MGRS_TILE'].isin(tiles_filter)])
    
    # starts the dates to be used
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date   = datetime.datetime.strptime(end_date,   '%Y-%m-%d')
    
    final_date = end_date   + datetime.timedelta(days=1)
    end_date   = start_date + datetime.timedelta(days=interval)

    # creates a list with the dates intervals to be used
    dates = acquire_dates(start_date, final_date, interval)
    
    # iterates through the paths and tiles
    print('Downloading...', flush=True)
    for key_tile in tiles['key tiles']:
        print(f'Key tile:{key_tile}\n----------', flush=True)

        # getting the dates from the key tile
        key_tile_df = df[(df['MGRS_TILE'] == key_tile) & 
                         ((df['SENSING_TIME']>=str(start_date)) & 
                          (df['SENSING_TIME']<str(final_date)))].copy()
        # creates the SENSING_DATE field, because SENSING_TIME has different values even for images in the same day
        sensing_dates = []
        for d in key_tile_df['SENSING_TIME'].values:
            sensing_dates.append(d[:10])
        key_tile_df['SENSING_DATE'] = sensing_dates
        # sorts the created dataframe according to SENSING_DATE and then GENERATION_TIME
        key_tile_df.sort_values(by=['SENSING_DATE', 'GENERATION_TIME'], inplace=True)
        # deletes rows for images acquired on the same day, keeping only the last processed one.
        key_tile_df.drop_duplicates(subset=['SENSING_DATE'], inplace=True, keep='last')

        # iterates through tiles in the path
        for tile in tqdm(tiles['tiles']):
            # iterating through the dates
            for start_date_, end_date_ in dates:
                # search for scenes of interest
                # select from dataframe scenes for the specific tile and time interval
                sub_df = df[(df['MGRS_TILE'] == tile) & ((df['SENSING_TIME']>=str(start_date_)) & (df['SENSING_TIME']<str(end_date_)))].copy()
                # creates the SENSING_DATE field, because SENSING_TIME has different values even for images in the same day
                sensing_dates = []
                for d in sub_df['SENSING_TIME'].values:
                    sensing_dates.append(d[:10])
                sub_df['SENSING_DATE'] = sensing_dates
                # sorts the created dataframe according to SENSING_DATE and then GENERATION_TIME
                sub_df.sort_values(by=['SENSING_DATE', 'TOTAL_SIZE'], inplace=True)
                # deletes rows for images acquired on the same day, keeping only the last processed one.
                sub_df.drop_duplicates(subset=['SENSING_DATE'], inplace=True, keep='last')
                sub_df = sub_df[sub_df['SENSING_DATE'].isin(key_tile_df['SENSING_DATE'].values)]

                if len(sub_df)>0:
                    # acquires url and granule id values
                    url = sub_df[sub_df['CLOUD_COVER'] == sub_df['CLOUD_COVER'].min()].BASE_URL.values[0]
                    granule_id = sub_df[sub_df['CLOUD_COVER'] == sub_df['CLOUD_COVER'].min()].GRANULE_ID.values[0]
                    # iterates through bands to download
                    for band in bands:
                        if not os.path.exists(os.path.join(save_folder, f'T{tile}_{url.split("_", 3)[2]+"_"+band+".jp2"}')):
                            # downloads the necessary files
                            command = f'gsutil -m cp -R {url}/GRANULE/{granule_id}/IMG_DATA/T{tile}_{url.split("_", 3)[2]+"_"+band+".jp2"} {save_folder}'
                            subprocess.call(command, shell=True)
                else:
                    # spit error
                    # print(f'Tile {tile} did not have any candidate images between {start_date} and {end_date}.')
                    pass
    
    # final statement   
    print('Download finished!\n')


# download images from BDC
def download_images_BDC(save_folder, access_token, start_date, end_date, bands, grid_images, collection='LC8_SR-1'):
    """
    Help downloading images from Brazil Data Cube.

    Parameters
    ----------
    save_folder: string
        Path to the folder where the images must be saved.
    access_token: string
        The token required to download from the Brazil Data Cube servers.
    start_date: string
        Start date for images to be downloaded. The date must be inserted in 
        the following format: 'YYYY-MM-DD'.
    end_date: string
        End date for the images to be downlaoded. The date must be inserted in
        the following format: 'YYYY-MM-DD'.
    bands: list of strings
        Bands to download. 
        Ex.: ['sr_band2', 'sr_band3', 'sr_band4', 'sr_band5'] for Landsat-8
        surface reflectance, ['B02', 'B03', 'B04', 'B08'] for Sentinel-2 
        surface reflectance, and ['B2', 'B3', 'B4', 'B5'] for Landsat-8 digital
        number.
    grid_images: list of strings or list of lists of numbers, optional
        The list of Sentinel-2 tiles (string) or Landsat-8 path-rows that
        are within the study area.
        Ex.: ['23LQH', '23LQJ', '23LQK'] or
             [[214, 64], [214, 65], [214, 66]].
    collection: string, optional
        The collection used to create the data cube stacks. Currrent supported
        options are:
        - 'LC8_DN-1' (Landsat-8 digital number)
            https://brazildatacube.dpi.inpe.br/stac/collections/LC8_DN-1?access_token=nZ97OpESX5DTWuFhkh3aZ3o4g4vBGocxz7Gzju3twv
        - 'LC8_SR-1' (Landsat-8 surface reflectance)
            https://brazildatacube.dpi.inpe.br/stac/collections/LC8_SR-1?access_token=nZ97OpESX5DTWuFhkh3aZ3o4g4vBGocxz7Gzju3twv
        - 'S2_L2A-1' (Sentinel-2 surface reflectance).
            https://brazildatacube.dpi.inpe.br/stac/collections/S2_L2A-1?access_token=nZ97OpESX5DTWuFhkh3aZ3o4g4vBGocxz7Gzju3twv
        The images' availability can vary drastically depending on the
        collection.

    Returns
    -------
    None
    """

    print('- Download Images from BDC -', flush=True)

    def download_items(items, save_folder, bands, grid_image_str, collection):
        # downloads the items in the search
        i=1
        for item in items:
            error_num = 0
            print(i,'/',len(items.features))
            if collection == 'LC8_SR-1' or collection == 'S2_L2A-1':
                for band in bands:
                    assets = item.assets
                    asset = assets[band]
                    file_name = asset['href'].split('/')[-1].rsplit('?')[0]
                    
                    is_in_tile = True
                    if collection == 'LC8_SR-1':
                        is_in_tile = file_name[10:16]==grid_image_str
                    elif collection == 'S2_L2A-1':
                        is_in_tile = file_name[1:6]==grid_image_str

                    not_downloaded = not os.path.exists(os.path.join(save_folder, file_name)) and is_in_tile
                    while not_downloaded:
                        try:
                            asset.download(save_folder)
                            not_downloaded = False
                        except Exception as error:
                            if error_num >= 5:
                                raise error
                            print('An error happened while downloading. Trying again in 5 seconds...')
                            time.sleep(5)
                            error_num+=1
            elif collection == 'LC8_DN-1' or collection == 'S2_L1C-1':
                not_downloaded = True
                while not_downloaded:
                    try:
                        item.download(save_folder)
                        not_downloaded = False

                        print('Extracting...', end='')
                        command = f"tar -xf '{os.path.join(save_folder, item['assets']['asset']['href'].split('/')[-1].split('?')[0])}' -C '{save_folder}' && rm -rf '{os.path.join(save_folder, item['assets']['asset']['href'].split('/')[-1].split('?')[0])}'"
                        subprocess.call(command, shell=True)
                        print(' Done!')
                    except Exception as error:
                        if error_num >= 5:
                            raise error
                        print('An error happened while downloading. Trying again in 5 seconds...')
                        time.sleep(5)
                        error_num+=1

            i+=1

    # create service
    service = stac.STAC('https://brazildatacube.dpi.inpe.br/stac/', access_token=access_token)
    collection_ = service.collection(collection)

    if collection=='LC8_SR-1' or collection=='LC8_DN-1':
        db = gpd.read_file(f'{package_directory}/aux/landsat_grid.shp')
    elif collection=='S2_L2A-1':
        db = gpd.read_file(f'{package_directory}/aux/sentinel_grid.shp')

    for grid_image in grid_images:

        if collection=='LC8_SR-1' or collection=='LC8_DN-1':
            print(f'\nPath: {grid_image[0]} - Row:{grid_image[1]}\n-----------------')
            rep_point = db[(db['PATH']==int(grid_image[0])) & (db['ROW']==int(grid_image[1]))].representative_point()
            grid_image_str = f'{str(grid_image[0]).zfill(3)}{str(grid_image[1]).zfill(3)}'
        elif collection=='S2_L2A-1' or collection=='S2_L1C-1':
            print(f'\nTile: {grid_image}\n-----------------')
            rep_point = db[db['Name']==grid_image].representative_point()
            grid_image_str = grid_image
    
        items = collection_.get_items(
                filter={
                        'bbox':f'{rep_point.x.values[0]-.0001},{rep_point.y.values[0]-.0001},{rep_point.x.values[0]+.0001},{rep_point.y.values[0]+.0001}', 
                        'datetime': f'{start_date}/{end_date}',
                        'limit':5000
                    }
        )

        download_items(items, save_folder, bands, grid_image_str, collection)
        

# reproject the bands 
def reproject_bands(files, save_folder, proj4 = '"+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +units=m +no_defs +type=crs"', nodata = -9999):
    """
    Reprojects tif files according to a give proj4.

    Parameters
    ----------
    files: list of strings
        List containing the paths of the tif files to be reprojected.
    save_folder: string
        Path to the folder where the reprojected images must be saved.
    proj4: string, optional
        The target projection to be used. To be inserted in the proj4 format.
        It defaults to the ALbers Equal Area projection used by the IBGE
        for the whole Brazilian territory.
    nodata: int, optional
        The 'no data' value for the files.
    """
    
    print('- Reprojecting Bands -', flush=True)
        
    # iterate and reproject bands
    for file in tqdm(files):
        command = f'gdalwarp -wo NUM_THREADS=4 -wm 4096 -co BIGTIFF=YES -srcnodata {nodata} -dstnodata {nodata} -overwrite -t_srs {proj4} -of GTiff {file} {os.path.join(save_folder, file.split("/")[-1])}'
        subprocess.call(command, shell=True)
        
    # final statement   
    print('Reprojection finished!\n')
        
        
# creates a mosaic with reprojected bands
def mosaic_bands(bands_folder, save_folder, bands, start_date, end_date, interval, date_charsinterval_in_bandnames, nodata=-9999):
    """
    Mosaic bands.

    Parameters
    ----------
    bands_folder: string
        Path to the folder that contains the tif bands to be mosaiced.
    save_folder: string
        Path to the folder where the mosaics must be saved.
    bands: list of strings
        Bands to mosaic. 
        Ex.: ['sr_band2', 'sr_band3', 'sr_band4', 'sr_band5'] for Landsat-8
        surface reflectance, ['B02', 'B03', 'B04', 'B08'] for Sentinel-2 
        surface reflectance, and ['B2', 'B3', 'B4', 'B5'] for Landsat-8 digital
        number.
    start_date: string
        Start date for the mosaics creation. The date must be inserted in the
        following format: 'YYYY-MM-DD'.
    end_date: string
        End date for the mosaics creation. The date must be inserted in the
        following format: 'YYYY-MM-DD'.
    interval: int
        Time interval in days to mosaic the bands.
    date_charsinterval_in_bandnames: list of ints, shape [2]
        The interval in the filename that contain the image acquisition date.
        Ex.: [17,25]
    nodata: int, optional
        The 'no data' value for the files.

    Returns
    -------
    None
    """

    print('- Mosaicing Bands -', flush=True)
        
    # starts the dates to be used
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date   = datetime.datetime.strptime(end_date,   '%Y-%m-%d')
    
    final_date = end_date   + datetime.timedelta(days=1)
    end_date   = start_date + datetime.timedelta(days=interval)

    # creates a list with the dates intervals to be used
    dates = acquire_dates(start_date, final_date, interval)
    
    # iterating through the bands
    for band in bands:
        print(f'Band {band}\n----------', flush=True)
        # iterating through the time intervals
        for [start_date, end_date] in tqdm(dates):
            # get files from the band and year specified.
            files_mosaic = glob.glob(f'{bands_folder}/*{band}*')
            files_mosaic.sort()

            # deletes the file paths from bands outside the 15-days interval
            files_ = files_mosaic.copy()
            for file in files_mosaic:
                file_date = file.split('/')[-1][date_charsinterval_in_bandnames[0]:date_charsinterval_in_bandnames[1]]
                if file_date >= start_date.strftime("%Y%m%d") and file_date<end_date.strftime("%Y%m%d"):
                    pass
                else:
                    files_.remove(file)
            files_mosaic = files_.copy()
            del files_

            # creating the vrt
            vrt_path = 'aux.vrt'
            command  = f'gdalbuildvrt -vrtnodata {nodata} {vrt_path} '+' '.join(files_mosaic)
            subprocess.call(command, shell=True)

            # translate vrt to tif
            translate_path = os.path.join(save_folder,f'S{start_date.strftime("%Y%m%d")}-E{end_date.strftime("%Y%m%d")}_{band}.tif')
            command = f'gdal_translate -co BIGTIFF=YES -co COMPRESS=PACKBITS -of GTiff {vrt_path} {translate_path}'
            subprocess.call(command, shell=True)
            
    # delete created vrt
    os.remove(vrt_path)
    
    # final statement   
    print('Mosaicing finished!\n') 
    

# crop to shapefile
def clip_shapefile(files, shapefile, save_folder):
    """
    Clip files with given shapefile.

    Parameters
    ----------
    files: list of strings
        A list with the path to the tif files to be clipped.
    shapefile: string
        Path to the shapefile used to clip the tif files.
    save_folder: string
        Path to the folder where the clipped images must be saved.

    Returns
    -------
    None
    """

    print('- Clip with Shapefile -', flush=True)

    # reproject shapefile to the same crs as the mosaics
    shp = gpd.read_file(shapefile)
    mosaic = r.open(files[0])
    shp.to_crs(mosaic.crs.to_dict())
    shp.to_file(os.path.join(save_folder, 'reprojected_shapefile.shp'))
    
    # iterate through files and crop them with gdal
    for file in tqdm(files):
        command = f'gdalwarp -co BIGTIFF=YES -co COMPRESS=PACKBITS -of GTiff -cutline "{os.path.join(save_folder, "reprojected_shapefile.shp")}" -crop_to_cutline "{file}" "{os.path.join(save_folder,file.split("/")[-1])}"'
        subprocess.call(command, shell=True)
    
    # removing reprojected shapefile
    if os.path.exists(os.path.join(save_folder, 'reprojected_shapefile.shp')):
        os.remove(os.path.join(save_folder, 'reprojected_shapefile.shp'))
    if os.path.exists(os.path.join(save_folder, 'reprojected_shapefile.cpg')):
        os.remove(os.path.join(save_folder, 'reprojected_shapefile.cpg'))
    if os.path.exists(os.path.join(save_folder, 'reprojected_shapefile.dbf')):
        os.remove(os.path.join(save_folder, 'reprojected_shapefile.dbf'))
    if os.path.exists(os.path.join(save_folder, 'reprojected_shapefile.prj')):
        os.remove(os.path.join(save_folder, 'reprojected_shapefile.prj'))
    if os.path.exists(os.path.join(save_folder, 'reprojected_shapefile.shx')):
        os.remove(os.path.join(save_folder, 'reprojected_shapefile.shx'))

    # final statement   
    print('Clipping finished!\n')
    
    
# creates the stack
def create_stacks(bands_folder, save_folder, bands):
    """
    Creates temporal stacks.

    Parameters
    ----------
    bands_folder: string
        Path to the folder that contains the tif bands to be mosaiced.
    save_folder: string
        Path to the forder to save the stacks.
    bands: list of strings
        Bands used to create the data cube stacks. Each band in the list will
        have a stack in the final process, inside save_folder. 
        Ex.: ['sr_band2', 'sr_band3', 'sr_band4', 'sr_band5'] for Landsat-8
        surface reflectance, ['B02', 'B03', 'B04', 'B08'] for Sentinel-2 
        surface reflectance, and ['B2', 'B3', 'B4', 'B5'] for Landsat-8 digital
        number.

    Returns
    -------
    None
    """

    print('- Creating Cubes -', flush=True)
    
    # checks bands_folder's consistency
    if bands_folder[-1]=='/':
        bands_folder = bands_folder[:-1]
    
    # iterating through bands
    for band in bands:
        # print header and get time at start
        print(f'----------------------------\nBand: {band}')
        
        # acquires the images to create the cubes
        files = glob.glob(f'{bands_folder}/*_{band}.tif')
        files.sort()
        print(len(files),'files...')
        
        # creates a VRT with the bands to be used in the cube
        command = f'gdalbuildvrt -vrtnodata 0 -separate aux.vrt {" ".join(files)}'
        subprocess.call(command, shell=True)
        
        # translates the VRT into a GeoTIFF with the bands
        command = f'gdal_translate -co BIGTIFF=YES -co COMPRESS=PACKBITS -of GTiff aux.vrt {os.path.join(save_folder, files[0].split("/")[-1][:9]+"-"+files[-1].split("/")[-1][10:19])}_{band}.tif'
        subprocess.call(command, shell=True)
    
    # delete created vrt
    os.remove('aux.vrt')
    
    # final statement
    print('Stacks finished!\n')
    
                    
# creates a list with the dates intervals, used when creating the cubes
def acquire_dates(start_date, final_date, interval):
    """
    Creates a list with the dates intervals, used when creating the cubes.

    Parameters
    ----------
    start_date: string
        Start date for the list. The date must be inserted in the following
        format: 'YYYY-MM-DD'.
    end_date: string
        End date for the list. The date must be inserted in the following
        format: 'YYYY-MM-DD'.
    interval: int, optional
        The interval used to create the list, in days.

    Returns
    -------
    dates: list
        List of date intervals. Each interval has a start and end date, in the
        datetime format.
    """

    dates = []
    
    end_date = start_date + datetime.timedelta(days=interval)

    while end_date<=final_date:
        dates.append([start_date, end_date])

        start_date = end_date
        end_date += datetime.timedelta(days=interval)
        
    return dates


# update metadata
def update_metadata(save_folder = '.'):
    """
    Downloads the metadata of the images in Google Cloud Storage.

    Parameters
    ----------
    save_folder: string
        Path of the folder to save the metadata files.

    Returns
    -------
    None
    """
    
    # downloads the data from google cloud
    command = f'gsutil -m cp -R gs://gcp-public-data-sentinel-2/index.csv.gz {save_folder}'
    subprocess.call(command, shell=True)
    
    # extract the data
    command = f'gzip -d '+ os.path.join(save_folder,'index.csv.gz')
    subprocess.call(command, shell=True)
    
    # rename metadata file
    os.rename(os.path.join(save_folder,'index.csv'), os.path.join(save_folder, 'Sentinel2-L1.csv'))