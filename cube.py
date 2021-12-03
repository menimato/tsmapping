# TODO: - implement check if mosaicing is really necessary before proceeding, and adapt everything for it.
#       - add functions help and description.
#       - missing dates are being filled with the previous available image
#           - the error is happening in the mosaicing phase.
#       - implement download from BDC.
#       - make sure if the problem wiht tqdm starting at the second path row persists in other runs of the code.
#       - in 'download_images_BDC':
#           - all bands in the colletion that contains Sentinel L1 in BDC is available in a single compressed file,
#             not separated according to the bands, like other collections. 
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
import shapely.geometry
from shapely import wkt as wkt_
from rasterio.warp import calculate_default_transform, reproject, Resampling

# absolute path
package_directory = os.path.dirname(os.path.abspath(__file__))

# to be used when downloading from google cloud
predefined_tiles = {'caatinga': {'number of paths': 5,
                                 'paths': {'path 1': {'key tile': '23LNL', 
                                                           'tiles': ['23LNG','23LNH','23LNJ','23LNK','23LNL','23LPK','23LPL','23MPM']},
                                                'path 2': {'key tile': '23LQL',
                                                           'tiles': ['23KPB', '23LMC', '23LNC', '23LND', '23LNE', '23LNF', '23LNG', '23LNH', '23LPC', '23LPD', 
                                                                        '23LPE', '23LPF', '23LPG', '23LPH', '23LPJ', '23LPK', '23LPL', '23LQC', '23LQD', '23LQE', 
                                                                        '23LQF', '23LQG', '23LQH', '23LQJ', '23LQK', '23LQL', '23LRG', '23LRH', '23LRJ', '23LRK', 
                                                                        '23LRL', '23MPM', '23MQM', '23MQN', '23MRM', '23MRN', '23MRR', '23MRS', '24LTR', '24MTA', 
                                                                        '24MTB', '24MTS', '24MTT', '24MTU', '24MTV', '24MUA', '24MUB', '24MUU', '24MUV']},
                                                'path 3': {'key tile': '24LUP',
                                                           'tiles': ['23LQD', '23LQE', '23LRD', '23LRE', '23LRF', '23LRG', '23LRH', '24LTJ', '24LTK', '24LTL', 
                                                                        '24LTM', '24LTN', '24LTP', '24LTQ', '24LTR', '24LUJ', '24LUK', '24LUL', '24LUM', '24LUN', 
                                                                        '24LUP', '24LUQ', '24LUR', '24LVL', '24LVM', '24LVN', '24LVP', '24LVQ', '24LVR', '24LWQ', 
                                                                        '24LWR', '24MTS', '24MUA', '24MUB', '24MUS', '24MUT', '24MUU', '24MUV', '24MVA', '24MVB', 
                                                                        '24MVS', '24MVT', '24MVU', '24MVV', '24MWA', '24MWB', '24MWS', '24MWT', '24MWU', '24MWV', 
                                                                        '24MXA', '24MXV']},
                                                'path 4': {'key tile': '24MYT',
                                                           'tiles': ['24LVM', '24LVN', '24LVP', '24LWM', '24LWN', '24LWP', '24LWQ', '24LWR', '24LXN', '24LXP', 
                                                                        '24LXQ', '24LXR', '24LYP', '24LYQ', '24LYR', '24LZR', '24MWS', '24MWT', '24MXS', '24MXT', 
                                                                        '24MXU', '24MXV', '24MYS', '24MYT', '24MYU', '24MYV', '24MZS', '24MZT', '24MZU', '24MZV']},
                                                'path 5': {'key tile': '25LBL',
                                                           'tiles': ['24LZR', '24MZS', '25MBM', '25MBN', '25MBP']}
                                          }
                                }
                   }

# to be used when downloading from Brazil Data Cube
predefined_paths_rows = {'caatinga': [[214, 64], [214, 65], [214, 66], [214, 67], [219, 62], [219, 63], [219, 64], [219, 65], [219, 66], [219, 67], [219, 68],
                                      [219, 69], [219, 70], [219, 71], [217, 62], [217, 63], [217, 64], [217, 65], [217, 66], [217, 67], [217, 68], [217, 69],
                                      [217, 70], [217, 71], [215, 63], [215, 64], [215, 65], [215, 66], [215, 67], [215, 68], [220, 66], [220, 67], [220, 68],
                                      [218, 62], [218, 63], [218, 64], [218, 65], [218, 66], [218, 67], [218, 68], [218, 69], [218, 70], [218, 71], [218, 72],
                                      [216, 63], [216, 64], [216, 65], [216, 66], [216, 67], [216, 68], [216, 69], [216, 70]]}


# create sentinel cubes with images from gcloud: all in one function
def create_cubes_gcloudSentinel(save_folder, bands, start_date, end_date, metadata_path, delete_auxiliary=True, tiles=predefined_tiles['caatinga'], interval=5, proj4 = '"+proj=aea +lat_0=-12 +lon_0=-54 +lat_1=-2 +lat_2=-22 +x_0=5000000 +y_0=10000000 +ellps=GRS80 +units=m +no_defs +type=crs"', clip_shapefile_path=None):
    
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
def download_images_gcloud(save_folder, metadata_path, bands, start_date, end_date, tiles=predefined_tiles['caatinga'], interval=5):
    print('- Download Images -', flush=True)
    print('Loading metadata...', flush=True)
    
    # creates list with the tiles of interest to download
    tiles_filter = []
    for i in range(1, tiles['number of paths']+1, 1):
        tiles_filter.extend([tiles['paths'][f'path {i}']['key tile']])
        tiles_filter.extend( tiles['paths'][f'path {i}']['tiles'])
    
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
    for path in range(1, tiles['number of paths']+1, 1):
        print(f'Path:{path}\n----------', flush=True)

        # getting the dates from the jey tile
        key_tile_df = df[(df['MGRS_TILE'] == tiles['paths'][f'path {path}']['key tile']) & 
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
        for tile in tqdm(tiles['paths'][f'path {path}']['tiles']):
            # iterating through the dates
            for [start_date, end_date] in dates:
                # search for scenes of interest
                # select from dataframe scenes for the specific tile and time interval
                sub_df = df[(df['MGRS_TILE'] == tile) & ((df['SENSING_TIME']>=str(start_date)) & (df['SENSING_TIME']<str(end_date)))].copy()
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
                    print(f'Tile {tile} did not have any candidate images between {start_date} and {end_date}.')
    
    # final statement   
    print('Download finished!\n')


# download images from BDC
def download_images_BDC(save_folder, access_token, start_date, end_date, bands=None, wkt=None, grid_images=None, collection='LC8_SR-1'):
    # TODO
    # - implement download from collections 'LC8_DN-1' and 'S2_L1C-1', which are available but are compressed.
    # - explain in the help for this function that wkt must not be too complex.
    # - implement the sentinel download from BDC correctly

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
                        print(file_name[10:16])
                        return 0
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
    
    # check if a method was chosen to search for the images
    if wkt is None and grid_images is None:
        raise ValueError("'wkt' or 'grid_images' must be given.")

    if collection=='LC8_SR-1' or collection=='LC8_DN-1':
        db = gpd.read_file(f'{package_directory}/aux/landsat_grid.shp')
    elif collection=='S2_L2A-1':
        db = gpd.read_file(f'{package_directory}/aux/sentinel_grid.shp')

    if not wkt is None:
            geom = wkt_.loads(wkt)
            items = collection_.get_items(
                filter = dict(intersects=shapely.geometry.mapping(geom))
                    # filter={
                    #         'wkt':wkt,
                    #         'datetime': f'{start_date}/{end_date}',
                    #         'limit':15000
                    #     }
            )

            return items

            # download_items(items, save_folder, bands)
    else:
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
    print('- Reprojecting Bands -', flush=True)
    # checks save_folder's consistency
    if save_folder[-1]!='/':
        save_folder = save_folder+'/'
        
    # iterate and reproject bands
    for file in tqdm(files):
        command = f'gdalwarp -wo NUM_THREADS=4 -wm 4096 -co BIGTIFF=YES -srcnodata {nodata} -dstnodata {nodata} -overwrite -t_srs {proj4} -of GTiff {file} {save_folder+file.split("/")[-1]}'
        subprocess.call(command, shell=True)
        
    # final statement   
    print('Reprojection finished!\n')
        
        
# creates a mosaic with reprojected bands
def mosaic_bands(bands_folder, save_folder, bands, start_date, end_date, interval, date_charsinterval_in_bandnames, nodata=-9999):
    print('- Mosaicing Bands -', flush=True)
    
    # checks save_folder's consistency
    if save_folder[-1]=='/':
        save_folder = save_folder[:-1]
        
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
            translate_path = f'{save_folder}/S{start_date.strftime("%Y%m%d")}-E{end_date.strftime("%Y%m%d")}_{band}.tif'
            command = f'gdal_translate -co BIGTIFF=YES -co COMPRESS=PACKBITS -of GTiff {vrt_path} {translate_path}'
            subprocess.call(command, shell=True)
            
    # delete created vrt
    os.remove(vrt_path)
    
    # final statement   
    print('Mosaicing finished!\n')
    

# crop to shapefile
def clip_shapefile(files, shapefile, save_folder):
    print('- Clip with Shapefile -', flush=True)
    
    # checks save_folder's consistency
    if save_folder[-1]=='/':
        save_folder = save_folder[:-1]
    
    # iterate through files and crop them with gdal
    for file in tqdm(files):
        command = f'gdalwarp -co BIGTIFF=YES -co COMPRESS=PACKBITS -of GTiff -cutline "{shapefile}" -crop_to_cutline "{file}" "{save_folder}/{file.split("/")[-1]}"'
        subprocess.call(command, shell=True)
        
    # final statement   
    print('Clipping finished!\n')
    
    
# creates the stack
def create_stacks(bands_folder, save_folder, bands):
    print('- Creating Cubes -', flush=True)
    
    # checks bands_folder's consistency
    if bands_folder[-1]=='/':
        bands_folder = bands_folder[:-1]
        
    # checks save_folder's consistency
    if save_folder[-1]=='/':
        save_folder = save_folder[:-1]
    
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
        command = f'gdal_translate -co BIGTIFF=YES -co COMPRESS=PACKBITS -of GTiff aux.vrt {save_folder}/{files[0].split("/")[-1][:9]}-{files[-1].split("/")[-1][10:19]}_{band}.tif'
        subprocess.call(command, shell=True)
    
    # delete created vrt
    os.remove('aux.vrt')
    
    # final statement
    print('Stacks finished!\n')
    
                    
# creates a list with the dates intervals, used when creating the cubes
def acquire_dates(start_date, final_date, interval):
    dates = []
    
    end_date = start_date + datetime.timedelta(days=interval)

    while end_date<=final_date:
        dates.append([start_date, end_date])

        start_date = end_date
        end_date += datetime.timedelta(days=interval)
        
    return dates


# update metadata
def update_metadata(save_folder = '.'):
    # checks save_folder's consistency
    if save_folder[-1]=='/':
        save_folder = save_folder[:-1]
    
    # downloads the data from google cloud
    command = f'gsutil -m cp -R gs://gcp-public-data-sentinel-2/index.csv.gz {save_folder}'
    subprocess.call(command, shell=True)
    
    # extract the data
    command = f'gzip -d {save_folder}/index.csv.gz'
    subprocess.call(command, shell=True)
    
    # rename metadata file
    os.rename(f'{save_folder}/index.csv', f'{save_folder}/Sentinel2-L1.csv')