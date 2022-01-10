import numpy as np
import rasterio as r
import geopandas as gpd
import os
import subprocess

def create_samples_LSTM(shapefile, stacks_paths, reference_attribute_name, save_folder, samples_amount=None, divide10000=True, shuffle_samples=True):
    # load the samples vector reference
    shp = gpd.read_file(shapefile)

    # load first stack
    stack = r.open(stacks_paths[0])

    # reproject vector reference to match stack
    shp = shp.to_crs(stack.crs.to_dict())

    # converting reference to int with wach number following each other
    u = np.unique(shp[reference_attribute_name].values)
    ref = shp[reference_attribute_name].values
    for i in range(len(u)):
        ref[ref==u[i]] = i
    ref = np.asarray(ref)

    # print which number corresponds to each class
    print('---------------------------------------------------------\nClasses Reference\n---------------------------------------------------------')
    for i in range(len(u)):
        print(i, '-', u[i])
    print('---------------------------------------------------------')
    
    # behaves differently depending on the shapefile geometry
    # if geometry is point
    if shp.geom_type[0]=='Point':
        # convert samples to row col
        samples_ind = r.transform.rowcol(transform=stack.transform, xs=shp['geometry'].x, ys=shp['geometry'].y)
        samples_ind = np.transpose(np.asarray(samples_ind))

    # if geometry is polygon
    elif shp.geom_type[0]=='Polygon':
        # verify if samples_amount is correctly given
        if type(samples_amount)!=int:
            raise ValueError('samples_amount must be given when using polygon references. It must be an int value greater than 0.')
        elif samples_amount<=0:
            raise ValueError('samples_amount must be a value greater than 0.')

        # rasterizes the reference
        shp['class_num'] = ref
        shp.to_file(os.path.join(save_folder, 'reprojected_reference.shp'))
        command = ['gdal_rasterize',
                   '-a class_num',
                   f'-ts {stack.width}.0 {stack.height}.0',
                   '-init -1.0',
                   '-a_nodata -1.0',
                   f'-te {stack.bounds.left} {stack.bounds.bottom} {stack.bounds.right} {stack.bounds.top}',
                   '-ot Int16',
                   '-of GTiff',
                   '-co COMPRESS=PACKBITS',
                   os.path.join(save_folder, 'reprojected_reference.shp'),
                   os.path.join(save_folder, 'rasterized_reference.tif')]
        command = ' '.join(command)
        subprocess.call(command, shell=True)
        
        # selecting the coordinates
        print('selecting the coordinates still have to be implemented.')
        return 0

    else:
        raise FileExistsError('The shapefile exists, but it must be uniquely composed of Points or Polygons.')

    # extracting samples from cubes
    samples = np.zeros([len(ref), stack.count, len(stacks_paths)], dtype=int)
    print('Extracting samples...')
    for stack_ind in range(len(stacks_paths)):
        stack = r.open(stacks_paths[stack_ind])

        for i in range(len(samples_ind)):
            samples[i,:,stack_ind] = np.squeeze(stack.read(window=r.windows.Window(samples_ind[i,1],samples_ind[i,0],1,1)))

    # optional dividing by 10000
    print('optional division by 10000 is yet to be made')

    # saving the samples
    print('saving is yet to be implemented')

    # plot the samples
    print('plot is yet to be implemented')

def create_samples_ConvLSTM():
    pass

#####################3
def shuffle_samples():
    pass