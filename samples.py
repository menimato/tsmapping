from binascii import Error
from datetime import datetime
from random import sample
import numpy as np
import rasterio as r
import geopandas as gpd
import os
import subprocess
import matplotlib.pyplot as plt
import io
import reportlab
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib import colors
from matplotlib.patches import Rectangle

def select_samples_LSTM(shapefile, stacks_paths, reference_attribute_name, save_folder, samples_amount=None, save_samples_location=False, divide10000=True, plot_map=False, selection_mode='random'):
    """
    Creates the training samples to be used by the LSTM. A shapefile with the
    reference in points or polygons can be used.
    
    Parameters
    ----------
    shapefile: string
        The path to the shapefile with the reference. It can contain points or 
        polygons in a known EPSG. It must not contain reference outside the
        used data cube stacks or in areas with 'no data' in them.
    stacks_paths: list of strings
        A list with the paths of the stacks, from which the time series data
        are extracted to create the samples. Their order represent the order
        the time series are aranged in the samples.
    reference_attribute_name: string
        Name of the attibute that contains the name of the class represented
        in the reference shapefile.
    save_folder: string
        Path to the folder used to storage the samples created.
    samples_amount: int or None, optional
        Number of samples to be selected. If None is given, then the maximum
        possible amount of unique samples is created. None is given by default.
    save_samples_location: bool, optional
        Wether or not to save the samples location in a points shapefile. False
        by default.
    divide10000: bool, optional
        Wether or not to divide the  series by 10,000. True by default.
    plot_map: bool, optional
        Wether or not to plot a figure showing where the samples selected are
        laid over the data cube stacks. False by default.
    selection_mode: string, optional
        The strategy to use when selecting the samples. Possible values are
        'random' and 'balanced'.

    Returns
    -------
    None
    """

    # gets current time to create the samples id
    now = datetime.now()

    print('reading files and preparing for samples selection...')
    # load the samples vector reference
    shp = gpd.read_file(shapefile)

    # load first stack
    stack = r.open(stacks_paths[0])

    # reproject vector reference to match stack
    shp = shp.to_crs(stack.crs.to_dict())

    # converting reference to int with the numbers following each other
    u = np.unique(shp[reference_attribute_name].values)
    ref = shp[reference_attribute_name].values
    for i in range(len(u)):
        ref[ref==u[i]] = i
    ref = np.asarray(ref)
    
    # behaves differently depending on the shapefile geometry
    # if geometry is point
    if shp.geom_type[0]=='Point':
        print('retrieving all possible samples...')
        # convert samples to row col
        possible_ind = np.transpose(np.asarray(r.transform.rowcol(transform=stack.transform, xs=shp['geometry'].x, ys=shp['geometry'].y)))

    # if geometry is polygon
    elif shp.geom_type[0]=='Polygon' or shp.geom_type[0]=='MultiPolygon':

        print('rasterizing polygons...')
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
        
        # opening rasterized array
        ref_raster = r.open(os.path.join(save_folder, 'rasterized_reference.tif')).read(1)

        print('retrieving all possible samples...')
        # retrieving all possible coordinates
        possible_ind = np.transpose(np.asarray(np.where(ref_raster!=-1)))
        ref = ref_raster[possible_ind[:,0], possible_ind[:,1]]

        # removing auxiliary files
        if os.path.exists(os.path.join(save_folder, 'reprojected_reference.shp')):
            os.remove(os.path.join(save_folder, 'reprojected_reference.shp'))
        if os.path.exists(os.path.join(save_folder, 'reprojected_reference.cpg')):
            os.remove(os.path.join(save_folder, 'reprojected_reference.cpg'))
        if os.path.exists(os.path.join(save_folder, 'reprojected_reference.dbf')):
            os.remove(os.path.join(save_folder, 'reprojected_reference.dbf'))
        if os.path.exists(os.path.join(save_folder, 'reprojected_reference.prj')):
            os.remove(os.path.join(save_folder, 'reprojected_reference.prj'))
        if os.path.exists(os.path.join(save_folder, 'reprojected_reference.shx')):
            os.remove(os.path.join(save_folder, 'reprojected_reference.shx'))
        if os.path.exists(os.path.join(save_folder, 'rasterized_reference.tif')):
            os.remove(os.path.join(save_folder, 'rasterized_reference.tif'))
    else:
        raise FileExistsError('The shapefile exists, but it must be uniquely composed of Points or Polygons.')

    # verify if samples_amount is correctly given
    print('selecting samples...')
    if samples_amount==None:
        samples_amount = len(possible_ind)
    if type(samples_amount)!=int:
        raise ValueError('samples_amount must be given when using polygon references. It must be None to use all possible samples or an int value greater than 0.')
    elif type(samples_amount)==int:
        if samples_amount<=0:
            raise ValueError('samples_amount must be a value greater than 0.')
        elif samples_amount>len(possible_ind):
            raise Error(f'samples_amount is larger than the highest number of possible unique samples. In this case samples_amount must be lower than {len(possible_ind)}.')
        
        # checks if selection mode is given correctly
        print(f'Strategy to choose samples: {selection_mode}')
        if not selection_mode.lower() in ['random', 'balanced']:
            raise Error('selection_mode must be equal to "random" or "balanced".')
        # in case the 'random' strategy was chosen
        elif selection_mode.lower() == 'random':
            ids = np.random.default_rng().choice(len(possible_ind), size=samples_amount, replace=False)
            samples_ind = possible_ind[ids,:]
            ref = ref[ids]
        elif selection_mode.lower() == 'balanced':
            # getting the number of samples per class
            samples_per_class = int(samples_amount/len(u))
            # checking if it is possible
            count = []
            for i in range(len(u)):
                count.append(np.sum(ref==i))
            count = np.asarray(count)
            is_possible = np.sum(count>=samples_per_class)==len(u)
            
            # if not possible, change the number of samples per class to the maximum possible amount
            if not is_possible:
                if np.min(count)==0:
                    raise Error(f'The class {u[np.where(count==0)[0][0]]} has 0 available entries, os it is impossible to use selection_mode="balanced".')
                print(f'WARNING: Balancing not possible with given samples_amount. Switching to the maximum amount of samples possible without breaking the balancing, which is {np.min(count)*len(u)} samples.')
                samples_per_class = np.min(count)

            # selecting the samples
            samples_ind = np.zeros([samples_per_class*len(u),2], dtype=int)
            samples_ref = np.zeros([samples_per_class*len(u)], dtype=int)
            for i in range(len(u)):
                ids = np.random.default_rng().choice(np.sum(ref==i), size=samples_per_class, replace=False)
                samples_ind[i*samples_per_class:(i+1)*samples_per_class] = np.asarray(possible_ind[ref==i][ids])
                samples_ref[i*samples_per_class:(i+1)*samples_per_class] = np.asarray([i]*samples_per_class)
            
            ref = samples_ref.copy()
            del samples_ref

            # shuffling samples since they are organized by class
            samples_ind, ref = shuffle_samples(samples_ind, ref)

    # extracting samples from cubes
    print('extracting samples from stacks...')
    samples = np.zeros([len(ref), stack.count, len(stacks_paths)], dtype=int)
    for stack_ind in range(len(stacks_paths)):
        stack = r.open(stacks_paths[stack_ind])

        for i in range(len(samples_ind)):
            samples[i,:,stack_ind] = np.squeeze(stack.read(window=r.windows.Window(samples_ind[i,1],samples_ind[i,0],1,1)))
    print(f'{len(samples)} samples extracted!')

    # optional saving samples location
    if save_samples_location:
        print('saving the location of each sample...')
        xy = np.asarray(r.transform.xy(transform=stack.transform, rows=samples_ind[:,0], cols=samples_ind[:,1]))
        df = gpd.GeoDataFrame({'class_num':ref})
        df[reference_attribute_name] = df.apply(lambda row: u[row.class_num], axis=1)
        df['x'] = xy[0,:]
        df['y'] = xy[1,:]
        df = gpd.GeoDataFrame(df, 
                              geometry=gpd.points_from_xy(df.x, df.y))
        df = df.set_crs(crs=stack.crs.to_dict())
        df.to_file(os.path.join(save_folder, f'{now.strftime("%Y-%m-%d_%H-%M-%S")}_samples_location.shp'))

    # optional dividing by 10000
    if divide10000:
        print('dividing by 10,000...')
        samples = samples/10000

    # printing the proportion of each class
    u2,c = np.unique(ref, return_counts=True)
    u2 = np.asarray(u2)
    c = np.asarray(c)
    percentage = c*100/np.sum(c)

    # printing
    text_log = ''
    text_log += '---------------------------------------------------------\nCLASSES REFERENCE\n---------------------------------------------------------\n'
    text_log += 'ID\tPROPORTION\tCLASS NAME\n'
    for i in range(len(u)):
        if i in u2:
            text_log += f'{i}\t{"%.1f" % percentage[np.where(u2==i)][0]}%\t\t{u[i]}\n'
        else:
            text_log += f'{i}\t{"%.1f" % 0}%\t\t{u[i]}\n'
    text_log += '---------------------------------------------------------'

    print(text_log)

    # saving the samples
    print('saving samples...')
    np.save(os.path.join(save_folder, f'{now.strftime("%Y-%m-%d_%H-%M-%S")}_samples.npy'), samples)
    np.save(os.path.join(save_folder, f'{now.strftime("%Y-%m-%d_%H-%M-%S")}_reference.npy'), ref)
    print(f'saved in: {save_folder}')

    # plot the samples
    if plot_map:
        plt.figure(figsize=(10,10))
        band = stack.read(1)
        plt.imshow(band, cmap='gray', interpolation='nearest')
        for i in u2:
            plt.plot(samples_ind[ref==i,1], samples_ind[ref==i,0], 'o', label=u[i])
        plt.legend()
        plt.title('Samples Location', fontweight='bold', size=18)
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.tight_layout()
        plt.show()

    # create pdf report about samples
    print('saving report...')
    def add_text(text, style="Normal", fontsize=12, space_up=12, space_down=12, color='black'):
        """ Adds text with some spacing around it to  PDF report 

        Parameters
        ----------
        text : str
            The string to print to PDF

        style : str
            The reportlab style

        fontsize : int
            The fontsize for the text
        """
        Story.append(reportlab.platypus.Spacer(1, space_up))
        ptext = "<font size={} color={}>{}</color></font>".format(fontsize, color, text)
        Story.append(reportlab.platypus.Paragraph(ptext, styles[style]))
        Story.append(reportlab.platypus.Spacer(1, space_down))

    # Use basic styles and the SimpleDocTemplate to get started with reportlab
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(os.path.join(save_folder, f'{now.strftime("%Y-%m-%d_%H-%M-%S")}_report.pdf'),
                            pagesize=reportlab.lib.pagesizes.A4,
                            rightMargin=reportlab.lib.units.cm*2,
                            leftMargin=reportlab.lib.units.cm*2,
                            topMargin=reportlab.lib.units.cm*2,
                            bottomMargin=reportlab.lib.units.cm*2)
    
    # The "story" just holds "instructions" on how to build the PDF
    Story = []

    add_text("LSTM Training Samples Report", style="Heading1", fontsize=24, space_up=0, space_down=4)

    add_text(f'id = {now.strftime("%Y-%m-%d_%H-%M-%S")}', space_up=0, space_down=4, color='gray')
    add_text('date = '+now.strftime("%Y/%b/%d %H:%M:%S"), space_up=0, space_down=4, color='gray')

    add_text(f'Save folder: {save_folder}', space_up=12, space_down=4, color='black')
    add_text(f'Samples file: {now.strftime("%Y-%m-%d_%H-%M-%S")}_samples.npy', space_up=0, space_down=4, color='black')
    add_text(f'Reference file: {now.strftime("%Y-%m-%d_%H-%M-%S")}_reference.npy', space_up=0, space_down=0, color='black')

    add_text(' .', space_up=12, space_down=0, color='white')

    data = [['Classes Reference', '', ''],
            ['ID', 'Proportion', 'Class Name']]
    for i in range(len(u)):
        if i in u2:
            data.append([i,f'{"%.1f" % percentage[np.where(u2==i)][0]}%', u[i]])
        else:
            data.append([i,f'{"%.1f" % 0}%', u[i]])
    table=reportlab.platypus.Table(data,style=[('FONTNAME', (0,0), (-1,1), 'Helvetica-Bold'),
                                               ('ALIGN', (0,0), (-1,0), 'CENTER'),
                                               ('GRID',(0,0),(-1,-1),0.5,colors.grey),
                                               ('SPAN',(0,0),(2,0))])

    Story.append(table)
    Story.append(reportlab.platypus.Spacer(1, 12))

    # plotting where the samples were collected
    plt.figure(figsize=(10,10))
    band = stack.read(1)
    plt.imshow(band, cmap='gray', interpolation='nearest')
    for i in u2:
        plt.plot(samples_ind[ref==i,1], samples_ind[ref==i,0], 'o', label=u[i])
    plt.legend()
    plt.title('Samples Location', fontweight='bold', size=18)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()

    image_buffer1 = buf
    im = reportlab.platypus.Image(image_buffer1, 15*reportlab.lib.units.cm, 15*reportlab.lib.units.cm)
    Story.append(im)

    # plotting first sample
    plt.figure(figsize=(10,5))
    band = stack.read(1)
    for i in range(samples.shape[2]):
        plt.plot(samples[0,:,i], label=f'Stack {i+1}')
    plt.legend()
    plt.title(f'First Sample: {u[ref[0]]}', fontweight='bold', size=18)
    plt.xlabel('Time Entries')
    plt.ylabel('Value')
    plt.grid()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()

    image_buffer2 = buf
    im1 = reportlab.platypus.Image(image_buffer2, 15*reportlab.lib.units.cm, 7.5*reportlab.lib.units.cm)
    Story.append(im1)

    # plotting last sample
    plt.figure(figsize=(10,5))
    band = stack.read(1)
    for i in range(samples.shape[2]):
        plt.plot(samples[-1,:,i], label=f'Stack {i+1}')
    plt.legend()
    plt.title(f'Last Sample: {u[ref[0]]}', fontweight='bold', size=18)
    plt.xlabel('Time Entries')
    plt.ylabel('Value')
    plt.grid()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    plt.close()

    image_buffer3 = buf
    im2 = reportlab.platypus.Image(image_buffer3, 15*reportlab.lib.units.cm, 7.5*reportlab.lib.units.cm)
    Story.append(im2)

    # builds the PDF
    doc.build(Story)

    # close open buffers
    image_buffer1.close()
    image_buffer2.close()
    image_buffer3.close()

    # done
    print('DONE!')

def select_samples_ConvLSTM(shapefile, stacks_paths, reference_attribute_name, save_folder, samples_amount, sample_size, save_samples_location=False, divide10000=True, plot_map=False, selection_mode='random'):
    # gets current time to create the samples id
    now = datetime.now()

    print('reading files and preparing for samples selection...')
    # load the samples vector reference
    shp = gpd.read_file(shapefile)

    # load first stack
    stack = r.open(stacks_paths[0])

    # reproject vector reference to match stack
    shp = shp.to_crs(stack.crs.to_dict())

    # converting reference to int with the numbers following each other
    u = np.unique(shp[reference_attribute_name].values)
    ref = shp[reference_attribute_name].values
    for i in range(len(u)):
        ref[ref==u[i]] = i
    ref = np.asarray(ref)

    # if geometry is not polygon
    if shp.geom_type[0]!='Polygon' and shp.geom_type[0]!='MultiPolygon':
        raise FileExistsError('The shapefile exists, but it must be uniquely composed of Points or Polygons.')

    # rasterizes the reference
    print('rasterizing polygons...')
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
    
    # opening rasterized array
    ref_raster = r.open(os.path.join(save_folder, 'rasterized_reference.tif')).read(1)

    # removing auxiliary files
    if os.path.exists(os.path.join(save_folder, 'reprojected_reference.shp')):
        os.remove(os.path.join(save_folder, 'reprojected_reference.shp'))
    if os.path.exists(os.path.join(save_folder, 'reprojected_reference.cpg')):
        os.remove(os.path.join(save_folder, 'reprojected_reference.cpg'))
    if os.path.exists(os.path.join(save_folder, 'reprojected_reference.dbf')):
        os.remove(os.path.join(save_folder, 'reprojected_reference.dbf'))
    if os.path.exists(os.path.join(save_folder, 'reprojected_reference.prj')):
        os.remove(os.path.join(save_folder, 'reprojected_reference.prj'))
    if os.path.exists(os.path.join(save_folder, 'reprojected_reference.shx')):
        os.remove(os.path.join(save_folder, 'reprojected_reference.shx'))
    if os.path.exists(os.path.join(save_folder, 'rasterized_reference.tif')):
        os.remove(os.path.join(save_folder, 'rasterized_reference.tif'))

    # samples per class
    samples_per_class = int(samples_amount/len(u))

    # check if there are enough pixels to create samples from...
    print('testing if there are enough samples origins...')
    for class_id in range(len(u)):
        class_count = np.sum(ref_raster==class_id)
        if class_count<samples_per_class:
            if class_count==0:
                raise Error(f'The class {u[class_id]} has no area to be used as reference. Please check in your shapefile reference if this class is within the stack area or if this class is well represented.')
            print(f'Class {u[class_id]} did not pass. Changing number of samples per class to {class_count}...')
            samples_per_class = class_count

    samples_amount = samples_per_class*len(u)

    # getting origin of samples
    print('getting the samples origins...')
    origins = []
    for class_id in range(len(u)):
        row_col = np.transpose(np.asarray(np.where(ref_raster==class_id)))
        np.random.shuffle(row_col)
        
        ind_aux = 0
        for i in range(samples_per_class):
            not_found = True
            while not_found:
                if ind_aux==len(row_col):
                    raise Error(f'Reference for the class "{u[class_id]}" is too small given the amount of samples required.')
                ref_ = ref_raster[row_col[ind_aux,0]-int(sample_size/2):row_col[ind_aux,0]+int(sample_size/2), row_col[ind_aux,1]-int(sample_size/2):row_col[ind_aux,1]+int(sample_size/2)]
                if ref_.shape==(sample_size,sample_size) and np.sum(ref_==-1)==0:
                    origins.append(row_col[ind_aux])
                    not_found = False
                ind_aux += 1

    del row_col
    origins = np.asarray(origins)
    np.random.shuffle(origins)

    # optional plotting of the samples location
    if plot_map:
        plt.figure(figsize=(20,20))
        plt.imshow(ref_raster, cmap='summer', interpolation='nearest')
        currentAxis = plt.gca()
        colors = [(ii,jj,kk) for ii,jj,kk in np.random.rand(len(u),3)]
        for ii in range(len(u)):
            for i,j in origins[ii*samples_per_class:(ii+1)*samples_per_class]:
                currentAxis.add_patch(Rectangle((j-(sample_size/2), i-(sample_size/2)), sample_size, sample_size, fill=colors[ii], alpha=.5, color=colors[ii]))
        
            plt.plot([-1,-1], [-1,-1], color=colors[ii], label=u[ii])
        plt.legend()

        plt.xlim(0, ref_raster.shape[1])
        plt.ylim(ref_raster.shape[0],0)

        plt.title('Samples Location', fontweight='bold', size=18)

        plt.tight_layout()
        
        plt.legend()
        plt.show()

    # extracting samples from the cubes
    print('extracting samples from stacks...')
    samples = np.zeros([samples_amount, stack.count, sample_size, sample_size, len(stacks_paths)], dtype=int)
    for i in range(len(stacks_paths)):
        stack = r.open(stacks_paths[i])
        for j in range(len(origins)):
            samples[j,:,:,:,i] = stack.read(window=r.windows.Window(origins[j,1]-int(sample_size/2), origins[j,0]-int(sample_size/2), sample_size, sample_size))
    
    # collecting reference
    reference = np.zeros([samples_amount, sample_size, sample_size], dtype=int)
    for i in range(len(origins)):
        reference[i] = ref_raster[origins[i,0]-int(sample_size/2):origins[i,0]+int(sample_size/2), origins[i,1]-int(sample_size/2):origins[i,1]+int(sample_size/2)]
    
    print(f'{len(samples)} samples extracted!')

    # printing log
    # printing the proportion of each class
    u2,c = np.unique(reference, return_counts=True)
    u2 = np.asarray(u2)
    c = np.asarray(c)
    percentage = c*100/np.sum(c)

    text_log = ''
    text_log += '---------------------------------------------------------\nCLASSES REFERENCE\n---------------------------------------------------------\n'
    text_log += 'ID\tPROPORTION\tCLASS NAME\n'
    for i in range(len(u)):
        if i in u2:
            text_log += f'{i}\t{"%.1f" % percentage[np.where(u2==i)][0]}%\t\t{u[i]}\n'
        else:
            text_log += f'{i}\t{"%.1f" % 0}%\t\t{u[i]}\n'
    text_log += '---------------------------------------------------------'

    print(text_log)

    # optional divide by 10,000
    print('dividing by 10,000...')
    if divide10000:
        samples = np.asarray(samples, dtype=np.float32)/10000

    # saving the samples
    print('saving samples...')
    np.save(os.path.join(save_folder, f'{now.strftime("%Y-%m-%d_%H-%M-%S")}_samples.npy'), samples)
    np.save(os.path.join(save_folder, f'{now.strftime("%Y-%m-%d_%H-%M-%S")}_reference.npy'), reference)
    print(f'saved in: {save_folder}')

    print('DONE!')



#####################3
def shuffle_samples(samples, ref):
    '''
    Shuffles the samples and their reference, without mismatching them.

    Parameters
    ----------
    samples: numpy array
        Samples array.
    ref: numpy array.
        Reference array

    Return
    ------
    samples: numpy array
        Shuffled samples.
    ref: numpy array
        Shuffled reference.
    '''
    inds = np.random.default_rng().choice(len(ref), size=len(ref), replace=False)
    return samples[inds], ref[inds]