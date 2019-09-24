import numpy as np
import hyperspy.api as hs

from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.feature import canny
import matplotlib.pyplot as plt


def Find_number(string):
    '''Pulls the number out from between xmp flags.'''
    start = string.find('>') + 1
    finish = string.find('</')
    return string[start:finish]  

def Split_add_scale(all_tifs, files):
    '''Split the HAADF and EBIC images into different stacks. Corrects the spatial scale. The images need to be in a hyperspy stack.'''
    if np.shape(np.shape(all_tifs))[0] == 3:
        "for only one type of image"
        for i in range(0,2):
            all_tifs.axes_manager[i].scale = Pull_Meta_dictionary(files[0])['Scale']*1e09 #convert to nm
            all_tifs.axes_manager[i].units = 'nm'

        HAADFs = all_tifs.split(axis=0)
        return HAADFs
		
    elif np.shape(np.shape(all_tifs))[0] == 4:    

        for i in range(2,4):
            all_tifs.axes_manager[i].scale = Pull_Meta_dictionary(files[0])['Scale']*1e09 #convert to nm
            all_tifs.axes_manager[i].units = 'nm'

        HAADFs, EBIC_raw = all_tifs.split(axis=0)
        return HAADFs, EBIC_raw


def Pull_Meta_dictionary(file):
    '''Searches a tif file for the array of search words. Compiles a dictionary from these. The HAADF values are
    written first, then the EBIC values overwrite these.'''
    
    
    search_words = np.array(['Mag>', 'PixelSizeX>', '<ebic:Ooffset', '<ebic:Contr', '<ebic:InvIoffset',
                         '<ebic:PreampGain', '<cdev:HV', '<ebic:BeamCurrent', 'diAdj:A1Gain>',
                         '<diImg:VideoInCalibration'])

    Meta_dictionary = {}

    flag = 0

    for word in search_words:
        with open(file, 'r', encoding='utf-8', errors='ignore') as fd:
            for line in fd:
                if flag == 1:
                    #print(word)
                    Meta_dictionary.update({word : Find_number(line)})
                    flag = 0

                elif flag == 3:
                    Meta_dictionary.update({'Video_Offset' : Find_number(line)})
                    flag = 0

                elif flag == 2:
                    #print(word)
                    Meta_dictionary.update({word : Find_number(line)})
                    flag = 3




                elif word in line:
                    if word in ['Mag>', 'PixelSizeX>', '<ebic:InvIoffset', 'diAdj:A1Gain>']:
                        Meta_dictionary.update({word : Find_number(line)})
                    elif word in ['<ebic:Contr', '<ebic:Ooffset', '<ebic:PreampGain', '<cdev:HV',
                                  '<ebic:BeamCurrent']:
                        flag = 1

                    elif word in ['<diImg:VideoInCalibration']:
                        flag = 2
    
    Clean_meta_dictionary = {'Mag' : int(Meta_dictionary['Mag>']),
        'Scale' : np.float(Meta_dictionary['PixelSizeX>']),
        'InvIoffset' : np.float(Meta_dictionary['<ebic:InvIoffset']),
        'Ooffset' : np.float(Meta_dictionary['<ebic:Ooffset']),
        'Contrast' : np.float(Meta_dictionary['<ebic:Contr']),
        'PreampGain' : np.float(Meta_dictionary['<ebic:PreampGain']),
        'HV': np.float(Meta_dictionary['<cdev:HV']),
        'BeamCurrent' : np.float(Meta_dictionary['<ebic:BeamCurrent']),
        'A1Gain': np.float(Meta_dictionary['diAdj:A1Gain>']),
        'Video_gain' : np.float(Meta_dictionary['<diImg:VideoInCalibration']),
        'Video_offset' : np.float(Meta_dictionary['Video_Offset'])}
    return Clean_meta_dictionary

def greyscale_to_videoV(value, Metadata_dict):
    'V = ADC-Value / Gain – Offset'
    return (value / Metadata_dict['Video_gain']) - Metadata_dict['Video_offset']

def diffV_to_current(value, Metadata_dict):
    return (((value - Metadata_dict['Ooffset']-(0.5-0.5000))/Metadata_dict['Contrast'])
            - Metadata_dict['InvIoffset'])/np.power(10, Metadata_dict['PreampGain']) * 1e9 * 1.0047

def greyscale_to_current(value, Metadata_dict):
	'''Converts any greyscale value to an ebic current. Uses amp settings in Metadata_dict.'''
	return diffV_to_current(greyscale_to_videoV(value, Metadata_dict), Metadata_dict)

def Efficiency(greyval, Metadata_dict):
    A = ((greyval - Metadata_dict['Ooffset'])/Metadata_dict['Contrast']) - Metadata_dict['InvIoffset']
    
    B = Metadata_dict['HV'] * 250 * (Metadata_dict['BeamCurrent']/1000)
    return np.abs((A/np.power(Metadata_dict['PreampGain'], 10)) * 1e09 * 1.0047) / B



def Plot_return_Hough_transform(im, threshold=480):
    '''Plots the hough transform for the image and returns the hough peaks.'''
    image = im.data > threshold

    h, theta, d = hough_line(image)

    fig, axes = plt.subplots(1, 3, figsize=(15,6))

    ax = axes.ravel()

    ax[0].imshow(im.data)
    ax[0].set_axis_off()

    ax[1].imshow(np.log(1 + h),
                extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
                aspect=1/5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    #ax[1].axis('image')

    ax[2].imshow(image)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()

    accum, angles, dists = hough_line_peaks(h, theta, d)

    plt.savefig('Hough_transform_output.png', dpi=200)
    
    return accum, angles, dists

def Fit_ebic_profile(series, scale, ):
    '''Fits an ebic profile using offset, error function, and a lorenztian.'''
    
    s = hs.signals.Signal1D(series)
    s.axes_manager[0].scale = scale
    s.axes_manager[0].units = 'nm'
    
    m = s.create_model()

    s.axes_manager[0].offset = -np.argmax(s.data)* scale

    lorentzian = hs.model.components1D.Lorentzian() # Create a Lorentzian comp.
    offset = hs.model.components1D.Offset()
    erf = hs.model.components1D.Erf()


    lorentzian.A.value = 4000000
    lorentzian.gamma.value = 30 * scale

    offset.offset.value = np.min(series)

    erf.A.value = -9000
    erf.sigma.value = 50 * scale
    erf.origin.value = 1

    m.append(lorentzian) # Add it to the model
    m.append(offset)
    m.append(erf)
    
    m.fit()
    return m
