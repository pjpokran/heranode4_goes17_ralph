import sys
import os
import os.path
import copy
import numpy as np
from scipy import ndimage
from argparse import ArgumentParser
from netCDF4 import Dataset

from geo import *
import ahi_calc_lib as acl
import crefl_ahi as crefl
import combine_rgb as crgb
import ipdb

FILL_VALUE_FLOAT = -9999.0
missing_value = FILL_VALUE_FLOAT
DTOR = np.pi/180.0

# Copied from R. Kuehn ahi_calc_lib.py 2017-19-05
def compute_space_mask(lat, lon):
    idx = np.isnan(lat)
    if np.any(idx):
        lat[idx] = FILL_VALUE_FLOAT
        lon[idx] = FILL_VALUE_FLOAT
    space_mask = np.logical_or(np.less(lat, -90), np.greater(lat, 90))
    space_mask = np.logical_or(space_mask, np.less(lon, -180))
    space_mask = np.logical_or(space_mask, np.greater(lon, 180))
    return space_mask

# Copied from R. Kuehn satutil_lib.py 2017-19-05
def POSSOL(jday, tu, xlon, xlat, space_mask):
    """
    # Pythonized from geocat/src/program_utils.f90
    ! Compute solar angles
    ! input:
    !         jday = julian day
    !         tu   = time of day - fractional hours
    !         lon  = latitude in degrees
    !         lat  = latitude in degrees
    !      
    !  output:
    !         asol = solar zenith angle in degrees
    !         phis = solar azimuth angle in degrees
    """

    sz = np.shape(xlat)
    asol = np.ones(sz, dtype=float)*missing_value
    phis = np.ones(sz, dtype=float)*missing_value

    # Find the indexes where space mask is not set and only process those
    idx = np.where(np.logical_not(space_mask))
    tsm = tu + xlon[idx]/15.0
    #xlo = xlon[idx]*DTOR
    #xla = xlat[idx]*DTOR

    #  time equation (mn.dec)
    a1 = (1.00554*jday - 6.28306) * DTOR
    a2 = (1.93946*jday + 23.35089) * DTOR
    et = -7.67825*np.sin(a1) - 10.09176*np.sin(a2)

    #      true solar time
    tsv = tsm + et/60.0
    tsv = tsv - 12.0

    #     hour angle
    ah = tsv*15.0*DTOR

    #      solar declination (in radian)
    a3 = (0.9683*jday - 78.00878) * DTOR
    delta = 23.4856*np.sin(a3)*DTOR

    #    elevation, azimuth
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    cos_ah = np.cos(ah)
    sin_xla = np.sin(xlat[idx]*DTOR)
    cos_xla = np.cos(xlat[idx]*DTOR)

    amuzero = sin_xla*sin_delta + cos_xla*cos_delta*cos_ah
    elev = np.arcsin(amuzero)
    cos_elev = np.cos(elev)
    az = np.zeros(sz, dtype=float)
    az[idx] = cos_delta*np.sin(ah)/cos_elev
    caz = np.zeros(sz, dtype=float)
    caz[idx] = (-cos_xla*sin_delta + sin_xla*cos_delta*cos_ah) / cos_elev

    azim = np.zeros(sz, dtype=float)
    #if az >= 1.0:
    jdx = np.where(np.logical_and(np.greater_equal(az, 1), np.logical_not(space_mask)))
    if np.any(jdx):
        azim[jdx] = np.arcsin(1.0)
    #elif az <= -1.0:
    jdx = np.where(np.logical_and(np.less_equal(az, -1), np.logical_not(space_mask)))
    if np.any(jdx):
        azim[jdx] = np.arcsin(-1.0)
    #else
    jdx = np.where(np.logical_and(np.logical_and(np.greater(az, -1), np.less(az,1)), np.logical_not(space_mask)))
    if np.any(jdx):
        azim[jdx] = np.arcsin(az[jdx])

    #if caz <= 0.0:
    jdx = np.where(np.logical_and(np.greater_equal(caz, 0), np.logical_not(space_mask)))
    if np.any(jdx):
        azim[jdx] = np.pi - azim[jdx]

    #if caz > 0.0 and az <= 0.0:
    jdx = np.where(np.logical_and(np.logical_and(np.greater(caz, 0.0), np.less_equal(az, 0.0)),  np.logical_not(space_mask)))
    if np.any(jdx):
        azim[jdx] = 2 * np.pi + azim[jdx]
    azim[idx] = azim[idx] + np.pi
    #if azim > pi2:
    jdx = np.where(np.logical_and(np.greater(azim, np.pi*2), np.logical_not(space_mask)))
    if np.any(jdx):
        azim[jdx] = azim[jdx] - np.pi*2

    #         asol = solar zenith angle in degrees
    #         phis = solar azimuth angle in degrees
    #     conversion in degrees
    elev = elev / DTOR
    asol[idx] = 90.0 - elev
    #     phis(i,j) = azim / DTOR - 180.0
    phis[idx] = azim[idx] / DTOR  # akh - try to get 0 - 360

    return {'asol': np.reshape(asol, sz), 'phis': np.reshape(phis, sz)}

# Copied from R. Kuehn satutil_lib.py 2017-19-05
def COMPUTE_SATELLITE_ANGLES(satlon, satlat, xlon, xlat, space_mask):
    """
    Pythonized from geocat/src/program_utils.f90
    ! Subroutine to make geostationary satellite azimuth field
    !
    !     xlon = longitude of the location (positive for western hemisphere) 
    !     xlat = latitude of the location  (positive for northern hemisphere) 
    !
    !     zenith  = satellite zenith view angle 
    !     azimuth = satellite azimuth angle clockwise from north
    """

    sz = np.shape(xlat)
    zenith = np.ones(sz, dtype=float)*missing_value
    azimuth = np.ones(sz, dtype=float)*missing_value
    lon = np.ones(sz, dtype=float)*missing_value
    lat = np.ones(sz, dtype=float)*missing_value
    beta = np.ones(sz, dtype=float)*missing_value

    idx = np.where(np.logical_not(space_mask))

    lon[idx] = (xlon[idx] - satlon) * DTOR   # in radians
    lat[idx] = (xlat[idx] - satlat) * DTOR   # in radians

    beta[idx] = np.arccos( np.cos(lat[idx]) * np.cos(lon[idx]) )
    zenith[idx] = 42164.0 * np.sin(beta[idx]) / np.sqrt(1.808e09 - 5.3725e08 * np.cos(beta[idx]))
    #     zenith angle
    #OLD FORTRANISH IMPLEMENTATION, INCOMPLETE PYTHON 
    #zenith[idx] = np.arcsin(np.max(-1.0, np.min(1.0, temp))) / DTOR
    jdx = np.where(np.logical_and(np.greater(zenith, 1.0), np.logical_not(space_mask)))
    kdx = np.where(np.logical_and(np.less(zenith, -1.0), np.logical_not(space_mask)))
    ldx = np.where(np.logical_and(np.logical_and(np.greater_equal(zenith, -1.0), np.logical_not(space_mask)), np.less_equal(zenith, 1.0)))
    if np.any(jdx):
        zenith[jdx] = np.arcsin(1.0)/DTOR
    if np.any(kdx):
        zenith[kdx] = np.arcsin(-1.0)/DTOR
    if np.any(ldx):
        zenith[ldx] = np.arcsin(zenith[ldx])/DTOR

    #     azimuth angle
    azimuth[idx] = np.sin(lon[idx]) / np.sin(beta[idx])
    jdx = np.where(np.logical_and(np.less(azimuth, -1.0), np.logical_not(space_mask)))
    kdx = np.where(np.logical_and(np.greater(azimuth, 1.0), np.logical_not(space_mask)))
    ldx = np.where(np.logical_and(np.logical_and(np.greater_equal(azimuth, -1.0), np.logical_not(space_mask)), np.less_equal(azimuth, 1.0)))
    #azimuth[idx] = np.arcsin( min(1.0, max(-1.0, azimuth[idx])) ) / DTOR
    if np.any(jdx):
        azimuth[jdx] = np.arcsin(1.0)/DTOR
    if np.any(kdx):
        azimuth[kdx] = np.arcsin(-1.0)/DTOR
    if np.any(ldx):
        azimuth[ldx] = np.arcsin(azimuth[ldx])/DTOR

    #if lat < 0.0:
    temp_lat = np.zeros(sz, dtype=float)
    temp_lat[idx] = (xlat[idx] - satlat) * DTOR   # in radians
    jdx = np.where(np.logical_and(np.less(temp_lat, 0.0), np.logical_not(space_mask)))
    if np.any(jdx):
        azimuth[jdx] = 180.0 - azimuth[jdx]

    #if (azimuth(i,j) < 0.0) then
    jdx = np.where(np.logical_and(np.less(azimuth, 0.0), np.logical_not(space_mask)))
    if np.any(jdx):
        azimuth[jdx] = azimuth[jdx] + 360.0

    return {'zenith': np.reshape(zenith, sz), 'azimuth': np.reshape(azimuth, sz)}

# Copied from R. Kuehn ah_calc_lib.py 2017-19-05
def scat_calc_short(AHI_name, lat, lon, senzen, senazi, space_mask=None):
    # Computes the solar and satellite scattering angles.
    if space_mask is None:
        space_mask = compute_space_mask(lat, lon)

    date_dict = acl.get_ABI_time(AHI_name)

    print("Computing solar angles...")
    solar_ang = POSSOL(date_dict['jday'], date_dict['dec_hour'], lon, lat, space_mask)
    #         asol = solar zenith angle in degrees
    #         phis = solar azimuth angle in degrees
    print("Done.")
    data = {'solzen': solar_ang['asol'],
            'solazi': solar_ang['phis'],
            'space_mask': space_mask}
    return data

def compute_subset_indexes(ll_ur, input_resolution=1.0, output_resolution=None):
    # Takes the lower left and upper right lat,lon coordinates and 
    # computes the indecies in the geostationary projection. 
    # The resulting domain will alway be rectangular.
    # Input: 
    # ll_ur: Lower left, upper right bounding box corner points vector: [ll_lat, ll_lon, ur_lat, ur_lon] 
    
    print("output resolution is ",output_resolution)
    assert(output_resolution > 2.0, 'Output resolutions greater than 2km not supported')

    if output_resolution is None:
        output_resolution = input_resolution

    # TODO note that geo will return a NaN if the lat,lon pair is not in the field of view, not checking for that yet
    #
    
    sub_lon_degrees = -75.0
    proj = AbiProjection(subsat_lon=sub_lon_degrees, resolution=input_resolution)
    fact = output_resolution/input_resolution

    line, element = proj.index_from_location(np.asarray([ll_ur[0], ll_ur[2]]), np.asarray([ll_ur[1], ll_ur[3]]))
    if np.any(np.isnan(line)) or np.any(np.isnan(element)):
        print('One of the lat/lon corner points is one of bounds for the projection')
        print('line: ', line)
        print('element: ', element)
        raise ValueError()


    ele = [np.int(np.round(np.min(element))), np.int(np.round(np.max(element)))]
    line = [np.int(np.round(np.min(line))), np.int(np.round(np.max(line)))]
    if int(fact) > 1:
        if ele[0] % fact != 0:
            ele[0] = int(np.floor(ele[0] / fact)*fact)

        dx = (ele[1]-ele[0])
        if dx % fact != 0:
            dx = np.floor(dx / fact)*fact
            ele[1] = int(ele[0]+dx)
        
        if line[0] % fact != 0:
            line[0] = int(np.floor(line[0] / fact)*fact)

        dx = (line[1]-line[0])
        if dx % fact != 0:
            dx = np.floor(dx / fact)*fact
            line[1] = int(line[0]+dx)

    index={'resolution': input_resolution, 'data': {'element': ele, 'line': line}}

    return index

def load_data(filename=None, aslice=None):
    fn_obj = acl.ABI_fname(filename)
    fn_obj.display()
    ncfile = Dataset(filename,'r')
    for var in ncfile.variables:
        ncfile.variables[var].set_auto_maskandscale(False)
    
    band_id= int(ncfile.variables['band_id'][:])
    #b_id = ABI_BANDS_MAP[band_id-1]
    band_wavelength= ncfile.variables['band_wavelength']
    if fn_obj.band <= 6:
        kappa0 = ncfile.variables['kappa0'][:]
        if aslice is not None:
            ABI_data = acl.nc4_convert_data(ncfile.variables['Rad'], aslice=aslice)*kappa0
        else:
            ABI_data = acl.nc4_convert_data(ncfile.variables['Rad'])*kappa0

    ncfile.close()
    return ABI_data, band_id

def block_mean(ar, fact):
    # Copied from https://stackoverflow.com/questions/18666014/downsample-array-in-python
    # WARNING this does not appear to wor under python 3.6, the returned array 'res' 
    # does not have square dimensions, not sure what is up.
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy//fact * (X//fact) + Y//fact
    print('regions = ',regions)
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (int(sx/fact), int(sy/fact))
    return res

# This is a spin off of the the ahi_calc.py module, for ABI corrected reflectence processing only.
def abi_make_crefl(output_path=None, ahi_fnames=None, input_resolutions=[0.5, 1.0, 1.0], output_resolution=1.0, tmp_path=None, geo_fname=None, terrain_fname=None, ll_ur=None, BT=False, sector="FULLDISK", debug=0):
    # input resolutions: list that correspond to the sub-satellite point nominal pixel rsolution for the band
    # Output resolution: resolution of the output image data.

    print('in abi_make_crefl, output_resolution = ',output_resolution)
    print('in abi_make_crefl, sector = ', sector)

#    quit()
 

    sub_lon_degrees = -75.0
    reflectance=[]
    band_id = []
    proj = AbiProjection(subsat_lon=sub_lon_degrees, resolution=output_resolution)
    if ll_ur is not None:
        ox = compute_subset_indexes(ll_ur, output_resolution=output_resolution)
    else:
        if sector is "FULLDISK":
            base_dim = 10848/int(output_resolution) # This is the base 1km resolution
            ox={'resolution': output_resolution, 'data': {'element': [0, base_dim], 'line': [0, base_dim]}}
            oxr={'resolution': output_resolution, 'data': {'element': [0, base_dim], 'line': [0, base_dim]}}
        elif sector is "CONUS":
#            ncfile = Dataset(fn,'r')
#            yvals = ncfile.dimensions['y']
#            xvals = ncfile.dimensions['x']
#            ncfile.close()
            yline0 = int(proj.line_from_angle(0.12824)  )
            yline1 = int(proj.line_from_angle(0.04424)  )
            xline0 = int(proj.element_from_angle (-0.10136))
            xline1 = int(proj.element_from_angle ( 0.03864))
            print("PPP yline0,yline1,xline0,xline1 ",yline0,yline1,xline0,xline1)
#            print ("ylines[0], [-1]   :",ylines[0], ylines[-1])
#            print ("xelements[0], [-1]:",xelements[0], yxelements[-1])
            base1_dim = 3000//int(output_resolution) # This is the base 1km resolution
            base2_dim = 5000//int(output_resolution) # This is the base 1km resolution
            print ("PPP base1_dim,base2_dim ",base1_dim, base2_dim)
            ox={'resolution': output_resolution, 'data': {'element': [0, base2_dim], 'line': [0, base1_dim]}}
            oxr={'resolution': output_resolution, 'data': {'element': [xline0, xline1], 'line': [yline0, yline1]}}
        elif sector is "MESO":
            print("Meso")
            ncfile = Dataset(ahi_fnames[1],'r')
#            yvals = ncfile.dimensions['y']
#            xvals = ncfile.dimensions['x']
#            print("yvals[0],yvals[-1] ",yvals[0],yvals[-1] )
#            print("xvals[0],yvals[-1] ",xvals[0],yvals[-1] )
# ncfile.variables["y_image_bounds"][0],ncfile.variables["y_image_bounds"][-1]
            yline0 = int(proj.line_from_angle(ncfile.variables["y_image_bounds"][0]))
            yline1 = int(proj.line_from_angle(ncfile.variables["y_image_bounds"][-1]))
            xline0 = int(proj.element_from_angle (ncfile.variables["x_image_bounds"][0]))
            xline1 = int(proj.element_from_angle (ncfile.variables["x_image_bounds"][-1]))
            print("PPP yline0,yline1,xline0,xline1 ",yline0,yline1,xline0,xline1)
#            print ("ylines[0], [-1]   :",ylines[0], ylines[-1])
#            print ("xelements[0], [-1]:",xelements[0], yxelements[-1])
            base1_dim = len(ncfile.dimensions['y'])//int(output_resolution) # This is the base 1km resolution
            base2_dim = len(ncfile.dimensions['x'])//int(output_resolution) # This is the base 1km resolution
            ncfile.close()
            print ("PPP base1_dim,base2_dim ",base1_dim, base2_dim)
            ox={'resolution': output_resolution, 'data': {'element': [0, base2_dim], 'line': [0, base1_dim]}}
            oxr={'resolution': output_resolution, 'data': {'element': [xline0, xline1], 'line': [yline0, yline1]}}
        else:
            print("Unsupported sector: ",sector)
            quit()

    print('Output resolution struct:')
    print(ox)
    for i, fn in enumerate(ahi_fnames):
        #ipdb.set_trace()
        if not os.path.isfile(fn):
            print('Error: Input file does not exist, %s' % (fn))
            assert(False)

        ncfile = Dataset(fn,'r')
        dim_size = len(ncfile.dimensions['x'])
        ncfile.close()
        if ll_ur:
            #ipdb.set_trace()
            ix = compute_subset_indexes(ll_ur, input_resolution=input_resolutions[i], output_resolution=output_resolution)
            islice = np.s_[ix['data']['line'][0]:ix['data']['line'][1],
                ix['data']['element'][0]:ix['data']['element'][1]]
        else:
            print('This is going to be a full disk image... enjoy your wait!')
            islice = np.s_[0:dim_size, 0:dim_size]

        print('islice: ', islice)
        tmp, b_id = load_data(filename=fn, aslice=islice)
        band_id.append(b_id)

        factor = int(output_resolution/input_resolutions[i])
        print('Scale factor: %d' % (factor))
        if factor  > 1:
            print(np.shape(tmp))
            #ipdb.set_trace()
            reflectance.append(block_mean(tmp, factor)) 
        else:
            reflectance.append(copy.deepcopy(tmp))
        print('Refl: ', np.shape(reflectance[i]))
    #ipdb.set_trace()
    # Here is where x and y get set
    # X,Y for data
    X = np.arange(ox['data']['element'][0], ox['data']['element'][1])
    Y = np.arange(ox['data']['line'][0], ox['data']['line'][1])
    Y = Y[:, np.newaxis] # Essentially transpose the Y vector so that broadcast_arrays in location_from_index is happy
    print("np.shape(X)",np.shape(X))
    print("np.shape(Y)",np.shape(Y))
    # X,Y for radiative angles
    XR = np.arange(oxr['data']['element'][0], oxr['data']['element'][1])
    YR= np.arange(oxr['data']['line'][0], oxr['data']['line'][1])
    YR = YR[:, np.newaxis] # Essentially transpose the Y vector so that broadcast_arrays in location_from_index is happy
    print("np.shape(XR)",np.shape(XR))
    print("np.shape(YR)",np.shape(YR))
    proj = AbiProjection(subsat_lon=sub_lon_degrees, resolution=output_resolution)
    lat, lon = proj.location_from_index(Y, X)
    latr, lonr = proj.location_from_index(YR, XR)
    #lat = np.rot90(lat)
    #lon = np.rot90(lon)

# The plan:
    # Use slice objects to keep track of subset data if necessary.
    

    # Compute geolocation and scattering calculations
    space_mask = compute_space_mask(latr, lonr)
    sat_ang = COMPUTE_SATELLITE_ANGLES(-1.0*sub_lon_degrees, 0.0, -1.0*lonr, latr, space_mask)
    
    solar_data = scat_calc_short(ahi_fnames[0], latr, lonr, sat_ang['zenith'], sat_ang['azimuth'], space_mask=space_mask)
    data = {'lat': latr,
            'lon': lonr,
            'sub_lon_degrees': sub_lon_degrees,
            'senzen': sat_ang['zenith'],
            'senazi': sat_ang['azimuth'],
            'solzen': solar_data['solzen'],
            'solazi': solar_data['solazi'],
            'space_mask': space_mask}


    # Do crefl pre load
    crefl_pp_dict = crefl.crefl_ahi_preprocess(data, terrain_fname=terrain_fname)
    print('Finished preprocess')

    # Loop through data files and do crefl for each band
    crefl_files=[]
    for i, fn in enumerate(ahi_fnames):
        abi_name = acl.ABI_fname(fn)
        reflectance[i] = reflectance[i]/crefl_pp_dict['mus']
        corr_refl = crefl.crefl_ahi(preprocess_dict=crefl_pp_dict, band_number=band_id[i], reflectance=reflectance[i], space_mask=space_mask)

        sz = np.shape(reflectance[i])
        sx = sz[0]
        sy = sz[1]
        #oname = 'AHI_CREFL_' + fn_obj.output_date() + '_' + str(int(band_number)) + '.real4.' + str(sy) + '.' + str(sx)
        #TODO FIX crefl  file name
        oname = 'AHI_CREFL_' + abi_name.output_date() + '_' + str(int(band_id[i])) + '.real4.' + str(sy) + '.' + str(sx)
        ofile = os.path.join(tmp_path, oname) 
        crefl_files.append(ofile)
        print('Writing file temporary  file %s' % (ofile))
        np.float32(corr_refl).tofile(ofile)


    # Do combine rgb
    crgb.combine_rgb(output_path=output_path,
        fnred=crefl_files[0],
        fngreen=crefl_files[1],
        fnblue=crefl_files[2],
        no_nonlin_scale=False,renorm=1.1,ncfile=ahi_fnames[1])
    print('Finished making rgb image in directory %s ' % (output_path))

    for crfn in crefl_files:
        os.remove(crfn)

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-o', '--output-path')
    parser.add_argument('-i', '--ahi-fnames', nargs='+')
    parser.add_argument('-t', '--tmp-path')
    parser.add_argument('-r', '--terrain')
    parser.add_argument('-g', '--geo_fname')
    parser.add_argument('-s', '--section', nargs='+', help='max min lat max min lon')
    parser.add_argument('-x', '--debug')
    parser.add_argument('--BT', action='store_true', default=False, help='Flag for brightness temperature files. All files are processed as BT')
    args = parser.parse_args()
    isBT = False
    if args.BT:
        isBT=True

    if args.geo_fname:
        geo_fname=args.geo_fname
    else:
        geo_fname=None

    if args.debug:
        ahi_calc(args.output_path, args.ahi_fnames, args.tmp_path, args.geo_fname, args.terrain, section=float(args.section), debug=int(args.debug), BT=isBT)
    else:
        if args.section is not None:
            tmp = map(float,args.section)
            print(tmp)
            ahi_calc(args.output_path, args.ahi_fnames, args.tmp_path, geo_fname, args.terrain, section=tmp, BT=isBT)
        else:
            ahi_calc(args.output_path, args.ahi_fnames, args.tmp_path, geo_fname, args.terrain, BT=isBT)
