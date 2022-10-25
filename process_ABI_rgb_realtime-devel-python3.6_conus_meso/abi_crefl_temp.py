#!/usr/bin/env python
# #!/usr/bin/env /sw/bin/python2.6

import string
import numpy as np
#import math
#import ipdb
#import h5py
import os
import sys
import timeit
import time
import re
from argparse import ArgumentParser
from netCDF4 import Dataset
from datetime import timedelta, datetime

import collocate_follower.collocate_follower as cf
from geo import AbiProjection

FILL_VALUE_FLOAT = -9999.0
DTOR = np.pi/180.0
missing_value = FILL_VALUE_FLOAT

def nc4_convert_data(data):
    allatt = [foo for foo in data.ncattrs()]
    sf = 1.0
    offset = 0.0
    fv = -1.0
    if 'scale_factor' in allatt:
        sf = data.getncattr('scale_factor')
    if 'add_offset' in allatt:
        offset = data.getncattr('add_offset')
    if '_FillValue' in allatt:
        fv = data.getncattr('_FillValue')
        idx = np.equal(data[:], fv)
        out = np.float32(data[:])*sf + offset
        out[idx] = -FILL_VALUE_FLOAT
        return out
    else:
        return np.float32(data[:])*sf + offset

# Copied from R. Kuehn ahi_calc_lib.py 2017-19-05
def compute_space_mask(lat, lon):
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

# Copied from R. Kuehn satutil_lib.py 2017-19-05
class ABI_fname:
    def __init__(self,fname):
        #    OR_ABI-L1b-RadF-M3C01_G16_s20170711900028_e20170711910395_c20170711910436.nc
        (self.orig_dir,self.name) = os.path.split(fname)
        #ipdb.set_trace()
        foo = self.name.split('.')[0]
        self.long_name = foo
        foo = re.split('\W+|_',self.name) #matches any non-alphanumeric character OR '_'
        self.sat_name = foo[0] + '_' + foo[1] 
        ymd = foo[6]
        self.band = int(foo[4][-2:])

        self.yr = ymd[1:5]
        self.doy = ymd[5:8]
        #self.dd = ymd[7:9]
        self.stime = ymd[8:12]
        self.date_obj = datetime.strptime(ymd[1:8] + ' ' + self.stime, '%Y%j %H%M')
        self.mm = self.date_obj.strftime('%m')
        self.dd = self.date_obj.strftime('%d')
            

    def output_fname(self):
        return self.long_name

    def output_date(self):
        return self.date_obj.strftime('d%Y%m%d_t%H%M')

    def display(self):
        print((self.long_name))

# Copied from R. Kuehn ah_calc_lib.py 2017-19-05
def get_ABI_time(fname):
    fn_obj = ABI_fname(fname)
    fn_obj.display()

    ncfile = Dataset(fname,'r')
    for var in ncfile.variables:
        ncfile.variables[var].set_auto_maskandscale(False)

    date_obj = datetime.strptime('20000101 1200', '%Y%m%d %H%M')
    base_time = timedelta(seconds=ncfile.variables['time_bounds'][0])
    date_obj = date_obj + base_time
    #date_obj = datetime.utcfromtimestamp(base_time)
    test_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
    print('The current date for this granule is:' + test_date)
    jday = int(date_obj.strftime('%j'))
    hh = int(date_obj.strftime('%H'))
    mm = int(date_obj.strftime('%M'))
    ss = int(date_obj.strftime('%S'))
    tu = hh+mm/60.0 + ss/3600.0

    ncfile.close()
    return {'d_obj': date_obj, 'jday': jday, 'dec_hour':tu}

# Copied from R. Kuehn ah_calc_lib.py 2017-19-05
def scat_calc_short(AHI_name, lat, lon, senzen, senazi, space_mask=None):
    # Computes the solar and satellite scattering angles.
    if space_mask is None:
        space_mask = compute_space_mask(lat, lon)

    date_dict = get_ABI_time(AHI_name)

    print("Computing solar angles...")
    solar_ang = POSSOL(date_dict['jday'], date_dict['dec_hour'], lon, lat, space_mask)
    #         asol = solar zenith angle in degrees
    #         phis = solar azimuth angle in degrees
    print("Done.")
    data = {'solzen': solar_ang['asol'],
            'solazi': solar_ang['phis'],
            'space_mask': space_mask}
    return data

def compute_subset_indexes(ll, ur):
    # Takes the lower left and upper right lat,lon coordinates and 
    # computes the indecies in the geostationary projection. 
    # The resulting domain will alway be rectangular.
    # Input: 
    # ll: lower left [lat, lon] pair
    # lr: upper right [lat, lon] pair

    # TODO: input checking
    sub_lon_degrees = -89.5
    proj = AbiProjection(subsat_lon=sub_lon_degrees, resolution=ABI_RES)
    line, element = proj.index_from_location([ll[0], ur[0]], [ll[1], ur[1]])
    #x = np.zeros((2,), dtype=np.int32)
    #y = np.zeros((2,), dtype=np.int32)
    indx = {'element': [np.min(element), np.max(element)],
    'line': [np.min(line), np.max(line)]}

    return indx


def abi_subset(output_path='./', output_file=None, master_in_file=None, abi_resolution='1km',
    index=None, debug=False):

    # Note that the abi_resolution parameter is used to set the default set of output datasets.
    # For the moment if this is not set consistently with the input data sets, then the code will fail.

    global sys

    if master_in_file:
        bReadMaster = True
        print('Master file: %s' % master_in_file)

    print('Output file: %s' % output_file)

    # Kill numpy floating point errors for now
    np.seterr(all='ignore')


    if abi_resolution == '1km':
        ABI_RES = 1.0
        # For mapping the band_id to an index. E.g. band_id=1,  would get index 0, band_id=5, index 3
        NUM_REFL_BANDS = 4 # Constant value
        NUM_EM_BANDS = 0
        ABI_BANDS_MAP = [0,1,2,-9,3]
    elif abi_resolution == '2km':
        ABI_RES = 2.0
        NUM_REFL_BANDS = 2 # Constant value
        NUM_EM_BANDS = 4
        ABI_BANDS_MAP = [-9, -9, -9, 0, -9, 1, 2, -9, -9, -9, 3, -9, 4, -9, 5, -9, -9]
                        #[0,  1,  2, 3,  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ABI_BANDS_MAP_BT = [-9, -9, -9, -9, -9, -9, 0, -9, -9, -9, 1, -9, 2, -9, 3, -9, -9]
                        #[0,  1,  2, 3,  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    else:
        print('Error: The abi_resolution parameter needs to be 1km or 2km')
        sys.exit(-1)
     
    # Output file stuff
    oname = output_file
    ofile = os.path.join(output_path, oname + '.nc')
    dataset = Dataset(ofile, 'w', format='NETCDF4')
    dataset.description = "ABI-VIIRS match file"
    sx=np.size(master_index[:,0])
    master_obs = dataset.createDimension('master_obs', sx)
    dataset.close()

    if bReadMaster:
        # Count the number of reflective and emissive bands in ABI/Himawari input
        n_refl_bands = 0
        n_em_bands = 0
        for i, fn in enumerate(master_in_file):
            fn_obj = ABI_fname(fn)
            if fn_obj.band <= 7:
                n_refl_bands +=1
            else:
                n_em_bands =+1

        # This logic handles cases where there are no reflective or emissive bands requested
        if not follower_refSB:
            follower_refSB = []

        if not follower_emB:
            follower_emB = []

        n_follower_bands = np.max([len(follower_refSB), len(follower_emB)])

        dataset = Dataset(ofile, 'a', format='NETCDF4')
        dataset.description = 'ABI-VIIRS matchfile, care of UW-MADISON SSEC/CIMSS'
        dataset.history = 'Created ' + time.ctime(time.time())
        dataset.source = 'ABI_viirs_match.py'
        nbands = len(master_in_file)
        rband = dataset.createDimension('rband', NUM_REFL_BANDS)
        if NUM_EM_BANDS:
            eband = dataset.createDimension('eband', NUM_EM_BANDS)
        if n_follower_bands:
            follower_nbands = dataset.createDimension('follower_nbands', n_follower_bands)

        for dimname in list(dataset.dimensions.keys()):
            dim = dataset.dimensions[dimname]
            print("dimname: %s, length: %d, is_unlimited: %g" % (dimname, len(dim), dim.isunlimited()))
        varlon = dataset.createVariable('ABI_longitude', np.float32, ('master_obs'))
        varlat = dataset.createVariable('ABI_latitude', np.float32, ('master_obs'))
        var_sat_zen = dataset.createVariable('ABI_sensor_zenith', np.float32, ('master_obs'))
        var_sat_azi = dataset.createVariable('ABI_sensor_azimuth', np.float32, ('master_obs'))
        var_sol_zen = dataset.createVariable('solar_zenith', np.float32, ('master_obs'))
        var_sol_azi = dataset.createVariable('solar_azimuth', np.float32, ('master_obs'))
        var_refl_band = dataset.createVariable('ABI_reflectance', np.float32, ('master_obs', 'rband'), fill_value=FILL_VALUE_FLOAT)
        if NUM_EM_BANDS:
            var_em_band   = dataset.createVariable('ABI_radiance', np.float32, ('master_obs', 'eband'), fill_value=FILL_VALUE_FLOAT)
            var_bt   = dataset.createVariable('ABI_bt', np.float32, ('master_obs', 'eband'), fill_value=FILL_VALUE_FLOAT)
        var_sp_var = dataset.createVariable('ABI_spatial_variance', np.float32, ('master_obs'))

        if n_em_bands > 0:
            tmp_rad = np.ones((sx, NUM_EM_BANDS), dtype=np.float32)*FILL_VALUE_FLOAT
            tmp_bt = np.ones((sx, NUM_EM_BANDS), dtype=np.float32)*FILL_VALUE_FLOAT

        if n_refl_bands:
            tmp_alb = np.ones((sx, NUM_REFL_BANDS), dtype=np.float32)*FILL_VALUE_FLOAT

        RAD = 'Rad'
        rband_cnt = 0
        eband_cnt = 0
        band_attr_refl = []
        band_attr_em = []
        for i, fn in enumerate(master_in_file):
            fn_obj = ABI_fname(fn)
            fn_obj.display()
            ncfile = Dataset(fn,'r')
            for var in ncfile.variables:
                ncfile.variables[var].set_auto_maskandscale(False)
            
            band_id= int(ncfile.variables['band_id'][:])
            b_id = ABI_BANDS_MAP[band_id-1]
            band_wavelength= ncfile.variables['band_wavelength']
            if fn_obj.band <= 6:
                # Note that getncattr throws an error so I'm using the following syntax
                # kappa0 is used to convert radiance to albedo
                kappa0 = ncfile.variables['kappa0']
                ABI_data = nc4_convert_data(ncfile.variables[RAD])*kappa0
                print(('Band ID %d'% (b_id)))
                if fn_obj.band == 2:
                    var_refl_band[:, b_id] = cf.collocate_master_hires(ABI_data, master_index)
                else:
                    var_refl_band[:, b_id] = ABI_data[master_index[:,0],master_index[:,1]]
                rband_cnt += 1
                band_attr_refl.append(fn_obj.output_fname())
            else:
                b_id = ABI_BANDS_MAP_BT[band_id-1]
                # For emissive bands just save radiance
                print(('Computing BT for band %d, %d' % (band_id, b_id)))
                ABI_data = nc4_convert_data(ncfile.variables[RAD]).astype(np.float32)
                var_em_band[:, b_id] = ABI_data[master_index[:,0],master_index[:,1]]

                fk1 = ncfile.variables['planck_fk1'][:]
                fk2 = ncfile.variables['planck_fk2'][:]
                bc1 = ncfile.variables['planck_bc1'][:]
                bc2 = ncfile.variables['planck_bc2'][:]
                var_bt[:, b_id] = (fk2 / (np.log((fk1/var_em_band[:, b_id]) + 1.0)) - bc1) / bc2

            if abi_resolution == '1km' and fn_obj.band == 2:
                print("Computing master spatial variance")
                #start_time = timeit.default_timer()
                var_sp_var[:] = cf.spatial_variability(ABI_data, master_index, 2)
                #elapsed = timeit.default_timer() - start_time
                #print(elapsed)

            ncfile.close()

        # TODO write better band attribute information
        #var_refl_band[:] = tmp_alb
        var_refl_band.source = ', '.join(band_attr_refl)
        if n_em_bands > 0:
            var_em_band.source = ', '.join(band_attr_em)
            var_bt.source = ', '.join(band_attr_em)


        if master_geo_file:
            ncfile = Dataset(master_geo_file,'r')
            for var in ncfile.variables:
                ncfile.variables[var].set_auto_maskandscale(False)
                #print(var)

            #ipdb.set_trace()
            tmp = ncfile.variables['latitude'][:]
            lat = tmp[master_index[:,0], master_index[:,1]]
            varlat[:] = lat
            tmp = ncfile.variables['longitude'][:]
            lon = tmp[master_index[:,0], master_index[:,1]]
            varlon[:] = lon
            tmp = ncfile.variables['sensor_zenith'][:]
            senzen = tmp[master_index[:,0], master_index[:,1]]
            var_sat_zen[:] = senzen
            tmp = ncfile.variables['sensor_azimuth'][:]
            senazi = tmp[master_index[:,0], master_index[:,1]]
            var_sat_azi[:] = senazi
            ncfile.close()
            space_mask = None
        else:
            sub_lon_degrees = -89.5
            proj = AbiProjection(subsat_lon=sub_lon_degrees, resolution=ABI_RES)
            lat, lon = proj.location_from_index(master_index[:,0], master_index[:,1])
            space_mask = compute_space_mask(lat, lon)
            sat_ang = COMPUTE_SATELLITE_ANGLES(-1.0*sub_lon_degrees, 0.0, -1.0*lon, lat, space_mask)
            senzen = sat_ang['zenith']
            senazi = sat_ang['azimuth']
            
            varlat[:] = lat
            varlon[:] = lon
            var_sat_azi[:] = sat_ang['azimuth']
            var_sat_zen[:] = sat_ang['zenith']

        solar_angles = scat_calc_short(master_in_file[0], lat, lon, senzen, senazi, space_mask=space_mask)

        var_sol_azi[:] = solar_angles['solazi']
        var_sol_zen[:] = solar_angles['solzen']
        print("Writing data file.")
        dataset.close()


    if bReadFollower:
        import sys
        if not follower_name:
            print('Must provide (VIIRS) or (MODIS) for follower name')
            sys.exit(-1)
        print('Matching follower file')
        follower_angle_datasets = ['sensor_azimuth', 'sensor_zenith']
        #follower_angle_datasets = ['sensor_azimuth', 'sensor_zenith', 'solar_azimuth', 'solar_zenith']
        follower_geo_datasets = ['latitude', 'longitude']

        # M1 0.412 
        # M2 0.445 
        # M3 0.488 B
        # M4 0.555 G
        # M5 0.672 R
        # M6 0.746 
        # M7 0.865 * G
        # M8 1.240 
        # M9 1.378 
        # M10 1.61 
        # M11 2.25 
        # M12 3.7 
        # M13 4.05 
        # M14 8.55 
        # M15 10.763 
        # M16 12.013

        VB = {'M01': 0.412,
        'M02': 0.445, 
        'M03': 0.488,
        'M04': 0.555,
        'M05': 0.672,
        'M06': 0.746,
        'M07': 0.865,
        'M08': 1.240,
        'M09': 1.378,
        'M10': 1.61, 
        'M11': 2.25,
        'M12': 3.7,
        'M13': 4.05,
        'M14': 8.55, 
        'M15': 10.763,
        'M16': 12.013}


        # M3, M5, M7, M9, M10
        #
        # Get the collocated Follower pixels
        #
        vidx = follower_index[:,:,1]
        hidx = follower_index[:,:,0]

        nCollFov = np.shape(follower_index)[0]
        # Find the number of collocated fov in the master footprint
        no_data_mask = np.less(vidx, 1)
        z = np.ma.array(vidx, mask=no_data_mask)
        nfov = ((z/z).sum(axis=1)).astype(int).data

        max_nfov = int(max(nfov))
        #ipdb.set_trace()
        print('Max number of follower collocated with master %d\n' % max_nfov)
        sys.stdout.flush()

        # Open the output file for writing as we will do this incrementally
        dataset = Dataset(ofile, 'a', format='NETCDF4')
        ncfile = Dataset(follower_in_file, 'r')
        nc_data = ncfile.groups['observation_data']
        for var in nc_data.variables:
            nc_data.variables[var].set_auto_maskandscale(False)
            
        ncfile_geo = Dataset(follower_in_geo_file, 'r')
        geo_data = ncfile_geo.groups['geolocation_data']
        for var in geo_data.variables:
            geo_data.variables[var].set_auto_maskandscale(False)



        var_nfov = dataset.createVariable(follower_name+'_nfov', np.int32, ('master_obs'))
        var_nfov[:] = nfov

        if nearest_neighbor:
            print('Using nearest neighbor for follower.')
            nfov_nn = np.ones(np.shape(nfov), dtype=int)
        else:
            nfov_nn = nfov
            print('Averaging follower data, not nearest neighbor.')

        if debug:
            print('DEBUG MODE: collocating follower latitudes')
            # Create the output variable
            var_lat = dataset.createVariable(follower_name+'_Latitude', np.float32, ('master_obs'))
            var_lon = dataset.createVariable(follower_name+'_Longitude', np.float32, ('master_obs'))
            # Read the follower geo data
            lat = geo_data.variables['latitude'][:]
            lon = geo_data.variables['longitude'][:]
            #start_time = timeit.default_timer()
            (lat_out, lon_out) = cf.collocate_follower_geo(lat, lon, hidx, vidx, nfov_nn)
            #elapsed = timeit.default_timer() - start_time
            #print(elapsed)

            var_lat[:] = lat_out
            var_lon[:] = lon_out

        print('collocating satellite and solar angles')
        for g in follower_angle_datasets:
            # Create the output variable
            tmp_out = dataset.createVariable(follower_name+'_'+g, np.float32, ('master_obs'))
            # Read the follower geo data
            tmp_in = nc4_convert_data(geo_data.variables[g])
            #start_time = timeit.default_timer()
            tmp_out[:] = cf.collocate_follower_generic(tmp_in, hidx, vidx, nfov_nn)
            #elapsed = timeit.default_timer() - start_time
            #print(elapsed)

        #TODO in the middle of porting this over using dictionary of bands
        if follower_refSB:
            print('Matching reflective follower bands')
            band_center = dataset.createVariable('ReflectiveBandCenters', np.float32, ('follower_nbands'))
            tmp_band_center = np.zeros((n_follower_bands,), dtype=np.float32)
            # follower_refSB must be list of M-bands ie ['M03', 'M07']
            for j, item in enumerate(follower_refSB):
                var_name = follower_name+'_'+item
                print('Collocating follower band %s as %s' % ('{a:02d}'.format(a=j), var_name)) 
                tmp_out = dataset.createVariable(var_name, np.float32, ('master_obs'))
                # Read the follower geo data
                tmp_in = nc4_convert_data(nc_data.variables[item])
                tmp_band_center[j] = VB[item]
                #start_time = timeit.default_timer()
                tmp_out[:] = cf.collocate_follower_generic(tmp_in, hidx, vidx, nfov_nn)
                #elapsed = timeit.default_timer() - start_time
                #print(elapsed)

                print('band %s complete' % ('{a:02d}'.format(a=j)))

            band_center[:] = tmp_band_center
            print('Done!')

        if follower_emB:
            print('Matching emissive follower bands')
            band_center = dataset.createVariable('EmissiveBandCenters', np.float32, ('follower_nbands'))
            tmp_band_center = np.zeros((n_follower_bands,), dtype=np.float32)
            for j, item in enumerate(follower_emB):
                var_name = follower_name+'_'+item
                print('Collocating follower band %s as %s' % ('{a:02d}'.format(a=j), var_name)) 
                tmp_out = dataset.createVariable(var_name, np.float32, ('master_obs'))
                tmp_out_bt = dataset.createVariable(var_name + '_BT', np.float32, ('master_obs'))
                # Read the follower geo data
                tmp_in = nc4_convert_data(nc_data.variables[item])
                lut = nc_data.variables[item + '_brightness_temperature_lut']
                tmp_out[:] = cf.collocate_follower_generic(tmp_in, hidx, vidx, nfov_nn)

                tmp_in = nc_data.variables[item][:]
                sz = np.shape(tmp_in)
                tmp_in = np.reshape(tmp_in, (sz[0]*sz[1],))
                lut = nc_data.variables[item + '_brightness_temperature_lut'][:]
                tmp_in = lut[tmp_in]
                tmp_in = np.reshape(tmp_in, (sz[0],sz[1]))
                tmp_band_center[j] = VB[item]
                #start_time = timeit.default_timer()
                tmp_out_bt[:] = cf.collocate_follower_generic(tmp_in, hidx, vidx, nfov_nn)
                #elapsed = timeit.default_timer() - start_time
                #print(elapsed)
                print('band %s complete' % ('{a:02d}'.format(a=j)))

            band_center[:] = tmp_band_center
            print('Done!')

        dataset.close()
        ncfile.close()
        ncfile_geo.close()


if __name__ == '__main__':
    # Function ABI_viirs_match.py 
    # About: Takes the ABI - viirs collocation file and the matching data
    #        files to produce a collocated data file
    # Usage: collocation_file seviri_file modis_file gdas_file output_file

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('collocation_file',  help='Collocation hdf file (mandatory)')
    parser.add_argument('output_file', help='Output h5 file (mandatory)')
    parser.add_argument('master_resolution', help='resolution of ABI channels (mandatory). 1km, or 2km')
    parser.add_argument('-m', '--master-in-file', nargs='*', help='ABI input file')
    parser.add_argument('-g', '--master-geo-file', help='ABI input geolocation file')
    parser.add_argument('-f', '--follower-in-file', help='Follower input iff file ')
    parser.add_argument('-G', '--follower-in-geo-file', help='Follower input iff file ')
    parser.add_argument('-r', '--follower_refl_bands', nargs='*', help='Follower reflevtive band name i.e. M01')
    parser.add_argument('-e', '--follower_em_bands', nargs='*', help='Follower emissive band name')
    parser.add_argument('--follower_name', help='Follower instrument name, must be MODIS or VIIRS')
    parser.add_argument('-n', '--avg-follower', action='store_true', default=False, help='Average follower data instead of nearest neighbor')
    parser.add_argument('-x', '--debug', action='store_true', default=False, help='DEBUG mode, collocate foller geo data ')
    args = parser.parse_args()

    # Nearest neighbor for follower comparison is default
    nn = True
    if args.avg_follower:
        nn = False

    follower_refl_bands=None
    if args.follower_refl_bands:
        follower_refl_bands = args.follower_refl_bands

    follower_em_bands=None
    if args.follower_em_bands:
        follower_em_bands = args.follower_em_bands

    if  args.follower_name and (args.follower_name == 'VIIRS' or args.follower_name == 'MODIS'):
        follower_name = args.follower_name
    else:
        follower_name = None
        print('CAUTION: No follower intrument name supplied, if you are supplying follower data you MUST provide a name.')

    abi_viirs_match(collocation_file=args.collocation_file, output_file=args.output_file,
        master_in_file=args.master_in_file, master_geo_file=args.master_geo_file,
         follower_in_file=args.follower_in_file, follower_in_geo_file=args.follower_in_geo_file,
         follower_refSB=follower_refl_bands, 
         follower_emB=follower_em_bands, follower_name=follower_name,
         nearest_neighbor=nn, abi_resolution=args.master_resolution, debug=args.debug)
