import os
import re
import subprocess # For calipso alts
import distutils  # For calipso alts

from pyhdf.SD import *
from datetime import timedelta, datetime

import numpy as np
from numpy import floor,zeros,ones,size,std,mean,divide,bitwise_and,array,squeeze,\
                  concatenate,shape,min,max,nonzero,equal,not_equal,nan,reshape,\
                  log,arange,less,logical_or,logical_and,greater_equal,greater,\
                  cos,sin,arcsin,arccos,arctan2,pi,sqrt,ma,asarray,right_shift,int32, where, shape
#import ipdb
#import logging

DTOR = np.pi/180.0
missing_value = -999.0

def hdf4_dump_attr(data):
    allatt = set(f.strip() for f in data.attributes())
    for f in allatt:
        print(f, data.attributes()[f])

def convert_data(data):
    allatt = set(f.strip() for f in data.attributes())
    sf = 1
    offset = 0
    if 'scale_factor' in allatt: 
        sf = data.attributes()['scale_factor']
    if 'add_offset' in allatt: 
        offset = data.attributes()['add_offset']

    return data[:]*sf + offset

def average_modis_pixels(data):
    print(type(data), np.shape(data))
    (sx, sy) = np.shape(data)
    out = np.ones((sx/4,sy/4), dtype=float)*(-9999.9)
    print(np.shape(out))
    for i in range(sx/4):
        for j in range(sy/4):
            re = (i+1)*4
            ce = (j+1)*4
            #print i*4, re, j*4, ce
            #print data[i*4:re, j*4:ce]
            tmp = data[i*4:re, j*4:ce]    
            idx = np.where(np.greater(tmp, -0.1))
            if idx[0].any():
                out[i,j] = np.mean(tmp[idx[0], idx[1]])

            #print out[i,j]
    return out

def read_modis_reflectance_data(fname, sds, band=1):
    fp = SD(fname)
    data = fp.select(sds)
    allatt = set(f.strip() for f in data.attributes())
    sf = 1
    offset = 0
    if 'reflectance_scales' in allatt: 
        sf = data.attributes()['reflectance_scales'][band-1]
        print(sf)
    if 'relectance_offsets' in allatt: 
        offset = data.attributes()['add_offset']

    d = data[:]
    print(np.max(d))
    idx = np.where(np.greater(d[band-1,:,:],65532))
    fp.end()
    out = np.squeeze((d[band-1,:,:] - offset)*sf)
    out[idx[0],idx[1]] = -9999.9
    return out

def read_modis(fname, sds_list):
    fp = SD(fname)
    data_dict = {}
    for f in sds_list:
        hdf4_dump_attr(fp.select(f))
        data_dict[f] = convert_data(fp.select(f))

    fp.end()
    return data_dict

def compute_earth2sun(julday):
    # Pythonized from geocat/src/program_utils.f90
    earth2sun = 1.0 - 0.016729*np.cos(0.9856*(julday-4.0)*DTOR)
    return earth2sun


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
    az[idx] = cos_delta*sin(ah)/cos_elev
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
        azim[jdx] = pi - azim[jdx]

    #if caz > 0.0 and az <= 0.0:
    jdx = np.where(np.logical_and(np.logical_and(np.greater(caz, 0.0), np.less_equal(az, 0.0)),  np.logical_not(space_mask)))
    if np.any(jdx):
        azim[jdx] = 2 * pi + azim[jdx]
    azim[idx] = azim[idx] + pi
    #if azim > pi2:
    jdx = np.where(np.logical_and(np.greater(azim, pi*2), np.logical_not(space_mask)))
    if np.any(jdx):
        azim[jdx] = azim[jdx] - pi*2

    #         asol = solar zenith angle in degrees
    #         phis = solar azimuth angle in degrees
    #     conversion in degrees
    elev = elev / DTOR
    asol[idx] = 90.0 - elev
    #     phis(i,j) = azim / DTOR - 180.0
    phis[idx] = azim[idx] / DTOR  # akh - try to get 0 - 360

    return {'asol': np.reshape(asol, sz), 'phis': np.reshape(phis, sz)}


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


def compute_relaz(sataz, solaz, space_mask, adjust180):
    # Pythonized from geocat/src/program_utils.f90
    # Compute the relative azimuth angle.
    sz = np.shape(space_mask)
    relaz = np.ones(sz, dtype=float)*missing_value

    idx = np.where(np.logical_not(space_mask))
    relaz[idx] = abs(solaz[idx] - sataz[idx])
    #if (relaz(i,j) > 180.0) relaz(i,j) = 360.0 - relaz(i,j)
    jdx = np.where(np.logical_and(np.less(relaz, 0.0), np.logical_not(space_mask)))
    if jdx:
        relaz[jdx] = 360.0 - relaz[jdx]

    if adjust180:
        relaz[idx] = 180.0 - relaz[idx]

    return relaz

def relative_azimuth(sol_az, sen_az):
    rel_az = np.abs(sol_az - sen_az)
    idx = np.where(np.greater(rel_az,180.0))[0]
    rel_az[idx] = 360.0 - rel_az[idx]
    return (180.0 - rel_az)

def scatt_angle(sol_zen, sen_zen, rel_az):
    DTOR = np.pi/180.0
    scatt_ang = -1.0 * (np.cos(sol_zen*DTOR)*np.cos(sen_zen*DTOR) \
        - np.sin(sol_zen*DTOR)*np.sin(sen_zen*DTOR)*np.cos(rel_az*DTOR))
        
    idx = np.where(np.greater(scatt_ang, 1))[0]
    scatt_ang[idx] = 1.0
    idx = np.where(np.less(scatt_ang, -1))[0]
    scatt_ang[idx] = -1.0
    
    return(np.arccos(scatt_ang) / DTOR)
    
# def compute_scat_zen(cos_solzen, cos_satzen, sin_solzen, sin_satzen, cos_relaz):
#     # Compute scattering angle
#     scatzen = missing_value

#     scatzen = -1.0 * (cos_solzen*cos_satzen - sin_solzen*sin_satzen*cos_relaz)

#     if (scatzen > 1.0):
#         scatzen = 0.0
#     scatzen = acos(scatzen)/DTOR

#     return (scatzen)


def unmask_data(data, offset, bits):
    return np.bitwise_and(right_shift(data, offset), 2**bits - 1)


def unmask_all(data, data_ob):
    odata = []
    oname = []

    for j,masks in enumerate(data_ob['output']):
        offset = data_ob['offset'][j]
        bits = data_ob['mask_bits'][j]
        tmp = unmask_data(data, offset, bits)
        for i,m in enumerate(masks['masks']):
            fmask = np.logical_and(~ma.getmask(data),ma.getdata(tmp) == i)
            tsum = fmask.sum(axis=1)
            odata.append(tsum)
            oname.append(masks['name'] + m)

    return odata, oname


def great_circle_distance(latlong_a, latlong_b):
    EARTH_RADIUS = 6378137.0/1000.0     # earth circumference in meters
    lat1, lon1 = latlong_a
    lat2, lon2 = latlong_b

    dLat = (lat2 - lat1)*np.pi/180.0
    dLon = (lon2 - lon1)*np.pi/180.0
    AA = (np.sin(dLat / 2) * np.sin(dLat / 2) +
            np.cos(lat1*np.pi/180.0) * np.cos(lat2*np.pi/180.0) *
            np.sin(dLon / 2) * np.sin(dLon / 2))
    CC = 2 * np.arctan2(sqrt(AA), np.sqrt(1 - AA))

    return EARTH_RADIUS * CC


# For bug in pyhdf and the 'metadata' i.e. Vdata field in the CALIPSO data files use the following work around
# as hdp is usually provided with any standart working install of the hdf4 libraries this shouldn't be a huge
# problem. Note that hdp must be in the environment path.
def CalAltitudes(filename, alt_type='lidar'):
    hdp = distutils.spawn.find_executable('hdp')
    if not hdp:
        print("CalAltitudes: Unable to acquire locate hdp executable")
        sys.exit(1)

    if alt_type == 'lidar':
        alt_type = 'Lidar_Data_Altitudes'
    elif alt_type == 'met':
        alt_type = 'Met_Data_Altitudes'
    command = hdp + " dumpvd -d -n metadata -f %s %s" % (alt_type, filename)
    print(command)

    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    dump = ps.communicate()[0]
    altitudes = dump.split()

    return asarray([float(point) for point in altitudes])

def match_file_name(aname,alist):
    for i,item in enumerate(alist):
        v = viirs_fname(item)
        if aname == v.product:
            return v
    return None

def find_sds_index(aname,alist):
    idx = -1
    for i,item in enumerate(alist):
        if aname == item:
            idx = i
            break
    return idx

def find_data_index(aname,alist):
    idx = -1
    for i,item in enumerate(alist):
        if aname == item.var_name:
            idx = i
            break
    return idx


def check_file_exists(fname):
    if not os.path.isfile(fname):
        fname = os.path.expanduser(fname)
        if not os.path.isfile(fname):
            print("File %s does not exist. Exiting\n" % (fname))
            exit()
        else:
            print("Expanding ~, reading input file: %s\n" % (fname))
    else:
        print("Reading input file %s\n" % (fname))
    return fname

def convert_data(data,t_slice=(),mask_it=False):
    allatt = set(f.strip() for f in data.attributes())
    sf = 1
    offset = 0
    if 'scale_factor' in allatt:
        sf = data.attributes()['scale_factor']
    if 'add_offset' in allatt:
        offset = data.attributes()['add_offset']
    if mask_it:
        if '_FillValue' in allatt:
            fv = data.attributes()['_FillValue']
            amask = data[:] == fv
            data = ma.MaskedArray(data[:],mask=amask)
            ma.set_fill_value(data, fv)
        else:
            data = ma.asarray(data[:])

    if t_slice:
        return data[t_slice[0]:t_slice[1],t_slice[0]:t_slice[1]]*sf + offset
    else:
        return data[:]*sf + offset

def center_geolocation(lat, lon):
    """
    Provide a relatively accurate center lat, lon returned as a list pair, given
    a list of list pairs.
    ex: in: geolocations = ((lat1,lon1), (lat2,lon2),)
    out: (center_lat, center_lon)
    """
    x = 0
    y = 0
    z = 0
     
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    x = np.mean(x)
    y = np.mean(y)
    z = np.mean(z)
     
    return (np.arctan2(z, np.sqrt(x * x + y * y)) * 180/np.pi, np.arctan2(y, x) * 180/np.pi)

def latlon2cart(lat, lon, ecef=False):
    lat = lat * np.pi/180.0
    lon = lon * np.pi/180.0
    if ecef:
        # Cartesian ECEF following WGS84
        # See Astronomical Almanac in Appendix K.
        a = 6378137.0 # Meters
        c = 6356752.3142 # Meters
        ep2 = (a**2-c**2)/a**2

        denom = np.sqrt(1-np.sin(lat)**2*ep2)
        Rn = a/denom
        h = 0

        x = (Rn+h) * np.cos(lat) * np.cos(lon);
        y = (Rn+h) * np.cos(lat) * np.sin(lon);
        z = (Rn*c**2/a**2+h) * np.sin(lat);
    else:
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)

    return (x,y,z)

class geocat_l2_fname:
    def __init__(self,fname):
        #    geocatL2.Meteosat-9.2012207.164500.hdf
        foo = fname.split('.')[1]
        self.sat_long_name = foo
        self.sat_short_name = foo[0:3] + foo[9:] 
        yr_doy = fname.split('.')[2]
        stime = fname.split('.')[3]
        self.yr = yr_doy[0:4]
        self.doy = yr_doy[4:7]
        self.time = stime[0:4]
        self.date_obj= datetime(int(self.yr), 1, 1) + timedelta(int(self.doy) - 1)
        self.mm = self.date_obj.strftime('%m')
        self.dd = self.date_obj.strftime('%d')

    def output_fname(self,data_prod=''):
        astr = 'L2_' + self.sat_short_name + '_' + self.yr + '_' + self.mm + '_' + self.dd + '_' + self.time + '_' + data_prod + '.png'
        return astr

    def output_date(self):
        astr = self.sat_short_name + ' ' + self.yr + ' ' + self.mm + ' ' + self.dd + ' ' + self.time 
        return astr

    def display(self):
        print(' YR= %s JD= %s time= %s, mm= %s, dd= %s\n' % (self.yr, self.doy, self.time, self.mm, self.dd))
        print(self.output_fname())

def classify_ABI(fname):
    # Uses the filename to determine if it is ABI or AHI
    (orig_dir,aname) = os.path.split(fname)
    foo = re.split('\W+|_',aname) #matches any non-alphanumeric character OR '_'
    #print('classify_ABI: %s\n%s' % (fname, foo))
    for A in foo:
        if A=='H08':
            return 'AHI'
        elif A=='ABI':
            return 'ABI'

class AHI_fname:
    def __init__(self,fname):
        #    HS_H08_20150415_0130_B02_FLDK.nc
        #    HS_H08_20150415_0130_B03_FLDK_R05_S1010.DAT
        (self.orig_dir,self.name) = os.path.split(fname)
        #ipdb.set_trace()
        foo = self.name.split('.')[0]
        self.long_name = foo
        foo = foo.split('_')
        self.sat_name = foo[0] + '_' + foo[1] 
        ymd = foo[2]
        stime = foo[3]
        self.band = int(foo[4][1:])
        self.type = foo[5]
        self.resolution = None
        self.section = None
        self.special = None
        if len(foo) == 8:
            self.resolution = int(foo[6])
            self.section = foo[7]
        elif len(foo) == 7:
            self.special = foo[6]
            self.long_name = "_".join(foo[0:7])
        elif len(foo) == 9:
            self.special = foo[8]
            self.long_name = "_".join(foo[0:8])

        self.yr = ymd[0:4]
        self.mm = ymd[4:6]
        self.dd = ymd[6:]
        self.stime = stime[0:4]
        self.date_obj = datetime.strptime(ymd + ' ' + stime, '%Y%m%d %H%M')
            

    def output_fname(self):
        return self.long_name

    def output_date(self):
        return self.date_obj.strftime('d%Y%m%d_t%H%M')

    def display(self):
        print(self.long_name)


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
        self.ymd = ymd
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
        print(self.long_name)
# IVCTP_npp_d20131020_t1018425_e1020067_b10258_c20131020121506891342_noaa_ops.h5

class viirs_fname:

    def __init__(self, fname):
    # If the filename is given with directory path
    # strip it off
        self.name = fname
        self.short_name = fname.split('/')[-1]
        alist = self.short_name.split('_')
        self.product = alist[0]
        self.sat_id = alist[1]
        self.date = alist[2]
        self.stime = alist[3]
        self.etime = alist[4]
        self.orbitN = alist[5]
        self.creat_t = alist[6]
        self.origin = alist[7] + '_' + alist[8]
        self.start_date_obj = datetime.strptime(self.date + ' ' + self.stime[:-1], 'd%Y%m%d t%H%M%S')
        us = int(self.stime[-1])
        self.start_date_obj = self.start_date_obj.replace(microsecond=us * 100000)
        self.end_date_obj = datetime.strptime(self.date + ' ' + self.etime[:-1], 'd%Y%m%d e%H%M%S')
        us = int(self.etime[-1])
        self.end_date_obj = self.end_date_obj.replace(microsecond=us * 100000)

    def output_date(self):
        print(self.start_date_obj.isoformat())
        print(self.end_date_obj.isoformat())


class viirs_iff_fname:

    def __init__(self, fname):
    # If the filename is given with directory path
    # strip it off
        (self.orig_dir,self.name) = os.path.split(fname)
        alist = self.name.split('_')
        self.product = alist[0]
        self.sat_id = alist[1]
        self.date = alist[2]
        self.time = alist[3]
        self.ctime = alist[4]
        self.origin = alist[5] + '_' + alist[6]
        self.start_date_obj = datetime.strptime(self.date + ' ' + self.time[:-1], 'd%Y%m%d t%H%M%S')

    def output_date(self):
        print(self.start_date_obj.isoformat())

class viirs_iff_fbf_fname:

    def __init__(self, fname):
    # If the filename is given with directory path
    # strip it off
        # npp_viirs_visible_08_20140125_210500_omercZero.real4.2300.3150
        (self.orig_dir,self.name) = os.path.split(fname)
        alist = self.name.split('_')
        self.sat_id = alist[0]
        self.inst = alist[1]
        self.product = alist[2]
        self.chan = alist[3]
        self.date = alist[4]
        self.time = alist[5]
        self.suffix = alist[6]

        self.proj = self.suffix.split('.real.')[0]
        #print alist
        self.start_date_obj = datetime.strptime(self.date + ' ' + self.time, '%Y%m%d %H%M%S')

    def output_date(self):
        print(self.start_date_obj.isoformat())

    def reconstruct(self):
        tmp = [self.sat_id,self.inst,self.product,self.chan,self.date,self.time,self.suffix]
        return '_'.join(tmp)
