import  numpy as np
from netCDF4 import Dataset
from argparse import ArgumentParser
import os
from datetime import timedelta, datetime
#import py_fgf_to_earth as geoloc
import satutil_lib as su
import pprint
import pdb
import crefl_ahi as crefl
import logging
import re

# Used for the fixed-grid to earth, and vice-versa, computations.
ABI_ID = int(1)
AHI_ID = int(3)
FILL_VALUE_FLOAT = -9999.0
DTOR = np.pi/180.0
missing_value = FILL_VALUE_FLOAT

class ABI_fname:
    def __init__(self,fname):
        #    OR_ABI-L1b-RadF-M3C01_G16_s20170711900028_e20170711910395_c20170711910436.nc
        (self.orig_dir,self.name) = os.path.split(fname)
        #ipdb.set_trace()
        ncfile = Dataset(fname,'r')
        print(ncfile)
        print("y_image, ",ncfile.variables["y_image_bounds"][0],ncfile.variables["y_image_bounds"][-1])
        print("x_image ",ncfile.variables["x_image_bounds"][0],ncfile.variables["x_image_bounds"][-1])
#        foo = self.name.split('.')[0]
        foo = ncfile.dataset_name
#        print('foo is ',foo)
        self.long_name = foo
#        foo = re.split('\W+|_',self.name) #matches any non-alphanumeric character OR '_'
        foo = re.split('\W+|_',ncfile.dataset_name) #matches any non-alphanumeric character OR '_'
        ncfile.close()
#        print('foo is ',foo)
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
        print(self.long_name)

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

def nc4_convert_data(data, aslice=None):
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
        
        if aslice is not None:
            idx = np.equal(data[aslice], fv)
            out = np.float32(data[aslice])*sf + offset
            out[idx] = -FILL_VALUE_FLOAT
        else:
            idx = np.equal(data[:], fv)
            out = np.float32(data[:])*sf + offset
            out[idx] = -FILL_VALUE_FLOAT
        return out
    else:
        return np.float32(data[:])*sf + offset

# def nc4_convert_data(data, aslice=None):
#     allatt = [foo for foo in data.ncattrs()]
#     sf = 1
#     offset = 0
#     if 'scale_factor' in allatt:
#         sf = data.getncattr('scale_factor')
#     if 'add_offset' in allatt:
#         offset = data.getncattr('add_offset')
    
#     if aslice is not None:
#         return data[aslice]*sf + offset
#     else:
#         return data[:]*sf + offset

def subset_data(input_dir, output_path, section=None, AHI_geo_name=None, AHI_name=None, debug=False):
    # TODO: FIXME --->Right now the geo data is always calculated
    np.seterr(all='ignore')
    log = logging.getLogger(__name__)

    ahi_name = AHI_name[0]
    if isinstance(AHI_name, basestring):
        ahi_name = AHI_name
    
    inst = su.classify_ABI(ahi_name)


    if inst=='AHI':
        input_fname = os.path.join(input_dir, ahi_name)
        fn_obj = su.AHI_fname(input_fname)
        fn_obj.display()

        ncfile = Dataset(input_fname,'r')
        for var in ncfile.variables:
            ncfile.variables[var].set_auto_maskandscale(False)

        if debug:
            for var in ncfile.variables:
                attr = [foo for foo in ncfile.variables[var].ncattrs()]
                print_attr = [var]
                print_attr.append(ncfile.variables[var].shape)
                for foo in attr:
                    
                    print_attr.append([foo + '  ::  ' + str(ncfile.variables[var].getncattr(foo))])

                pprint.pprint(print_attr,indent=4,width=4)

        cgms_x = ncfile.variables['cgms_x'][:]
        cgms_y = ncfile.variables['cgms_y'][:]
        COFF1 = ncfile.variables['cgms_x'].getncattr('COFF')
        CFAC = ncfile.variables['Projection'].getncattr('CFAC')
        COFF = ncfile.variables['Projection'].getncattr('COFF')
        LFAC = ncfile.variables['Projection'].getncattr('LFAC')
        LOFF = ncfile.variables['Projection'].getncattr('LOFF')
        if COFF1 > COFF:
            log.debug('!!! acl.subset_data: Adjusting cgms_x and cgms_y for hsdnc convert bug !!!')
            log.debug([COFF1, COFF])
            div = 2.0
            cgms_x /= div
            cgms_y /= div
        semi_major_axis = ncfile.variables['Projection'].getncattr('semi_major_axis')
        inv_flat = ncfile.variables['Projection'].getncattr('inverse_flattening')
        sub_lon_degrees = np.float(ncfile.variables['Projection'].getncattr('longitude_of_projection_origin'))
        pp_height = ncfile.variables['Projection'].getncattr('perspective_point_height')

        scale_x  = CFAC
        offset_x = np.float(COFF)
        scale_y  = LFAC
        offset_y = np.float(LOFF)

        if section is not None:
            lats = np.array([section[0], section[1], section[0], section[1]])
            lons = np.array([section[2], section[2], section[3], section[3]])
            xx = np.zeros((4,), dtype=int)
            yy = np.zeros((4,), dtype=int)
            (xx, yy) = geoloc.py_earth_to_fgf(lats, lons,  scale_x, offset_x, scale_y, offset_y, sub_lon_degrees, AHI_ID)
            # Pure cython implementation that is still bugged
            # for i in range(len(lats)):
            #     [x,y] = geoloc.earth_to_fgf_cy(sat, lons[i], lats[i], scale_x, offset_x, scale_y, offset_y, sub_lon_degrees)
            #     xx[i] = x
            #     yy[i] = y
            #     print([x,y])

            xlim = np.array([np.min(xx), np.max(xx)])
            ylim = np.array([np.min(yy), np.max(yy)])

            ylim = ylim - cgms_y[0]

            log.debug([xlim, ylim])
            # Calculate subsetted geolocation (lat, lon) data
            log.debug('Calculating geolocation data')
            cgms_x = cgms_x[xlim[0]:xlim[1]]
            cgms_y = cgms_y[ylim[0]:ylim[1]]
            
        (lat, lon) = geoloc.py_fgf_to_earth(cgms_x, cgms_y, scale_x, offset_x, scale_y, offset_y, sub_lon_degrees, AHI_ID)

    elif inst=='ABI':
        input_fname = os.path.join(input_dir, ahi_name)
        fn_obj = su.ABI_fname(input_fname)
        fn_obj.display()

        ncfile = Dataset(input_fname,'r')
        for var in ncfile.variables:
            ncfile.variables[var].set_auto_maskandscale(False)

        if debug:
            for var in ncfile.variables:
                attr = [foo for foo in ncfile.variables[var].ncattrs()]
                print_attr = [var]
                print_attr.append(ncfile.variables[var].shape)
                for foo in attr:
                    
                    print_attr.append([foo + '  ::  ' + str(ncfile.variables[var].getncattr(foo))])

                pprint.pprint(print_attr,indent=4,width=4)

        cgms_x = ncfile.variables['x'][:]
        scale_x = ncfile.variables['x'].getncattr('scale_factor')
        offset_x = ncfile.variables['x'].getncattr('add_offset')
        cgms_y = ncfile.variables['y'][:]
        scale_y = ncfile.variables['y'].getncattr('scale_factor')
        offset_y = ncfile.variables['y'].getncattr('add_offset')
        
        proj = 'goes_imager_projection'
        semi_major_axis = ncfile.variables[proj].getncattr('semi_major_axis')
        inv_flat = ncfile.variables[proj].getncattr('inverse_flattening')
        sub_lon_degrees = np.float(ncfile.variables[proj].getncattr('longitude_of_projection_origin'))
        pp_height = ncfile.variables[proj].getncattr('perspective_point_height')

        lats = np.array([section[0], section[1], section[0], section[1]])
        lons = np.array([section[2], section[2], section[3], section[3]])

        xx = np.zeros((4,), dtype=int)
        yy = np.zeros((4,), dtype=int)
        (xx, yy) = geoloc.py_earth_to_fgf(lats, lons,  scale_x, offset_x, scale_y, offset_y, sub_lon_degrees, ABI_ID)
        # Pure cython implementation that is still bugged
        # for i in range(len(lats)):
        #     [x,y] = geoloc.earth_to_fgf_cy(sat, lons[i], lats[i], scale_x, offset_x, scale_y, offset_y, sub_lon_degrees)
        #     xx[i] = x
        #     yy[i] = y
        #     print([x,y])

        xlim = np.array([np.min(xx), np.max(xx)])
        ylim = np.array([np.min(yy), np.max(yy)])

        log.debug('TODO: This ylim calculation needs checked!!!!!!')
        ylim = ylim - cgms_y[0]

        log.debug([xlim, ylim])
        # Calculate subsetted geolocation (lat, lon) data
        log.debug('Calculating geolocation data')
        cgms_x = cgms_x[xlim[0]:xlim[1]]
        cgms_y = cgms_y[ylim[0]:ylim[1]]
        
        (lat, lon) = geoloc.py_fgf_to_earth(cgms_x, cgms_y, scale_x, offset_x, scale_y, offset_y, sub_lon_degrees)
    
    space_mask = compute_space_mask(lat, lon)
    log.debug("Computing sensor view angles...")
    sat_ang = su.COMPUTE_SATELLITE_ANGLES(-1.0*sub_lon_degrees, 0.0, -1.0*lon, lat, space_mask)
    #     zenith  = satellite zenith view angle 
    #     azimuth = satellite azimuth angle clockwise from north
    log.debug("Done.")

    log.debug("Writing temporary geo data file.")
    #fn_obj = su.AHI_fname(AHI_name[0])
    oname = fn_obj.output_fname() + '_' + 'geotemp.nc'
    ofile = os.path.join(output_path, oname)
    new_geo_fname = oname

    dataset = Dataset(ofile, 'w', format='NETCDF4')
    dataset.description = "Geolocation file for ABI or AHI"
    sx=len(cgms_x)
    sy=len(cgms_y)
    #print(sx,sy)
    x = dataset.createDimension('x', sx)
    y = dataset.createDimension('y', sy)
    prj = dataset.createDimension('prj', 1)

    for dimname in dataset.dimensions.keys():
        dim = dataset.dimensions[dimname]
        log.debug("dimname: %s, length: %d, is_unlimited: %g" % (dimname, len(dim), dim.isunlimited()))
    varlon = dataset.createVariable('longitude', np.float32, ('y', 'x'))
    varlat = dataset.createVariable('latitude', np.float32, ('y', 'x'))
    var_sat_zen = dataset.createVariable('sensor_zenith', np.float32, ('y', 'x'))
    var_sat_azi = dataset.createVariable('sensor_azimuth', np.float32, ('y', 'x'))
    varproj = dataset.createVariable('Projection', np.uint8, ('prj',)) # Empty variable
  
    varlat[:] = lat
    varlon[:] = lon
    var_sat_azi[:] = sat_ang['azimuth']
    var_sat_zen[:] = sat_ang['zenith']

    # Copies 'Projection' attributes
    for ncattr in ncfile.variables['Projection'].ncattrs():
        aval = ncfile.variables['Projection'].getncattr(ncattr)
        if aval:
            varproj.setncattr(ncattr, aval)

    dataset.close()
    ncfile.close()


    # Subset channel data and write out to temp files
    # The following if-else blocks are almost identical, however due to the 
    # small but significant differences in the dataset names the code
    # was split up this way. TODO: With a little effort this could and
    # should be made more generic and rolled into one.

    if inst == 'AHI':
        AHI_new_names=[]
        nmRAD = 'RAD'
        log.debug("Writing temporary data files.")
        for afile in AHI_name:
            fn_obj=su.AHI_fname(afile)
            input_fname = os.path.join(input_dir, afile)
            ncfile_in = Dataset(input_fname,'r')
            for var in ncfile_in.variables:
                ncfile.variables[var].set_auto_maskandscale(False)


            oname = fn_obj.output_fname() + '_' + 'temp.nc'
            AHI_new_names.append(oname)
            ofile = os.path.join(output_path, oname)
            ncfile_out = Dataset(ofile,'w', format='NETCDF4')
            x = ncfile_out.createDimension('x', sx)
            y = ncfile_out.createDimension('y', sy)
            var_rad = ncfile_out.createVariable(nmRAD, np.int16, ('y', 'x'))
            var_line_time = ncfile_out.createVariable('line_time_offset', np.float64, ('y'))

            # Copies 'RAD' attributes
            for ncattr in ncfile_in.variables[nmRAD].ncattrs():
                aval = ncfile_in.variables[nmRAD].getncattr(ncattr)
                if aval:
                    var_rad.setncattr(ncattr, aval)

            RAD = ncfile_in.variables[nmRAD][:]
            if section is not None:
                var_rad[:] = RAD[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            else:
                var_rad[:] = RAD

            for ncattr in ncfile_in.variables['line_time_offset'].ncattrs():
                aval = ncfile_in.variables['line_time_offset'].getncattr(ncattr)
                if aval:
                    var_line_time.setncattr(ncattr, aval)


            ncfile_in.close()
            ncfile_out.close()

    elif inst=='ABI':
        AHx_new_names=[]
        nmRAD = 'Rad'
        log.debug("Writing temporary data files.")
        for afile in AHI_name:
            fn_obj=su.ABI_fname(afile)
            input_fname = os.path.join(input_dir, afile)
            ncfile_in = Dataset(input_fname,'r')
            for var in ncfile_in.variables:
                ncfile.variables[var].set_auto_maskandscale(False)

            oname = fn_obj.output_fname() + '_' + 'temp.nc'
            AHx_new_names.append(oname) #Hmmm what is this used for ???
            ofile = os.path.join(output_path, oname)
            ncfile_out = Dataset(ofile,'w', format='NETCDF4')
            x = ncfile_out.createDimension('x', sx)
            y = ncfile_out.createDimension('y', sy)
            var_rad = ncfile_out.createVariable(nmRAD, np.int16, ('y', 'x'))
            # line_time_offset doesn't exist for GOES-R (G16), and there doen't look like
            # there is a functional equivalent. 
            #var_line_time = ncfile_out.createVariable('line_time_offset', np.float64, ('y'))

            # Copies 'RAD' attributes
            for ncattr in ncfile_in.variables[nmRAD].ncattrs():
                aval = ncfile_in.variables[nmRAD].getncattr(ncattr)
                if aval:
                    var_rad.setncattr(ncattr, aval)

            RAD = ncfile_in.variables[nmRAD][:]
            var_rad[:] = RAD[ylim[0]:ylim[1], xlim[0]:xlim[1]]

            # for ncattr in ncfile_in.variables['line_time_offset'].ncattrs():
            #     aval = ncfile_in.variables['line_time_offset'].getncattr(ncattr)
            #     if aval:
            #         var_line_time.setncattr(ncattr, aval)


            ncfile_in.close()
            ncfile_out.close()

 
    log.debug("Done")
    return [new_geo_fname, AHI_new_names]

def compute_space_mask(lat, lon):
    space_mask = np.logical_or(np.less(lat, -90), np.greater(lat, 90))
    space_mask = np.logical_or(space_mask, np.less(lon, -180))
    space_mask = np.logical_or(space_mask, np.greater(lon, 180))
    return space_mask

def get_AHI_time(AHI_name):
    log = logging.getLogger(__name__)
    np.seterr(all='ignore')

    ahi_name = AHI_name[0]
    if isinstance(AHI_name, basestring):
        ahi_name = AHI_name
    
    inst = su.classify_ABI(ahi_name)

    if inst=='AHI':
        fn_obj = su.AHI_fname(ahi_name)
        fn_obj.display()

        ncfile = Dataset(ahi_name,'r')
        for var in ncfile.variables:
            ncfile.variables[var].set_auto_maskandscale(False)

        base_time = ncfile.variables['line_time_offset'].getncattr('base_time')

        date_obj = datetime.utcfromtimestamp(base_time)
        test_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
        log.debug('The current date for this granule is:' + test_date)
        jday = int(date_obj.strftime('%j'))
        hh = int(date_obj.strftime('%H'))
        mm = int(date_obj.strftime('%M'))
        ss = int(date_obj.strftime('%S'))
        tu = hh+mm/60.0 + ss/3600.0

        ncfile.close()
        return {'d_obj': date_obj, 'jday': jday, 'dec_hour':tu}
    elif inst=='ABI':
        from datetime import timedelta

        fn_obj = su.ABI_fname(ahi_name)
        fn_obj.display()

        ncfile = Dataset(ahi_name,'r')
        for var in ncfile.variables:
            ncfile.variables[var].set_auto_maskandscale(False)

        date_obj = datetime.strptime('20000101 1200', '%Y%m%d %H%M')
        base_time = timedelta(seconds=ncfile.variables['time_bounds'][0])
        date_obj = date_obj + base_time
        #date_obj = datetime.utcfromtimestamp(base_time)
        test_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
        log.debug('The current date for this granule is:' + test_date)
        jday = int(date_obj.strftime('%j'))
        hh = int(date_obj.strftime('%H'))
        mm = int(date_obj.strftime('%M'))
        ss = int(date_obj.strftime('%S'))
        tu = hh+mm/60.0 + ss/3600.0

        ncfile.close()
        return {'d_obj': date_obj, 'jday': jday, 'dec_hour':tu}

def compute_AHI_geolocation(output_path=None, AHI_name=None, debug=0):

    log = logging.getLogger(__name__)
    # Is the AHI_name a list of filenames or just a filename
    ahi_name = AHI_name[0]
    if isinstance(AHI_name, basestring):
        ahi_name = AHI_name
    
    inst = su.classify_ABI(ahi_name)

    log.debug(inst)

    np.seterr(all='ignore')
    ncfile = Dataset(ahi_name,'r')
    for var in ncfile.variables:
        ncfile.variables[var].set_auto_maskandscale(False)

    if debug:
        for var in ncfile.variables:
            attr = [foo for foo in ncfile.variables[var].ncattrs()]
            print_attr = [var]
            print_attr.append(ncfile.variables[var].shape)
            for foo in attr:
                
                print_attr.append([foo + '  ::  ' + str(ncfile.variables[var].getncattr(foo))])

            pprint.pprint(print_attr,indent=4,width=4)
    if inst == 'AHI':
        fn_obj = su.AHI_fname(ahi_name)
        fn_obj.display()
        SAT_ID = AHI_ID

        cgms_x = ncfile.variables['cgms_x'][:]
        cgms_y = ncfile.variables['cgms_y'][:]
        CFAC = ncfile.variables['Projection'].getncattr('CFAC')
        COFF = ncfile.variables['Projection'].getncattr('COFF')
        LFAC = ncfile.variables['Projection'].getncattr('LFAC')
        LOFF = ncfile.variables['Projection'].getncattr('LOFF')
        div = np.floor(5501.0/COFF)
        cgms_x /= div
        cgms_y /= div
 
        scale_x  = CFAC
        offset_x = np.float(COFF)
        scale_y  = LFAC
        offset_y = np.float(LOFF)
        proj = 'Projection'

    elif inst == 'ABI':
        fn_obj = su.ABI_fname(AHI_name)
        fn_obj.display()
        SAT_ID = ABI_ID
        cgms_x = ncfile.variables['x'][:]
        scale_x = ncfile.variables['x'].getncattr('scale_factor')
        offset_x = ncfile.variables['x'].getncattr('add_offset')
        cgms_y = ncfile.variables['y'][:]
        scale_y = ncfile.variables['y'].getncattr('scale_factor')
        offset_y = ncfile.variables['y'].getncattr('add_offset')
        proj = 'goes_imager_projection'

    semi_major_axis = ncfile.variables[proj].getncattr('semi_major_axis')
    inv_flat = ncfile.variables[proj].getncattr('inverse_flattening')
    sub_lon_degrees = np.float(ncfile.variables[proj].getncattr('longitude_of_projection_origin'))
    pp_height = ncfile.variables[proj].getncattr('perspective_point_height')

    if debug:
        cgms_x = cgms_x[::debug]
        cgms_y = cgms_y[::debug]

    log.debug("Computing lat/lon values...")
    (lat, lon) = geoloc.py_fgf_to_earth(cgms_x, cgms_y, scale_x, offset_x, scale_y, offset_y, sub_lon_degrees, SAT_ID)
    log.debug("Done")

    space_mask = compute_space_mask(lat, lon)
    log.debug("Computing sensor view angles...")
    sat_ang = su.COMPUTE_SATELLITE_ANGLES(-1.0*sub_lon_degrees, 0.0, -1.0*lon, lat, space_mask)
    #     zenith  = satellite zenith view angle 
    #     azimuth = satellite azimuth angle clockwise from north
    log.debug("Done.")

    log.debug("Writing data file.")
    oname = fn_obj.output_fname() + '_' + 'geolocation'
    ofile = os.path.join(output_path, oname + '.nc')

    dataset = Dataset(ofile, 'w', format='NETCDF4')
    dataset.description = "Geolocation file for AHI/ABI"
    sx=np.size(cgms_x)
    sy=np.size(cgms_y)
    x = dataset.createDimension('x', sx)
    y = dataset.createDimension('y', sy)
    prj = dataset.createDimension('prj', 1)

    for dimname in dataset.dimensions.keys():
        dim = dataset.dimensions[dimname]
        log.debug("dimname: %s, length: %d, is_unlimited: %g",(dimname, len(dim), dim.isunlimited()))
    varlon = dataset.createVariable('longitude', np.float32, ('y', 'x'))
    varlat = dataset.createVariable('latitude', np.float32, ('y', 'x'))
    var_sat_zen = dataset.createVariable('sensor_zenith', np.float32, ('y', 'x'))
    var_sat_azi = dataset.createVariable('sensor_azimuth', np.float32, ('y', 'x'))
    varproj = dataset.createVariable('Projection', np.uint8, ('prj',)) # Empty variable
  
    varlat[:] = lat
    varlon[:] = lon
    var_sat_azi[:] = sat_ang['azimuth']
    var_sat_zen[:] = sat_ang['zenith']

    # Copies 'Projection' attributes
    for ncattr in ncfile.variables[proj].ncattrs():
        aval = ncfile.variables[proj].getncattr(ncattr)
        if aval:
            varproj.setncattr(ncattr, aval)

    dataset.close()
    ncfile.close()

def AHI_read_geo(AHI_geo_name, debug=0):
    np.seterr(all='ignore')

    ahi_name = AHI_geo_name[0]
    if isinstance(AHI_geo_name, basestring):
        ahi_name = AHI_geo_name
    
    inst = su.classify_ABI(ahi_name)

    if inst=='AHI':
        fn_geo_obj = su.AHI_fname(ahi_name)
    elif inst=='ABI':
        fn_geo_obj = su.ABI_fname(ahi_name)

    fn_geo_obj.display()

    ncfile = Dataset(ahi_name,'r')

    for var in ncfile.variables:
        ncfile.variables[var].set_auto_maskandscale(False)

    if debug:
        for var in ncfile.variables:
            attr = [foo for foo in ncfile.variables[var].ncattrs()]
            print_attr = [var]
            print_attr.append(ncfile.variables[var].shape)
            for foo in attr:
                
                print_attr.append([foo + '  ::  ' + str(ncfile.variables[var].getncattr(foo))])

            pprint.pprint(print_attr,indent=4,width=4)

    lat = ncfile.variables['latitude'][:,:]
    lon = ncfile.variables['longitude'][:,:]
    senzen = ncfile.variables['sensor_zenith'][:,:]
    senazi = ncfile.variables['sensor_azimuth'][:,:]

    sub_lon_degrees = np.float(ncfile.variables['Projection'].getncattr('longitude_of_projection_origin'))
    ncfile.close()
    if debug:
        lat = lat[::debug,::debug]
        lon = lon[::debug,::debug]
        senzen = senzen[::debug,::debug]
        senazi = senazi[::debug,::debug]

    return lat, lon, sub_lon_degrees, senzen, senazi

def AHI_scat_calc_short(AHI_name, lat, lon, senzen, senazi):
    # Computes the solar and satellite scattering angles.
    space_mask = compute_space_mask(lat, lon)

    date_dict = get_AHI_time(AHI_name)

    #print("Computing solar angles...")
    solar_ang = su.POSSOL(date_dict['jday'], date_dict['dec_hour'], lon, lat, space_mask)
    #         asol = solar zenith angle in degrees
    #         phis = solar azimuth angle in degrees
    #print("Done.")
    data = {'solzen': solar_ang['asol'],
            'solazi': solar_ang['phis'],
            'space_mask': space_mask}
    return data

def AHI_scat_calc(output_path=None, AHI_geo_name=None, AHI_name=None, module_output='data', debug=0):
    # Computes the solar and satellite scattering angles.
    lat, lon, sub_lon_degrees, senzen, senazi = AHI_read_geo(AHI_geo_name, debug=debug)
    space_mask = compute_space_mask(lat, lon)

    date_dict = get_AHI_time(AHI_name)

    inst = su.classify_ABI(AHI_name)
    if inst=='AHI':
        fn_obj = su.AHI_fname(AHI_name)
    elif inst=='ABI':
        fn_obj = su.ABI_fname(AHI_name)

    # TODO Read in lat, lon, space_mask

    #print("Computing solar angles...")
    solar_ang = su.POSSOL(date_dict['jday'], date_dict['dec_hour'], lon, lat, space_mask)
    #         asol = solar zenith angle in degrees
    #         phis = solar azimuth angle in degrees
    #print("Done.")

    sz = np.shape(lat)
    sx=sz[0]
    sy=sz[1]
    if module_output == 'netcdf':
        #print("Writing netcdf file.")
        oname = fn_obj.output_fname() + '_' + 'scatt_ang'
        ofile = os.path.join(output_path, oname + '.nc')

        dataset = Dataset(ofile, 'w', format='NETCDF4')
        dataset.description = "Scattering angle file for ABI/AHI"
        x = dataset.createDimension('x', sx)
        y = dataset.createDimension('y', sy)

        for dimname in dataset.dimensions.keys():
            dim = dataset.dimensions[dimname]
            #print(dimname, len(dim), dim.isunlimited())
        var_sol_zen = dataset.createVariable('solar_zenith', np.float32, ('y', 'x'))
        var_sol_azi = dataset.createVariable('solar_azimuth', np.float32, ('y', 'x'))

        var_sol_azi[:] = solar_ang['phis']
        var_sol_zen[:] = solar_ang['asol']

        dataset.close()
        return ofile
    elif module_output == 'binary':
        #print("Writing binary files.")
        oname = 'AxI_SCATCALC_' + fn_obj.output_date() + '_solzen' + '.real4.' + str(sx) + '.' + str(sy)
        ofile_zen = os.path.join(output_path, oname_zen) 
        np.float32(solar_ang['asol']).tofile(ofile)

        oname = 'AxI_SCATCALC_' + fn_obj.output_date() + '_solazi' + '.real4.' + str(sx) + '.' + str(sy)
        ofile_azi = os.path.join(output_path, oname) 
        np.float32(solar_ang['phis']).tofile(ofile_azi)
        return [ofile_zen, ofile_azi]
    elif module_output == 'data':
        data = {'lat': lat,
                'lon': lon,
                'sub_lon_degrees': sub_lon_degrees,
                'senzen': senzen,
                'senazi': senazi,
                'solzen': solar_ang['asol'],
                'solazi': solar_ang['phis'],
                'space_mask': space_mask}
        return data


def ahi_crefl_preload(module_input='data', data_dict=None, AHI_geo_name=None, AHI_scat_dict=None, terrain_fname=None):
    # Module calls the crefl pre-processor that only needs called once per scene
    # Input: module_input='data', passes the data dictionary directly from the solar scattering angle code
    #    module_input='file' reads the data from files TO BE COMPLETED
    if module_input == 'file':
        #lat, lon, sub_lon_degrees, senzen, senazi = AHI_read_geo(AHI_geo_name)

        # TODO READ DATA, PACK INTO DICT

        #adict = crefl.crefl_ahi_preprocess(data_dict, terrain_fname=terrain_fname, debug=debug)
        print('ERROR ahi_crefl_preload: Not implemented yet!')
        exit()
    elif module_input == 'data':
        crefl_pp_dict = crefl.crefl_ahi_preprocess(data_dict, terrain_fname=terrain_fname)

    return crefl_pp_dict
   

def ahi_rad_to_BT(output_path=None, AHI_name=None, debug=0):
    # geo_scat_dict needs lat, lon, solzen, solazi, senzen, senazi, space_mask
    # Refer to ahi_crefl below if you make changes to this code
    
    log = logging.getLogger(__name__)
    wl = 11.2395*1.0e-6 # Central wavelenth of band 14. TODO fix me
    np.seterr(all='ignore')

    out_names = []
    for afile in AHI_name:
        fn_obj=su.AHI_fname(afile)
        band_number = fn_obj.band
        ncfile = Dataset(afile,'r')
        for var in ncfile.variables:
            ncfile.variables[var].set_auto_maskandscale(False)

        c = ncfile.variables['RAD'].c # getncattr throws an error, confusing
        h = ncfile.variables['RAD'].h
        k = ncfile.variables['RAD'].k
        bt_c0 = ncfile.variables['RAD'].bt_c0
        bt_c1 = ncfile.variables['RAD'].bt_c1
        bt_c2 = ncfile.variables['RAD'].bt_c2

        I = nc4_convert_data(ncfile.variables['RAD'])[:]
        K1 = 2*h*c**2/(wl**5)/1e6
        K2 = h*c/(k*wl)
        BT = K2/np.log(K1/I + 1.0)
        BT = bt_c0 + bt_c1*BT + bt_c2*np.power(BT,2)

        ncfile.close()

        #corr_refl = corr_refl*data_dict['mus']

        log.debug("Writing BT file.")

        sz = np.shape(BT)
        sx = sz[0]
        sy = sz[1]
        oname = 'AHI_BT_' + fn_obj.output_date() + '_' + str(int(band_number)) + '.real4.' + str(sy) + '.' + str(sx)
        ofile = os.path.join(output_path, oname) 
        out_names.append(ofile)
        np.float32(BT).tofile(ofile)

    return out_names

def ahi_crefl(output_path=None, AHI_name=None, data_dict=None, debug=0):
    # geo_scat_dict needs lat, lon, solzen, solazi, senzen, senazi, space_mask
    

    np.seterr(all='ignore')
    log = logging.getLogger(__name__)

    space_mask = data_dict['space_mask']
    out_names = []
    for afile in AHI_name:
        fn_obj=su.AHI_fname(afile)
        band_number = fn_obj.band
        ncfile = Dataset(afile,'r')
        for var in ncfile.variables:
            ncfile.variables[var].set_auto_maskandscale(False)

        cprime = ncfile.variables['RAD'].cprime # getncattr throws an error, confusing
        albedo = nc4_convert_data(ncfile.variables['RAD'])*cprime
        ncfile.close()
        if debug:
            albedo = albedo[::debug, ::debug]

        reflectance = albedo/data_dict['mus']
        corr_refl = crefl.crefl_ahi(preprocess_dict=data_dict, band_number=band_number, reflectance=reflectance, space_mask=space_mask)

        #corr_refl = corr_refl*data_dict['mus']

        log.debug("Writing corrected reflectance file.")

        sz = np.shape(reflectance)
        sx = sz[0]
        sy = sz[1]
        oname = 'AHI_CREFL_' + fn_obj.output_date() + '_' + str(int(band_number)) + '.real4.' + str(sy) + '.' + str(sx)
        ofile = os.path.join(output_path, oname) 
        out_names.append(ofile)
        np.float32(corr_refl).tofile(ofile)

    return out_names


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    #group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-o', '--output-path')
    parser.add_argument('-g', '--create-geo', action='store_true')
    #parser.add_argument('-iffcth','--IFF-CTH', action='store_true', default=False, help='VIIRS IFF CTH imagery')
    parser.add_argument('-f','--AHI-name', help='AHI netcdf file name', required=True)
    parser.add_argument('-c', '--corr-refl', action='store_true', default=False, help='Compute corrected reflectances')
    parser.add_argument('-b','--band-number', help='Band number', required=False)
    args = parser.parse_args()
    # if args.corr_refl:
    #     # TODO this is broken, and a hack
    #     if not hasattr(args,'band_number'):
    #         print('Error: Oops! Need a band number to compute corrected reflectance')
    #         exit()
    # ahi_geoloc_scat_calc(output_path=args.output_path, AHI_name=args.AHI_name, compute_crefl=args.corr_refl, band_number=float(args.band_number))

    if args.create_geo:
        compute_AHI_geolocation(output_path=args.output_path, AHI_name=args.AHI_name, debug=0)
