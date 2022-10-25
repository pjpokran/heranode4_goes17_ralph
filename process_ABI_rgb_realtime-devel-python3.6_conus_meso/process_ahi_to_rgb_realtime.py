from datetime import timedelta, datetime
import logging
import logging.config
import time
import os
import yaml
import subprocess
#import ipdb
from argparse import ArgumentParser
import numpy as np
import init_ahi_log
import add_timestamp
import ahi_calc

Sections = ['S0110',
    'S0210',
    'S0310',
    'S0410',
    'S0510',
    'S0610',
    'S0710',
    'S0810',
    'S0910',
    'S1010']

def ahi_sections(lats):
    logger = logging.getLogger(__name__)
    max_lat = max(lats)
    min_lat = min(lats)
    if max_lat == min_lat:
        raise ValueError('User input error: latitude values limits are the same.')
    sections = [47, 33, 20, 11, 1, -11, -21, -33, -47, -90]
    idxs = np.where(np.greater(max_lat, sections))[0][0] # Return first index of value 
    idxe = np.where(np.greater(min_lat, sections))[0][0] # Return first index of value 
    logger.debug(idxs, idxe)
    return [idxs, idxe]

def ahi_fldk_fname(date_obj, band):
    astr = ['HS', 'H08', date_obj.strftime('%Y%m%d'), date_obj.strftime('%H%M'), band, 'FLDK.nc']
    return '_'.join(astr)

#HS_H08_20150417_0700_B01_FLDK_R10_S0110.DAT

def ahi_dat_fname(date_obj, band, res, sect):
    astr = ['HS', 'H08', date_obj.strftime('%Y%m%d'), date_obj.strftime('%H%M'), band, 'FLDK', res, sect + '.DAT']
    return '_'.join(astr)

def ahi_partial_fname(date_obj):
    astr = ['HS', 'H08', date_obj.strftime('%Y%m%d'), date_obj.strftime('%H%M')]
    return '_'.join(astr)

#AHI_CREFL_d20160428_t0450_3_rgb.png

def crefl_partial_fname(date_obj):
    astr = ['AHI', 'CREFL', date_obj.strftime('d%Y%m%d'), date_obj.strftime('t%H%M'),'*','rgb.jpg']
    return '_'.join(astr)

def crefl_web_fname(date_obj):
    astr = ['CREFL','H08', date_obj.strftime('%Y%m%d'), date_obj.strftime('%H%M'),'rgb.jpg']
    return '_'.join(astr)

def BT_partial_fname(date_obj):
    astr = ['AHI', 'BT', date_obj.strftime('d%Y%m%d'), date_obj.strftime('t%H%M'),'*','grayscale.jpg']
    return '_'.join(astr)

def BT_web_fname(date_obj):
    astr = ['BT','H08', date_obj.strftime('%Y%m%d'), date_obj.strftime('%H%M'),'BT11.jpg']
    return '_'.join(astr)


def remove_nc_files(nc_dir, adate, ahour):
    logger = logging.getLogger(__name__)
    adate = adate
    nc = ['HS', 'H08', adate, ahour, '*']
    astr = '_'.join(nc)
    apath = os.path.join(nc_dir,adate,'1km',astr)
    cmd = ' '.join(['rm',apath])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

    apath = os.path.join(nc_dir,adate,'2km',astr)
    cmd = ' '.join(['rm',apath])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)


def remove_temp_files(temp_dir, adate, ahour):
    logger = logging.getLogger(__name__)
    bt = ['AHI', 'BT', 'd'+adate, 't'+ahour,'*']
    bt = '_'.join(bt)
    apath = os.path.join(temp_dir,bt)
    cmd = ' '.join(['rm',apath])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

    crefl = ['AHI', 'CREFL', 'd'+adate, 't'+ahour,'*']
    crefl = '_'.join(crefl)
    apath = os.path.join(temp_dir,crefl)
    cmd = ' '.join(['rm',apath])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

    nc = ['HS', 'H08', adate, ahour, '*']
    nc = '_'.join(nc)
    apath = os.path.join(temp_dir,nc)
    cmd = ' '.join(['rm',apath])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

init_ahi_log.setup_log()
logger = logging.getLogger(__name__)

#
# Setup for this processing run, i.e. input parameters
#

parser = ArgumentParser(description=__doc__)
parser.add_argument('-d', '--nodel', action='store_true', default=False, help='Do not delete temporary files')
parser.add_argument('-i', '--input', nargs='+')
args = parser.parse_args()

logger.info('Processing: {%s}', args.input[0])

kepler_local_dir = '/data/kuehn/AHI_rt/'
kepler_tmp_dir = os.path.join(kepler_local_dir,'tmp')
kepler_DAT_dir = os.path.join(kepler_local_dir,'DAT')
kepler_output_dir = os.path.join(kepler_local_dir,'rgb')
kepler_nc_dir = os.path.join(kepler_local_dir,'nc')
terrain_fname='/data/kuehn/AHI/geo/CMGDEM.hdf'

# The resolutions of netcdf to generate
hsdnc_script = ['/home/kuehn/work/himawari_devel/himawari/py/hsd2nc1km_rk_devel.sh',
                '/home/kuehn/work/himawari_devel/himawari/py/hsd2nc2km_rk_devel.sh']
ncres = ['1km', '2km'] 

saved_cwd = os.getcwd()
bands = ['B01', 'B02', 'B03', 'B04', 'B14']
ress = ['R10', 'R10', 'R05', 'R10', 'R20']

const_text_str = ['Himawari 8','UW-Madison SSEC CIMSS']
# If you want full disk, comment out the lats,lons defined lists.
lats=[] #[Max, min]
lons=[] #[Max, Min]

# DEFAULT for KORUS-AQ
#lats = [45, 30]
#lons = [133, 109]

# Philippines
lats = [22.5, 2]
lons = [128, 105]
#lats = [5, 19.9]
#lons = [115, 130]

# New Guinea
#lats = [-5, -10]
#lons = [160, 155]
if lats:
    ldx = ahi_sections(lats)
    sections = Sections[ldx[0]:ldx[1]+1]
    lat_lon = [lats[0], lats[1], lons[0], lons[1]]
else:
    sections = Sections

#
# Process the data
#

idir, ifname = os.path.split(args.input[0])
logger.debug('Input directory and filename: %s, %s',idir, ifname)

tmp = ifname.split('_')
s_time = '_'.join((tmp[2],tmp[3]))
this_t = datetime.strptime(s_time, '%Y%m%d_%H%M')

adate = this_t.strftime('%Y%m%d')
ahour = this_t.strftime('%H%M')
logger.debug('*** Copying AHI data to local directory')
for band, res in zip(bands, ress):
    for s in sections:
        zfiles = os.path.join(idir, ahi_dat_fname(this_t, band, res, s))
        # scp kuehn@zara:DIR/FNAME /data/kuehn/AHI/tmp_combine
        cmd = ' '.join(['cp', zfiles, kepler_DAT_dir])
        logger.debug(cmd)
        status = subprocess.call(cmd, shell=True)
        logger.debug('Command status: ', status)  

# Change to DAT dir and convery to netcdf format
os.chdir(kepler_DAT_dir)

logger.debug('*** Converting DAT file to netcdf')
for i, hsdcmd in enumerate(hsdnc_script):    
    cmd = ' '.join([hsdcmd, ahi_partial_fname(this_t)])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)


    ncdir = os.path.join(kepler_nc_dir, adate)
    if not os.path.isdir(ncdir):
        os.mkdir(ncdir)

    ncdir = os.path.join(kepler_nc_dir, adate, ncres[i])
    if not os.path.isdir(ncdir):
        os.mkdir(ncdir)


    cmd = ' '.join(['mv',ahi_partial_fname(this_t)+'*.nc', ncdir])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)


# Remove DAT files
logger.debug('*** Removing DAT files')
for band, res in zip(bands, ress):
    for s in sections:
        zfiles = os.path.join(kepler_DAT_dir, ahi_dat_fname(this_t, band, res, s))
        cmd = ' '.join(['rm',zfiles])
        logger.debug(cmd)
        status = subprocess.call(cmd, shell=True)

# Come back to python working dir
os.chdir(saved_cwd)

#
# 1. 
# For 1km data make true color image, large
#

# Generate list of files to process
logger.info('Making 1km, full resolution, rgb')
ncdir = os.path.join(kepler_nc_dir, adate, '1km')
files = []
for band in bands:
    if not 'B14' in band:
        files.append(os.path.join(ncdir,ahi_fldk_fname(this_t, band)))

if 'lat_lon' in locals():
    try:
        isDay = ahi_calc.ahi_calc(output_path=kepler_output_dir, ahi_fnames=files, tmp_path=kepler_tmp_dir, terrain_fname=terrain_fname, section=lat_lon)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception, e:
        logger.error('ahi_calc failed', exc_info=True)
    #cmd = ['python', 'ahi_calc.py', '-o', kepler_output_dir,'-i',' '.join(files),'-t', kepler_tmp_dir, '-r', terrain_fname, '-s',  ' '.join(map(str,lat_lon))]
else:
    isDay = ahi_calc.ahi_calc(output_path=kepler_output_dir, ahi_fnames=files, tmp_path=kepler_tmp_dir, terrain_fname=terrain_fname)
    #cmd = ['python', 'ahi_calc.py', '-o', kepler_output_dir,'-i',' '.join(files),'-t', kepler_tmp_dir, '-r', terrain_fname]

#cmd = ' '.join(cmd)
#logger.debug(cmd)
#status = subprocess.call(cmd, shell=True)
#status = subprocess.check_output(cmd, shell=True)
# Is this a night file?
#isDay = False
#if not "NIGHT FILE" in status:
if isDay:
    #isDay = True
    rgbdir = os.path.join(kepler_output_dir, adate)
    if not os.path.isdir(rgbdir):
        os.mkdir(rgbdir)

    rgbdir = os.path.join(rgbdir,'1km_rgb_large','large')
    try:
        os.makedirs(rgbdir)
    except OSError:
        logger.debug('Directory already exists')
    except:
        raise

    cmd = ' '.join(['mv',os.path.join(kepler_output_dir,crefl_partial_fname(this_t)), os.path.join(rgbdir,crefl_web_fname(this_t))])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)
    
    text_str = list(const_text_str)
    text_str.append('RGB')
    add_timestamp.timestamp_ahi(os.path.join(rgbdir,crefl_web_fname(this_t)), text_str=text_str)
    #cmd = ' '.join(['python','add_timestamp.py','-i',os.path.join(rgbdir,crefl_web_fname(this_t)), '-t', 'RGB'])
    #logger.debug(cmd)
    #status = subprocess.call(cmd, shell=True)

    #cmd = ' '.join(['mv',os.path.join(kepler_output_dir,crefl_partial_fname(this_t)), os.path.join(rgbdir,'latest_rgb.png')])
    #print(cmd)
    #status = subprocess.call(cmd, shell=True)
    #AHI_CREFL_d20160428_t0450_3_rgb.png

#
# 2. 
# For 2km data make true color image, large
#

logger.debug('*** For 2km data make large true color image')
if isDay:
    logger.info('Making 2km rgb')
    # Generate list of files to process
    ncdir = os.path.join(kepler_nc_dir, adate, '2km')
    files = []
    for band in bands:
        if not 'B14' in band:
            files.append(os.path.join(ncdir,ahi_fldk_fname(this_t, band)))

    try:
        null = ahi_calc.ahi_calc(output_path=kepler_output_dir, ahi_fnames=files, tmp_path=kepler_tmp_dir, terrain_fname=terrain_fname, section=lat_lon)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception, e:
        logger.error('ahi_calc failed', exc_info=True)
    #cmd = ['python', 'ahi_calc.py', '-o', kepler_output_dir,'-i',' '.join(files),'-t', kepler_tmp_dir, '-r', terrain_fname, '-s',  ' '.join(map(str,lat_lon))]
    #cmd = ' '.join(cmd)
    #logger.debug(cmd)
    #status = subprocess.call(cmd, shell=True)


    rgbdir = os.path.join(kepler_output_dir, adate)
    rgbdir = os.path.join(rgbdir,'2km_rgb_large','large')
    try:
        os.makedirs(rgbdir)
    except OSError:
        logger.debug('Directory already exists')
    except:
        raise

    cmd = ' '.join(['cp',os.path.join(kepler_output_dir,crefl_partial_fname(this_t)), os.path.join(kepler_output_dir,'latest_rgb.jpg')])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)
    cmd = ' '.join(['cp',os.path.join(kepler_output_dir,crefl_partial_fname(this_t)), os.path.join(kepler_output_dir,'latest.jpg')])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)
    cmd = ' '.join(['mv',os.path.join(kepler_output_dir,crefl_partial_fname(this_t)), os.path.join(rgbdir,crefl_web_fname(this_t))])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

    text_str = list(const_text_str)
    text_str.append('RGB')
    add_timestamp.timestamp_ahi(os.path.join(rgbdir,crefl_web_fname(this_t)), text_str=text_str)
    #cmd = ' '.join(['python','add_timestamp.py','-i',os.path.join(rgbdir,crefl_web_fname(this_t)), '-t', 'RGB'])
    #logger.debug(cmd)
    #status = subprocess.call(cmd, shell=True)

    #AHI_CREFL_d20160428_t0450_3_rgb.png

#
# 3. 
# For 2km data make brightness temperature image, large
# ONLY WORKS FOR B14 or 11um, needs a little work to be general... sorry future me, or whomever.

# Generate list of files to process
logger.debug('*** Making BT imag1e')
logger.info('Making BT image')
ncdir = os.path.join(kepler_nc_dir, adate, '2km')
files = []
for band in bands:
    if 'B14' in band:
        files.append(os.path.join(ncdir,ahi_fldk_fname(this_t, band)))

try:
    null = ahi_calc.ahi_calc(output_path=kepler_output_dir, ahi_fnames=files, tmp_path=kepler_tmp_dir, BT=True, section=lat_lon)
except (SystemExit, KeyboardInterrupt):
    raise
except Exception, e:
    logger.error('ahi_calc failed', exc_info=True)
#cmd = ['python', 'ahi_calc.py', '-o', kepler_output_dir,'-i',' '.join(files),'-t', kepler_tmp_dir, '--BT', '-s', ' '.join(map(str,lat_lon))]
#cmd = ' '.join(cmd)
#logger.debug(cmd)
#status = subprocess.call(cmd, shell=True)

rgbdir = os.path.join(kepler_output_dir, adate)
rgbdir = os.path.join(rgbdir,'2km_BT11_large','large')
try:
    os.makedirs(rgbdir)
except OSError:
    logger.debug('Directory already exists')
except:
    raise

# Changes file names
cmd = ' '.join(['cp',os.path.join(kepler_output_dir,BT_partial_fname(this_t)), os.path.join(kepler_output_dir,'latest_bt11.jpg')])
logger.debug(cmd)
status = subprocess.call(cmd, shell=True)
if not isDay:
    cmd = ' '.join(['cp',os.path.join(kepler_output_dir,BT_partial_fname(this_t)), os.path.join(kepler_output_dir,'latest.jpg')])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)
cmd = ' '.join(['mv',os.path.join(kepler_output_dir,BT_partial_fname(this_t)), os.path.join(rgbdir,BT_web_fname(this_t))])
logger.debug(cmd)
status = subprocess.call(cmd, shell=True)


text_str = list(const_text_str)
text_str.append('BT 11um')
add_timestamp.timestamp_ahi(os.path.join(rgbdir,BT_web_fname(this_t)), text_str=text_str)
#cmd = ' '.join(['python','add_timestamp.py','-i',os.path.join(rgbdir,BT_web_fname(this_t)), '-t', '"BT 11um"'])
#logger.debug(cmd)
#status = subprocess.call(cmd, shell=True)


#
# 4. 
# For 1km data make true color image, medium
#

logger.debug('*** For 1km data make medium true color image')
if isDay:
    logger.info('Making subset rgb image')
    lats = [5, 19.9]
    lons = [115, 130]
    #lats = [-5, -10]
    #lons = [160, 155]
    #lats = [40, 33]
    #lons = [130, 120]
    lat_lon = [lats[0], lats[1], lons[0], lons[1]]
    ncdir = os.path.join(kepler_nc_dir, adate, '1km')
    files = []
    for band in bands:
        if not 'B14' in band:
            files.append(os.path.join(ncdir,ahi_fldk_fname(this_t, band)))

    try:
        null = ahi_calc.ahi_calc(output_path=kepler_output_dir, ahi_fnames=files, tmp_path=kepler_tmp_dir, terrain_fname=terrain_fname, section=lat_lon)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception, e:
        logger.error('ahi_calc failed', exc_info=True)
    #cmd = ['python', 'ahi_calc.py', '-o', kepler_output_dir,'-i',' '.join(files),'-t', kepler_tmp_dir, '-r', terrain_fname, '-s',  ' '.join(map(str,lat_lon))]
    #cmd = ' '.join(cmd)
    #logger.debug(cmd)
    #status = subprocess.call(cmd, shell=True)

    rgbdir = os.path.join(kepler_output_dir, adate)
    rgbdir = os.path.join(rgbdir,'1km_rgb_small','large')

    try:
        os.makedirs(rgbdir)
    except OSError:
        print('Directory already exists')
    except:
        raise

    cmd = ' '.join(['mv',os.path.join(kepler_output_dir,crefl_partial_fname(this_t)), os.path.join(rgbdir,crefl_web_fname(this_t))])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

    text_str = list(const_text_str)
    text_str.append('RGB')
    add_timestamp.timestamp_ahi(os.path.join(rgbdir,crefl_web_fname(this_t)), text_str=text_str)
    #cmd = ' '.join(['python','add_timestamp.py','-i',os.path.join(rgbdir,crefl_web_fname(this_t)), '-t', 'RGB'])
    #logger.debug(cmd)
    #status = subprocess.call(cmd, shell=True)

    #cmd = ' '.join(['mv',os.path.join(kepler_output_dir,crefl_partial_fname(this_t)), os.path.join(rgbdir,'latest_rgb.png')])
    #print(cmd)
    #status = subprocess.call(cmd, shell=True)


# Copy files to web server
copy_to_web = False
if copy_to_web:
    logger.info('Copying data to web server')
    rgbdir2 = os.path.join(kepler_output_dir, adate)
    cmd = ' '.join(['rsync  -h -v -r -P -t -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"',
        rgbdir2,'kuehn@birch:/var/apache/cimss/htdocs/korus-aq/images/_ahirgb'])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

    cmd = ' '.join(['rsync  -h -v -r -P -t -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"',
        rgbdir2,'kuehn@cypress:/data/htdocs/korus-aq/images/_ahirgb'])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

    rgb_file = os.path.join(kepler_output_dir, 'latest_rgb.jpg')
    destination = os.path.join('/var/apache/cimss/htdocs/korus-aq/layout-images', 'latest-rgb.jpg')
    cmd = ' '.join(['scp', rgb_file, ''.join(['kuehn@birch:',destination])])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

    ir_file = os.path.join(kepler_output_dir, 'latest_bt11.jpg')
    destination = os.path.join('/var/apache/cimss/htdocs/korus-aq/layout-images', 'latest-bt11.jpg')
    cmd = ' '.join(['scp', ir_file, ''.join(['kuehn@birch:',destination])])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

    ir_file = os.path.join(kepler_output_dir, 'latest.jpg')
    destination = os.path.join('/var/apache/cimss/htdocs/korus-aq/layout-images', 'latest.jpg')
    cmd = ' '.join(['scp', ir_file, ''.join(['kuehn@birch:',destination])])
    logger.debug(cmd)
    status = subprocess.call(cmd, shell=True)

if not args.nodel:
    remove_nc_files(kepler_nc_dir, adate, ahour)
    remove_temp_files(kepler_tmp_dir, adate, ahour)
