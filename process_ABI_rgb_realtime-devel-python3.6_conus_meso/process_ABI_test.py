from datetime import timedelta, datetime
import logging
import logging.config
import time
import os
import yaml
from argparse import ArgumentParser
import init_ahi_log
import abi_make_crefl
import glob


init_ahi_log.setup_log()
logger = logging.getLogger(__name__)

#
# Setup for this processing run, i.e. input parameters
#

parser = ArgumentParser(description=__doc__)
parser.add_argument('-k', '--input-dir')
parser.add_argument('-i', '--input', nargs='+')
args = parser.parse_args()



logger.info('Processing: {%s}', args.input[0])


kepler_local_dir = '/data/kuehn/ABI_rt/'
#kepler_local_dir = '/Data/ABI_rt/'

kepler_tmp_dir = os.path.join(kepler_local_dir,'tmp')
kepler_output_dir = os.path.join(kepler_local_dir,'rgb')

terrain_fname='/data/kuehn/AHI/geo/CMGDEM.hdf'
#terrain_fname='/Users/kuehn/Box Sync/work/polar2grid/polar2grid/viirs_crefl/CMGDEM.hdf'

# The resolutions of netcdf to generate
saved_cwd = os.getcwd()

os.chdir(args.input_dir)
# Info 
# 1   0.47  1 km
# 2   0.64  0.5 km
# 3   0.86  1 km
# 4   1.38  2 km
# 5   1.61  1 km
# 6   2.26  2 km
# 7   3.90  .
# 8   6.15  .
# 9   7.0   .
# 10  7.4   .
# 11  8.5   .
# 12  9.7   .
# 13  10.35 .
# 14  11.2  .
# 15  12.3  .
# 16  13.3  .


const_text_str = ['GOES-16','UW-Madison SSEC CIMSS']
# Lower left, upper right  lat,lon pair
#ll_ur = [20, -40, 35, -75]
#ll_ur = [10.0, -120.0, 60.0, -50.0]
ll_ur = [10.0, -90.0, 30.0, -60.0]

(IDIR,fnC01) = os.path.split(args.input[0])
print(IDIR, fnC01)
fnC02 = fnC01.replace('C01', 'C02')
fnC02 = glob.glob(fnC02[:41]+"*")

fnC03 = fnC01.replace('C01', 'C03')
fnC03 = glob.glob(fnC03[:41]+"*")
files = [fnC02[0], fnC03[0], fnC01]
print(files)
"""
files = ['/Data/ABI/20170312/OR_ABI-L1b-RadF-M3C02_G16_s20170711900028_e20170711910395_c20170711910432.nc',
    '/Data/ABI/20170312/OR_ABI-L1b-RadF-M3C03_G16_s20170711900028_e20170711910395_c20170711910438.nc',
    '/Data/ABI/20170312/OR_ABI-L1b-RadF-M3C01_G16_s20170711900028_e20170711910395_c20170711910436.nc']
"""
isDay = abi_make_crefl.abi_make_crefl(
    output_path=kepler_output_dir,
    ahi_fnames=files,
    tmp_path=kepler_tmp_dir,
    output_resolution=2.0,
    geo_fname=None,
    terrain_fname=terrain_fname,
    #ll_ur=ll_ur,
    BT=False,
    debug=0)

