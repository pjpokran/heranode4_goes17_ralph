from datetime import timedelta, datetime
import logging
import logging.config
import time
import os
import yaml
from argparse import ArgumentParser
import init_ahi_log
import abi_make_crefl_fulldisk
import glob


init_ahi_log.setup_log()
logger = logging.getLogger(__name__)

#
# Setup for this processing run, i.e. input parameters
#

parser = ArgumentParser(description=__doc__)
#parser.add_argument('-k', '--input-dir')
parser.add_argument('-i', '--input-time', nargs='+')
args = parser.parse_args()
print("args ",args)
#
#
#
#logger.info('Processing: {%s}', args.input-time[0])
#

#kepler_local_dir = '/data/kuehn/ABI_rt/'
kepler_local_dir = '/home/poker/ABI_rt/'
kepler_local_dir = '/dev/shm/'
save_dir = '/whirlwind/goes16/test/ABI_rt/'
#save_dir = '/dev/shm/'
#kepler_local_dir = '/Data/ABI_rt/'

#kepler_tmp_dir = os.path.join(kepler_local_dir,'tmp')
kepler_tmp_dir = kepler_local_dir
kepler_output_dir = os.path.join(save_dir,'rgb')

terrain_fname='/data/kuehn/AHI/geo/CMGDEM.hdf'
terrain_fname='/home/poker/resources/CMGDEM.hdf'
#terrain_fname='/Users/kuehn/Box Sync/work/polar2grid/polar2grid/viirs_crefl/CMGDEM.hdf'

# The resolutions of netcdf to generate
saved_cwd = os.getcwd()

#os.chdir(args.input_dir)
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


const_text_str = ['GOES-16','UW-Madison SSEC CIMSS AOS']
# Lower left, upper right  lat,lon pair
#ll_ur = [20, -40, 35, -75]
#ll_ur = [10.0, -120.0, 60.0, -50.0]
ll_ur = [10.0, -90.0, 30.0, -60.0]

# worked for fulldisk 1km ll_ur = [29.9, -94.9, 40.0, -85.0]
# worked for fulldisk 1km ll_ur = [-29.89, -101.01, 50.0, -75.0]
# worked for fulldisk 1km ll_ur = [10.89, -111.01, 50.01, -65.02]
# worked for fulldisk 1km ll_ur = [0.89, -115.01, 55.01, -65.02]
# 1km NAMER ll_ur = [0.89, -115.01, 55.01, -65.02]
#ll_ur = [0.89, -115.01, 55.01, -65.02]
# 1km WI ll_ur = [40.0, -95.0, 50.0, -85.0]
ll_ur = [40.0, -95.0, 50.0, -85.0]
#ll_ur = [10.89, -111.01, 50.01, -65.02]

# For geting time/files from input

(fnC01) = glob.glob("/home/ldm/data/grb/fulldisk/01/*RadF*s" + (args.input_time[0]) + "*")
print(fnC01)
#fnC02 = fnC01.replace('C01', 'C02')
#fnC02 = glob.glob("/home/ldm/data/grb/fulldisk/02/"+fnC02[:41]+"*")
(fnC02) = glob.glob("/home/ldm/data/grb/fulldisk/02/*RadF*s" + (args.input_time[0]) + "*")
#
#fnC03 = fnC01.replace('C01', 'C03')
#fnC03 = glob.glob("/home/ldm/data/grb/fulldisk/03"+fnC03[:41]+"*")
(fnC03) = glob.glob("/home/ldm/data/grb/fulldisk/03/*RadF*s" + (args.input_time[0]) + "*")

#files = [fnC02[0], fnC03[0], fnC01]
files = [fnC02[0], fnC03[0], fnC01[0]]
print(files)

###################
##quit()
# File order should be red, green, blue: C02, C03, C01
#files = ['/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C02_G16_s20180811900446_e20180811911213_c20180811911252.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C03_G16_s20180811900446_e20180811911213_c20180811911261.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C01_G16_s20180811900446_e20180811911213_c20180811911259.nc']
#files = ['/home/ldm/data/grb/fulldisk/02/OR_ABI-L1b-RadF-M3C02_G16_s20180821530453_e20180821541220_c20180821541256.nc',
#        '/home/ldm/data/grb/fulldisk/03/OR_ABI-L1b-RadF-M3C03_G16_s20180821530453_e20180821541220_c20180821541264.nc',
#        '/home/ldm/data/grb/fulldisk/01/OR_ABI-L1b-RadF-M3C01_G16_s20180821530453_e20180821541220_c20180821541263.nc']
#files = ['/home/ldm/data/grb/fulldisk/02/OR_ABI-L1b-RadF-M3C02_G16_s20180881900386_e20180881911153_c20180881911189.nc',
#        '/home/ldm/data/grb/fulldisk/03/OR_ABI-L1b-RadF-M3C03_G16_s20180881900386_e20180881911152_c20180881911197.nc',
#        '/home/ldm/data/grb/fulldisk/01/OR_ABI-L1b-RadF-M3C01_G16_s20180881900386_e20180881911152_c20180881911193.nc']
#files = ['/home/ldm/data/grb/fulldisk/02/OR_ABI-L1b-RadF-M3C02_G16_s20180921800390_e20180921811157_c20180921811192.nc',
#        '/home/ldm/data/grb/fulldisk/03/OR_ABI-L1b-RadF-M3C03_G16_s20180921800390_e20180921811157_c20180921811202.nc',
#        '/home/ldm/data/grb/fulldisk/01/OR_ABI-L1b-RadF-M3C01_G16_s20180921800390_e20180921811157_c20180921811201.nc']
#files = ['/home/ldm/data/grb/fulldisk/02/OR_ABI-L1b-RadF-M3C02_G16_s20181232030381_e20181232041148_c20181232041182.nc',
#        '/home/ldm/data/grb/fulldisk/03/OR_ABI-L1b-RadF-M3C03_G16_s20181232030381_e20181232041148_c20181232041191.nc',
#        '/home/ldm/data/grb/fulldisk/01/OR_ABI-L1b-RadF-M3C01_G16_s20181232030381_e20181232041148_c20181232041191.nc']
#files = ['/home/ldm/data/grb/fulldisk/02/latest.nc',
#        '/home/ldm/data/grb/fulldisk/03/latest.nc',
#        '/home/ldm/data/grb/fulldisk/01/latest.nc']
#files = ['/home/ldm/data/grb/fulldisk/02/OR_ABI-L1b-RadF-M3C02_G16_s20181361800408_e20181361811175_c20181361811206.nc',
#        '/home/ldm/data/grb/fulldisk/03/OR_ABI-L1b-RadF-M3C03_G16_s20181361800408_e20181361811175_c20181361811221.nc',
#        '/home/ldm/data/grb/fulldisk/01/OR_ABI-L1b-RadF-M3C01_G16_s20181361800408_e20181361811175_c20181361811221.nc']
#files = ['/home/ldm/data/grb/conus/02/latest.nc',
#        '/home/ldm/data/grb/conus/03/latest.nc',
#        '/home/ldm/data/grb/conus/01/latest.nc']
#files = ['/home/ldm/data/grb/conus/02/OR_ABI-L1b-RadC-M3C02_G16_s20181281802200_e20181281804573_c20181281805025.nc',
#        '/home/ldm/data/grb/conus/03/OR_ABI-L1b-RadC-M3C03_G16_s20181281802200_e20181281804573_c20181281805025.nc',
#        '/home/ldm/data/grb/conus/01/OR_ABI-L1b-RadC-M3C01_G16_s20181281802200_e20181281804573_c20181281805026.nc']
#files = ['/home/ldm/data/grb/meso/02/OR_ABI-L1b-RadM2-M3C02_G16_s20181282059580_e20181282100038_c20181282100071.nc',
#        '/home/ldm/data/grb/meso/03/OR_ABI-L1b-RadM2-M3C03_G16_s20181282059580_e20181282100038_c20181282100082.nc',
#        '/home/ldm/data/grb/meso/01/OR_ABI-L1b-RadM2-M3C01_G16_s20181282059580_e20181282100038_c20181282100079.nc']
#files = ['/home/ldm/data/grb/meso/02/OR_ABI-L1b-RadM2-M3C02_G16_s20181281130579_e20181281131036_c20181281131068.nc',
#        '/home/ldm/data/grb/meso/03/OR_ABI-L1b-RadM2-M3C03_G16_s20181281130579_e20181281131036_c20181281131077.nc',
#        '/home/ldm/data/grb/meso/01/OR_ABI-L1b-RadM2-M3C01_G16_s20181281130579_e20181281131037_c20181281131076.nc']
#files = ['/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C02_G16_s20172021300380_e20172021311147_c20172021311184.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C03_G16_s20172021300380_e20172021311147_c20172021311193.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C01_G16_s20172021300380_e20172021311147_c20172021311192.nc']
#files = ['/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C02_G16_s20172022000378_e20172022011145_c20172022011177.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C03_G16_s20172022000378_e20172022011145_c20172022011191.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C01_G16_s20172022000378_e20172022011145_c20172022011192.nc']
#
#files = ['/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C02_G16_s20172021600380_e20172021611147_c20172021611182.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C03_G16_s20172021600380_e20172021611147_c20172021611191.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C01_G16_s20172021600380_e20172021611147_c20172021611189.nc']
#
##files = ['/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C02_G16_s20172021400380_e20172021411147_c20172021411182.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C03_G16_s20172021400380_e20172021411147_c20172021411190.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C01_G16_s20172021400380_e20172021411147_c20172021411192.nc']
#
#files = ['/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C02_G16_s20172021200380_e20172021211147_c20172021211180.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C03_G16_s20172021200380_e20172021211147_c20172021211191.nc',
#        '/data/kuehn/ABI/OR_ABI-L1b-RadF-M3C01_G16_s20172021200380_e20172021211147_c20172021211192.nc']
#
#"""
#files = ['/data/kuehn/ABI/2017_11_03_307/OR_ABI-L1b-RadF-M3C01_G16_s20173071600382_e20173071611149_c20173071611193.nc',
#        '/data/kuehn/ABI/2017_11_03_307/OR_ABI-L1b-RadF-M3C02_G16_s20173071600382_e20173071611149_c20173071611185.nc',
#        '/data/kuehn/ABI/2017_11_03_307/OR_ABI-L1b-RadF-M3C03_G16_s20173071600382_e20173071611149_c20173071611194.nc']
#"""
#"""
#files = ['/Data/ABI/20170312/OR_ABI-L1b-RadF-M3C02_G16_s20170711900028_e20170711910395_c20170711910432.nc',
#    '/Data/ABI/20170312/OR_ABI-L1b-RadF-M3C03_G16_s20170711900028_e20170711910395_c20170711910438.nc',
#    '/Data/ABI/20170312/OR_ABI-L1b-RadF-M3C01_G16_s20170711900028_e20170711910395_c20170711910436.nc']
#"""
isDay = abi_make_crefl_fulldisk.abi_make_crefl(
    output_path=kepler_output_dir,
    ahi_fnames=files,
    tmp_path=kepler_tmp_dir,
    input_resolutions=[0.5,1.0,1.0],
    output_resolution=4.0,
    geo_fname=None,
    terrain_fname=terrain_fname,
#    ll_ur=ll_ur,
    BT=False,
    sector="FULLDISK",
    debug=0)

