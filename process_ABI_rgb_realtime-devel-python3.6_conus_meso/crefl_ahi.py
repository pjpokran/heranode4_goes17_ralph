# -*- coding: utf-8 -*-
import numpy as np
from pyhdf.SD import *
from argparse import ArgumentParser
from satutil_lib import *
import os
import logging

MAXSOLZ = 86.5
MAXAIRMASS = 18
SCALEHEIGHT = 8000
FILL_INT16 =32767
DEG2RAD = np.pi/180.0

TAUSTEP4SPHALB = 0.0003
MAXNUMSPHALBVALUES = 4000    # with no aerosol taur <= 0.4 in all bands everywhere

# band_num = np.array([1, 2, 3, 4, 5, 6])
# bands = np.array([0.47, 0.51, 0.64, 0.86, 1.6, 2.3])
rg_fudge = 0.55
# Abs_O3_coeff = np.array([4.2869e-003, 17.8177e-003*rg_fudge, 25.6509e-003*rg_fudge, 802.4319e-006, 0.0000e+000, 2e-5])
# Abs_H20_coeff = np.array([2.4111e-003, 25.1537e-003*rg_fudge, 7.8454e-003*rg_fudge, 7.9258e-3, 9.3392e-003, 2.53e-2])
# Abs_O2_coeff = np.array([1.2360e-003, 1.0993e-003, 3.7296e-003, 177.7161e-006, 10.4899e-003, 1.63e-2])
# taur0 = np.array([184.7200e-003, 132.1000e-003, 52.3490e-003, 15.8450e-003, 1.3074e-003, 311.2900e-006])

# STILL missing the 1.37 band info, so using above 0.86 is bad 10/17/2017
band_num = np.array([1, 2, 3, 4, 5])
bands = np.array([0.47, 0.64, 0.86, 1.6, 2.3])
Abs_O3_coeff = np.array([4.2869e-003, 25.6509e-003*rg_fudge, 802.4319e-006, 0.0000e+000, 2e-5])
Abs_H20_coeff = np.array([2.4111e-003, 7.8454e-003*rg_fudge, 7.9258e-3, 9.3392e-003, 2.53e-2])
Abs_O2_coeff = np.array([1.2360e-003, 3.7296e-003, 177.7161e-006, 10.4899e-003, 1.63e-2])
taur0 = np.array([184.7200e-003, 52.3490e-003, 15.8450e-003, 1.3074e-003, 311.2900e-006])
missing_value = -999.0


def csalbr(tau):
    # Previously 3 functions csalbr fintexp1, fintexp3
    a= [ -.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857]
    xx = a[0] + a[1]*tau + a[2]*tau**2 + a[3]*tau**3 + a[4]*tau**4 + a[5]*tau**5

    # xx = a[0]
    # xftau = 1.0
    # for i in xrange(5):
    #     xftau = xftau*tau
    #     xx = xx + a[i] * xftau
    fintexp1 = xx-np.log(tau)
    fintexp3 = (np.exp(-tau) * (1.0 - tau) + tau**2 * fintexp1) / 2.0

    return (3.0 * tau - fintexp3 * (4.0 + 2.0 * tau) + 2.0 * np.exp(-tau)) / (4.0 + 3.0 * tau)
 
def G_calc(zenith, a_coeff):
    return (np.cos(zenith * DEG2RAD)+(a_coeff[0]*(zenith**a_coeff[1])*(a_coeff[2]-zenith)**a_coeff[3]))**-1

def crefl_ahi_preprocess(geo_scat_dict=None, terrain_fname=None):
    lat = geo_scat_dict['lat']
    lon = geo_scat_dict['lon']
    SensorZenith = geo_scat_dict['senzen']
    SensorAzimuth = geo_scat_dict['senazi']
    SolarZenith = geo_scat_dict['solzen']
    SolarAzimuth = geo_scat_dict['solazi']
    space_mask = geo_scat_dict['space_mask']
    #lat, lon, SensorAzimuth, SensorZenith, SolarAzimuth, SolarZenith, space_mask
    fp = SD(terrain_fname)
    dem = fp.select('averaged elevation')[:]
    fp.end()

    # Get digital elevation map data for our granule, set ocean fill value to 0
    #ipdb.set_trace()
    sz = np.shape(lat)
    height = np.ones(sz, dtype=float)*missing_value
    idx = np.where(np.logical_not(space_mask))
    row = np.int32((90.0 - lat[idx]) *  np.shape(dem)[0]/ 180.0)-1
    col = np.int32((lon[idx] + 180.0) * np.shape(dem)[1]/ 360.0)-1
    h=np.float64(dem[row,col])
    jdx = np.where(np.less(h,0.0))
    h[jdx] = 0.0
    height[idx] = h

    mus = np.ones(sz, dtype=float)*missing_value
    muv = np.ones(sz, dtype=float)*missing_value
    phios = np.ones(sz, dtype=float)*missing_value
    solar_zenith_mask = np.zeros(sz, dtype=np.int32)
    mus[idx] = np.cos(SolarZenith[idx] * DEG2RAD)
    muv[idx] = np.cos(SensorZenith[idx] * DEG2RAD)
    phios[idx] = (SolarAzimuth[idx] - SensorAzimuth[idx]) + 180.0

    bNewAirmassCalc = True
    if bNewAirmassCalc:
        a_O3  = [268.45, 0.5, 115.42, -3.2922]
        a_H2O = [0.0311, 0.1, 92.471, -1.3814]
        a_O2  = [0.4567, 0.007, 96.4884, -1.6970]

        G_O3 =  np.ones(sz, dtype=float)*missing_value
        G_H2O = np.ones(sz, dtype=float)*missing_value
        G_O2 =  np.ones(sz, dtype=float)*missing_value

        G_O3[idx] = G_calc(SolarZenith[idx], a_O3) + G_calc(SensorZenith[idx], a_O3)
        G_H2O[idx] = G_calc(SolarZenith[idx], a_H2O) + G_calc(SensorZenith[idx], a_H2O)
        G_O2[idx] = G_calc(SolarZenith[idx], a_O2) + G_calc(SensorZenith[idx], a_O2)

        air_mass = {'G_O3': G_O3, 'G_H2O': G_H2O, 'G_O2': G_O2}
    else:
        air_mass = np.ones(sz, dtype=float)*missing_value
        air_mass[idx] = 1.0/mus[idx] + 1/muv[idx];
        jdx = np.where(np.greater(air_mass,MAXAIRMASS))
        air_mass[jdx] = -1.0

    tau_step = np.linspace(TAUSTEP4SPHALB, MAXNUMSPHALBVALUES*TAUSTEP4SPHALB, MAXNUMSPHALBVALUES)
    sphalb0 = csalbr(tau_step);
    jdx = np.where(np.logical_and(np.greater(SolarZenith,86), np.logical_not(space_mask)))
    solar_zenith_mask[jdx] = 1.0
    return {'height': height,
             'mus': mus,
             'muv': muv,
             'phios': phios,
             'sphalb0': sphalb0,
             'bNewAirmassCalc': bNewAirmassCalc,
             'air_mass': air_mass,
             'space_mask': space_mask,
             'solar_zenith_mask': solar_zenith_mask}

def crefl_ahi(preprocess_dict=None, band_number=None, reflectance=None, space_mask=None):
    log = logging.getLogger(__name__)
    # See CREFL_VIIRS_IFF.PY for original port information, most c comments deleted
    # FROM FUNCTION CHAND
    # phi: azimuthal difference between sun and observation in degree
    #      (phi=0 in backscattering direction)
    # mus: cosine of the sun zenith angle
    # muv: cosine of the observation zenith angle
    # taur: molecular optical depth
    # rhoray: molecular path reflectance
    # constant xdep: depolarization factor (0.0279)
    #          xfd = (1-xdep/(2-xdep)) / (1 + 2*xdep/(2-xdep)) = 2 * (1 - xdep) / (2 + xdep) = 0.958725775
    # */
    if band_number > 6 or band_number < 1:
        print("ERROR: Invalid band number for AHI, must be 1-6, %g given" % (band_number))
        exit()

    xfd = 0.958725775
    xbeta2 = 0.5
    as0 = [0.33243832, 0.16285370, -0.30924818, -0.10324388, 0.11493334,
        -6.777104e-02, 1.577425e-03, -1.240906e-02, 3.241678e-02, -3.503695e-02]
    as1 = [0.19666292, -5.439061e-02]
    as2 = [0.14545937, -2.910845e-02]

    idx = np.where(np.logical_and(np.logical_not(space_mask), np.logical_not(preprocess_dict['solar_zenith_mask'])))

    height   = preprocess_dict['height'][idx]
    mus      = preprocess_dict['mus'][idx]
    muv      = preprocess_dict['muv'][idx]
    phios    = preprocess_dict['phios'][idx]
    sphalb0  = preprocess_dict['sphalb0']
    #phios = phi + 180.0 DONE IN PREPROCESSOR
    xcos1 = 1.0
    xcos2 = np.cos(phios * DEG2RAD)
    xcos3 = np.cos(2.0 * phios * DEG2RAD)
    xph1 = 1.0 + (3.0 * mus * mus - 1.0) * (3.0 * muv * muv - 1.0) * xfd / 8.0
    xph2 = -xfd * xbeta2 * 1.5 * mus * muv * np.sqrt(1.0 - mus * mus) * np.sqrt(1.0 - muv * muv)
    xph3 = xfd * xbeta2 * 0.375 * (1.0 - mus * mus) * (1.0 - muv * muv)

    fs01 = as0[0] + (mus + muv)*as0[1] + (mus * muv)*as0[2] + (mus * mus + muv * muv)*as0[3] + (mus * mus * muv * muv)*as0[4]
    fs02 = as0[5] + (mus + muv)*as0[6] + (mus * muv)*as0[7] + (mus * mus + muv * muv)*as0[8] + (mus * mus * muv * muv)*as0[9]

    log.debug("Processing band:")
    ib = band_number-1
    taur = taur0[ib] * np.exp(-height / SCALEHEIGHT);
    xlntaur = np.log(taur)
    fs0 = fs01 + fs02 * xlntaur
    fs1 = as1[0] + xlntaur * as1[1]
    fs2 = as2[0] + xlntaur * as2[1]
    del xlntaur
    trdown = np.exp(-taur / mus)
    trup= np.exp(-taur / muv)
    xitm1 = (1.0 - trdown * trup) / 4.0 / (mus + muv)
    xitm2 = (1.0 - trdown) * (1.0 - trup)
    xitot1 = xph1 * (xitm1 + xitm2 * fs0)
    xitot2 = xph2 * (xitm1 + xitm2 * fs1)
    xitot3 = xph3 * (xitm1 + xitm2 * fs2)
    rhoray = xitot1 * xcos1 + xitot2 * xcos2 * 2.0 + xitot3 * xcos3 * 2.0

    sphalb = sphalb0[np.int32(taur / TAUSTEP4SPHALB + 0.5)]
    Ttotrayu = ((2 / 3. + muv) + (2 / 3. - muv) * trup) / (4 / 3. + taur)
    Ttotrayd = ((2 / 3. + mus) + (2 / 3. - mus) * trdown) / (4 / 3. + taur)

    if preprocess_dict['bNewAirmassCalc']:
        air_mass = preprocess_dict['air_mass']
        if Abs_O3_coeff[ib] != 0:
            tO3 =  np.exp(-air_mass['G_O3'][idx] * Abs_O3_coeff[ib])
        if Abs_H20_coeff[ib] != 0:
            tH2O = np.exp(-air_mass['G_H2O'][idx] * Abs_H20_coeff[ib])
        tO2 =  np.exp(-air_mass['G_O2'][idx] * Abs_O2_coeff[ib])
    else:
        air_mass = preprocess_dict['air_mass'][idx]
        if Abs_O3_coeff[ib] != 0:
            tO3 = np.exp(-air_mass * Abs_O3_coeff[ib])
        if Abs_H20_coeff[ib] != 0:
            tH2O = np.exp(-air_mass * Abs_H20_coeff[ib])
        tO2 = np.exp(-air_mass * Abs_O2_coeff[ib])

    TtotraytH2O = Ttotrayu * Ttotrayd * tH2O
    tOG = tO3 * tO2

    sz = np.shape(reflectance)
#    tOG_out = np.ones(sz, dtype=float)*missing_value
#    Ttotray_out = np.ones(sz, dtype=float)*missing_value
#    rhoray_out = np.ones(sz, dtype=float)*missing_value
#    tOG_out[idx] = tOG
#    Ttotray_out[idx] = TtotraytH2O
#    rhoray_out[idx] = rhoray
#    fname = 'test_crefl_tOG'+ str(int(band_number)) + '.real4.' + str(sz[0]) + '.' + str(sz[1]) 
#    np.float32(tOG_out).tofile(fname)
#    fname = 'test_crefl_Ttotray'+ str(int(band_number)) + '.real4.' + str(sz[0]) + '.' + str(sz[1]) 
#    np.float32(Ttotray_out).tofile(fname)
#    fname = 'test_crefl_rhoray'+ str(int(band_number)) + '.real4.' + str(sz[0]) + '.' + str(sz[1]) 
#    np.float32(rhoray_out).tofile(fname) 

    
    ii = np.where(np.greater_equal(reflectance[idx], 65528))
    corr_refl = (reflectance[idx] / tOG - rhoray) / TtotraytH2O
    corr_refl = corr_refl/(1.0 + corr_refl * sphalb);
    corr_refl[ii] = np.float32(-999.00)
    cr = np.ones(sz, dtype=float)*missing_value
    cr[idx] = corr_refl
    #cr = np.reshape(cr, sz)

    return cr
