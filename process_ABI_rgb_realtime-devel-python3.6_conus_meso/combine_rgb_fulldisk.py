# -*- coding: utf-8 -*-
"""
This function converts flat-binary-format data output from polar2grid to a rgb or grayscale image.

It is used in two main modes rgb, and grayscale:
    RGB
    The three channel input files must be supplied for the red, green, and blue channels.
    The data can be from any channel but they way they are assigned is how the image
    will be assembled.
    GRAYSCALE
    For a grayscale image, you must only supply the red channel. If you want to display 
    brightness temperature data then supply the bt flag as shown in the help.
    For orbits that are descending you can supply the rotation flag to correctly orientate 
    the data. This is paritcularly important if you are working with data in the oblique
    mercator projection.
"""

import matplotlib
matplotlib.use('Agg')
from cartopy.io.shapereader import Reader
import numpy as np
from pyhdf.SD import *
import matplotlib.image as mpimg
from argparse import ArgumentParser
#from satplotlib import *
from satutil_lib import *
import pprint
import os
import logging

#idx = np.array([0, 30, 60, 120, 190, 255])/255.0
#sc  = np.array([0, 110, 160, 210, 240, 255])/255.0

idx = np.array([0,  30,  60, 120, 190, 255])/255.0
#sc  = np.array([0, 100, 145, 190, 210, 255])/255.0
sc  = np.array([0,  100, 128, 188, 223, 255])/255.0

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def silentrename(filename1, filename2):
    try:
        os.rename(filename1, filename2)
    except OSError:
        pass


def mask_RGB(R,G,B):
    ii = np.where(~np.isfinite(R))
    R[ii] = -999.0
    ii = np.where(~np.isfinite(G))
    G[ii] = -999.0
    ii = np.where(~np.isfinite(B))
    B[ii] = -999.0
    bIs_FV = logical_or(less(R,0.0),less(G,0.0))
    bIs_FV = logical_or(bIs_FV,less(B,0.0))

    #ii = np.where(~bIs_FV)
    #print np.median(R[ii]),np.std(R[ii])

    ii = np.where(greater(R,1.0))
    R[ii] = 1.0
    ii = np.where(greater(G,1.0))
    G[ii] = 1.0
    ii = np.where(greater(B,1.0))
    B[ii] = 1.0

    ii = np.where(bIs_FV)
    R[ii] = 1.0/255.0
    G[ii] = 1.0/255.0
    B[ii] = 1.0/255.0
    A = np.ones(np.shape(R),dtype=np.uint8) * 255
    A[ii] = 0
    return R,G,B,A

def rescale(R,G,B):
    ii = np.where(~np.isfinite(R))
    R[ii] = -999.0
    ii = np.where(~np.isfinite(G))
    G[ii] = -999.0
    ii = np.where(~np.isfinite(B))
    B[ii] = -999.0
    bIs_FV = logical_or(less(R,0.0),less(G,0.0))
    bIs_FV = logical_or(bIs_FV,less(B,0.0))

    #ii = np.where(~bIs_FV)
    #print np.median(R[ii]),np.std(R[ii])

    R = R/1.30
    G = G/1.30
    B = B/1.30

    ii = np.where(greater(R,1.0))
    R[ii] = 1.0
    ii = np.where(greater(G,1.0))
    G[ii] = 1.0
    ii = np.where(greater(B,1.0))
    B[ii] = 1.0


    R2 = np.copy(R)
    G2 = np.copy(G)
    B2 = np.copy(B)
    
    for i in range(1,len(idx)):
        ii = np.where(np.logical_and(np.greater(R,idx[i-1]),np.less_equal(R,idx[i])))
        rsc = (sc[i] - sc[i-1])/(idx[i] - idx[i-1])
        R2[ii] = (R[ii] - idx[i-1])*rsc + sc[i-1]
        ii = np.where(np.logical_and(np.greater(G,idx[i-1]),np.less_equal(G,idx[i])))
        G2[ii] = (G[ii] - idx[i-1])*rsc + sc[i-1]
        ii = np.where(np.logical_and(np.greater(B,idx[i-1]),np.less_equal(B,idx[i])))
        B2[ii] = (B[ii] - idx[i-1])*rsc + sc[i-1]

    ii = np.where(bIs_FV)
    R2[ii] = 0
    G2[ii] = 0
    B2[ii] = 0
    A = np.ones(np.shape(R2),dtype=np.uint8) * 255
    A[ii] = 0
    return R2,G2,B2,A


def rescale_single(R):
    ii = np.where(greater(R,1.0))
    R[ii] = 1.0
    R2 = np.copy(R)
    for i in range(1,len(idx)):
        ii = np.where(np.logical_and(np.greater(R,idx[i-1]),np.less_equal(R,idx[i])))
        rsc = (sc[i] - sc[i-1])/(idx[i] - idx[i-1])
        R2[ii] = (R[ii] - idx[i-1])*rsc + sc[i-1]

    return R2

def get_dim_from_filename(fname,atype='real4'):
    # Returs the dimensions from the flat binary file filename
    (adir,fname) = os.path.split(fname)
    dims_txt = fname.split(atype)[1].split('.')
    dims = [int(dims_txt[2]),int(dims_txt[1])]

    return dims


def combine_rgb_data(red=None,grn=None,blu=None, renorm=1.1, no_nonlin_scale=False, rotate=None,alpha=True):
    log = logging.getLogger(__name__)
    dims = np.shape(red)
    red /= renorm
    grn /= renorm
    blu /= renorm

    if not no_nonlin_scale:
        log.debug("Applying nonlinear channel rescaling")
        red,grn,blu,A = rescale(red,grn,blu)
    else:
        log.debug("Simple reflectance masking")
        red,grn,blu,A = mask_RGB(red, grn, blu)


    dim1 = dims[0]
    dim2 = dims[1]

    if rotate == '90':
        red = np.rot90(red,k=1)
        grn = np.rot90(grn,k=1)
        blu = np.rot90(blu,k=1)
        A = np.rot90(A,k=1)
        dim1 = dims[1]
        dim2 = dims[0]
    elif rotate == '-90':
        dim1 = dims[1]
        dim2 = dims[0]
        red = np.rot90(red,k=3)
        grn = np.rot90(grn,k=3)
        blu = np.rot90(blu,k=3)
        A = np.rot90(A,k=3)

    if alpha:
        out = np.zeros((dim1,dim2,4),dtype=np.uint8)
        out[:,:,3] = A
    else:
        out = np.zeros((dim1,dim2,3),dtype=np.uint8)
    out[:,:,0] = np.uint8(red *255)
    out[:,:,1] = np.uint8(grn *255)
    out[:,:,2] = np.uint8(blu *255)

#    print('max of red   ', np.max(out[:,:,0]))
#    print('max of green ', np.max(out[:,:,2]))
#    print('max of blue  ', np.max(out[:,:,3]))

    #mpimg.imsave(os.path.join(output_path,oname),out)
    return out



def combine_rgb(output_path='./',fnred=None,fngreen=None,fnblue=None,fnir=None, fake_green=True, no_nonlin_scale=False,rotate=None,alpha=True,renorm=1.1,ncfile=None):
    log = logging.getLogger(__name__)
    dims = get_dim_from_filename(fnred)
    red = np.fromfile(fnred,dtype=np.float32)/renorm
    grn = np.fromfile(fngreen,dtype=np.float32)/renorm
    blu = np.fromfile(fnblue,dtype=np.float32)/renorm
#    print('max of red   ', np.max(red))
#    print('max of green ', np.max(grn))
#    print('max of blue  ', np.max(blu))
    
#    ipdb.set_trace()

    is_jpg = True
    if alpha and is_jpg:
        alpha = False

    if fake_green:
        #new_grn = np.zeros_like(grn)
        #TODO only apply to unmasked (not space) pixels
#        print("making fake green ??")
        new_grn = 0.45*red + 0.10*grn + 0.45*blu
        grn = new_grn


    if fnir:
        # From /Users/kuehn/Dropbox/MatlabCode/plot_ahi_loc_scatt.m line 140
        #idx = d > 2 & rho4 > 0.27 & rho4 < 0.45 & d < 1.17*d2;
        log.debug('Applying near-ir green correction to rgb image')
#        print('Applying near-ir green correction to rgb image')
        nir = np.fromfile(fnir,dtype=np.float32)
        x = nir/grn
        x2 = nir/red
        A = np.greater(x,2.0)
        B = np.logical_and(np.greater(nir, 0.27), A)
        A = np.logical_and(np.less(nir, 0.45), B)
        B = np.logical_and(np.less(x,1.17*x2), A)
        idx = np.where(B)[0]
        #rho2p(idx) = rho2(idx)*0.4 + rho4(idx)*0.22

        #grn[idx] = grn[idx]*0.6 + nir[idx]*0.12
        grn = grn+nir*0.035
        

    #ipdb.set_trace()
    if not no_nonlin_scale:
#        print("Applying nonlinear channel rescaling")
        log.debug("Applying nonlinear channel rescaling")
        red,grn,blu,A = rescale(red,grn,blu)
    else:
#        print("Applying nonlinear channel rescaling")
        log.debug("Simple reflectance masking")
        red,grn,blu,A = mask_RGB(red, grn, blu)



    red = np.reshape(red,(dims[0],dims[1]))
    grn = np.reshape(grn,(dims[0],dims[1]))
    blu = np.reshape(blu,(dims[0],dims[1]))
    A = np.reshape(A,(dims[0],dims[1]))

    dim1 = dims[0]
    dim2 = dims[1]

    if rotate == '90':
        red = np.rot90(red,k=1)
        grn = np.rot90(grn,k=1)
        blu = np.rot90(blu,k=1)
        A = np.rot90(A,k=1)
        dim1 = dims[1]
        dim2 = dims[0]
    elif rotate == '-90':
        dim1 = dims[1]
        dim2 = dims[0]
        red = np.rot90(red,k=3)
        grn = np.rot90(grn,k=3)
        blu = np.rot90(blu,k=3)
        A = np.rot90(A,k=3)

    if alpha:
        out = np.zeros((dim1,dim2,4),dtype=np.uint8)
        out[:,:,3] = A
    else:
        out = np.zeros((dim1,dim2,3),dtype=np.uint8)
    out[:,:,0] = np.uint8(red *255)
    out[:,:,1] = np.uint8(grn *255)
    out[:,:,2] = np.uint8(blu *255)
#    out[:,:,0] = red
#    out[:,:,1] = grn
#    out[:,:,2] = blu 
#    print('plotted max of red   ', np.max(out[:,:,0]))
#    print('plotted max of green ', np.max(out[:,:,1]))
#    print('plotted max of blue  ', np.max(out[:,:,2]))

    # npp_viirs_visible_04_20140125_174000_omerc.real4.2300.3150
    (adir,fname) = os.path.split(fnred)
    prefix = fname.split('.real4.')[0]
    oname = prefix + '_rgb_nofilt.jpg'
    oname1 = prefix + '_rgb_wi.jpg'
    oname2 = prefix + '_rgb_mw.jpg'
    oname4 = prefix + '_rgb_ne.jpg'
    oname8 = prefix + '_rgb_namer.jpg'
    oname9 = prefix + '_rgb_fulldisk.jpg'
    oname9full = prefix + '_rgb_fulldisk.jpg'
    #oname = prefix + '_rgb_nofilt.png'
    print("saving 4km full res image")
    mpimg.imsave("/dustdevil/goes16/goes17/grb/rgb/fulldisk_4km/" + oname9full,out)
    print("saved  4km full res image")

    import netCDF4
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import shutil
    f = netCDF4.Dataset(ncfile)
    fxa = f.variables['x'][:]
    fya = f.variables['y'][:]
    xa = fxa[::4]
    ya = fya[::4]
    xa = xa*35785831
    ya = ya*35785831
    globe = ccrs.Globe(semimajor_axis=6378137.,semiminor_axis=6356752.)
    proj = ccrs.Geostationary(central_longitude=-137.0,
                          satellite_height=35785831,
                          globe=globe,sweep_axis='x')
    wi_image_crop_top=96
    wi_image_crop_bottom=-2306
    wi_image_crop_left=2080
    wi_image_crop_right=-1980

    mw_image_crop_top=46
    mw_image_crop_bottom=-1906
    mw_image_crop_left=1550
    mw_image_crop_right=-1956

    ne_image_crop_top=50
    ne_image_crop_bottom=-1450
    ne_image_crop_left=2550
    ne_image_crop_right=-250

    npac_image_crop_top=0
    npac_image_crop_bottom=-1900
    npac_image_crop_left=400
    npac_image_crop_right=-800
#
##    npac_image_size_y=(image_rows+npac_image_crop_bottom-npac_image_crop_top)
##    npac_image_size_x=(image_columns+npac_image_crop_right-npac_image_crop_left)
    npac_image_size_x=1512
    npac_image_size_y=1012
#
#    print("npac image size")
#    print(npac_image_size_x, npac_image_size_y)
#
    npac_image_size_x=npac_image_size_x*.015
    npac_image_size_y=npac_image_size_y*.015

    ak_image_crop_top=5
    ak_image_crop_bottom=-2400
    ak_image_crop_left=700
    ak_image_crop_right=-1200
##
##    ak_image_size_y=(image_rows+ak_image_crop_bottom-ak_image_crop_top)
##    ak_image_size_x=(image_columns+ak_image_crop_right-ak_image_crop_left)
#    ak_image_size_x=1624
#    ak_image_size_y=614
    ak_image_size_x=812
    ak_image_size_y=307
##
#    print("ak image size")
#    print(ak_image_size_x, ak_image_size_y)
#
#    ak_image_size_x=ak_image_size_x*.014
#    ak_image_size_y=ak_image_size_y*.014
    ak_image_size_x=ak_image_size_x*.028
    ak_image_size_y=ak_image_size_y*.028


    namer_image_crop_top=0
    namer_image_crop_bottom=-1200
    namer_image_crop_left=200
    namer_image_crop_right=-700

#    fig1 = plt.figure(figsize=(16.,16.))
#    fig2 = plt.figure(figsize=(18.,18.))
##    fig3 = plt.figure(figsize=(18.,18.))
#    fig4 = plt.figure(figsize=(18.,18.))
    fig3 = plt.figure(figsize=(npac_image_size_x,npac_image_size_y),dpi=80.)
    fig4 = plt.figure(figsize=(ak_image_size_x,ak_image_size_y),dpi=80.)
    fig8 = plt.figure(figsize=(15.1,15.1))
    fig9 = plt.figure(figsize=(14.98,14.98))

#    ax1 = fig1.add_subplot(1, 1, 1, projection=proj)
#    ax1.outline_patch.set_edgecolor('none')
#    ax1.background_patch.set_fill(False)
#    ax1.outline_patch.set_edgecolor('black')
#    ax1.patch.set_facecolor('none')
#    ax2 = fig2.add_subplot(1, 1, 1, projection=proj)
#    ax2.outline_patch.set_edgecolor('none')
#    ax2.background_patch.set_fill(False)
#    ax2.outline_patch.set_edgecolor('black')
#    ax2.patch.set_facecolor('none')
##    ax3 = fig3.add_subplot(1, 1, 1, projection=proj)
##    ax3.outline_patch.set_edgecolor('none')
##    ax3.background_patch.set_fill(False)
##    ax3.outline_patch.set_edgecolor('black')
##    ax3.patch.set_facecolor('none')
##    ax4 = fig4.add_subplot(1, 1, 1, projection=proj)
##    ax4.outline_patch.set_edgecolor('none')
##    ax4.background_patch.set_fill(False)
##    ax4.outline_patch.set_edgecolor('black')
##    ax4.patch.set_facecolor('none')

    ax3 = fig3.add_subplot(1, 1, 1, projection=proj)
    ax3.outline_patch.set_edgecolor('none')
    ax3.background_patch.set_fill(False)
    ax3.patch.set_facecolor('none')

    ax4 = fig4.add_subplot(1, 1, 1, projection=proj)
    ax4.outline_patch.set_edgecolor('none')
    ax3.background_patch.set_fill(False)
    ax3.patch.set_facecolor('none')

    ax8 = fig8.add_subplot(1, 1, 1, projection=proj)
    ax8.outline_patch.set_edgecolor('none')
    ax8.background_patch.set_fill(False)
    ax8.patch.set_facecolor('none')

    ax9 = fig9.add_subplot(1, 1, 1, projection=proj)
    ax9.outline_patch.set_edgecolor('none')
    ax9.background_patch.set_fill(False)
    ax9.patch.set_facecolor('none')

# for s/wisc/n/ill meso
#    ax9.set_extent((-91.3,-87.0,40.8,45.0))
#    ax9.set_extent((-92.3,-85.0,38.0,45.7))
#    ax9.set_extent((-92.3,-82.0,35.0,45.7))

#    im = ax9.imshow(out[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper', cmap='Greys_r', vmin=0., vmax=1., transform=proj)
#    im = ax8.imshow(out[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper')
#    im = ax1.imshow(out[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right],extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper')
#    im = ax2.imshow(out[mw_image_crop_top:mw_image_crop_bottom,mw_image_crop_left:mw_image_crop_right],extent=(xa[mw_image_crop_left],xa[mw_image_crop_right],ya[mw_image_crop_bottom],ya[mw_image_crop_top]), origin='upper')
##    im = ax3.imshow(out[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right],extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper')
#    im = ax4.imshow(out[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right],extent=(xa[ne_image_crop_left],xa[ne_image_crop_right],ya[ne_image_crop_bottom],ya[ne_image_crop_top]), origin='upper')
    im = ax3.imshow(out[npac_image_crop_top:npac_image_crop_bottom,npac_image_crop_left:npac_image_crop_right],extent=(xa[npac_image_crop_left],xa[npac_image_crop_right],ya[npac_image_crop_bottom],ya[npac_image_crop_top]), origin='upper')
    im = ax4.imshow(out[ak_image_crop_top:ak_image_crop_bottom,ak_image_crop_left:ak_image_crop_right],extent=(xa[ak_image_crop_left],xa[ak_image_crop_right],ya[ak_image_crop_bottom],ya[ak_image_crop_top]), origin='upper')
    im = ax8.imshow(out[namer_image_crop_top:namer_image_crop_bottom,namer_image_crop_left:namer_image_crop_right],extent=(xa[namer_image_crop_left],xa[namer_image_crop_right],ya[namer_image_crop_bottom],ya[namer_image_crop_top]), origin='upper')

    im = ax9.imshow(out[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper')

    import cartopy.feature as cfeat
    fname = '/home/poker/resources/cb_2016_us_county_5m.shp'
    counties = Reader(fname)
# only for close up
#    ax9.add_geometries(counties.geometries(), ccrs.PlateCarree(), edgecolor='darkgreen', facecolor='None')
#    ax8.coastlines(resolution='50m', color='green')
#    ax8.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#    ax1.coastlines(resolution='50m', color='green')
#    ax2.coastlines(resolution='50m', color='green')
#    ax4.coastlines(resolution='50m', color='green')
    ax3.coastlines(resolution='50m', color='green')
    ax4.coastlines(resolution='50m', color='green')
    ax8.coastlines(resolution='50m', color='green')
    ax9.coastlines(resolution='50m', color='green')
#    ax1.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#    ax2.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#    ax4.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#    ax3.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
#    ax4.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
    ax8.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
    ax9.add_feature(cfeat.BORDERS, linewidth=1, edgecolor='green')
    state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

    state_boundaries2 = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='10m', facecolor='none', edgecolor='red')
#    ax8.add_feature(state_boundaries, linestyle=':')
#    ax1.add_feature(state_boundaries, linestyle=':')
#    ax2.add_feature(state_boundaries, linestyle=':')
#    ax4.add_feature(state_boundaries, linestyle=':')
    ax3.add_feature(state_boundaries, linestyle=':')
    ax4.add_feature(state_boundaries, linestyle=':')
    ax8.add_feature(state_boundaries, linestyle=':')
    ax9.add_feature(state_boundaries, linestyle=':')


    time_var = f.time_coverage_start

    iyear = time_var[0:4]
#    print("iyear ",iyear)
    imonth = time_var[5:7]
#    print("imonth ",imonth)
    import calendar
    cmonth = calendar.month_abbr[int(imonth)]
#    print("cmonth ",cmonth)
    iday = time_var[8:10]
#    print("iday ",iday)
    itime = time_var[11:19]
    itimehr = time_var[11:13]
    itimemn = time_var[14:16]
    itimesc = time_var[17:19]
    
    sunique = "swisc"

    ctime_string = iyear +' '+cmonth+' '+iday+'  '+itime+' GMT'
    ctime_file_string = iyear + imonth + iday + itimehr + itimemn + itimesc + "_" + sunique
    list_string = sunique + '.jpg'

    if f.platform_ID == "G17":
        time_string = 'GOES-17 Rayleigh Corrected Reflectance\nRed/Veggie Pseudo Green/Blue Color\n%s '%ctime_string
    elif f.platform_ID == "G18":
        time_string = 'GOES-18 Rayleigh Corrected Reflectance\nRed/Veggie Pseudo Green/Blue Color\n%s '%ctime_string
    else:
        time_string = 'GOES-West Rayleigh Corrected Reflectance\nRed/Veggie Pseudo Green/Blue Color\n%s '%ctime_string

    from matplotlib import patheffects
    outline_effect = [patheffects.withStroke(linewidth=2, foreground='black')]

#    text1 = ax1.text(0.50, 0.90, time_string,
#        horizontalalignment='center', transform = ax1.transAxes,
#        color='yellow', fontsize='large', weight='bold')
#    text1.set_path_effects(outline_effect)
#    text2 = ax2.text(0.50, 0.92, time_string,
#        horizontalalignment='center', transform = ax2.transAxes,
#        color='yellow', fontsize='large', weight='bold')
#    text2.set_path_effects(outline_effect)
#    text4 = ax4.text(0.50, 0.92, time_string,
#        horizontalalignment='center', transform = ax4.transAxes,
#        color='yellow', fontsize='large', weight='bold')
#    text4.set_path_effects(outline_effect)
    text3 = ax3.text(0.005, 0.92, time_string,
        horizontalalignment='left', transform = ax3.transAxes,
        color='yellow', fontsize='12', weight='bold')
    text3.set_path_effects(outline_effect)
    text4 = ax4.text(0.005, 0.91, time_string,
        horizontalalignment='left', transform = ax4.transAxes,
        color='yellow', fontsize='12', weight='bold')
    text4.set_path_effects(outline_effect)
    text8 = ax8.text(0.005, 0.92, time_string,
        horizontalalignment='left', transform = ax8.transAxes,
        color='yellow', fontsize='12', weight='bold')
    text8.set_path_effects(outline_effect)
    text9 = ax9.text(0.005, 0.95, time_string,
        horizontalalignment='left', transform = ax9.transAxes,
        color='yellow', fontsize='12', weight='bold')
    text9.set_path_effects(outline_effect)

    from PIL import Image


    aoslogo = Image.open('/home/poker/uw-aoslogo.png')
    aoslogoheight = aoslogo.size[1]
    aoslogowidth = aoslogo.size[0]

# We need a float array between 0-1, rather than
# a uint8 array between 0-255
    aoslogo = np.array(aoslogo).astype(np.float) / 255

#    oname1 = "/dustdevil/goes17/grb/rgb/wi/"+iyear+imonth+iday+itimehr+itimemn+"_wi.jpg"
#    oname2 = "/dustdevil/goes17/grb/rgb/mw/"+iyear+imonth+iday+itimehr+itimemn+"_mw.jpg"
#    oname4 = "/dustdevil/goes17/grb/rgb/ne/"+iyear+imonth+iday+itimehr+itimemn+"_ne.jpg"
    oname3 = "/dustdevil/goes16/goes17/grb/rgb/npac/"+iyear+imonth+iday+itimehr+itimemn+"_npac.jpg"
    oname4 = "/dustdevil/goes16/goes17/grb/rgb/ak/"+iyear+imonth+iday+itimehr+itimemn+"_ak.jpg"
    oname8 = "/dustdevil/goes16/goes17/grb/rgb/namer/"+iyear+imonth+iday+itimehr+itimemn+"_namer.jpg"
    oname9 = "/dustdevil/goes16/goes17/grb/rgb/fulldisk/"+iyear+imonth+iday+itimehr+itimemn+"_fulldisk.jpg"


##    fig8.savefig('test8.jpg', bbox_inches='tight', pad_inches=0)
##    fig9.savefig('test9.jpg', bbox_inches='tight', pad_inches=0)
#    fig1.figimage(aoslogo,  0, 0, zorder=10)
#    fig1.savefig(oname1, bbox_inches='tight', pad_inches=0)
##    print("done saving wi")
#    fig2.figimage(aoslogo,  0, 0, zorder=10)
#    fig2.savefig(oname2, bbox_inches='tight', pad_inches=0)
##    print("done saving mw")
#    fig4.figimage(aoslogo,  0, 0, zorder=10)
#    fig4.savefig(oname4, bbox_inches='tight', pad_inches=0)
##    print("done saving ne")
    fig3.figimage(aoslogo,  0, 0, zorder=10)
    fig3.savefig(oname3, bbox_inches='tight', pad_inches=0)
    fig4.figimage(aoslogo,  0, 0, zorder=10)
    fig4.savefig(oname4, bbox_inches='tight', pad_inches=0)
    fig8.figimage(aoslogo,  0, 0, zorder=10)
    fig8.savefig(oname8, bbox_inches='tight', pad_inches=0)
#    print("done saving namer")
    fig9.figimage(aoslogo,  0, 0, zorder=10)
    fig9.savefig(oname9, bbox_inches='tight', pad_inches=0)
    f.close
#    print("done saving fulldisk")
    print("done saving images")


    silentremove("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_99.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_98.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_99.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_97.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_98.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_96.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_97.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_95.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_96.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_94.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_95.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_93.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_94.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_92.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_93.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_91.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_92.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_90.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_91.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_89.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_90.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_88.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_89.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_87.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_88.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_86.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_87.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_85.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_86.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_84.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_85.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_83.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_84.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_82.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_83.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_81.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_82.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_80.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_81.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_79.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_80.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_78.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_79.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_77.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_78.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_76.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_77.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_75.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_76.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_74.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_75.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_73.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_74.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_72.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_73.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_71.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_72.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_70.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_71.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_69.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_70.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_68.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_69.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_67.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_68.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_66.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_67.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_65.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_66.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_64.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_65.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_63.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_64.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_62.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_63.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_61.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_62.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_60.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_61.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_59.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_60.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_58.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_59.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_57.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_58.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_56.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_57.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_55.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_56.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_54.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_55.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_53.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_54.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_52.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_53.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_51.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_52.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_50.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_51.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_49.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_50.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_48.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_49.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_47.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_48.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_46.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_47.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_45.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_46.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_44.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_45.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_43.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_44.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_42.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_43.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_41.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_42.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_40.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_41.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_39.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_40.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_38.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_39.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_37.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_38.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_36.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_37.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_35.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_36.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_34.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_35.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_33.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_34.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_32.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_33.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_31.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_32.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_30.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_31.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_29.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_30.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_28.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_29.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_27.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_28.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_26.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_27.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_25.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_26.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_24.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_25.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_23.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_24.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_22.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_23.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_21.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_22.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_20.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_21.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_19.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_20.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_18.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_19.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_17.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_18.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_16.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_17.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_15.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_16.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_14.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_15.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_13.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_14.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_12.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_13.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_11.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_12.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_10.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_11.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_9.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_10.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_8.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_9.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_7.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_8.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_6.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_7.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_5.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_6.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_4.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_5.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_3.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_4.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_2.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_3.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_1.jpg", "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_2.jpg")
    shutil.copy(oname3, "/dustdevil/goes16/goes17/grb/rgb/npac/latest_npac_1.jpg")

    silentremove("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_99.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_98.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_99.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_97.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_98.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_96.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_97.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_95.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_96.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_94.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_95.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_93.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_94.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_92.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_93.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_91.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_92.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_90.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_91.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_89.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_90.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_88.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_89.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_87.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_88.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_86.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_87.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_85.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_86.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_84.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_85.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_83.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_84.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_82.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_83.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_81.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_82.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_80.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_81.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_79.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_80.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_78.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_79.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_77.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_78.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_76.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_77.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_75.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_76.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_74.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_75.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_73.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_74.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_72.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_73.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_71.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_72.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_70.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_71.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_69.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_70.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_68.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_69.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_67.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_68.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_66.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_67.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_65.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_66.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_64.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_65.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_63.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_64.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_62.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_63.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_61.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_62.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_60.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_61.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_59.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_60.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_58.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_59.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_57.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_58.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_56.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_57.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_55.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_56.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_54.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_55.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_53.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_54.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_52.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_53.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_51.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_52.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_50.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_51.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_49.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_50.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_48.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_49.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_47.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_48.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_46.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_47.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_45.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_46.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_44.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_45.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_43.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_44.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_42.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_43.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_41.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_42.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_40.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_41.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_39.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_40.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_38.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_39.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_37.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_38.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_36.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_37.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_35.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_36.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_34.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_35.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_33.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_34.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_32.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_33.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_31.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_32.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_30.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_31.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_29.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_30.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_28.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_29.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_27.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_28.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_26.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_27.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_25.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_26.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_24.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_25.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_23.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_24.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_22.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_23.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_21.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_22.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_20.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_21.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_19.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_20.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_18.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_19.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_17.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_18.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_16.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_17.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_15.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_16.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_14.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_15.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_13.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_14.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_12.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_13.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_11.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_12.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_10.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_11.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_9.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_10.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_8.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_9.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_7.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_8.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_6.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_7.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_5.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_6.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_4.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_5.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_3.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_4.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_2.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_3.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_1.jpg", "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_2.jpg")
    shutil.copy(oname4, "/dustdevil/goes16/goes17/grb/rgb/ak/latest_ak_1.jpg")

    silentremove("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_99.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_98.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_99.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_97.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_98.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_96.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_97.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_95.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_96.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_94.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_95.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_93.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_94.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_92.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_93.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_91.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_92.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_90.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_91.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_89.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_90.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_88.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_89.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_87.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_88.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_86.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_87.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_85.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_86.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_84.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_85.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_83.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_84.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_82.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_83.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_81.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_82.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_80.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_81.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_79.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_80.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_78.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_79.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_77.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_78.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_76.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_77.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_75.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_76.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_74.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_75.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_73.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_74.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_72.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_73.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_71.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_72.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_70.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_71.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_69.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_70.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_68.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_69.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_67.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_68.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_66.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_67.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_65.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_66.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_64.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_65.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_63.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_64.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_62.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_63.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_61.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_62.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_60.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_61.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_59.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_60.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_58.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_59.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_57.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_58.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_56.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_57.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_55.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_56.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_54.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_55.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_53.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_54.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_52.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_53.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_51.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_52.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_50.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_51.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_49.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_50.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_48.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_49.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_47.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_48.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_46.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_47.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_45.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_46.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_44.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_45.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_43.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_44.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_42.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_43.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_41.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_42.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_40.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_41.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_39.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_40.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_38.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_39.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_37.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_38.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_36.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_37.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_35.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_36.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_34.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_35.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_33.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_34.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_32.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_33.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_31.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_32.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_30.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_31.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_29.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_30.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_28.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_29.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_27.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_28.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_26.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_27.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_25.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_26.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_24.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_25.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_23.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_24.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_22.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_23.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_21.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_22.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_20.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_21.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_19.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_20.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_18.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_19.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_17.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_18.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_16.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_17.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_15.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_16.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_14.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_15.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_13.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_14.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_12.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_13.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_11.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_12.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_10.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_11.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_9.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_10.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_8.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_9.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_7.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_8.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_6.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_7.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_5.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_6.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_4.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_5.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_3.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_4.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_2.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_3.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_1.jpg", "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_2.jpg")
    shutil.copy(oname8, "/dustdevil/goes16/goes17/grb/rgb/namer/latest_namer_1.jpg")

    silentremove("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_99.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_98.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_99.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_97.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_98.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_96.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_97.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_95.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_96.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_94.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_95.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_93.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_94.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_92.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_93.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_91.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_92.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_90.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_91.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_89.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_90.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_88.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_89.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_87.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_88.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_86.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_87.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_85.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_86.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_84.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_85.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_83.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_84.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_82.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_83.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_81.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_82.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_80.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_81.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_79.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_80.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_78.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_79.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_77.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_78.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_76.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_77.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_75.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_76.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_74.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_75.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_73.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_74.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_72.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_73.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_71.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_72.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_70.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_71.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_69.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_70.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_68.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_69.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_67.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_68.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_66.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_67.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_65.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_66.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_64.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_65.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_63.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_64.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_62.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_63.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_61.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_62.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_60.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_61.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_59.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_60.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_58.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_59.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_57.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_58.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_56.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_57.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_55.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_56.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_54.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_55.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_53.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_54.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_52.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_53.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_51.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_52.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_50.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_51.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_49.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_50.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_48.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_49.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_47.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_48.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_46.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_47.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_45.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_46.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_44.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_45.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_43.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_44.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_42.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_43.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_41.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_42.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_40.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_41.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_39.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_40.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_38.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_39.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_37.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_38.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_36.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_37.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_35.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_36.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_34.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_35.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_33.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_34.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_32.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_33.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_31.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_32.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_30.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_31.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_29.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_30.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_28.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_29.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_27.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_28.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_26.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_27.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_25.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_26.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_24.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_25.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_23.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_24.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_22.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_23.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_21.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_22.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_20.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_21.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_19.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_20.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_18.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_19.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_17.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_18.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_16.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_17.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_15.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_16.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_14.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_15.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_13.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_14.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_12.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_13.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_11.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_12.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_10.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_11.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_9.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_10.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_8.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_9.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_7.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_8.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_6.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_7.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_5.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_6.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_4.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_5.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_3.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_4.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_2.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_3.jpg")
    silentrename("/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_1.jpg", "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_2.jpg")

    shutil.copy(oname9, "/dustdevil/goes16/goes17/grb/rgb/fulldisk/latest_fulldisk_1.jpg")

    import glob

# NPAC 6h, 24h loops
    file_list = glob.glob('/dustdevil/goes16/goes17/grb/rgb/ak/2*jpg')
    file_list.sort()
    thefile = open('/dustdevil/goes16/goes17/grb/rgb/ak/ak_24h_temp.list', 'w')
    thelist = file_list[-96:]
    #print ("thelist is ",thelist)

    for item in thelist:
        head, tail = os.path.split(item)
        thefile.write(tail + '\n')
    thefile.close
    os.rename('/dustdevil/goes16/goes17/grb/rgb/ak/ak_24h_temp.list','/dustdevil/goes16/goes17/grb/rgb/ak/ak_24h.list')
     
    thefile = open('/dustdevil/goes16/goes17/grb/rgb/ak/ak_6h_temp.list', 'w')
    thelist = file_list[-24:]
    #print ("thelist is ",thelist)

    for item in thelist:
        head, tail = os.path.split(item)
        thefile.write(tail + '\n')
    thefile.close
    os.rename('/dustdevil/goes16/goes17/grb/rgb/ak/ak_6h_temp.list','/dustdevil/goes16/goes17/grb/rgb/ak/ak_6h.list')
#    print("done with AK manipulation")

# NPAC 6h, 24h loops
    file_list = glob.glob('/dustdevil/goes16/goes17/grb/rgb/npac/2*jpg')
    file_list.sort()
    thefile = open('/dustdevil/goes16/goes17/grb/rgb/npac/npac_24h_temp.list', 'w')
    thelist = file_list[-96:]
    #print ("thelist is ",thelist)

    for item in thelist:
        head, tail = os.path.split(item)
        thefile.write(tail + '\n')
    thefile.close
    os.rename('/dustdevil/goes16/goes17/grb/rgb/npac/npac_24h_temp.list','/dustdevil/goes16/goes17/grb/rgb/npac/npac_24h.list')
     
    thefile = open('/dustdevil/goes16/goes17/grb/rgb/npac/npac_6h_temp.list', 'w')
    thelist = file_list[-24:]
    #print ("thelist is ",thelist)

    for item in thelist:
        head, tail = os.path.split(item)
        thefile.write(tail + '\n')
    thefile.close
    os.rename('/dustdevil/goes16/goes17/grb/rgb/npac/npac_6h_temp.list','/dustdevil/goes16/goes17/grb/rgb/npac/npac_6h.list')
#    print("done with NPAC manipulation")

# NAMER 6h, 24h loops
    file_list = glob.glob('/dustdevil/goes16/goes17/grb/rgb/namer/2*jpg')
    file_list.sort()
    thefile = open('/dustdevil/goes16/goes17/grb/rgb/namer/namer_24h_temp.list', 'w')
    thelist = file_list[-96:]
    #print ("thelist is ",thelist)

    for item in thelist:
        head, tail = os.path.split(item)
        thefile.write(tail + '\n')
    thefile.close
    os.rename('/dustdevil/goes16/goes17/grb/rgb/namer/namer_24h_temp.list','/dustdevil/goes16/goes17/grb/rgb/namer/namer_24h.list')
     
    thefile = open('/dustdevil/goes16/goes17/grb/rgb/namer/namer_6h_temp.list', 'w')
    thelist = file_list[-24:]
    #print ("thelist is ",thelist)

    for item in thelist:
        head, tail = os.path.split(item)
        thefile.write(tail + '\n')
    thefile.close
    os.rename('/dustdevil/goes16/goes17/grb/rgb/namer/namer_6h_temp.list','/dustdevil/goes16/goes17/grb/rgb/namer/namer_6h.list')
#    print("done with GULF manipulation")

# FULLDISK 6h,24h loops
    file_list = glob.glob('/dustdevil/goes16/goes17/grb/rgb/fulldisk/2*jpg')
    file_list.sort()
    thefile = open('/dustdevil/goes16/goes17/grb/rgb/fulldisk/fulldisk_24h_temp.list', 'w')
    thelist = file_list[-96:]
    #print ("thelist is ",thelist)

    for item in thelist:
        head, tail = os.path.split(item)
        thefile.write(tail + '\n')
    thefile.close
    os.rename('/dustdevil/goes16/goes17/grb/rgb/fulldisk/fulldisk_24h_temp.list','/dustdevil/goes16/goes17/grb/rgb/fulldisk/fulldisk_24h.list')
     
    thefile = open('/dustdevil/goes16/goes17/grb/rgb/fulldisk/fulldisk_6h_temp.list', 'w')
    thelist = file_list[-24:]
    #print ("thelist is ",thelist)

    for item in thelist:
        head, tail = os.path.split(item)
        thefile.write(tail + '\n')
    thefile.close
    os.rename('/dustdevil/goes16/goes17/grb/rgb/fulldisk/fulldisk_6h_temp.list','/dustdevil/goes16/goes17/grb/rgb/fulldisk/fulldisk_6h.list')
#    print("done with CONUS manipulation")

    print("done with file manipulation")
     

    



#    from PIL import Image
#    from PIL import ImageFilter
#    from PIL import ImageEnhance
#    image = Image.open(os.path.join(output_path,oname))
#    #image = image.filter(ImageFilter.SHARPEN)
#    saturation = 1.25
#    if saturation:
#        #print(saturation)
#        enhancer = ImageEnhance.Color(image)
#        image = enhancer.enhance(saturation)
#    brightness = 1.05
#    if brightness:
#        #print(brightness)
#        enhancer = ImageEnhance.Brightness(image)
#        image = enhancer.enhance(brightness)
#
#    oname = prefix + '_rgb.jpg'
#    image.save(os.path.join(output_path,oname))
##    oname = prefix + '_rgb.jpg'
##    image.save(os.path.join(output_path,oname))

def fbf2grayscale(output_path='./',fnred=None, isBT=False, rotate=None):
    log = logging.getLogger(__name__)
    from PIL import Image
    from pylab import cm
    dims = get_dim_from_filename(fnred)
    log.debug('fbf2grayscale: ', fnred)
    #print(dims)
    if isBT:
        red = np.fromfile(fnred,dtype=np.float32)
    else:
        red = np.fromfile(fnred,dtype=np.float32)/1.1

    #if nonlin_scale:
    #    red = rescale_single(red)

    ii = np.where(~np.isfinite(red))
    red[ii] = 0
    ii = np.where(less(red,0.0))
    red[ii] = 0
    if isBT:
        ii = np.where(greater(red,0.0))
        tmin = np.min(red[ii])
        if tmin < 150.0:
            tmin = 150.0
        log.debug("Min value %f" % (tmin))
        tmin = 200.0
        log.debug("Set Min value %f" % (tmin))
        sf = np.round(np.max(red))
        log.debug("Max value %f" % (sf))
        sf = 300.0
        log.debug("Set Max value %f" % (sf))
        red[ii] = red[ii] - tmin
        sf = np.round(sf-tmin)
        #sf = np.round(np.max(red))
        log.debug("Max shifted value %f" % (sf))
        red[ii] = red[ii]/sf
        log.debug("Max value %f" % (np.max(red)))
        log.debug("Min value %f" % (np.min(red)))

    red = np.reshape(red,(dims[0],dims[1]))

    if rotate == '90':
        red = np.rot90(red,k=1)
    elif rotate == '-90':
        red = np.rot90(red,k=3)

    
    if isBT:
        im = Image.fromarray(cm.gray_r(red, bytes=True))
    else:
        im = Image.fromarray(cm.gray(red, bytes=True))

    (adir,fname) = os.path.split(fnred)
    prefix = fname.split('.real4.')[0]
    #im.save(output_path + '/' + prefix + '_grayscale.png')
    im.save(output_path + '/' + prefix + '_grayscale.jpg')
#    out = np.zeros((dims[0],dims[1],3),dtype=np.uint8)
#    out[:,:,0] = np.uint8(red *255)
#    out[:,:,1] = np.uint8(red *255)
#    out[:,:,2] = np.uint8(red *255)
 
#    (adir,fname) = os.path.split(fnred)
#    prefix = fname.split('.real4.')[0]
#    mpimg.imsave(output_path + '/' + prefix + '_rgb.png',out)

if __name__ == '__main__':
    import init_ahi_log
    init_ahi_log.setup_log()
    log = logging.getLogger(__name__)
    parser = ArgumentParser(description=__doc__)
    #group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-o', '--output_path', help='Directory for output file (mandatory)', required=True)
    #parser.add_argument('-iffcth','--IFF-CTH', action='store_true', default=False, help='VIIRS IFF CTH imagery')
    parser.add_argument('-r','--red', help='Red', required=True)
    parser.add_argument('-g','--green', help='Green')
    parser.add_argument('-b','--blue', help='Blue')
    parser.add_argument('-z','--no-nonlin-scale', action='store_true', default=False)
    parser.add_argument('-t','--is-bt',action='store_true', default=False, help='Flag to rescale brightness temperature bands appropriately')
    parser.add_argument('--rot-night',action='store_true', default=False, help='Rotation for descending orbit... night, else it rotates for day ')
    parser.add_argument('--renorm-value', help='Renormalization reflectance value, default is 1.1')
    args = parser.parse_args()

    log.debug(args.no_nonlin_scale)

    if args.renorm_value:
        renorm = float(args.renorm_value)
    else:
        renorm = 1.1

    #ipdb.set_trace()
    if not args.green and not args.blue:
        if args.is_bt:
            is_bt = True
        else:
            is_bt = False
        #print is_bt
        if args.rot_night:
            rot='-90'
        else:
            rot='90'
        fbf2grayscale(output_path=args.output_path,fnred=args.red,isBT=is_bt,rotate=rot)
    else:
        combine_rgb(output_path=args.output_path,fnred=args.red,fngreen=args.green,fnblue=args.blue,no_nonlin_scale=args.no_nonlin_scale,renorm=renorm)
