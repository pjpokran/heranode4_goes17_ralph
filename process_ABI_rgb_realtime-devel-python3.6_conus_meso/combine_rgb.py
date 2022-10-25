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
    print('max of red   ', np.max(red))
    print('max of green ', np.max(grn))
    print('max of blue  ', np.max(blu))
    
#    ipdb.set_trace()

    is_jpg = True
    if alpha and is_jpg:
        alpha = False

    if fake_green:
        #new_grn = np.zeros_like(grn)
        #TODO only apply to unmasked (not space) pixels
        print("making fake green ??")
        new_grn = 0.45*red + 0.10*grn + 0.45*blu
        grn = new_grn


    if fnir:
        # From /Users/kuehn/Dropbox/MatlabCode/plot_ahi_loc_scatt.m line 140
        #idx = d > 2 & rho4 > 0.27 & rho4 < 0.45 & d < 1.17*d2;
        log.debug('Applying near-ir green correction to rgb image')
        print('Applying near-ir green correction to rgb image')
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
        print("Applying nonlinear channel rescaling")
        log.debug("Applying nonlinear channel rescaling")
        red,grn,blu,A = rescale(red,grn,blu)
    else:
        print("Applying nonlinear channel rescaling")
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
    print('plotted max of red   ', np.max(out[:,:,0]))
    print('plotted max of green ', np.max(out[:,:,1]))
    print('plotted max of blue  ', np.max(out[:,:,2]))

    # npp_viirs_visible_04_20140125_174000_omerc.real4.2300.3150
    (adir,fname) = os.path.split(fnred)
    prefix = fname.split('.real4.')[0]
    oname = prefix + '_rgb_nofilt.jpg'
    oname1 = prefix + '_rgb_wi.jpg'
    oname2 = prefix + '_rgb_mw.jpg'
    oname4 = prefix + '_rgb_ne.jpg'
    oname8 = prefix + '_rgb_gulf.jpg'
    oname9 = prefix + '_rgb_conus.jpg'
    #oname = prefix + '_rgb_nofilt.png'
#    mpimg.imsave(os.path.join(output_path,oname),out)

    import netCDF4
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    f = netCDF4.Dataset(ncfile)
    xa = f.variables['x'][:]
    ya = f.variables['y'][:]
    xa = xa*35785831
    ya = ya*35785831
    globe = ccrs.Globe(semimajor_axis=6378137.,semiminor_axis=6356752.)
    proj = ccrs.Geostationary(central_longitude=-75.0,
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

    gulf_image_crop_top=856
    gulf_image_crop_bottom=-10
    gulf_image_crop_left=1298
    gulf_image_crop_right=-152

    fig1 = plt.figure(figsize=(16.,16.))
    fig2 = plt.figure(figsize=(18.,18.))
#    fig3 = plt.figure(figsize=(18.,18.))
    fig4 = plt.figure(figsize=(18.,18.))
    fig8 = plt.figure(figsize=(30.,30.))
    fig9 = plt.figure(figsize=(20.,20.))

    ax1 = fig1.add_subplot(1, 1, 1, projection=proj)
    ax1.outline_patch.set_edgecolor('none')
    ax1.background_patch.set_fill(False)
    ax1.outline_patch.set_edgecolor('black')
    ax1.patch.set_facecolor('none')
    ax2 = fig2.add_subplot(1, 1, 1, projection=proj)
    ax2.outline_patch.set_edgecolor('none')
    ax2.background_patch.set_fill(False)
    ax2.outline_patch.set_edgecolor('black')
    ax2.patch.set_facecolor('none')
#    ax3 = fig3.add_subplot(1, 1, 1, projection=proj)
#    ax3.outline_patch.set_edgecolor('none')
#    ax3.background_patch.set_fill(False)
#    ax3.outline_patch.set_edgecolor('black')
#    ax3.patch.set_facecolor('none')
    ax4 = fig4.add_subplot(1, 1, 1, projection=proj)
    ax4.outline_patch.set_edgecolor('none')
    ax4.background_patch.set_fill(False)
    ax4.outline_patch.set_edgecolor('black')
    ax4.patch.set_facecolor('none')
    ax8 = fig8.add_subplot(1, 1, 1, projection=proj)
    ax8.outline_patch.set_edgecolor('none')
    ax8.background_patch.set_fill(False)
    ax8.outline_patch.set_edgecolor('black')
    ax8.patch.set_facecolor('none')
    ax9 = fig9.add_subplot(1, 1, 1, projection=proj)
    ax9.outline_patch.set_edgecolor('none')
    ax9.background_patch.set_fill(False)
    ax9.outline_patch.set_edgecolor('black')
    ax9.patch.set_facecolor('none')
# for s/wisc/n/ill meso
#    ax9.set_extent((-91.3,-87.0,40.8,45.0))
#    ax9.set_extent((-92.3,-85.0,38.0,45.7))
#    ax9.set_extent((-92.3,-82.0,35.0,45.7))

#    im = ax9.imshow(out[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper', cmap='Greys_r', vmin=0., vmax=1., transform=proj)
#    im = ax8.imshow(out[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper')
    im = ax1.imshow(out[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right],extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper')
    im = ax2.imshow(out[mw_image_crop_top:mw_image_crop_bottom,mw_image_crop_left:mw_image_crop_right],extent=(xa[mw_image_crop_left],xa[mw_image_crop_right],ya[mw_image_crop_bottom],ya[mw_image_crop_top]), origin='upper')
#    im = ax3.imshow(out[wi_image_crop_top:wi_image_crop_bottom,wi_image_crop_left:wi_image_crop_right],extent=(xa[wi_image_crop_left],xa[wi_image_crop_right],ya[wi_image_crop_bottom],ya[wi_image_crop_top]), origin='upper')
    im = ax4.imshow(out[ne_image_crop_top:ne_image_crop_bottom,ne_image_crop_left:ne_image_crop_right],extent=(xa[ne_image_crop_left],xa[ne_image_crop_right],ya[ne_image_crop_bottom],ya[ne_image_crop_top]), origin='upper')
    im = ax8.imshow(out[gulf_image_crop_top:gulf_image_crop_bottom,gulf_image_crop_left:gulf_image_crop_right],extent=(xa[gulf_image_crop_left],xa[gulf_image_crop_right],ya[gulf_image_crop_bottom],ya[gulf_image_crop_top]), origin='upper')

    im = ax9.imshow(out[:], extent=(xa[0],xa[-1],ya[-1],ya[0]), origin='upper')

    import cartopy.feature as cfeat
    fname = '/home/poker/resources/cb_2016_us_county_5m.shp'
    counties = Reader(fname)
# only for close up
#    ax9.add_geometries(counties.geometries(), ccrs.PlateCarree(), edgecolor='darkgreen', facecolor='None')
#    ax8.coastlines(resolution='50m', color='green')
#    ax8.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
    ax1.coastlines(resolution='50m', color='green')
    ax2.coastlines(resolution='50m', color='green')
    ax4.coastlines(resolution='50m', color='green')
    ax8.coastlines(resolution='50m', color='green')
    ax9.coastlines(resolution='50m', color='green')
    ax1.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
    ax2.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
    ax4.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
    ax8.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
    ax9.add_feature(cfeat.BORDERS, linewidth='1', edgecolor='green')
    state_boundaries = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='50m', facecolor='none', edgecolor='red')

    state_boundaries2 = cfeat.NaturalEarthFeature(category='cultural',
                                             name='admin_1_states_provinces_lakes',
                                             scale='10m', facecolor='none', edgecolor='red')
#    ax8.add_feature(state_boundaries, linestyle=':')
    ax1.add_feature(state_boundaries, linestyle=':')
    ax2.add_feature(state_boundaries, linestyle=':')
    ax4.add_feature(state_boundaries, linestyle=':')
    ax8.add_feature(state_boundaries, linestyle=':')
    ax9.add_feature(state_boundaries, linestyle=':')


    time_var = f.time_coverage_start

    iyear = time_var[0:4]
    print("iyear ",iyear)
    imonth = time_var[5:7]
    print("imonth ",imonth)
    import calendar
    cmonth = calendar.month_abbr[int(imonth)]
    print("cmonth ",cmonth)
    iday = time_var[8:10]
    print("iday ",iday)
    itime = time_var[11:19]
    itimehr = time_var[11:13]
    itimemn = time_var[14:16]
    itimesc = time_var[17:19]
    
    sunique = "swisc"

    ctime_string = iyear +' '+cmonth+' '+iday+'  '+itime+' GMT'
    ctime_file_string = iyear + imonth + iday + itimehr + itimemn + itimesc + "_" + sunique
    list_string = sunique + '.jpg'
    time_string = 'GOES-16 Rayleigh Corrected Reflectance\nRed/Veggie Pseudo Green/Blue Color\n %s '%ctime_string
    from matplotlib import patheffects
    outline_effect = [patheffects.withStroke(linewidth=2, foreground='black')]

    text1 = ax1.text(0.50, 0.90, time_string,
        horizontalalignment='center', transform = ax1.transAxes,
        color='yellow', fontsize='large', weight='bold')
    text1.set_path_effects(outline_effect)
    text2 = ax2.text(0.50, 0.92, time_string,
        horizontalalignment='center', transform = ax2.transAxes,
        color='yellow', fontsize='large', weight='bold')
    text2.set_path_effects(outline_effect)
    text4 = ax4.text(0.50, 0.92, time_string,
        horizontalalignment='center', transform = ax4.transAxes,
        color='yellow', fontsize='large', weight='bold')
    text4.set_path_effects(outline_effect)
    text8 = ax8.text(0.50, 0.95, time_string,
        horizontalalignment='center', transform = ax8.transAxes,
        color='yellow', fontsize='large', weight='bold')
    text8.set_path_effects(outline_effect)
    text9 = ax9.text(0.50, 0.93, time_string,
        horizontalalignment='center', transform = ax9.transAxes,
        color='yellow', fontsize='large', weight='bold')
    text9.set_path_effects(outline_effect)

    from PIL import Image


    aoslogo = Image.open('/home/poker/uw-aoslogo.png')
    aoslogoheight = aoslogo.size[1]
    aoslogowidth = aoslogo.size[0]

# We need a float array between 0-1, rather than
# a uint8 array between 0-255
    aoslogo = np.array(aoslogo).astype(np.float) / 255



#    fig8.savefig('test8.jpg', bbox_inches='tight', pad_inches=0)
#    fig9.savefig('test9.jpg', bbox_inches='tight', pad_inches=0)
    fig1.figimage(aoslogo,  0, 0, zorder=10)
    fig1.savefig(os.path.join(output_path,oname1), bbox_inches='tight', pad_inches=0)
    fig2.figimage(aoslogo,  0, 0, zorder=10)
    fig2.savefig(os.path.join(output_path,oname2), bbox_inches='tight', pad_inches=0)
    fig4.figimage(aoslogo,  0, 0, zorder=10)
    fig4.savefig(os.path.join(output_path,oname4), bbox_inches='tight', pad_inches=0)
    fig8.figimage(aoslogo,  0, 0, zorder=10)
    fig8.savefig(os.path.join(output_path,oname8), bbox_inches='tight', pad_inches=0)
    fig9.figimage(aoslogo,  0, 0, zorder=10)
    fig9.savefig(os.path.join(output_path,oname9), bbox_inches='tight', pad_inches=0)
    f.close




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
