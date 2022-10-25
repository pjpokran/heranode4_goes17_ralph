from numpy import floor,zeros,size,std,mean,divide,bitwise_and,array,squeeze,\
                  concatenate,shape,min,max,nonzero,equal,not_equal,nan,reshape,\
                  log,arange,less,logical_or,logical_and,greater_equal,greater,cos,sin,arctan2,pi,sqrt,linspace
# from matplotlib.toolkits.basemap import Basemap, shiftgrid
from mpl_toolkits.basemap import Basemap
#from pylab import meshgrid, show, clf, cm, colorbar, nx, savefig, axes, setp, gca, figure, title
from pylab import meshgrid, show, clf, cm, colorbar, ma, savefig, axes, setp, gca, figure, title, close
from matplotlib import colors
from matplotlib import colorbar
from matplotlib.colors import LogNorm
from satutil_lib import *
import pprint
#import ipdb
#from matplotlib.pyplot import scatter, imshow

def my_jet_data():
    _jet_data =   {'red':   ((0.,0.,0.),
        (0.1, 0.6, 0.6),
        (0.18, 0., 0.),
        (0.45, 0, 0),
        (0.66, 1, 1),
        (0.89,1, 1),
        (1, 0.5, 0.5)),
        'green': ((0., 0., 0.),
        (0.1,0.,0.0),
        (0.18, 0, 0),
        (0.225,0, 0),
        (0.375,1, 1),
        (0.64,1, 1),
        (0.91,0,0),
        (1, 0, 0)),
        'blue':  ((0. ,0. ,0.),
        (0.1,0.6,0.6),
        (0.18, 0.4, 0.4),
        (0.21, 0.6, 0.6),
        (0.34, 1, 1),
        (0.65,0, 0),
        (1, 0, 0))}
    return(_jet_data)

# From  http://stackoverflow.com/questions/18704353/correcting-matplotlib-colorbar-ticks
# http://wiki.scipy.org/Cookbook/Matplotlib/ColormapTransformations
def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str or type(cmap) == str:
        cmap = cm.get_cmap(cmap)
    colors_i = concatenate((linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

# From  http://stackoverflow.com/questions/18704353/correcting-matplotlib-colorbar-ticks
# http://wiki.scipy.org/Cookbook/Matplotlib/ColormapTransformations
def colorbar_index(ncolors, cmap):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    mcolorbar = colorbar(mappable)
    mcolorbar.set_ticks(linspace(0, ncolors, ncolors))
    mcolorbar.set_ticklabels(list(range(ncolors)))

class sat_image:
    def __init__(self,lat,lon,data,vmax,title_txt,coastline_color='white',cmap='gnuplot'):

        self.vmax = vmax
        self.fig=figure(figsize=(8.5,8.0))
        self.ax = self.fig.add_axes([0.015,0.015,0.97,0.95])
        self.fig.patch.set_alpha(0.0)
        self.ax.set_aspect('auto')
        print('Creating basemap...')
        self.m = Basemap(projection='geos',lon_0=0,resolution='c')
        x,y = self.m(lon,lat)
        # From profiling  pcolor takes almost 85% of the time so optimizing anything
        # else not worth while
        self.img_inst = self.m.pcolor(x,y,data,shading='flat', cmap=cmap,vmin=0,vmax=vmax)

        print('Draw coastlines')
        self.m.drawcoastlines(color=coastline_color)
        self.m.drawcountries(color=coastline_color)
        self.m.drawstates(color=coastline_color)
        self.m.drawmapboundary(color=coastline_color,fill_color='#303030')
        parallels = arange(-80.,90,10.)
        meridians = arange(0.,360.,10.)
        self.m.drawparallels(parallels,labels=[0,0,0,0],color='lightgrey')
        self.m.drawmeridians(meridians,labels=[0,0,0,0],fontsize=8,fmt='%3d',color='lightgrey')

        title( title_txt, color=coastline_color)
        print('Draw colorbar')
        self.ax = gca()

    def add_colorbar(self,cb_ticks,text_color='k'):
        axes(self.ax)

        cb_ax = self.fig.add_axes([0.94, 0.015, 0.015, 0.3])

        cb = self.fig.colorbar(self.img_inst,ticks=cb_ticks,cax=cb_ax)
        for t in cb.ax.get_yticklabels():
            setp(t,'color',text_color,'fontsize',8)


class granule_sat_image:
    def __init__(self,lat,lon,proj='ortho',width=2300, height=3100, fill_color='#303030',cgeo='Geo'):
        self.mLat = [max(lat), min(lat)]
        self.mLon = [max(lon), min(lon)]
        self.ascending= ''
        self.projparams= ''
        self.fill_color=fill_color
        
        n,m = shape(lat)
        idx1 = m/2-1 # Midpoint along the second dim
        idx2 = n-1  # Endpoint of first dim
        idx0 = n/2-1  # Mid of first dim
        print((idx1, idx2, idx0))
        if n % 2 == 0:
            idx11 = [idx1, idx1+1]
        else:
            idx11 = idx1
        if m % 2 == 0:
            idx00 = [idx0, idx0+1]
        else:
            idx00 = idx0
        print((idx11, idx2, idx00))
        if cgeo == 'Geo':
            import time
            t = time.time()
            print('Calculating center of grid from spherical coordinates, can take some time if grid is large')
            [lat_0, lon_0] = center_geolocation(lat*pi /180,lon * pi/180)
            print('Elapsed time %f' % (time.time()-t))
            print('Center Lat,Lon %8.3f,%8.3f' % (lat_0,lon_0))
        elif cgeo == 'Idx':
            lat_0, lon_0 = float(mean(lat[idx00,idx11])),float(mean(lon[idx00,idx11]))
            print('Center Lat,Lon from index %8.3f,%8.3f' % (lat_0,lon_0))
            print('Absolute Lat limits %8.3f,%8.3f' % (self.mLat[0],self.mLat[1]))
            print('Absolute Lon limits %8.3f,%8.3f' % (self.mLon[0],self.mLon[1]))
            print('Creating basemap...')
            if proj == 'goes':
                print('Using geostationary projection.')
                self.m1 = Basemap(projection='geos',lon_0=0,resolution=None)
            elif proj == 'ortho':
                print('Using orthographic projection.')
                self.m1 = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution=None)
            elif proj == 'omerc':
                lat_2 = float(mean(lat[0, idx11]))
                lon_2 = float(mean(lon[0, idx11]))
                lat_1 = float(mean(lat[-1, idx11]))
                lon_1 = float(mean(lon[-1, idx11]))
                if lat_1 > lat_2:
                    self.ascending = True
                else:
                    self.ascending = False
                # ll_lon = lon[0,-1]
                # ll_lat = lat[0,-1]
                # ur_lon = lon[-1,0]
                # ur_lat = lat[-1,0]
                ll_lon = lon[0,0]
                ll_lat = lat[0,0]
                ur_lon = lon[-1,-1]
                ur_lat = lat[-1,-1]
                self.width = width * 1000
                self.height = height * 1000
                print('Corner points from data (ll, ur), lon, lat %8.3f, %8.3f, %8.3f, %8.3f'% (ll_lon, ll_lat, ur_lon, ur_lat))
                self.m1 = Basemap(projection=proj,lon_0=lon_0,lat_0=lat_0,resolution=None,\
                    lon_1=lon_1, lat_1=lat_1, lon_2=lon_2, lat_2=lat_2, no_rot=True,\
                    llcrnrlon=ll_lon, llcrnrlat=ll_lat, urcrnrlon=ur_lon, urcrnrlat=ur_lat)
                pprint.pprint(self.m1.projparams,indent=4,width=1)
                x_0, y_0 = self.m1(lon_0, lat_0)
                ii = [0,0,-1,-1]
                jj = [0,-1,-1,0]
                xx, yy =self.m1(lon[ii,jj], lat[ii,jj])
                print('xMax,xMin distance from x_o (%10.1f, %10.1f)' % (max(xx-x_0),min(xx-x_0)))
                print('yMax,yMin distance from y_o (%10.1f, %10.1f)' % (max(yy-y_0),min(yy-y_0)))
                #pdb.set_trace()
                ur_lon, ur_lat = self.m1(x_0 - 0.5*self.width, y_0 - 0.5*self.height, inverse=True)
                ul_lon, ul_lat = self.m1(x_0 + 0.5*self.width, y_0 - 0.5*self.height, inverse=True)
                lr_lon, lr_lat = self.m1(x_0 - 0.5*self.width, y_0  + 0.5*self.height, inverse=True)
                ll_lon, ll_lat = self.m1(x_0 + 0.5*self.width, y_0  + 0.5*self.height, inverse=True)
                print('x +/- 0.5 width  = (%10.1f, %10.1f)' % (x_0 + 0.5*self.width,x_0 - 0.5*self.width))
                print('y +/- 0.5 height = (%10.1f, %10.1f)' % (y_0 + 0.5*self.height,y_0 - 0.5*self.height))
                print('From inverse')
                print('Lons (%8.3f, %8.3f, %8.3f, %8.3f)' % (ur_lon, ul_lon, ll_lon, lr_lon))
                print('Lats (%8.3f, %8.3f, %8.3f, %8.3f)\n' % (ur_lat, ul_lat, ll_lat, lr_lat))
            elif proj == 'nsper_spec':
                print('Using special near-sided projection.')
                self.m1 = Basemap(projection='nsper',lon_0=lon_0,lat_0=lat_0,resolution=None,\
                    satellite_height=700*1000.0 )
            else:
                print('Oops! Invalid projection specified, exiting.')
                sys.exit()

            if proj == 'goes' or proj == 'ortho' or proj == 'nsper_spec':
                if proj == 'goes':
                    xfudge = 0.03
                else:
                    xfudge = 0
                m1x,m1y = self.m1(lon,lat)
                dx = 1.0e4

                mx = [max(m1x)+dx, min(m1x)-dx]
                my = [max(m1y)+dx,min(m1y)-dx]
                mx_0 = (mx[0]-mx[1])/2+mx[1]
                my_0 = (my[0]-my[1])/2+my[1]
                lon_0, lat_0 = self.m1(mx_0,my_0,inverse=True)
                print(self.m1(mx,my,inverse=True))
                mx[0] = mx[0]-0.5*self.m1.xmax
                mx[1] = mx[1]-0.5*self.m1.xmax# - xfudge*self.m1.xmax
                my[0] = my[0]-0.5*self.m1.ymax
                my[1] = my[1]-0.5*self.m1.ymax

            #pdb.set_trace()
            if proj == 'goes':
                self.m = Basemap(projection='geos',lon_0=0,resolution='l',\
                    llcrnrx=mx[1],llcrnry=my[1],urcrnrx=mx[0],urcrnry=my[0])
            elif proj == 'ortho':
                self.m = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution='l',\
                    llcrnrx=mx[1],llcrnry=my[1],urcrnrx=mx[0],urcrnry=my[0])
            elif proj == 'omerc':
                self.m = Basemap(projection=proj,lon_0=lon_0,lat_0=lat_0,resolution='l',\
                    lon_1=lon_1, lat_1=lat_1, lon_2=lon_2, lat_2=lat_2, no_rot=True,\
                    llcrnrlon=ll_lon, llcrnrlat=ll_lat, urcrnrlon=ur_lon, urcrnrlat=ur_lat)
                pprint.pprint(self.m.projparams,indent=4,width=1)
                x1, y1 = self.m(lon_1, lat_1)
                x2, y2 = self.m(lon_2, lat_2)
                print('x,y of gc points (%10.1f %10.1f) (%10.2f, %10.1f)' % (x1, y1, x2, y2))
                #print  self.m.projparams
                self.projparams = self.m.projparams
                self.projparams_o = self.m1.projparams
                self.lat_0 = lat_0
                self.lon_0 = lon_0
                self.corner_points = {'ll_lon': ll_lon, 'll_lat': ll_lat, 'ur_lon': ur_lon, 'ur_lat': ur_lat,
                'lr_lon': lr_lon, 'lr_lat': lr_lat, 'ul_lon': ul_lon, 'ul_lat': ul_lat }
                x_0, y_0 = self.m(lon_0, lat_0)
                print('lon_0, lat_0, x_0, y_0 (%8.3f, %8.3f) (%10.1f, %10.1f)\n' % (lon_0, lat_0, x_0, y_0))
                self.grid_origin_x, self.grid_origin_y = (x_0 - 0.5*self.width, y_0 + 0.5*self.height)
                self.grid_origin_x2, self.grid_origin_y2 = (x_0 - 0.5*self.width, y_0 - 0.5*self.height)
            elif proj == 'nsper':
                print('Using near-sided (satellite) projection.')
                self.m = Basemap(projection='nsper',lon_0=lon_0,lat_0=lat_0,\
                    satellite_height=702*1000.,resolution='l')
            elif proj == 'nsper_spec':
                self.m = Basemap(projection='nsper',lon_0=lon_0,lat_0=lat_0,\
                    satellite_height=35786*1000,llcrnrx=mx[1],llcrnry=my[1],\
                    urcrnrx=mx[0],urcrnry=my[0],resolution='l')

    def plot_data(self,lat,lon,data,vmax,title_txt='',marker_size=3,vmin=[0],figsize=(9.5,7.05),\
        axes_lim=[0.015,0.015,0.93,0.95], coastline_color='white',cmap=['gnuplot'],\
        proj='ortho',norm='linear',two_cmap=False,z2=0,ncolors=None,mask_names=None,\
        cb_label=None, units=None, mask_applied_name=None,dpi=72):
        r = self.width/self.height
        fig_width = self.width/dpi/1000
        fig_height = fig_width/r
        if not figsize:
            self.fig=figure(figsize=(fig_width,fig_height),dpi=dpi)
        else:
            self.fig=figure("GranuleData",figsize=figsize,dpi=100)
        self.vmax = vmax
        # add_axes(left,bottom, width, height)
        self.ax = self.fig.add_axes(axes_lim)
        self.fig.patch.set_alpha(0.0)
        self.ax.set_aspect('auto')
        self.cmap = cmap
        self.vmax = vmax
        self.vmin = vmin
        self.norm = norm
        self.ncolors= ncolors
        self.cb_label = cb_label
        self.units = units
        # For a masked data set (i.e. cloud mask) with discrete colormap
        self.mask_names = mask_names
        # For data set that is plotted with two distinct colormaps masked by a second data set
        # i.e. cloud OD with an 'ice' colormap and 'water' colormap
        self.mask_a_name = mask_applied_name

        x,y = self.m(lon,lat)
        print('Plotting image...')
        if not two_cmap:
            if norm == 'linear':
                self.img_inst1 = self.m.scatter(x,y, ax=self.ax,s=marker_size,c=data,marker=',', edgecolor='none', cmap=cmap[0], vmin=vmin[0], vmax=vmax[0])
            elif norm == 'log':
                self.img_inst1 = self.m.scatter(x,y, ax=self.ax,s=marker_size,c=data,marker=',', edgecolor='none', cmap=cmap[0], norm=LogNorm(vmin=vmin[0], vmax=vmax[0]))
            elif norm == 'discrete':
                #self.cmap[0] = cm.get_cmap(self.cmap[0],ncolors)
                self.cmap[0] = cmap_discretize(self.cmap[0], ncolors)
                self.mask_names = mask_names
                self.img_inst1 = self.m.scatter(x,y, ax=self.ax,s=marker_size,c=data,marker=',', edgecolor='none', cmap=cmap[0], vmin=vmin[0], vmax=vmax[0])
        else:
            if norm == 'linear':
                self.img_inst1 = self.m.scatter(x,y, ax=self.ax,s=marker_size,c=data,marker=',', edgecolor='none', cmap=cmap[0], vmin=vmin[0], vmax=vmax[0])
                self.img_inst2 = self.m.scatter(x,y, ax=self.ax,s=marker_size,c=z2,marker=',', edgecolor='none', cmap=cmap[1], vmin=vmin[1], vmax=vmax[1])
            elif norm == 'log':
                self.img_inst1 = self.m.scatter(x,y, ax=self.ax,s=marker_size,c=data,marker=',', edgecolor='none', cmap=cmap[0], norm=LogNorm(vmin=vmin[0], vmax=vmax[0]))
                self.img_inst2 = self.m.scatter(x,y, ax=self.ax,s=marker_size,c=z2,marker=',', edgecolor='none', cmap=cmap[1], norm=LogNorm(vmin=vmin[1], vmax=vmax[1]))

        print('Draw coastlines')
        self.m.drawcoastlines(color=coastline_color)
        self.m.drawcountries(color=coastline_color)
        self.m.drawstates(color=coastline_color)
        self.m.drawmapboundary(color=None,fill_color=self.fill_color)
        parallels = arange(-80.,90,10.)
        meridians = arange(0.,360.,10.)
        self.m.drawparallels(parallels,labels=[0,0,0,0],color='lightgrey')
        self.m.drawmeridians(meridians,labels=[0,0,0,0],fontsize=8,fmt='%3d',color='lightgrey')

        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()

        title( title_txt, color='k')
        print('Draw colorbar')
        self.ax = gca()

    def img_data(self,lat,lon,rgb,title_txt,coastline_color='white'):
        self.fig=figure(figsize=(9.5,7.5))
        # add_axes(left,bottom, width, height)
        self.ax = self.fig.add_axes([0.015,0.015,0.94,0.95])
        self.fig.patch.set_alpha(0.0)
        self.ax.set_aspect('auto')

        print('Plotting image...')
        x,y = self.m(lon,lat)
        self.img_inst1 = self.m.imshow(rgb, interpolation='nearest', origin='upper')

        print('Draw coastlines')
        self.m.drawcoastlines(color=coastline_color)
        self.m.drawcountries(color=coastline_color)
        self.m.drawstates(color=coastline_color)
        self.m.drawmapboundary(color=coastline_color,fill_color='#303030')
        parallels = arange(-80.,90,10.)
        meridians = arange(0.,360.,10.)
        self.m.drawparallels(parallels,labels=[0,0,0,0],color='lightgrey')
        self.m.drawmeridians(meridians,labels=[0,0,0,0],fontsize=8,fmt='%3d',color='lightgrey')

        title( title_txt, color='k')
        print('Draw colorbar')
        self.ax = gca()

    def img_p2g_coastline(self,title_txt,width=2300, height=3250, dpi=72,coastline_color='white'):
        r = width/height
        fig_width = width/dpi
        fig_height = fig_width/r
        self.fig=figure(figsize=(fig_width,fig_height),dpi=dpi)
        # add_axes(left,bottom, width, height)
        self.ax = self.fig.add_axes([0.0,0.0,1.0,1.0])
        self.fig.patch.set_alpha(0.0)
        self.ax.set_aspect('auto')

        print('Draw coastlines')
        self.m.drawcoastlines(color=coastline_color)
        self.m.drawcountries(color=coastline_color)
        self.m.drawstates(color=coastline_color)
        self.m.drawmapboundary(color=coastline_color,fill_color='#303030')
        parallels = arange(-80.,90,10.)
        meridians = arange(0.,360.,10.)
        self.m.drawparallels(parallels,labels=[0,0,0,0],color='lightgrey')
        self.m.drawmeridians(meridians,labels=[0,0,0,0],fontsize=8,fmt='%3d',color='lightgrey')
        self.fig.patch.set_visible(False)
        self.ax.axis('off')

        title( title_txt, color='k')
        self.ax = gca()

    def plot_track(self,lat,lon,figsize=(9.5,7.5),axes_lim=[0.015,0.015,0.93,0.95],\
        coastline_color='white',color='blue',linewidth=3):
        if not self.m:
            print("ERROR: satplotlib:plot_track() No projection set, returning.")

        self.fig_track=figure("TrackData",figsize=figsize,dpi=100)
        self.ax_track = self.fig_track.add_axes(axes_lim)
        self.fig_track.patch.set_alpha(0.0)
        self.ax_track.set_aspect('auto')

        x,y = self.m(lon,lat)
        print('Plotting track...')
        self.img_inst1 = self.m.plot(x, y, '.',ax=self.ax_track, color=color, linewidth=3)
        self.ax_track.set_xlim(self.xlim)
        self.ax_track.set_ylim(self.ylim)
        self.fig_track.patch.set_visible(False)
        self.ax_track.axis('off')

    def make_global_thumbnail(self,lat,lon,proj='eck4'):
        ii = [0,0,-1,-1,0]
        jj = [0,-1,-1,0,0]
        Lats = lat[ii,jj]
        Lons = lon[ii,jj]
        idx1 = shape(lat)[1]/2-1 # Midpoint along the second dim
        xLon = lon[-1,idx1]
        xLat = lat[-1,idx1]
        if proj == 'ortho':
            [lat_0, lon_0] = center_geolocation(lat *pi /180,lon * pi/180)
            self.mth = Basemap(projection='ortho',lon_0=lon_0,lat_0=lat_0,resolution='c')
            self.fig = figure(figsize=(3.0,3.0))
        else:
            self.mth = Basemap(projection='eck4',lon_0=0,resolution='c')
            self.fig = figure(figsize=(3.5,2.5))

        # add_axes(left,bottom, width, height)
        self.ax = self.fig.add_axes([0,0,1,1])
        self.fig.patch.set_alpha(0.0)
        self.ax.set_aspect('auto')

        print('Plotting image...')
        x,y = self.mth(Lons,Lats)
        xx, yy = self.mth(xLon,xLat)
        self.img_inst1 = self.mth.plot(x,y,'-',color='r',linewidth=2.0)
        self.img_inst1 = self.mth.plot(xx,yy,'o',color='y',markersize=6.0)

        print('Draw coastlines')
        coastline_color = 'white'
        self.mth.drawcoastlines(color=coastline_color)
        self.mth.drawmapboundary(color=coastline_color,fill_color='#303030')
        parallels = arange(-80.,90,30.)
        meridians = arange(0.,360.,30.)
        self.mth.drawparallels(parallels,labels=[0,0,0,0],color='lightgrey')
        self.mth.drawmeridians(meridians,labels=[0,0,0,0],fontsize=8,fmt='%3d',color='lightgrey')

        print('Draw colorbar')
        self.ax = gca()

    def add_colorbar(self,cb_ticks,text_color='k',cb_ticks2=None):
        axes(self.ax)

        cb_ax = self.fig.add_axes([0.95, 0.015, 0.015, 0.3])

        cb = self.fig.colorbar(self.img_inst1,ticks=cb_ticks,cax=cb_ax)
        for t in cb.ax.get_yticklabels():
            setp(t,'color',text_color,'fontsize',8)

        if hasattr(self, 'img_inst2'):
            cb_ax = self.fig.add_axes([0.9, 0.015, 0.015, 0.3])

            cb = self.fig.colorbar(self.img_inst2,ticks=cb_ticks2,cax=cb_ax)
            for t in cb.ax.get_yticklabels():
                setp(t,'color',text_color,'fontsize',8)

    def separate_colorbar(self, orientation='vertical'):
        if hasattr(self, 'fig'):
            self.fig.clf()
        self.fig=figure(figsize=(3.0,3.0),dpi=100)
        self.fig.patch.set_alpha(0.0)
        ax_loc = []
        height = 0.05
        width = 0.9
        if orientation == 'horizontal':
            left = [0.05, 0.05]
            bottom = [0.8, 0.475]
            ax_loc.append([left[0], bottom[0], width, height])
            ax_loc.append([left[1], bottom[1], width, height])
        else:
            left = [0.05, 0.475]
            bottom = [0.05, 0.05]
            ax_loc.append([left[0], bottom[0], height, width])
            ax_loc.append([left[1], bottom[1], height, width])

        for i in range(len(self.cmap)):
            if self.norm == 'discrete':
                ticks = linspace(0, self.ncolors, self.ncolors)
                cnorm = colors.Normalize(vmin=0, vmax=self.ncolors)
            else:
                cnorm = colors.Normalize(vmin=self.vmin[i], vmax=self.vmax[i])
                diff = self.vmax[i] - self.vmin[i]
                if diff < 1.0:
                    ticks = linspace(self.vmin[i], self.vmax[i], 4.0)
                elif diff < 10.0:
                    ticks = arange(self.vmin[i], self.vmax[i] +1.0)
                else:
                    ticks = None

            ax = self.fig.add_axes(ax_loc[i])
            cb1 = colorbar.ColorbarBase(ax, cmap=self.cmap[i], ticks=ticks, norm=cnorm, orientation=orientation)
            if self.cb_label and self.units:
                if self.mask_a_name:
                    cb1.set_label(self.mask_a_name[i] + " " + self.cb_label + ' (' + self.units + ')')
                else:
                    cb1.set_label(self.cb_label + ' (' + self.units + ')')

            if self.norm == 'discrete':
                cb1.set_array([])
                cb1.set_clim(-0.5, self.ncolors+0.5)
                #cb1.set_ticklabels(range(self.ncolors))
                cb1.set_ticklabels(self.mask_names)
                if orientation == 'horizontal':
                    for t in cb1.ax.get_xticklabels():
                        #setp(t,'color',text_color,'fontsize',8)
                        setp(t,'fontsize',6)
                else:
                    for t in cb1.ax.get_yticklabels():
                        #setp(t,'color',text_color,'fontsize',8)
                        setp(t,'fontsize',6)








class global_gridded_data_set:
    def __init__(self,lat,lon,proj='eck4'):
        self.mLat = [max(lat), min(lat)]
        self.mLon = [max(lon), min(lon)]

        print('Creating basemap...')
        if proj == 'eck4':
            print('Using eck4 projection.')
            self.m = Basemap(projection='eck4',lon_0=0,resolution='c')
        elif proj == 'foo':
            print('Using foo projection.')
        else:
            print('Oops! Invalid projection specified, exiting.')
            sys.exit()

        self.x,self.y = self.m(lon,lat)

    def plot_data(self,data,vmax,title_txt,vmin=0,coastline_color='black',cmap='jet'):
        self.vmax = vmax
        self.fig=figure(figsize=(9.5,7.5))
        # add_axes(left,bottom, width, height)
        self.ax = self.fig.add_axes([0.015,0.015,0.94,0.95])
        self.fig.patch.set_alpha(0.0)
        self.ax.set_aspect('auto')

        print('Plotting image...')
        #self.img_inst = self.m.scatter(self.x,self.y,s=marker_size,c=data,marker=',', edgecolor='none',cmap=cmap,vmin=vmin,vmax=vmax)
        self.img_inst = self.m.pcolor(self.x, self.y, data, cmap=cmap, vmin=vmin, vmax=vmax)

        print('Draw coastlines')
        self.m.drawcoastlines(color=coastline_color)
        #self.m.drawcountries(color=coastline_color)
        #self.m.drawstates(color=coastline_color)
        self.m.drawmapboundary(color=coastline_color,fill_color='#303030')
        parallels = arange(-80.,90,30.)
        meridians = arange(0.,360.,30.)
        self.m.drawparallels(parallels,labels=[0,0,0,0],color='lightgrey')
        self.m.drawmeridians(meridians,labels=[0,0,0,0],fontsize=8,fmt='%3d',color='lightgrey')

        title( title_txt, color='k')
        print('Draw colorbar')
        self.ax = gca()

    def add_colorbar(self,cb_ticks,text_color='k'):
        axes(self.ax)

        cb_ax = self.fig.add_axes([0.96, 0.015, 0.015, 0.3])

        cb = self.fig.colorbar(self.img_inst,ticks=cb_ticks,cax=cb_ax)
        for t in cb.ax.get_yticklabels():
            setp(t,'color',text_color,'fontsize',8)

    def close(self):
        close(self.fig)

