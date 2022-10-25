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
import pprint
import ipdb


def img_p2g_coastline(title_txt,width=2300, height=3250, dpi=72,coastline_color='#555555'):

    print('Using geostationary projection.')
    #m = Basemap(projection='geos',lon_0=140.7,resolution='h',rmajor=(6378.1370*1.0e3),
    #        rminor=(6356.75231414 * 1.0e3),satellite_height=(35786.0230*1.0e3))
    ipdb.set_trace()
    m = Basemap(projection='geos',lon_0=140.7,resolution='h',satellite_height=(35786.0230*1.0e3))
    m.rmajor = (6378.1370*1.0e3)
    m.rminor = (6356.75231414 * 1.0e3)
    r = width/height
    fig_width = width/dpi
    fig_height = fig_width/r
    fig=figure(figsize=(fig_width,fig_height),dpi=dpi)
    # add_axes(left,bottom, width, height)
    ax = fig.add_axes([0.0,0.0,1.0,1.0])
    fig.patch.set_alpha(0.0)
    ax.set_aspect('auto')

    print('Draw coastlines')
    m.drawcoastlines(color=coastline_color, linewidth=10.0)
#    m.drawcountries(color=coastline_color)
#    m.drawstates(color=coastline_color)
    #m.drawmapboundary(color=coastline_color,fill_color='#000000')
 #   parallels = arange(-80.,90,10.)
 #   meridians = arange(0.,360.,10.)
 #   m.drawparallels(parallels,labels=[0,0,0,0],color='lightgrey')
#    m.drawmeridians(meridians,labels=[0,0,0,0],fontsize=8,fmt='%3d',color='lightgrey')
    fig.patch.set_visible(False)
    ax.axis('off')

    title( title_txt, color='k')
    ax = gca()


width = 11000
height = width
xfname = 'boundaries.png'
#I = granule_sat_image(lat=None, lon=None, proj='goes', width=width, height=height, fill_color='#555555', cgeo='Idx')
img_p2g_coastline('Political Boundaries',width=width, height=height, dpi=72, coastline_color='#FF0000')
ofile = '../test_output' + '/' + xfname
savefig(ofile, dpi=72, transparent=True)
