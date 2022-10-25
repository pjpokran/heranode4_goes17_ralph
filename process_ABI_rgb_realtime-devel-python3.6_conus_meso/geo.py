import numpy as np
from pyproj import Proj

class GeostationaryProjection(object):
    """Convert between geostationary sensor Y/X view angles and earth locations
    
    View angles are given/returned in radians (as provided in GOES 16 L1b data),
    while latitudes and longitudes are in degrees. View angles are interpreted as
    positive-north, positive-east. Equatorial
    and polar earth radii and also satellite altitude can be optionally specified
    in km. If not given they default to the values used in the GOES 16 fixed grid.
    Sweep angle axis can also be specified as 'x' (like ABI) or 'y' (like SEVIRI);
    it defaults to 'x'.
    """

    default_r_eq = 6378.1370
    default_r_pol = 6356.75231414
    default_sat_alt = 35786.0230

    def __init__(self, sat_lon, sat_alt=default_sat_alt, sweep='x',
                 r_eq=default_r_eq, r_pol=default_r_pol):
        self.proj = Proj(proj='geos', lon_0=sat_lon, h=(sat_alt * 1e3), sweep=sweep,
                         a=(r_eq * 1e3), b=(r_pol * 1e3), no_defs=True)
        self.meters_to_radians = sat_alt * 1e3  # from inspection of proj.4 source

    def location_from_angles(self, y, x):
        y, x = np.broadcast_arrays(y, x)
        y = y * self.meters_to_radians
        x = x * self.meters_to_radians
        lon, lat = self.proj(x, y, inverse=True)
        bad = (lat == 1e30)
        lat[bad] = np.nan
        lon[bad] = np.nan
        return lat, lon

    def angles_from_location(self, lat, lon):
        x, y = self.proj(lon, lat)
        x = np.asarray(x)
        y = np.asarray(y)
        bad = (x == 1e30)
        x[bad] = np.nan
        y[bad] = np.nan
        return (y / self.meters_to_radians,
                x / self.meters_to_radians)


class AbiProjection(GeostationaryProjection):
    """Conversions for ABI imager index to/from earth location
    
    Subsatellite longitude must be provided to constructor (e.g., use -89.5 for
    early GOES 16 data). Default resolution is 2 km but any of 0.5, 1.0, 2.0 can
    be specified.
    """

    def __init__(self, subsat_lon, resolution=2.0):
        super(AbiProjection, self).__init__(subsat_lon)
        self.pixels_per_dim = int(round(self.pixels_per_dim_2km * 2 / resolution))
        self.angle_step = self.angle_step_2km * resolution / 2
        self.max_angle = (self.pixels_per_dim - 1) * self.angle_step / 2

    def location_from_index(self, line, element):
        y = self.angle_from_line(line)
        x = self.angle_from_element(element)
        return self.location_from_angles(y, x)

    def index_from_location(self, lat, lon):
        y, x = self.angles_from_location(lat, lon)
        line = self.line_from_angle(y)
        element = self.element_from_angle(x)
        return line, element

    def angle_from_line(self, line):
        return self.max_angle - line * self.angle_step

    def angle_from_element(self, element):
        return -self.max_angle + element * self.angle_step

    def line_from_angle(self, y):
        return (self.max_angle - y) / self.angle_step

    def element_from_angle(self, x):
        return (x + self.max_angle) / self.angle_step

    pixels_per_dim_2km = 5424
    angle_step_2km = 5.6e-5

