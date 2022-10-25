# draw timestamp on each .JPG in this folder
import os,time
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS
from datetime import timedelta, datetime
from argparse import ArgumentParser
import subprocess
import logging

def get_exif(fn):
    '''returns all EXIF data as dictionary'''
    ret = {}
    i = Image.open(fn)
    info = i._getexif()
    try:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            ret[decoded] = value
    except:
        pass
    return ret


def get_datetime(fn):
    tmp = fn.split('_')
    s_time = '_'.join((tmp[2],tmp[3]))
    return datetime.strptime(s_time, '%Y%m%d_%H%M')


def add_timestrip(ifname=None, date_obj=None,text_str=[]):
    log = logging.getLogger(__name__)
    date_str = ' '.join([date_obj.strftime('%Y-%m-%d'), date_obj.strftime('%H:%M')])
    
    if text_str:
        astr = ', '.join(text_str)
    else:
        astr = 'unknown'

    astr = ', '.join([astr, date_str])
    log.debug(astr)

    try:
        im = Image.open(ifname)
        imgwidth = im.size[0]    # get width of first image
        imgheight = im.size[1]
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception, e:
        log.error('Failed to open file', exc_info=True)

    cmd = ['fc-match', '-f', '"%{file}"', 'FreeMono']
    cmd = ' '.join(cmd)
    log.debug(cmd)
    #cmd = 'fc-match -f "%{file}\n" FreeMono'
    fontPath = subprocess.check_output(cmd, shell=True)
    log.debug(fontPath)


    #fontPath = "C:\\WINDOWS\\Fonts\\LUCON.TTF"  # font file path
    myfont = ImageFont.truetype ( fontPath, 20 ) # load font and size
    (textw, texth) = myfont.getsize(astr)   # get size of timestamp
    

    x = imgwidth/2 - textw/2
    y = 2 

    draw = ImageDraw.Draw ( im )
    # thin border
    
    background = 'black'
    x0 = 0
    y0 = 0
    x1 = imgwidth
    y1 = texth + y + 2

    draw.rectangle([x0, y0, x1, y1], fill=background)
    # draw.text((x-1, y-1), date_str, font=myfont, fill=background)
    # draw.text((x+1, y-1), date_str, font=myfont, fill=background)
    # draw.text((x-1, y+1), date_str, font=myfont, fill=background)
    # draw.text((x+1, y+1), date_str, font=myfont, fill=background)
    # # text
    draw.text ( (x, y), astr, font=myfont, fill="white" )
    im.save (ifname) # save in new dir


def add_timestamp(ifname=None, date_obj=None):
    log = logging.getLogger(__name__)

    try:
        im = Image.open(ifname)
        imgwidth = im.size[0]    # get width of first image
        imgheight = im.size[1]
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception, e:
        log.error('Failed to open file', exc_info=True)

    cmd = ['fc-match', '-f', '"%{file}"', 'FreeMono']
    cmd = ' '.join(cmd)
    log.debug(cmd)
    #cmd = 'fc-match -f "%{file}\n" FreeMono'
    #TODO check that the font exists
    fontPath = subprocess.check_output(cmd, shell=True)
    log.debug(fontPath)


    #fontPath = "C:\\WINDOWS\\Fonts\\LUCON.TTF"  # font file path
    myfont = ImageFont.truetype ( fontPath, 18 ) # load font and size
    (textw, texth) = myfont.getsize('0000-00-00 00:00')   # get size of timestamp
    offset = -10
    x = imgwidth - textw + offset   # position of text
    y = imgheight - texth + offset

    draw = ImageDraw.Draw ( im )
    # thin border
    date_str = ' '.join([date_obj.strftime('%Y-%m-%d'), date_obj.strftime('%H:%M')])
    background = 'black'
    x0 = imgwidth - textw + offset - 2
    y0 = imgheight - texth + offset - 2
    x1 = imgwidth + offset + 2
    y1 = imgheight + offset + 2

    draw.rectangle([x0, y0, x1, y1], fill=background)
    # draw.text((x-1, y-1), date_str, font=myfont, fill=background)
    # draw.text((x+1, y-1), date_str, font=myfont, fill=background)
    # draw.text((x-1, y+1), date_str, font=myfont, fill=background)
    # draw.text((x+1, y+1), date_str, font=myfont, fill=background)
    # # text
    date_str = ' '.join([date_obj.strftime('%Y-%m-%d'), date_obj.strftime('%H:%M')])
    draw.text ( (x, y), date_str, font=myfont, fill="white" )
    im.save (ifname, quality=95 ) # save in new dir

def timestamp_ahi(ifname=None,text_str=[]):
    idir, fn = os.path.split(ifname)

    date_obj = get_datetime(fn)
    #add_timestamp(ifname=ifname, date_obj=date_obj)
    add_timestrip(ifname=ifname, date_obj=date_obj,text_str=text_str)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) 
    log = logging.getLogger(__name__)
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--ahi-fname', nargs='+')
    parser.add_argument('-t', '--text')
    args = parser.parse_args()

    log.debug(args.ahi_fname)
    text_str = ['Himawari 8','UW-Madison SSEC CIMSS',args.text]
    timestamp_ahi(args.ahi_fname[0],text_str=text_str)
