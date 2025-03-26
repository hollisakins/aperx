import subprocess
from astropy.io import fits
from astropy.wcs import WCS
import sys
import numpy as np


class Image:

    def __init__(self, band, field_name, pixel_scale, version, tile):
        base_dir = config.mosaic_dir
        mosaic_filename_format = #...
        # 'mosaic_nircam_[filter]_[field_name]_[pixel_scale]_[version]_[tile]_[ext]'

        base_filename = config.mosaic_filename_format + '.fits'
        base_filename = base_filename.replace('[filter]', band)
        base_filename = base_filename.replace('[field_name]', field_name)
        base_filename = base_filename.replace('[pixel_scale]', pixel_scale)
        base_filename = base_filename.replace('[version]', version)
        base_filename = base_filename.replace('[tile]', tile)

        sci_filename = base_filename.replace('[ext]', 'sci')
        err_filename = base_filename.replace('[ext]', 'err')
        wht_filename = base_filename.replace('[ext]', 'wht')
        
        sci_filepath = os.path.join(base_dir, sci_filename)
        err_filepath = os.path.join(base_dir, err_filename)
        wht_filepath = os.path.join(base_dir, wht_filename)

        psf_filename = base_filename.replace('[ext]', 'psf')
        if not os.path.exists(os.path.join(base_dir, 'psfs')):
            os.mkdir(os.path.join(base_dir, 'psfs'))
        psf_filepath = os.path.join(base_dir, 'psfs', psf_filename)

        # Load in the FITS images
        self.sci = sci
        self.wht = wht


    @property 
    def has_psf(self):
        return os.path.exists(self.psf_filepath)

