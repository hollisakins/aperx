import subprocess
from astropy.io import fits
from astropy.wcs import WCS
import sys, os, warnings
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Image:
    sci_file: str
    err_file: str
    wht_file: str
    psf_file: str = None
    psfmatched_file: str = None

    _sci = field(init=False, repr=False, default=None)
    _err = field(init=False, repr=False, default=None)
    _wht = field(init=False, repr=False, default=None)
    _hdr = field(init=False, repr=False, default=None)
    _wcs = field(init=False, repr=False, default=None)
    _pixel_scale = field(init=False, repr=False, default=None)
    _shape = field(init=False, repr=False, default=None)
    _psf = field(init=False, repr=False, default=None)
    _psfmatched = field(init=False, repr=False, default=None)

    @property 
    def base_file(self):
        return os.path.commonprefix([self.sci_file, self.err_file, self.wht_file])

    @property
    def exists(self):
        sci_exists = os.path.exists(self.sci_file)
        err_exists = os.path.exists(self.err_file) or (self.err_file is None)
        wht_exists = os.path.exists(self.wht_file) or (self.wht_file is None)
        return sci_exists and err_exists and wht_exists

    @property 
    def has_psf(self):
        if self.psf_file is None:
            return False
        return os.path.exists(self.psf_file)

    @property 
    def has_psfmatched_equivalent(self):
        if self.psfmatched_file is None:
            return False
        return os.path.exists(self.psfmatched_file)

    @property
    def sci(self):
        """
        Load the science image for the given band using memory mapping.
        """
        if self._sci is None:
            with fits.open(self.sci_file, memmap=True) as hdul:
                sci_image = hdul['SCI'].data
            self._sci = sci_image
        return self._sci
        
    @property
    def err(self):
        """
        Load the error image for the given band using memory mapping.
        """
        if self._err is None:
            with fits.open(self.err_file, memmap=True) as hdul:
                err_image = hdul['ERR'].data
            self._err = err_image
        return self._err

    @property
    def wht(self):
        """
        Load the weight image for the given band using memory mapping.
        """
        if self._wht is None:
            with fits.open(self.wht_file, memmap=True) as hdul:
                wht_image = hdul['WHT'].data
            self._wht = wht_image
        return self._wht

    def close(self):
        """
        Force close all memmapped arrays
        """
        del self._sci, self._err, self._wht, self._psf, self._psfmatched
        self._sci, self._err, self._wht, self._psf, self._psfmatched = None, None, None, None, None

    @property
    def hdr(self):
        if self._hdr is None:
            self._hdr = fits.getheader(self.sci_file)
        return self._hdr

    @property
    def wcs(self):
        if self._wcs is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self._wcs = WCS(self.hdr)
        return self._wcs

    @property
    def pixel_scale(self):
        if self._pixel_scale is None:
            self._pixel_scale = self.wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value
        return self._pixel_scale

    @property
    def shape(self):
        if self._shape is None:
            shape = (int(self.hdr['NAXIS2']), int(self.hdr['NAXIS1']))
            self._shape = shape
        return self._shape

    def __repr__(self):
        return f"Image({os.path.basename(self.base_file)})"
