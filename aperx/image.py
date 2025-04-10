import subprocess
from astropy.io import fits
from astropy.wcs import WCS
import sys, os, warnings
import numpy as np
from dataclasses import dataclass, field

# import warnings
# warnings.simplefilter('ignore')
import tqdm, time
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from photutils.aperture import SkyCircularAperture, aperture_photometry
from typing import Tuple
from astropy.convolution import convolve_fft

from .psf import PSF

@dataclass
class Image:
    sci_file: str
    err_file: str
    wht_file: str
    mask_file: str
    filter: str
    psf_file: str = None
    psfmatched_file: str = None

    _sci : np.ndarray = field(init=False, repr=False, default=None)
    _err : np.ndarray = field(init=False, repr=False, default=None)
    _wht : np.ndarray = field(init=False, repr=False, default=None)
    _mask : np.ndarray = field(init=False, repr=False, default=None)
    _hdr : np.ndarray = field(init=False, repr=False, default=None)
    _wcs : np.ndarray = field(init=False, repr=False, default=None)
    _pixel_scale : float = field(init=False, repr=False, default=None)
    _shape : Tuple[int,int] = field(init=False, repr=False, default=None)
    _psf : np.ndarray = field(init=False, repr=False, default=None)
    _psfmatched : np.ndarray = field(init=False, repr=False, default=None)

    @property 
    def base_file(self):
        return os.path.commonprefix([self.sci_file, self.err_file, self.wht_file])

    @property 
    def sci_exists(self):
        return os.path.exists(self.sci_file)

    @property 
    def err_exists(self):
        return os.path.exists(self.err_file) 

    @property
    def wht_exists(self):
        return os.path.exists(self.wht_file)

    @property
    def exists(self):
        return self.sci_exists and self.err_exists and self.wht_exists

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
                try:
                    sci_image = hdul['SCI'].data
                except:
                    sci_image = hdul[0].data
            self._sci = sci_image
        return self._sci
        
    @property
    def err(self):
        """
        Load the error image for the given band using memory mapping.
        """
        if self._err is None:
            with fits.open(self.err_file, memmap=True) as hdul:
                try:
                    err_image = hdul['ERR'].data
                except:
                    err_image = hdul[0].data
            self._err = err_image
        return self._err

    @property
    def wht(self):
        """
        Load the weight image for the given band using memory mapping.
        """
        if self._wht is None:
            with fits.open(self.wht_file, memmap=True) as hdul:
                try:
                    wht_image = hdul['WHT'].data
                except:
                    wht_image = hdul[0].data
            self._wht = wht_image
        return self._wht
    
    @property
    def mask(self):
        """
        Load the source mask for the given band using memory mapping.
        """
        if self._mask is None:
            with fits.open(self.mask_file, memmap=True) as hdul:
                mask_image = hdul['SRCMASK'].data
            self._mask = mask_image
        return self._mask

    @property
    def psf(self):
        """
        Load the PSF for the given band.
        """
        if not self.has_psf:
            raise FileNotFoundError(f"PSF file not found: {self.psf_file}")
        if self._psf is None:
            self._psf = PSF(self.psf_file)
        return self._psf

    @property
    def psfmatched(self):
        """
        Load the psfmatched science image for the given band using memory mapping.
        """
        if not self.has_psfmatched_equivalent:
            raise FileNotFoundError(f"PSF-matched file not found: {self.psfmatched_file}")
        if self._psfmatched is None:
            with fits.open(self.psfmatched_file, memmap=True) as hdul:
                psfmatched_image = hdul['SCI'].data
            self._psfmatched = psfmatched_image
        return self._psfmatched

    def close(self):
        """
        Force close all memmapped arrays
        """
        del self._sci, self._err, self._wht, self._psf, self._psfmatched
        self._sci, self._err, self._wht, self._psf, self._psfmatched = None, None, None, None, None

    def convert_byteorder(self):
        if self._sci is None: self.sci # call self.sci to load sci to self._sci
        self._sci = self._sci.astype(self._sci.dtype.newbyteorder('='))
        if self._err is None: self.err 
        self._err = self._err.astype(self._err.dtype.newbyteorder('='))
        if self._wht is None: self.wht
        self._wht = self._wht.astype(self._wht.dtype.newbyteorder('='))

        if self.has_psfmatched_equivalent:
            if self._psfmatched is None: self.psfmatched
            self._psfmatched = self._psfmatched.astype(self._psfmatched.dtype.newbyteorder('='))

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

 


    def generate_psfmatched_image(self, target_filter, target_psf_file, reg_fact=1e-4, overwrite=False):
        """
        Convolve the science image with the PSF and save the result to a new file.
        """
        if not self.has_psf:
            raise FileNotFoundError(f"PSF file not found: {self.psf_file}")

        if os.path.exists(self.psfmatched_file) and not overwrite:
            print('PSF matched file already exists, use overwrite=True to regenerate.')
            return

        print(f'Generating homogenization kernel {self.filter} -> {target_filter} for {self}')
        kernel_file = self.psf_file.replace('.fits', f'_kernel_to_{target_filter}.fits')
        kernel_log_file = kernel_file.replace('.fits', '.log')
        if not os.path.exists(kernel_file) or overwrite:
            cmd = ['pypher', self.psf_file, target_psf_file, kernel_file, '-r', str(reg_fact)] 
            subprocess.run(cmd, check=True)

        print(f'\t Convolving {self}...')
        start = time.time()

        kernel = fits.getdata(kernel_file)
        convolved = convolve_fft(self.sci, kernel, normalize_kernel=False, allow_huge=True)
        end = time.time()
        t = end-start
        print(f"\t Done in {int(np.floor(t/60))}m{int(t-60*int(np.floor(t/60)))}s")

        convolved[np.isnan(self.sci)|(self.wht==0)] = np.nan
        convolved = convolved.astype(self.sci.dtype) # convolution sets dtype to float64 

        print(f'\t Writing to {self.psfmatched_file}')
        hdu = fits.PrimaryHDU(convolved, header=self.hdr)
        hdul = fits.HDUList([hdu])
        hdul.writeto(self.psfmatched_file, overwrite=True)

        os.remove(kernel_file)
        os.remove(kernel_log_file)
        del hdu, hdul, convolved, kernel

    @property
    def area(self):
        """
        On-sky area covered this image, in square arcmin
        """
        n_valid_pixels = np.sum(np.isfinite(self.sci))
        return n_valid_pixels * self.pixel_scale**2 / 3600 # in sq. arcmin

    def _compute_unit_conv(self, output_unit):
        output_unit = u.Unit(output_unit)

        if not 'spectral flux density' in output_unit.physical_type:
            raise ValueError('Currently only fnu units are supported')

        current_unit = self.hdr['BUNIT']
        if current_unit == 'MJy/sr':
            conversion = 1e6*u.Jy/u.sr
            conversion *= ((self.pixel_scale * u.arcsec)**2).to(u.sr)
            conversion = conversion.to(output_unit).value
        else:
            raise ValueError(f"Couldn't parse units ({current_unit}) for {image}")

        return conversion

    def get_background_pixels(
        self, 
        psfmatched = False,
    ):

        if psfmatched:
            nsci = self.psfmatched * np.sqrt(self.wht)
        else:
            nsci = self.sci * np.sqrt(self.wht)
        
        nsci = nsci[(self.mask==0)&(np.isfinite(nsci))]
        return nsci


    def get_random_aperture_fluxes(
        self,
        aperture_diameter: float,
        num_apertures: int,
        psfmatched = False,
    ):
        
        if psfmatched: 
            nsci = self.psfmatched * np.sqrt(self.wht)
        else:
            nsci = self.sci * np.sqrt(self.wht)

        x, y = np.arange(np.shape(nsci)[1]), np.arange(np.shape(nsci)[0])
        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        mask = (self.mask == 0) & np.isfinite(nsci)
        mask = mask.flatten()
        x, y = x[mask], y[mask]
        i = np.random.randint(low=0,high=len(x),size=num_apertures)
        x, y = x[i], y[i]
        centroids = self.wcs.pixel_to_world(x, y)

        d = aperture_diameter*u.arcsec
        aperture = SkyCircularAperture(centroids, r=d/2)
        tbl = aperture_photometry(nsci, aperture, mask=~np.isfinite(nsci), wcs=self.wcs) # mask=self.mask>0
        flux = np.array(tbl['aperture_sum'],dtype=float)

        return flux




