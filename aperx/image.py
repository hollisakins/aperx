import subprocess
from astropy.io import fits
from astropy.wcs import WCS
import sys, os, warnings
import numpy as np
from dataclasses import dataclass, field

# import warnings
# warnings.simplefilter('ignore')
import tqdm
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from photutils.aperture import SkyCircularAperture, aperture_photometry



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



    def generate_psfmatched_image(self, target_filter, target_psf_file, overwrite=False):
        """
        Convolve the science image with the PSF and save the result to a new file.
        """
        if not self.has_psf:
            raise FileNotFoundError(f"PSF file not found: {self.psf_file}")

        if os.path.exists(self.psfmatched_file) and not overwrite:
            print('PSF matched file already exists, use overwrite=True to regenerate.')
            return

        kernel_file = self.psf_file.replace('.fits', f'_kernel_to_{target_filter}.fits')
        kernel_log_file = kernel_file.replace('.fits', '.log')
        cmd = f"""
            pypher {self.psf_file} {target_psf_file} {kernel_file}
        """
        # TODO add some regularization? 
        subprocess.run(cmd, shell=True)

        kernel = fits.getdata(kernel_file)
        convolved = convolve_fft(self.sci, kernel, normalize_kernel=False)
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
        return np.prod(self.shape) * self.pixel_scale**2 / 3600 # in sq. arcmin



    def get_random_aperture_fluxes(
        aperture_diameters: np.ndarray,
        num_apertures_per_sq_arcmin: int,
        source_mask,
        psfmatched = False,
    ):
        
        num_apertures = int(num_apertures_per_sq_arcmin * self.area)

        num_diameters = len(aperture_diameters)
        flux = np.zeros((num_apertures,num_diameters,))

        if psfmatched: 
            nsci = self.psfmatched * np.sqrt(self.wht)
        else:
            nsci = self.sci * np.sqrt(self.wht)

        x, y = np.arange(np.shape(nsci)[1]), np.arange(np.shape(nsci)[0])
        x, y = np.meshgrid(x, y)
        x, y = x.flatten(), y.flatten()
        mask = ~source_mask & np.isfinite(nsci.flatten())
        x, y = x[mask], y[mask]
        i = np.random.randint(low=0,high=len(x),size=num_apertures)
        x, y = x[i], y[i]
        centroids = wcs.pixel_to_world(x, y)

        for i in range(num_diameters):
            d = aperture_diameters[i]*u.arcsec
            aperture = SkyCircularAperture(centroids, r=d/2)
            tbl = aperture_photometry(nsci, aperture, mask=mask, wcs=wcs)
            flux[:,i] = np.array(tbl['aperture_sum'],dtype=float)

        return flux




