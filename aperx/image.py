import subprocess
from astropy.io import fits
from astropy.wcs import WCS
import sys, os, warnings
import numpy as np
from dataclasses import dataclass, field
from contextlib import redirect_stdout

# import warnings
# warnings.simplefilter('ignore')
import tqdm, time
import astropy.units as u
from astropy.constants import c as speed_of_light
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from photutils.aperture import SkyCircularAperture, aperture_photometry
from typing import Tuple
from astropy.convolution import convolve_fft

from .psf import PSF
import glob
import re

from rich.table import Table as RichTable
from rich.console import Console
console = Console()
console._log_render.omit_repeated_times = False


@dataclass
class Image:
    filter: str
    tile: str
    sci_file: str
    err_file: str
    wht_file: str
    hom_file: str = None
    psf_file: str = None
    seg_file: str = None
    # mask_file: str
    memmap: bool = False

    sci_extension: str = 'sci'
    err_extension: str = 'err'
    wht_extension: str = 'wht'
    psf_extension: str = 'psf'


    _sci : np.ndarray = field(repr=False, default=None)
    _err : np.ndarray = field(repr=False, default=None)
    _wht : np.ndarray = field(repr=False, default=None)
    _psf : np.ndarray = field(repr=False, default=None)
    _hom : np.ndarray = field(repr=False, default=None)
    _seg : np.ndarray = field(repr=False, default=None)
    
    _hdr : np.ndarray = field(repr=False, default=None)
    _wcs : np.ndarray = field(repr=False, default=None)
    _pixel_scale : float = field(repr=False, default=None)
    _shape : Tuple[int,int] = field(repr=False, default=None)


    @property 
    def base_file(self):
        if self.wht_file is not None:
            return os.path.commonprefix([self.sci_file, self.err_file, self.wht_file])
        else:
            return os.path.commonprefix([self.sci_file, self.err_file])

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
    def psf_exists(self):
        if self.psf_file is None:
            return False
        return os.path.exists(self.psf_file)
    
    @property
    def hom_exists(self):
        if self.hom_file is None:
            return False
        return os.path.exists(self.hom_file)
    
    @property
    def seg_exists(self):
        if self.seg_file is None:
            return False
        return os.path.exists(self.seg_file)

    @property
    def exists(self):
        return self.sci_exists and self.err_exists and self.wht_exists

    @property
    def sci(self):
        """
        Load the science image for the given band using memory mapping.
        """
        if self._sci is None:
            with fits.open(self.sci_file, memmap=self.memmap) as hdul:
                try:
                    sci_image = hdul['SCI'].data
                except:
                    sci_image = hdul[0].data
            self._sci = sci_image
        return self._sci.astype(self._sci.dtype.newbyteorder('='))
        
    @property
    def err(self):
        """
        Load the error image for the given band using memory mapping.
        """
        if self._err is None:
            with fits.open(self.err_file, memmap=self.memmap) as hdul:
                try:
                    err_image = hdul['ERR'].data
                except:
                    err_image = hdul[0].data
            self._err = err_image
        return self._err.astype(self._err.dtype.newbyteorder('='))

    @property
    def wht(self):
        """
        Load the weight image for the given band using memory mapping.
        """
        if self._wht is None:
            with fits.open(self.wht_file, memmap=self.memmap) as hdul:
                try:
                    wht_image = hdul['WHT'].data
                except:
                    wht_image = hdul[0].data
            self._wht = wht_image
        return self._wht.astype(self._wht.dtype.newbyteorder('='))
    
    @property
    def seg(self):
        """
        Load the source mask for the given band using memory mapping.
        """
        if self._seg is None:
            with fits.open(self.seg_file, memmap=self.memmap) as hdul:
                seg_image = hdul[0].data
            self._seg = seg_image
        return self._seg.astype(self._seg.dtype.newbyteorder('='))

    @property
    def psf(self):
        """
        Load the PSF for the given band.
        """
        if not self.psf_exists:
            raise FileNotFoundError(f"PSF file not found: {self.psf_file}")
        if self._psf is None:
            self._psf = PSF(self.psf_file)
        return self._psf

    @property
    def hom(self):
        """
        Load the PSF-homogenized science image for the given band using memory mapping.
        """
        if not self.hom_exists:
            raise FileNotFoundError(f"PSF-homogenized file not found: {self.hom_file}")
        if self._hom is None:
            with fits.open(self.hom_file, memmap=self.memmap) as hdul:
                try:
                    hom_image = hdul['SCI'].data
                except:
                    hom_image = hdul[0].data
            self._hom = hom_image
        return self._hom.astype(self._hom.dtype.newbyteorder('='))

    def close(self):
        """
        Force close all memmapped arrays
        """
        del self._sci, self._err, self._wht, self._psf, self._hom, self._seg
        self._sci, self._err, self._wht, self._psf, self._hom, self._seg = None, None, None, None, None, None
    
    def __setattr__(self, key, value):
        if key in ['sci', 'err', 'wht', 'psf', 'hom', 'seg']:
            super().__setattr__('_'+key, value)
        else:
            super().__setattr__(key, value)


    def convert_byteorder(self):
        if self._sci is None: self.sci # call self.sci to load sci to self._sci
        self._sci = self._sci.astype(self._sci.dtype.newbyteorder('='))
        if self.err_exists:
            if self._err is None: self.err 
            self._err = self._err.astype(self._err.dtype.newbyteorder('='))
        if self.wht_exists:
            if self._wht is None: self.wht
            self._wht = self._wht.astype(self._wht.dtype.newbyteorder('='))
        if self.hom_exists:
            if self._hom is None: self.hom
            self._hom = self._hom.astype(self._hom.dtype.newbyteorder('='))
        if self.seg_exists:
            if self._seg is None: self.seg
            self._seg = self._seg.astype(self._seg.dtype.newbyteorder('='))

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

    @property
    def area(self):
        """
        On-sky area covered this image, in square arcmin
        """
        n_valid_pixels = np.sum(np.isfinite(self.sci))
        return n_valid_pixels * self.pixel_scale**2 / 3600 # in sq. arcmin

    def __repr__(self):
        return f"Image({self.filter}, {self.tile})"

    def generate_psfhomogenized_image(
        self, 
        target_filter: str, 
        target_psf_file: str, 
        output_file: str, 
        reg_fact: float = 1e-4, 
        overwrite: bool = False,
        logger = None,
    ):
        """
        Convolve the science image with the PSF and save the result to a new file.
        """

        if not self.psf_exists:
            raise FileNotFoundError(f"PSF file not found: {self.psf_file}")

        if os.path.exists(output_file) and not overwrite:
            logger.info(f'{self}: psf-homogenized file already exists, skipping generation')
            return

        logger.info(f'Generating homogenization kernel {self.filter} -> {target_filter} for {self}')
        kernel_file = self.psf_file.replace('.fits', f'_kernel_to_{target_filter}.fits')
        kernel_log_file = kernel_file.replace('.fits', '.log')
        if not os.path.exists(kernel_file) or overwrite:
            cmd = ['pypher', self.psf_file, target_psf_file, kernel_file, '-r', str(reg_fact)] 
            with redirect_stdout(None):
                subprocess.run(cmd, check=True)

        logger.info(f'\t Convolving {self}...')
        start = time.time()

        kernel = fits.getdata(kernel_file)
        convolved = convolve_fft(self.sci, kernel, normalize_kernel=False, allow_huge=True)
        end = time.time()
        t = end-start
        logger.info(f"\t Done in {int(np.floor(t/60))}m{int(t-60*int(np.floor(t/60)))}s")

        convolved[np.isnan(self.sci)|(self.wht==0)] = np.nan
        convolved = convolved.astype(self.sci.dtype) # convolution sets dtype to float64 

        logger.info(f'\t Writing to {output_file}')
        hdu = fits.PrimaryHDU(convolved, header=self.hdr)
        hdul = fits.HDUList([hdu])
        hdul.writeto(output_file, overwrite=True)

        os.remove(kernel_file)
        os.remove(kernel_log_file)
        del hdu, hdul, convolved, kernel

    def _compute_unit_conv(self, output_unit):
        output_unit = u.Unit(output_unit)

        if not 'spectral flux density' in output_unit.physical_type:
            raise ValueError('Currently only fnu units are supported')

        current_unit = self.hdr['BUNIT']
        if current_unit == 'MJy/sr':
            conversion = 1e6*u.Jy/u.sr
            conversion *= ((self.pixel_scale * u.arcsec)**2).to(u.sr)
            conversion = conversion.to(output_unit).value
        elif current_unit == 'ELECTRONS/S':
            photflam = self.hdr['PHOTFLAM'] * u.erg/u.cm**2/u.angstrom
            photplam = self.hdr['PHOTPLAM'] * u.angstrom
            conversion = ((1/u.s) * photflam * photplam**2 / speed_of_light).to(output_unit).value
        else:
            raise ValueError(f"Couldn't parse units ({current_unit}) for {self}")

        return conversion

    # def get_random_aperture_fluxes(
    #     self,
    #     aperture_diameter: float,
    #     num_apertures: int,
    #     psfmatched = False,
    # ):
        
    #     if psfmatched: 
    #         nsci = self.psfmatched * np.sqrt(self.wht)
    #     else:
    #         nsci = self.sci * np.sqrt(self.wht)

    #     x, y = np.arange(np.shape(nsci)[1]), np.arange(np.shape(nsci)[0])
    #     x, y = np.meshgrid(x, y)
    #     x, y = x.flatten(), y.flatten()
    #     mask = (self.mask == 0) & np.isfinite(nsci)
    #     mask = mask.flatten()
    #     x, y = x[mask], y[mask]
    #     i = np.random.randint(low=0,high=len(x),size=num_apertures)
    #     x, y = x[i], y[i]
    #     centroids = self.wcs.pixel_to_world(x, y)

    #     d = aperture_diameter*u.arcsec
    #     aperture = SkyCircularAperture(centroids, r=d/2)
    #     tbl = aperture_photometry(nsci, aperture, mask=~np.isfinite(nsci), wcs=self.wcs) # mask=self.mask>0
    #     flux = np.array(tbl['aperture_sum'],dtype=float)

    #     return flux



class Images:
    def __init__(self, images: list[Image], memmap=False):
        self.images = images
        self.memmap = memmap

    @classmethod
    def load(cls, 
             image_patterns: list[str], 
             filter: str | list[str] = '*',
             tile: str | list[str] = '*', 
             version: str = None, 
             pixel_scale: str = None, 
             sci_extension: str = 'sci',
             err_extension: str = 'err',
             wht_extension: str = 'wht',
             psf_extension: str = 'psf',
             psf_tile: str = 'master',
             memmap: bool = False,
        ):
        """
        Loads images from the given file patterns and returns an Images object.

        File patterns can contain wildcards in brackets, e.g. [filter], [pixelscale], [version], [tile].
        By default, these will be treated as proper wildcards, and all images matching the pattern will be loaded.
        If a specific value (or values) of a wildcard is given as a keyword argument, only images matching that 
        value will be loaded.
        Note that the [ext] keyword must be present, and is not treated as a wildcard, as the same image but with
        different extensions are loaded as the same Image object.

        Args:
            image_patterns (list[str]): List of file patterns to load images from.
            filter (str or list[str]): Filter(s) to match against the image patterns.
            pixel_scale (str or list[str]): Pixel scale(s) to match against the image patterns.
            version (str or list[str]): Version(s) to match against the image patterns.
            tile (str or list[str]): Tile(s) to match against the image patterns.

        Example:

            ```
            Images.load(
                image_patterns = ['~/mosaics/[filter]/extensions/mosaic_nircam_[filter]_cosmos_[pixelscale]_[version]_[tile]_[ext].fits'],
                filter = ['f115w', 'f150w'],
                pixel_scale = '30mas',
                version = 'v1.0',
                tile = '*', 
            )
            ```
        """
        if isinstance(filter, str):
            filter = [filter]
        if isinstance(pixel_scale, str):
            pixel_scale = [pixel_scale]
        if isinstance(version, str):
            version = [version]
        if isinstance(tile, str):
            tile = [tile]

        image_dict = {}
        for image_pattern in image_patterns:
            print('====================')
            print(f'Loading images from {image_pattern}')
            print('====================')

            if '[version]' in image_pattern and version is None:
                raise ValueError('[version] wildcard detected in image pattern but not specified in kwargs')
            if '[pixelscale]' in image_pattern and pixel_scale is None:
                raise ValueError('[pixelscale] wildcard detected in image pattern but not specified in kwargs')

            image_pattern_regex = image_pattern.replace('[filter]',r'([^_]+)')
            image_pattern_regex = image_pattern_regex.replace('[pixelscale]',r'([^_]+)')
            image_pattern_regex = image_pattern_regex.replace('[version]',r'([^_]+)')
            image_pattern_regex = image_pattern_regex.replace('[tile]',r'([^_]+)')
            image_pattern_regex = image_pattern_regex.replace('[ext]','(.+)')
            image_pattern_regex = fr'{image_pattern_regex}'
            print(image_pattern_regex)
            match = re.search(image_pattern_regex, image_pattern)
            if match is None:
                raise ValueError(f"Could not parse image pattern {image_pattern}")
            else:
                pattern_groups = list(match.groups())
                # print(pattern_groups)

            # Replace bracketed wildcards with glob wildcards
            glob_pattern = image_pattern.replace('[filter]', '*')
            glob_pattern = glob_pattern.replace('[pixelscale]', '*')
            glob_pattern = glob_pattern.replace('[version]', '*')
            glob_pattern = glob_pattern.replace('[tile]', '*')
            glob_pattern = glob_pattern.replace('[ext]', sci_extension)

            # Find all files matching the pattern
            image_files = glob.glob(glob_pattern)
            for image_file in image_files:
                image_file_regex = image_pattern.replace('[filter]',r'([^_]+)')
                image_file_regex = image_file_regex.replace('[pixelscale]',r'([^_]+)')
                image_file_regex = image_file_regex.replace('[version]',r'([^_]+)')
                image_file_regex = image_file_regex.replace('[tile]',r'([^_]+)')
                image_file_regex = image_file_regex.replace('[ext]',f'({sci_extension})')
                image_file_regex = fr'{image_file_regex}'

                match = re.search(image_file_regex, image_file)
                if match is None:
                    raise ValueError(f"Could not parse image file {image_file}")
                else:
                    image_file_groups = list(match.groups())

                filter_i = image_file_groups[pattern_groups.index('[filter]')]
                tile_i = image_file_groups[pattern_groups.index('[tile]')]
                ext_i = image_file_groups[pattern_groups.index('[ext]')]

                if '[pixelscale]' in pattern_groups:
                    pixel_scale_i = image_file_groups[pattern_groups.index('[pixelscale]')]
                else:
                    pixel_scale_i = ''
                if '[version]' in pattern_groups:
                    version_i = image_file_groups[pattern_groups.index('[version]')]
                else:
                    version_i = ''

                if filter != ['*'] and filter_i not in filter:
                    continue
                if tile != ['*'] and tile_i not in tile:
                    continue
                # if pixel_scale_i not in pixel_scale:
                    # continue
                # if version != ['*'] and version_i not in version:
                    # continue
                
                # print(f'+ {image_file}')
                # print(f'  | detected filter={filter_i}, pixel_scale={pixel_scale_i}, version={version_i}, tile={tile_i}, ext={ext_i}')

                if tile_i not in image_dict:
                    image_dict[tile_i] = {}
                if filter_i not in image_dict[tile_i]:
                    image_dict[tile_i][filter_i] = {}
                
                file = image_pattern.replace('[filter]', filter_i)
                file = file.replace('[pixelscale]', pixel_scale_i)
                file = file.replace('[version]', version_i)
                file = file.replace('[tile]', tile_i)
            
                image_dict[tile_i][filter_i][sci_extension] = image_file
                image_dict[tile_i][filter_i][err_extension] = file.replace('[ext]', err_extension)
                image_dict[tile_i][filter_i][wht_extension] = file.replace('[ext]', wht_extension)
                # image_dict[tile_i][filter_i][hom_extension] = file.replace('[ext]', hom_extension)
                
                file = image_pattern.replace('[filter]', filter_i)
                file = file.replace('[pixelscale]', pixel_scale_i)
                file = file.replace('[version]', version_i)
                file = file.replace('[tile]', psf_tile)
                file = file.replace('[tile]', tile_i)
                image_dict[tile_i][filter_i][psf_extension] = file.replace('[ext]', psf_extension)
                

            images = []
            for tile in image_dict:
                for filt in image_dict[tile]:
                    sci_file = image_dict[tile][filt][sci_extension]
                    err_file = image_dict[tile][filt][err_extension]
                    wht_file = image_dict[tile][filt][wht_extension]
                    psf_file = image_dict[tile][filt][psf_extension]
                    # hom_file = image_dict[tile][filt][hom_extension]
                    
                    images.append(
                        Image(
                            tile=tile, 
                            filter=filt, 
                            sci_file = sci_file, 
                            err_file = err_file, 
                            wht_file = wht_file, 
                            psf_file=psf_file, 
                            memmap=memmap,
                            sci_extension = sci_extension, 
                            err_extension = err_extension, 
                            wht_extension = wht_extension, 
                            psf_extension = psf_extension, 

                        )
                    )
        
        return cls(images, memmap=memmap)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def __repr__(self):
        return f"Images({[image.__repr__() for image in self.images]})"
    
    def __str__(self):
        return f"Images({[image.__repr__() for image in self.images]})"
    
    def __add__(self, other):
        if isinstance(other, Image):
            return Images(self.images + [other])
        elif isinstance(other, Images):
            return Images(self.images + other.images)
        else:
            raise ValueError

    def __radd__(self, other):
        return Images(self.images + other.images)

    def append(self, addition: Image):
        self.images.append(addition)

    def __iadd__(self, other):
        if isinstance(other, Image):
            self.images += [other]
        elif isinstance(other, Images):
            self.images += other.images
        else:
            raise ValueError
        return self

    def __contains__(self, item):
        return item in self.images

    def __iter__(self):
        return iter(self.images)

    def get(self, tile=None, filter=None):
        if tile is None and filter is None:
            raise ValueError('Why?')
        
        elif tile is None:
            if isinstance(filter, str):
                filter = [filter]
            images = []
            for f in filter:
                images += [image for image in self.images if image.filter == f]

        elif filter is None:
            if isinstance(tile, str):
                tile = [tile]
            images = []
            for t in tile:
                images += [image for image in self.images if image.tile == t]
        
        else:
            if isinstance(tile, str):
                tile = [tile]
            if isinstance(filter, str):
                filter = [filter]
            images = []
            for t in tile:
                for f in filter:
                    images += [image for image in self.images if image.tile == t and image.filter == f]
                
        if len(images)==0:
            return None
        # elif len(images)==1:
            # return images[0]
        else:
            return Images(images)

    def pprint(self, filenames=False):
        for tile in self.tiles:
            table = RichTable()
            table.add_column(tile)

            images = self.get(tile=tile)
            images.sort('filter')
            for im in images:
                table.add_column(im.filter, justify="center", style="cyan", no_wrap=True)
            
            def label(file):
                l = ''
                if os.path.exists(file):
                    l += ':white_check_mark:'
                else:
                    l += ':x:'
                if filenames:
                    l += ' ' + os.path.basename(file)
                return l

            table.add_row('SCI', *[label(im.sci_file) for im in images])
            table.add_row('ERR', *[label(im.err_file) for im in images])
            table.add_row('WHT', *[label(im.wht_file) for im in images])
            table.add_row('PSF', *[label(im.psf_file) for im in images])
            # table.add_row('HOM', *[label(im.hom_file) for im in images])

            console.print(table)
    
    @property
    def tiles(self):
        return list(set([image.tile for image in self.images]))

    @property
    def filters(self):
        return list(set([image.filter for image in self.images]))

    @property
    def all_exist(self):
        all_exist = True
        for image in self.images:
            all_exist = all_exist and image.exists
        return all_exist

    def sort(self, key):
        if key=='filter':
            self.images.sort(key=lambda im: im.filter)
        elif key=='tile':
            self.images.sort(key=lambda im: im.tile)