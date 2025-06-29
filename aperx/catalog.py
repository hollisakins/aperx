from .image import Image, Images
from . import utils

from collections import defaultdict
from typing import List
import os, warnings, tqdm, sys
import pickle
from copy import copy

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from astropy.wcs import WCS
from astropy.table import Table, vstack
from photutils.aperture import aperture_photometry, CircularAperture, ApertureStats
from astropy.coordinates import SkyCoord

from multiprocessing import Pool
from functools import partial

import sep
sep.set_extract_pixstack(int(1e7))
sep.set_sub_object_limit(4096)

def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5*(x-mu)**2/sigma**2)


def generate_psfs(
    images: Images,
    filters: list,
    fwhm_min: dict,
    fwhm_max: dict,
    max_ellip: float = 0.1,
    min_snr: float = 8.0,
    max_snr: float = 1000.0,
    psf_size: int = 301,
    checkplots: bool = False,
    overwrite: bool = False,
    plot: bool = False,
    az_average: bool = False,
    logger = None
):
    """
    Generate PSFs for all filters across all tiles.
    
    Args:
        images: Images collection containing all mosaic images
        filters: List of filter names to process
        fwhm_min: Dict mapping filter names to minimum FWHM scales
        fwhm_max: Dict mapping filter names to maximum FWHM scales
        max_ellip: Maximum ellipticity for PSF generation
        min_snr: Minimum SNR for PSF generation
        max_snr: Maximum SNR for PSF generation
        psf_size: Size of PSF image, should be odd
        checkplots: Whether to generate checkplots during PSF generation
        overwrite: Whether to overwrite existing PSF files
        plot: Plot the final PSFs for inspection
        az_average: Whether to azimuthally average PSFs
        logger: Logger instance for output
    """
    if logger is None:
        logger = utils.setup_logger(name='aperx.psf_generation')
    
    for filt in filters:
        filter_images = images.get(filter=filt)
        psf_files = [image.psf_file for image in filter_images]

        if len(filter_images) == 1:
            master_psf_file = None
            if filter_images[0].psf_exists and not overwrite:
                logger.info(f'All PSF files for {filt} already exist, skipping generation')
                continue

        elif len(set(psf_files)) == 1:
            # Build master PSF
            master_psf_file = psf_files[0]
            if os.path.exists(master_psf_file) and not overwrite:
                logger.info(f'Master PSF for {filt} already exists, skipping generation')
                continue
        else:
            master_psf_file=None
            all_exist = True
            for psf_file in psf_files:
                all_exist = all_exist and os.path.exists(psf_file)
            if all_exist and not overwrite:
                logger.info(f'All PSF files for {filt} already exist, skipping generation')
                continue

        # Build PSFs
        from .psf import PSF
        psf = PSF.build(
            images=filter_images,
            fwhm_min=fwhm_min[filt],
            fwhm_max=fwhm_max[filt],
            max_ellip=max_ellip, 
            min_snr=min_snr,
            max_snr=max_snr,
            checkplots=checkplots,
            psf_size=psf_size,
            overwrite=overwrite,
            master_psf_file=master_psf_file,
            az_average=az_average,
            logger = logger,
        )
        
        if plot:
            if len(filter_images)==1:
                psf.plot(save=filter_images[0].psf_file.replace('.fits','.png'))
            elif master_psf_file:
                psf.plot(save=master_psf_file.replace('.fits','.png'))
                for image in filter_images:
                    try:
                        psf = PSF(image.base_file + 'psf.fits', logger=logger)
                        psf.plot(save=image.base_file + 'psf.png')
                    except FileNotFoundError:
                        pass


def distance_to_nearest_edge(x, y, image_width, image_height):
    """
    Compute the distance to the nearest edge for a point (x, y) in an image.

    Parameters:
    x (float): X-coordinate of the point.
    y (float): Y-coordinate of the point.
    image_width (int): Width of the image.
    image_height (int): Height of the image.

    Returns:
    float: Distance to the nearest edge.
    """
    left_edge_dist = x
    right_edge_dist = image_width - x
    top_edge_dist = y
    bottom_edge_dist = image_height - y

    if np.ndim(x)==1:
        return np.min([left_edge_dist, right_edge_dist, top_edge_dist, bottom_edge_dist], axis=0)
    else:
        return min(left_edge_dist, right_edge_dist, top_edge_dist, bottom_edge_dist)


def merge_catalogs(
    catalogs: dict,
    matching_radius: float = 0.1,
    edge_mask: int = 300,
    logger = None
):
    """
    Merge multiple per-tile catalogs into a single catalog.
    
    Args:
        catalogs: Dictionary mapping tile names to Catalog objects
        matching_radius: Radius in arcsec for matching sources in overlap regions
        edge_mask: Distance in pixels from tile edge to mask spurious sources
        logger: Logger instance for output
        
    Returns:
        Catalog: New catalog with merged results
    """
    if logger is None:
        logger = utils.setup_logger(name='aperx.merge_catalogs')
    
    tile_names = list(catalogs.keys())
    
    if len(tile_names) == 0:
        logger.warning('No catalogs to merge')
        return None
        
    if len(tile_names) == 1:
        logger.info('Only one catalog, skipping merge')
        tile = tile_names[0]
        catalog = catalogs[tile]
        # Create merged catalog with the single tile's data
        merged_catalog = Catalog(catalog.images, flux_unit=catalog.flux_unit)
        merged_catalog.objects = catalog.objects[tile]
        merged_catalog.detection_images = catalog.detection_images
        return merged_catalog

    logger.info(f'Merging {len(tile_names)} tile catalogs')
    
    # Start with first catalog
    tile0 = tile_names[0]
    catalog0 = catalogs[tile0]
    objs0 = catalog0.objects.copy()
    objs0['tile'] = tile0
    
    # Remove objects too close to tile edges
    detec0 = catalog0.detection_image
    height0, width0 = detec0.shape
    objs0_distance_to_edge = distance_to_nearest_edge(objs0['x'], objs0['y'], width0, height0)
    objs0 = objs0[objs0_distance_to_edge > edge_mask]
    
    logger.info(f'{tile0}: {len(objs0)} objects')
    coords0 = SkyCoord(objs0['ra'], objs0['dec'], unit='deg')

    # Merge with remaining catalogs
    for tile1 in tile_names[1:]:
        catalog1 = catalogs[tile1]
        objs1 = catalog1.objects.copy()
        objs1['tile'] = tile1

        # Remove objects too close to tile edges
        detec1 = catalog1.detection_image
        height1, width1 = detec1.shape
        objs1_distance_to_edge = distance_to_nearest_edge(objs1['x'], objs1['y'], width1, height1)
        objs1 = objs1[objs1_distance_to_edge > edge_mask]

        logger.info(f'{tile1}: {len(objs1)} objects')
        coords1 = SkyCoord(objs1['ra'], objs1['dec'], unit='deg')

        # Perform source matching
        idx, d2d, d3d = coords0.match_to_catalog_sky(coords1)
        match = d2d < matching_radius*u.arcsec
        
        objs0_unique = objs0[~match]
        objs0_matched = objs0[match]
        objs1_matched = objs1[idx[match]]
        unique = np.ones(len(objs1), dtype=bool)
        unique[idx[match]] = False
        objs1_unique = objs1[unique]

        # Handle matched sources - keep the one farther from tile edge
        if len(objs0_matched) > 0:
            objs0_distance_to_edge = distance_to_nearest_edge(objs0_matched['x'], objs0_matched['y'], width0, height0)
            objs1_distance_to_edge = distance_to_nearest_edge(objs1_matched['x'], objs1_matched['y'], width1, height1)
            which = np.argmax([objs0_distance_to_edge, objs1_distance_to_edge], axis=0)
            
            # Combine all sources
            objs_merged = vstack([objs0_unique, objs0_matched[which==0], objs1_matched[which==1], objs1_unique], join_type='outer')
        else:
            # No matches, just combine unique sources
            objs_merged = vstack([objs0_unique, objs1_unique], join_type='outer')

        # Update for next iteration
        tile0 = tile1
        objs0 = objs_merged
        coords0 = SkyCoord(objs0['ra'], objs0['dec'], unit='deg')
        height0, width0 = height1, width1

    logger.info(f'Final merged catalog: {len(objs_merged)} objects')
    
    # Reassign IDs
    objs_merged['id'] = np.arange(len(objs_merged)).astype(int)
    
    # Create new merged catalog
    # Combine all images from all catalogs
    all_images = Images([])
    all_detection_images = Images([])

    for catalog in catalogs.values():
        all_images += catalog.images
        all_detection_images += catalog.detection_image
    
    merged_catalog = Catalog(all_images, flux_unit=catalog0.flux_unit)
    merged_catalog.detection_images = all_detection_images
    merged_catalog.objects = objs_merged
    merged_catalog.psfhom_target_filter = catalog.psfhom_target_filter
    merged_catalog.metadata['psfhom_target_filter'] = merged_catalog.psfhom_target_filter
    merged_catalog.psfhom_inverse_filters = catalog.psfhom_inverse_filters
    merged_catalog.metadata['psfhom_inverse_filters'] = merged_catalog.psfhom_inverse_filters
    merged_catalog.aperture_diameters = catalog.aperture_diameters
    merged_catalog.metadata['aperture_diameters'] = merged_catalog.aperture_diameters
    merged_catalog.metadata['detec_sci'] = all_detection_images[0].sci_file.replace(all_detection_images[0].tile, '[tile]')
    merged_catalog.metadata['detec_err'] = all_detection_images[0].err_file.replace(all_detection_images[0].tile, '[tile]')
    merged_catalog.metadata['detec_seg'] = all_detection_images[0].seg_file.replace(all_detection_images[0].tile, '[tile]')
    
    return merged_catalog



def _fit_pixel_distribution(data, sigma_upper=1.0, maxiters=3, uncertainties=False):
    data = data.flatten()
    data = data[np.isfinite(data)]

    mean, median, std = sigma_clipped_stats(data, sigma_upper=sigma_upper, sigma_lower=10., maxiters=maxiters)

    bins = np.linspace(median-7*std, median+7*std, 100)
    y, bins = np.histogram(data, bins=bins)
    y = y/np.max(y)
    p0 = [1, median, std]
    bc = 0.5*(bins[1:]+bins[:-1])
    popt, pcov = curve_fit(gauss, bc[bc<median], y[bc<median], p0=p0, bounds=([0.5,-20*std,0],[2,20*std,10*std]))

    for i in range(maxiters):
        p0 = popt
        bins = np.linspace(popt[1]-5*popt[2], popt[1]+5*popt[2], 100)
        y, bins = np.histogram(data, bins=bins)
        y = y/np.max(y)
        bc = 0.5*(bins[1:]+bins[:-1])
        popt, pcov = curve_fit(gauss, bc[bc<median+sigma_upper*popt[2]], y[bc<median+sigma_upper*popt[2]], p0=p0, bounds=([0.5,-20*std,0],[2,20*std,10*std]))

    if uncertainties:
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    return popt

class Catalog:

    def __init__(
        self,
        images: Images,
        flux_unit: str = 'uJy', 
        tile: str = "None", 
    ):

        self.images = images
        self.filters = copy(sorted(self.images.filters))

        self.flux_unit = flux_unit
        self.tile = tile
        self.memmap = images.memmap

        self.detection_image = None
        self.objects = None
    
        self.logger = utils.setup_logger(name='aperx.catalog')

        self.metadata = {}
        self.metadata['tile'] = self.tile
        self.metadata['flux_unit'] = self.flux_unit

    @classmethod
    def load(cls, file):
        header = fits.getheader(file, ext=0)
        
        objects = Table.read(file, hdu=1)
        images_table = Table.read(file, hdu=2)
        images = []
        for t in images_table:
            im = Image(
                filter = t['filter'],
                tile = t['tile'],
                sci_file = t['sci_file'],
                err_file = t['err_file'],
                wht_file = t['wht_file'],
                hom_file = t['hom_file'],
                psf_file = t['psf_file'],
                seg_file = t['seg_file'],
                memmap = t['memmap'],
                sci_extension = t['sci_extension'],
                err_extension = t['err_extension'],
                wht_extension = t['wht_extension'],
                psf_extension = t['psf_extension'],
            )
            images.append(im)
        images = Images(images)

        cat = cls(images)
        cat.objects = objects

        cat.metadata = cat.header_to_metadata(header)
        cat.tile = cat.metadata['tile']
        cat.flux_unit = cat.metadata['flux_unit']
        cat.psfhom_target_filter = cat.metadata['psfhom_target_filter']
        cat.psfhom_inverse_filters = cat.metadata['psfhom_inverse_filters']
        cat.aperture_diameters = cat.metadata['aperture_diameters']
        cat._load_detection_image(cat.metadata['detec_sci'], cat.metadata['detec_sci'], cat.metadata['detec_seg'])

        return cat

    def write(
        self, 
        output_file,
    ):
        self.logger.info(f'Writing catalog to {output_file}')
        header = self.metadata_to_header(self.metadata)
        from astropy.io import fits
        hdul = fits.HDUList([
            fits.PrimaryHDU(header=header),
            fits.table_to_hdu(self.objects),
            fits.table_to_hdu(self.images_table), 
        ])
        hdul.writeto(output_file, overwrite=True)

    @staticmethod
    def metadata_to_header(metadata: dict):
        header = {}
        header['TILE'] = metadata['tile']
        header['FLUXUNIT'] = metadata['flux_unit']
        header['PSF_TARG'] = metadata['psfhom_target_filter']
        header['PSF_INV'] = ", ".join(metadata['psfhom_inverse_filters']) if len(metadata['psfhom_inverse_filters'])>1 else ""
        header['AP_DIAM'] = ", ".join([str(round(d,3)) for d in metadata['aperture_diameters']]) if 'aperture_diameters' in metadata else "None"
        header['DETECSCI'] = metadata['detec_sci']
        header['DETECERR'] = metadata['detec_sci']
        header['DETECSEG'] = metadata['detec_seg']
        return fits.Header(header)
    
    @staticmethod
    def header_to_metadata(header: fits.Header):
        metadata = {}
        metadata['tile'] = header['TILE']
        metadata['flux_unit'] = header['FLUXUNIT']
        metadata['psfhom_target_filter'] = str(header['PSF_TARG'])
        metadata['psfhom_inverse_filters'] = str(header['PSF_INV']).split(', ') if len(str(header['PSF_INV'])) > 1 else []
        metadata['aperture_diameters'] = [float(d) for d in str(header['AP_DIAM']).split(', ')]
        metadata['detec_sci'] = header['DETECSCI']
        metadata['detec_sci'] = header['DETECERR']
        metadata['detec_seg'] = header['DETECSEG']
        return metadata


    @property 
    def images_table(self):
        t = {
            'filter':[], 'tile':[], 
            'sci_file':[], 'err_file':[], 'wht_file':[], 
            'hom_file':[], 'psf_file':[], 'seg_file':[],
            'memmap':[],'sci_extension':[], 'err_extension':[], 
            'wht_extension':[], 'psf_extension':[],
        }
        for image in self.images:
            t['filter'].append(str(image.filter))
            t['tile'].append(str(image.tile))
            t['sci_file'].append(str(image.sci_file))
            t['err_file'].append(str(image.err_file))
            t['wht_file'].append(str(image.wht_file))
            t['hom_file'].append(str(image.hom_file))
            t['psf_file'].append(str(image.psf_file))
            t['seg_file'].append(str(image.seg_file))
            t['memmap'].append(str(image.memmap))
            t['sci_extension'].append(str(image.sci_extension))
            t['err_extension'].append(str(image.err_extension))
            t['wht_extension'].append(str(image.wht_extension))
            t['psf_extension'].append(str(image.psf_extension))
        t = Table(t)
        return t


    ############################################################################################################
    ################################################### PSFs ###################################################
    ############################################################################################################

    def generate_psfhomogenized_images(
        self, 
        target_filter, 
        inverse_filters = [],
        reg_fact: float = 1e-4,
        overwrite: bool = False,
    ):
        self.psfhom_target_filter = target_filter
        self.psfhom_inverse_filters = inverse_filters
        self.metadata['psfhom_target_filter'] = self.psfhom_target_filter
        self.metadata['psfhom_inverse_filters'] = self.psfhom_inverse_filters

        if target_filter not in self.filters:
            msg = f'Target filter must be in {self.filters}'
            self.logger.error(msg)
            sys.exit(1)
        
        # for filt in inverse_filters:
        #     if filt not in self.filters:
        #         msg = f'Inverse filter {filt} not in {self.filters}'
        #         self.logger.warning(msg)
        #         sys.exit(1)
            

        # For images with PSF smaller than the target, we perform normal PSF-homogenization 
        target_image = self.images.get(filter=target_filter)[0]
        other_filters = [f for f in self.filters if f != target_filter and f not in inverse_filters]
        images = self.images.get(filter=other_filters)

        if images is None:
            return
        
        for image in images:
            if 'mock' in image.sci_extension:
                image.hom_file = image.base_file + f'hom_mock_{target_filter}.fits'
            else:
                image.hom_file = image.base_file + f'hom_{target_filter}.fits'
            # image.hom_file = image.sci_file.
            image.generate_psfhomogenized_image(
                target_filter = target_filter,
                target_psf_file = target_image.psf_file,
                output_file = image.hom_file,
                reg_fact = reg_fact, 
                overwrite = overwrite,
                logger = self.logger, 
            )
        
        # For images with larger PSF than the target, generate target image PSF-matched to image
        if len(inverse_filters) != 0:
            images = self.images.get(filter=inverse_filters)
            
            for filt, image in zip(inverse_filters, images):
                if 'mock' in image.sci_extension:
                    image.hom_file = image.base_file.replace(filt, target_filter) + f'hom_mock_{filt}.fits'
                else:
                    image.hom_file = image.base_file.replace(filt, target_filter) + f'hom_{filt}.fits'
                target_image.generate_psfhomogenized_image(
                    target_filter = filt, 
                    target_psf_file = image.psf_file, 
                    output_file = image.hom_file,
                    reg_fact = reg_fact, 
                    overwrite = overwrite, 
                    logger = self.logger
                )


    ############################################################################################################
    ############################################# DETECTION IMAGES #############################################
    ############################################################################################################

    def _build_ivw_detection_image(
        self, 
        detection_bands: List[str], 
        sigma_upper: float = 1.0,
        maxiters: int = 3,
        psfhom: bool = True,
    ):
        """
        Build an inverse-variance weighted detection image from a list of detection bands.

        Args:
        - detection_bands (List[str]): List of detection bands to use.
        - sigma_upper (float): Sigma clipping threshold to esimate background rms.
        - maxiters (int): Maximum number of iterations to use when fitting the pixel distribution.
        """

        self.logger.info(f'Building IVW detection image')

        # Check that the detection bands are all the same shape
        shapes = []
        for band in detection_bands:
            shapes.append(self.images.get(filter=band)[0].shape)
        if len(set(shapes)) > 1:
            msg = f'All detection bands must have the same shape.'
            self.logger.error(msg)
            sys.exit(1)

        shape = shapes[0]

        nbands = np.zeros(shape, dtype=int)
        num, den = np.zeros(shape), np.zeros(shape)
        for i, band in enumerate(detection_bands):
            # make normalized science images by multiplying by square root of the weight images
            image = self.images.get(filter=band)[0]
            if psfhom and band!=self.psfhom_target_filter and band not in self.psfhom_inverse_filters:
                sci = image.hom
            else:
                sci = image.sci
            wht = image.wht # weight image

            num += sci * wht
            den += wht
            #num[wht == 0] = np.nan
            nbands[np.isfinite(sci)&(wht > 0)] += 1

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # mask = nbands < len(detection_bands)
            # num[mask] = np.nan
            # den[mask] = np.nan
            ivw = num/den
            nivw = ivw * np.sqrt(den)


        # fit pixel distribution of normalized science images to a Gaussian to get the rms
        A, mu, sigma = _fit_pixel_distribution(
            nivw, 
            sigma_upper = sigma_upper, 
            maxiters = maxiters
        )

        data = nivw.flatten()
        data = data[np.isfinite(data)]
        mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=3)
        bins = np.linspace(median-5*std, median+5*std, 101)
        
        with warnings.catch_warnings(): 
            warnings.simplefilter('ignore')    
            detec_sci = (nivw - mu) / np.sqrt(den)
            detec_err = sigma / np.sqrt(den)

        return detec_sci, detec_err


    def _build_chimean_detection_image(
        self, 
        detection_bands, 
        **kwargs
    ):
        """
        Build a chi-mean detection image from a list of detection bands.

        Args:
        - detection_bands (List[str]): List of detection bands to use.
        - **kwargs: Additional keyword arguments to pass to _build_chisq_detection_image.
        """

        # make chi-sq image
        chisq = self._build_chisq_detection_image(detection_bands, **kwargs)

        n = np.zeros(chisq.shape)
        for band in detection_bands:
            isvalid = np.isfinite(self.science_images[band]) & (self.weight_images[band] > 0)
            n += isvalid

        from scipy.special import gamma
        mu = np.sqrt(2) * gamma((n+1)/2) / gamma(n/2)
        
        chi_mean = (np.sqrt(chisq) - mu) / np.sqrt(n - mu**2)

        return chi_mean

    def _load_detection_image(self, sci, err, seg):

        self.logger.info(f'Loading {sci}')
        self.detection_image = Image(
            filter = 'detec', 
            tile = self.tile,
            sci_file = sci, 
            err_file = err, 
            seg_file = seg,
            wht_file = None,
            psf_file = None,
            hom_file = None,
            memmap = self.memmap,
        )


    def build_detection_image(
        self,
        tile, 
        output_dir: str, 
        output_filename: str,
        method: str, # ['ivw', 'chi-mean', 'chi-sq']
        filters: List[str],
        psfhom: bool,
        overwrite: bool = False,
        **kwargs
    ):
        """
        Builds a detection image for a given set of filters. 

        Args:

        """

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_file = os.path.join(output_dir, output_filename)
        detec_file = output_file.replace('[tile]', tile)
        detec_sci_file = detec_file.replace('[ext]', 'sci')
        detec_err_file = detec_file.replace('[ext]', 'err')
        self.metadata['detec_sci'] = detec_sci_file
        self.metadata['detec_err'] = detec_err_file
        
        self.logger.info('Building detection images')

        if method == 'chi-mean':
            return self._build_chimean_detection_image()
        elif method == 'chi-sq':
            return self._build_chisq_detection_image()
        
        # Inverse variance weighted detection image
        elif method == 'ivw':
            if any([self.images.get(filter=f) is None for f in filters]):
                self.logger.warning(f'Tile {tile} does not have images for all detection bands {filters}')
                filters = [f for f in filters if self.images.get(filter=f) is not None]
                # return False

            if (os.path.exists(detec_sci_file) and os.path.exists(detec_err_file)) and not overwrite:
                self.logger.info(f'Skipping detection image generation for tile {tile}, detection image already exists!')

                self._load_detection_image(detec_sci_file, detec_err_file, None)
                return True

            detec_sci, detec_err = self._build_ivw_detection_image(
                tile,
                detection_bands=filters, 
                sigma_upper=kwargs['sigma_upper'],
                maxiters=kwargs['maxiters'],
                psfhom=psfhom, 
            )
        
            # Write out the detection image to a file
            hom = self.images.get(filter=filters[-1])[0]
            dtype = hom.sci.dtype
            detec_sci = detec_sci.astype(dtype)
            detec_err = detec_err.astype(dtype)
            detec_hdr = hom.hdr
            
            fits.writeto(detec_sci_file, detec_sci, header=detec_hdr, overwrite=True)
            fits.writeto(detec_err_file, detec_err, header=detec_hdr, overwrite=True)
            
            self._load_detection_image(detec_sci_file, detec_err_file, None)
            return True
    
        else:
            msg = f'Invalid detection image method: {method}. Must be one of ["ivw", "chi-mean", "chi-sq"].'
            self.logger.error(msg)
            sys.exit(1)
        


    ############################################################################################################
    ############################################# SOURCE DETECTION #############################################
    ############################################################################################################

    def detect_sources(self, 
        detection_scheme: str = 'single',
        save_segmap: bool = False,
        **kwargs, 
    ):
        """
        Detects sources in the detection images. 

        Args:
            detection_scheme (str): 
            kwargs: passed to detection function
        """

        if self.detection_image is None:
            self.logger.warning(f"Detection image not found for tile {tile}; perhaps it doesn't have all the detection filters?")
            return 

        detec = self.detection_image.sci/self.detection_image.err
        mask_nan = np.isnan(detec)

        
        if detection_scheme == 'single':
            assert 'kernel_type' in kwargs, "`source_detection.kernel_type` must be specified"
            assert 'kernel_params' in kwargs, "`source_detection.kernel_params` must be specified"
            assert 'thresh' in kwargs, "`source_detection.thresh` must be specified"
            assert 'minarea' in kwargs, "`source_detection.minarea` must be specified"
            assert 'deblend_nthresh' in kwargs, "`source_detection.deblend_nthresh` must be specified"
            assert 'deblend_cont' in kwargs, "`source_detection.deblend_cont` must be specified"
            assert 'filter_type' in kwargs, "`source_detection.filter_type` must be specified"
            assert 'clean' in kwargs, "`source_detection.clean` must be specified"
            assert 'clean_param' in kwargs, "`source_detection.clean_param` must be specified"

            self.logger.info(f'Performing source detection')
            objs, segmap = self._detect_sources_single_stage(
                detec, 
                mask_nan,
                kernel_type=kwargs['kernel_type'], 
                kernel_params=kwargs['kernel_params'], 
                thresh=kwargs['thresh'],
                minarea=kwargs['minarea'],
                deblend_nthresh=kwargs['deblend_nthresh'],
                deblend_cont=kwargs['deblend_cont'],
                filter_type=kwargs['filter_type'],
                clean=kwargs['clean'],
                clean_param=kwargs['clean_param'],
            )
            self.logger.info(f"{len(objs)} detections")

        elif detection_scheme == 'hot+cold':
            assert 'cold' in kwargs, "`source_detection.cold` table not found"
            assert 'hot' in kwargs, "`source_detection.hot` table not found"

            if not 'cold_mask_scale_factor' in kwargs:
                kwargs['cold_mask_scale_factor'] = None
            if not 'cold_mask_min_radius' in kwargs:
                kwargs['cold_mask_min_radius'] = None

            # Check cold-mode parameters
            assert 'kernel_type' in kwargs['cold'], "`source_detection.cold.kernel_type` must be specified"
            assert 'kernel_params' in kwargs['cold'], "`source_detection.cold.kernel_params` must be specified"
            assert 'thresh' in kwargs['cold'], "`source_detection.cold.thresh` must be specified"
            assert 'minarea' in kwargs['cold'], "`source_detection.cold.minarea` must be specified"
            assert 'deblend_nthresh' in kwargs['cold'], "`source_detection.cold.deblend_nthresh` must be specified"
            assert 'deblend_cont' in kwargs['cold'], "`source_detection.cold.deblend_cont` must be specified"
            assert 'filter_type' in kwargs['cold'], "`source_detection.cold.filter_type` must be specified"
            assert 'clean' in kwargs['cold'], "`source_detection.cold.clean` must be specified"
            assert 'clean_param' in kwargs['cold'], "`source_detection.cold.clean_param` must be specified"
    
            # Check hot-mode parameters
            assert 'kernel_type' in kwargs['hot'], "`source_detection.hot.kernel_type` must be specified"
            assert 'kernel_params' in kwargs['hot'], "`source_detection.hot.kernel_params` must be specified"
            assert 'thresh' in kwargs['hot'], "`source_detection.hot.thresh` must be specified"
            assert 'minarea' in kwargs['hot'], "`source_detection.hot.minarea` must be specified"
            assert 'deblend_nthresh' in kwargs['hot'], "`source_detection.hot.deblend_nthresh` must be specified"
            assert 'deblend_cont' in kwargs['hot'], "`source_detection.hot.deblend_cont` must be specified"
            assert 'filter_type' in kwargs['hot'], "`source_detection.hot.filter_type` must be specified"
            assert 'clean' in kwargs['hot'], "`source_detection.hot.clean` must be specified"
            assert 'clean_param' in kwargs['hot'], "`source_detection.hot.clean_param` must be specified"

            self.logger.info(f'Performing cold-mode source detection')
            objs_cold, segmap_cold = self._detect_sources_single_stage(
                detec, 
                mask_nan,
                kernel_type=kwargs['cold']['kernel_type'], 
                kernel_params=kwargs['cold']['kernel_params'], 
                thresh=kwargs['cold']['thresh'],
                minarea=kwargs['cold']['minarea'],
                deblend_nthresh=kwargs['cold']['deblend_nthresh'],
                deblend_cont=kwargs['cold']['deblend_cont'],
                filter_type=kwargs['cold']['filter_type'],
                clean=kwargs['cold']['clean'],
                clean_param=kwargs['cold']['clean_param'],
            )
            objs_cold['mode'] = 'cold'

            self.logger.info(f'Performing hot-mode source detection')
            objs_hot, segmap_hot = self._detect_sources_single_stage(
                detec, 
                mask_nan,
                kernel_type=kwargs['hot']['kernel_type'], 
                kernel_params=kwargs['hot']['kernel_params'], 
                thresh=kwargs['hot']['thresh'],
                minarea=kwargs['hot']['minarea'],
                deblend_nthresh=kwargs['hot']['deblend_nthresh'],
                deblend_cont=kwargs['hot']['deblend_cont'],
                filter_type=kwargs['hot']['filter_type'],
                clean=kwargs['hot']['clean'],
                clean_param=kwargs['hot']['clean_param'],
            )
            objs_hot['mode'] = 'hot'
            
            self.logger.info(f'Merging hot+cold detections')
            objs, segmap = self._merge_detections(
                objs_cold, segmap_cold,
                objs_hot, segmap_hot, 
                mask_scale_factor = kwargs['cold_mask_scale_factor'], 
                mask_min_radius = kwargs['cold_mask_min_radius'], 
                dilate_kernel_size = kwargs['cold_mask_dilate_kernel_size']
            )

            self.logger.info(f"{len(objs)} detections, {len(objs[objs['mode']=='cold'])} cold, {len(objs[objs['mode']=='hot'])} hot")

        self.objects = objs

        seg_file = self.detection_image.sci_file.replace('sci.fits', 'seg.fits')
        fits.writeto(seg_file, data=segmap, header=self.detection_image.hdr, overwrite=True)
        self.detection_image.seg_file = seg_file
        self.metadata['detec_seg'] = seg_file

    def _detect_sources_single_stage(
        self, 
        detec,
        mask,
        kernel_type,
        kernel_params,
        thresh, 
        minarea, 
        deblend_nthresh, 
        deblend_cont, 
        filter_type='matched', 
        clean=True,
        clean_param=1.0,
    ):

        if kernel_type == 'tophat':
            from astropy.convolution import Tophat2DKernel
            if 'radius' not in kernel_params:
                raise ValueError('radius must be specified for Tophat2DKernel')
            if 'mode' not in kernel_params:
                kernel_params['mode'] = 'oversample'
            kernel = Tophat2DKernel(**kernel_params)
            kernel = kernel.array
            kernel = kernel/np.max(kernel)
        
        elif kernel_type == 'gaussian':
            if 'fwhm' not in kernel_params:
                raise ValueError('`fwhm` must be specified for Gaussian2DKernel')
            if 'size' not in kernel_params:
                kernel_params['size'] = kernel_params['fwhm'] * 3
            from photutils.segmentation import make_2dgaussian_kernel
            kernel = make_2dgaussian_kernel(**kernel_params).array

        else:
            raise ValueError('kernel_type must be one of `gaussian` or `tophat`')

        objs, segmap = sep.extract(
            detec, 
            thresh=thresh, minarea=minarea,
            deblend_nthresh=deblend_nthresh, deblend_cont=deblend_cont, mask=mask,
            filter_type=filter_type, filter_kernel=kernel, clean=clean, clean_param=clean_param,
            segmentation_map=True)

        ids = np.arange(len(objs))+1
        ids = ids.astype(int)
        objs = Table(objs)
        objs['id'] = ids

        # Fix angles
        objs['theta'][objs['theta']>np.pi/2] -= np.pi

        return objs, segmap    
    
    def _merge_detections(
        self,
        objs1, segmap1,
        objs2, segmap2,
        mask_scale_factor=None, # scale factor from objs1['a'] and objs['b'] that defines the elliptical mask
        mask_min_radius=None, # minimum circular radius for the elliptical mask (pixels)
        dilate_kernel_size=None,
    ):
        '''
        Merges two lists of detections, objs1 and objs2, where all detections 
        in objs1 are included and detections in objs2 are included only if 
        they do not fall in a mask created as the union of segmap1 and an elliptical 
        mask derived from mask_scale_factor and mask_min_radius. 
        '''

        if mask_min_radius is not None and mask_scale_factor is not None:
            # Construct elliptical mask around sources in objs1
            m = np.zeros(segmap1.shape, dtype=bool)
            a = mask_scale_factor * objs1['a']
            b = mask_scale_factor * objs1['b']
            a[a < mask_min_radius] = mask_min_radius
            b[b < mask_min_radius] = mask_min_radius
            sep.mask_ellipse(m, objs1['x'], objs1['y'], a, b, objs1['theta'], r=1.)

            # Combine mask with segmap1 area
            mask = m | (segmap1 > 0)
        
        else:
            mask = segmap1 > 0

        if dilate_kernel_size is not None:
            kernel = Tophat2DKernel(dilate_kernel_size)
            mask = binary_dilation(mask.astype(int), kernel.array).astype(bool)

        
        # Mask objects in objs2 with segmap pixels that overlap the mask
        ids_to_mask = np.unique(segmap2 * mask)[1:] 
        segmap2[np.isin(segmap2,ids_to_mask)] = 0 # set segmap to 0
        objs2 = objs2[~np.isin(objs2['id'], ids_to_mask)]

        segmap2[segmap2>0] += np.max(segmap1)
        segmap = segmap1 + segmap2

        objs2['id'] += np.max(objs1['id'])
        objs = vstack((objs1,objs2))

        return objs, segmap

    def plot_detections(self, output_dir):
        from regions import Regions, EllipseSkyRegion
        from photutils.utils.colormaps import make_random_cmap
        from matplotlib.patches import Ellipse
        from matplotlib import colors
        import matplotlib.pyplot as plt

        cmap1 = plt.colormaps['Greys_r']
        cmap1.set_bad('k')
        background_color='#000000ff'

        # Get tile name from the catalog's data
        self.logger.info(f'Plotting detections')
        objs = self.objects
        detec = self.detection_image.sci / self.detection_image.err
        segmap = self.detection_image.seg

        cmap2 = make_random_cmap(len(objs)+1)
        cmap2.colors[0] = colors.to_rgba(background_color)

        fig, ax = plt.subplots(1,2,figsize=(14,10),sharex=True,sharey=True,constrained_layout=True)
        ax[0].imshow(detec, norm=colors.LogNorm(vmin=1, vmax=300), cmap=cmap1)
        ax[1].imshow(segmap, cmap=cmap2)

        kronrad = objs['kronrad']
        for i in range(len(objs)):
            # reg = EllipseSkyRegion(center=center_sky[i], width=width_sky[i]*u.arcsec, height=height_sky[i]*u.arcsec, angle=(np.degrees(objs['theta'][i])+20)*u.deg)

            e = Ellipse(xy=(objs['x'][i], objs['y'][i]), 
                        width=2.5*kronrad[i]*objs['a'][i], height=2.5*kronrad[i]*objs['b'][i], angle=np.degrees(objs['theta'][i]))
            e.set_facecolor('none')
            e.set_linewidth(0.15)
            e.set_edgecolor('lime')
            if objs['flag'][i]>1:
                e.set_edgecolor('r')

            ax[0].add_artist(e)

        ax[0].axis('off')
        ax[1].axis('off')

        output_filename = 'checkplot_' + os.path.basename(self.detection_image.base_file) + 'apertures+segm.pdf'
        out = os.path.join(output_dir, 'plots', output_filename)
        plt.savefig(out, dpi=1000)
        plt.close()


    ############################################################################################################
    ########################################### MISCELLANEOUS CALCS ############################################
    ############################################################################################################

    def compute_kron_radius(self, windowed=False):
        self.logger.info(f'Computing kron radius')

        detec = self.detection_image.sci / self.detection_image.err
        mask = np.isnan(detec)
        segmap = self.detection_image.seg
        objs = self.objects

        if windowed:
            x, y = objs['xwin'], objs['ywin']
        else:
            x, y = objs['x'], objs['y']

        a, b, theta = objs['a'], objs['b'], objs['theta']
        kronrad, krflag = sep.kron_radius(detec, x, y, a, b, theta, r=6.0, mask=mask, seg_id=objs['id'], segmap=segmap)
        self.objects['kronrad'] = kronrad
        self.objects['flag'] |= krflag

    def compute_windowed_positions(self):
        self.logger.info(f'Computing windowed positions')
        objs = self.objects
        if 'kronrad' not in objs.columns:
            self.compute_kron_radius()
    
        detec = self.detection_image.sci / self.detection_image.err
        mask = np.isnan(detec)
        segmap = self.detection_image.seg

        flux, fluxerr, flag = sep.sum_ellipse(
            detec, objs['x'], objs['y'], 
            objs['a'], objs['b'], objs['theta'], 
            2.5*objs['kronrad'], subpix=1, mask=mask,
            seg_id=objs['id'], segmap=segmap,
        )

        rhalf, rflag = sep.flux_radius(
            detec, objs['x'], objs['y'], 
            6.*objs['a'], 0.5, 
            seg_id=objs['id'], segmap=segmap,
            mask=mask, normflux=flux, subpix=5
        )

        sig = 2. / 2.35 * rhalf  
        xwin, ywin, flag = sep.winpos(
            detec, objs['x'], objs['y'], sig, 
            mask=mask)

        self.objects['xwin'] = xwin
        self.objects['ywin'] = ywin

    def compute_rhalf(self):
        images = self.images
        segmap = self.detection_image.seg
        objs = self.objects
        if 'kronrad' not in objs.columns:
            self.compute_kron_radius()

        for image in images:
            band = image.filter
            self.logger.info(f'Computing half-light radii for band {band}')

            mask = np.isnan(image.sci)

            flux, fluxerr, flag = sep.sum_ellipse(
                image.sci, objs['x'], objs['y'], objs['a'], objs['b'], objs['theta'], 
                2.5*objs['kronrad'], subpix=1, mask=mask,
                seg_id=objs['id'], segmap=segmap,
            )

            rhalf, rflag = sep.flux_radius(
                image.sci, objs['x'], objs['y'], 6.*objs['a'], 0.5, 
                seg_id=objs['id'], segmap=segmap,
                mask=mask, normflux=flux, subpix=5
            )

            self.objects[f'rh_{band}'] = rhalf



    @property
    def windowed(self):
        return 'xwin' in self.objects.columns and 'ywin' in self.objects.columns

    def get_xy(self):
        if self.windowed:
            return self.objects['xwin'], self.objects['ywin']
        else:
            return self.objects['x'], self.objects['y']

    def add_ra_dec(self):        
        x, y = self.get_xy()            
        wcs = self.detection_image.wcs
        coords = wcs.pixel_to_world(x, y)
        ra = coords.ra.value
        dec = coords.dec.value
        self.objects['ra'] = ra
        self.objects['dec'] = dec

    def compute_weights(self):
        self.logger.info(f'Getting weight map values')
        x, y = self.get_xy()            
        for image in self.images:
            filt = image.filter
            aperture = CircularAperture(np.array([x, y]).T, r=5)
            aperstats = ApertureStats(image.wht, aperture)
            self.objects[f'wht_{filt}'] = aperstats.median
            aperstats = ApertureStats(image.err, aperture)
            self.objects[f'err_{filt}'] = aperstats.median


    ############################################################################################################
    ################################################ PHOTOMETRY ################################################
    ############################################################################################################

    @staticmethod
    def _compute_aper_photometry(
            objs: Table,
            sci: npt.ArrayLike,
            err: npt.ArrayLike | None,
            mask: npt.ArrayLike | None, 
            segmap: npt.ArrayLike, 
            aperture_diameters: list[float],
            windowed: bool = False
        ) -> tuple[np.ndarray, ...]:

        if windowed:
            x, y = objs['xwin'], objs['ywin']
        else:
            x, y = objs['x'], objs['y']

        flux = np.zeros((len(objs), len(aperture_diameters)))
        fluxerr = np.zeros((len(objs), len(aperture_diameters)))
        for i, diam in enumerate(aperture_diameters):

            flux_i, fluxerr_i, flag = sep.sum_circle(
                sci, x, y, 
                diam/2, 
                err=err, 
                mask=mask, 
                segmap=segmap, 
                seg_id=objs['id']
            )

            flux[:,i] = flux_i
            fluxerr[:,i] = fluxerr_i

        return flux, fluxerr, flag

    def compute_aper_photometry(
        self, 
        aperture_diameters: list[float], 
        psfhom: bool = False,
        flux_unit: str = 'uJy',
    ):
        self.aperture_diameters = aperture_diameters
        self.metadata['aperture_diameters'] = aperture_diameters

        images = self.images
        images.sort('filter')
        segmap = self.detection_image.seg

        for image in images:
            filt = image.filter

            if psfhom:
                self.logger.info(f'Computing aper_hom photometry for {filt} (psf-homogenized)')
            else:
                self.logger.info(f'Computing aper_nat photometry for {filt}, (native resolution)')

            objs = self.objects
            image.convert_byteorder()
            if psfhom and filt != self.psfhom_target_filter and filt not in self.psfhom_inverse_filters:
                sci = image.hom
            else:
                sci  = image.sci
            err = image.err
            mask = np.isnan(sci)

            aperture_diameters_pixels = np.array(aperture_diameters) / image.pixel_scale

            flux, fluxerr, flag = self._compute_aper_photometry(
                objs, sci, err, mask, segmap, aperture_diameters_pixels, windowed=self.windowed,
            )

            if psfhom and filt in self.psfhom_inverse_filters:
                self.logger.info(f'Correcting {filt} based on {self.psfhom_target_filter} homogenization')
                flux1, _, _ = self._compute_aper_photometry(
                    objs, image.hom, None, mask, segmap, aperture_diameters_pixels, windowed=self.windowed,
                ) # flux1 = aperture flux in <target_filter>, psf-homogenized to <filter>
                image2 = images.get(filter=self.psfhom_target_filter)[0]
                flux2, _, _ = self._compute_aper_photometry(
                    objs, image2.sci, None, mask, segmap, aperture_diameters_pixels, windowed=self.windowed,
                ) # flux2 = aperture flux in <target_filter>, native-resolution
                
                # correction factor for the flux lost due to the larger PSF
                corr_fact = flux2/flux1
                
                flux *= corr_fact
                fluxerr *= corr_fact
                self.objects[f'psf_corr_aper_{filt}'] = corr_fact

            conversion = image._compute_unit_conv(flux_unit)

            if psfhom:
                self.objects[f'f_aper_hom_{filt}'] = flux * conversion
                self.objects[f'e_aper_hom_{filt}'] = fluxerr * conversion
            else:
                self.objects[f'f_aper_nat_{filt}'] = flux * conversion
                self.objects[f'e_aper_nat_{filt}'] = fluxerr * conversion

            image.close()

    @staticmethod
    def _compute_auto_photometry(
            objs: Table,
            sci: npt.ArrayLike, 
            err: npt.ArrayLike, 
            mask: npt.ArrayLike,
            segmap: npt.ArrayLike, 
            kron_params: tuple[float,float],
            windowed: bool = False,
        ) -> tuple[np.ndarray, ...]:

        if windowed:
            x, y = objs['xwin'], objs['ywin']
        else:
            x, y = objs['x'], objs['y']
        a, b, theta = objs['a'], objs['b'], objs['theta']

        flux, fluxerr, flag = sep.sum_ellipse(
            sci, x, y, a, b, theta, 
            err=err,
            r=kron_params[0]*objs['kronrad'], 
            mask=mask, 
            segmap=segmap, 
            seg_id=objs['id']
        )

        a = kron_params[0]*objs['kronrad']*objs['a']
        b = kron_params[0]*objs['kronrad']*objs['b']

        use_circle = kron_params[0] * objs['kronrad'] * np.sqrt(a * b) < kron_params[1]/2
        cflux, cfluxerr, cflag = sep.sum_circle(
            sci, objs['x'][use_circle], objs['y'][use_circle],
            kron_params[1]/2, err=err, mask=mask, segmap=segmap, seg_id=objs['id'][use_circle])

        flux[use_circle] = cflux
        fluxerr[use_circle] = cfluxerr
        flag[use_circle] |= cflag
        a[use_circle] = kron_params[1]/2
        b[use_circle] = kron_params[1]/2

        return flux, fluxerr, flag, a, b

    def compute_auto_photometry(
        self, 
        kron_params: tuple[float,float] = (2.5,3.5),
        flux_unit: str = 'uJy',
    ):        
        images = self.images
        images.sort('filter')
        segmap = self.detection_image.seg

        for image in images:
            filt = image.filter
            
            self.logger.info(f'Computing auto photometry for {filt}')

            objs = self.objects
            if 'kronrad' not in objs.columns:
                raise ValueError('kron radius must be computed before auto photometry')
            
            image.convert_byteorder()
            if filt != self.psfhom_target_filter and filt not in self.psfhom_inverse_filters:
                sci = image.hom
            else:
                sci = image.sci
            err = image.err
            mask = np.isnan(sci)
            
            flux, fluxerr, flag, a, b = self._compute_auto_photometry(
                objs, sci, err, mask, segmap, kron_params, windowed=self.windowed,
            )

            if filt in self.psfhom_inverse_filters:
                self.logger.info(f'Correcting {filt} based on {self.psfhom_target_filter} homogenization')
                flux1, _, _, _, _ = self._compute_auto_photometry(
                    objs, image.hom, None, mask, segmap, kron_params, windowed=self.windowed,
                ) # flux1 = auto flux in <target_filter>, psf-homogenized to <filter>
                image2 = images.get(filter=self.psfhom_target_filter)[0]
                flux2, _, _, _, _ = self._compute_auto_photometry(
                    objs, image2.sci, None, mask, segmap, kron_params, windowed=self.windowed,
                ) # flux2 = aperture flux in <target_filter>, native-resolution
                
                # correction factor for the flux lost due to the larger PSF
                corr_fact = flux2/flux1
                
                flux *= corr_fact
                fluxerr *= corr_fact
                self.objects[f'psf_corr_auto_{filt}'] = corr_fact

                
            # convert units
            conversion = image._compute_unit_conv(flux_unit)

            self.objects[f'f_auto_{filt}'] = flux * conversion
            self.objects[f'e_auto_{filt}'] = fluxerr * conversion
            self.objects['flag'] |= flag
            
            image.close()


    def apply_kron_corr(
        self,
        filt: str, 
        thresh: float, 
        kron_params1: tuple[float,float],
        kron_params2: tuple[float,float],
    ):
        images = self.images
        detec = self.detection_image
        segmap = detec.seg

        # Compute the kron correction from the given filter
        if filt in images.filters:
            image = images.get(filter=filt)[0]
            image.convert_byteorder()
            sci = image.sci
            err = image.err
            pixel_scale = image.pixel_scale
        
        elif band == 'detec':
            sci = detec.sci
            err = detec.err
            pixel_scale = detec.pixel_scale
        else:
            raise ValueError(f'Cannot compute kron correction from {filt}')

        self.logger.info(f'Computing kron correction from {filt}')
        mask = np.isnan(sci)
        objs = self.objects
        if 'kronrad' not in objs.columns:
            raise ValueError('kron radius must be computed before auto photometry')

        flux1, fluxerr1, flag1, a1, b1 = self._compute_auto_photometry(objs, sci, err, mask, segmap, kron_params1, windowed=self.windowed)
        flux2, fluxerr2, flag2, a2, b2 = self._compute_auto_photometry(objs, sci, err, mask, segmap, kron_params2, windowed=self.windowed)
        kron_corr = flux2/flux1

        flag_blend = kron_corr > thresh
        kron_corr = np.where(kron_corr<1, 1, kron_corr)
        kron_corr = np.where(kron_corr>thresh, thresh, kron_corr)

        for filt_i in images.filters:
            self.objects[f'f_auto_{filt_i}'] *= kron_corr
            self.objects[f'e_auto_{filt_i}'] *= kron_corr

        self.objects['kron_corr'] = kron_corr
        self.objects['kron1_a'] = a1 * pixel_scale
        self.objects['kron1_b'] = b1 * pixel_scale
        self.objects['kron1_area'] = np.pi * a1 * b1
        self.objects['kron2_a'] = a2 * pixel_scale
        self.objects['kron2_b'] = b2 * pixel_scale
        self.objects['kron2_area'] = np.pi * a2 * b2



    ############################################################################################################
    ############################################## POST-PROCESSING #############################################
    ############################################################################################################

    def merge_tiles(
        self, 
        matching_radius: float = 0.1,
        edge_mask: int = 300, # pixels
    ):
        """
        Merges the catalogs constructed from different tiles into one final catalog. 

        Args:
            matching_radius (float): radius, in arcsec, used to match sources in overlap regions
            edge_mask (int): distance in pixels from the tile edge in which sources are assumed to be spurious
        """
        if len(self.tiles)==1:
            self.logger.info('Skipping merge_tiles, only one tile in catalog.')
            self.objects = self.objects[self.tiles[0]]
            return

        self.logger.info(f'Merging {len(self.tiles)} tiles')
        
        tile0 = self.tiles[0]
        objs0 = self.objects[tile0]
        objs0['tile'] = tile0
        
        # remove objects from objs0 if they are >edge_mask pixels from a boundary
        detec = self.detection_image
        height0, width0 = detec.shape
        objs0_distance_to_edge = distance_to_nearest_edge(objs0['x'], objs0['y'], width0, height0)
        objs0 = objs0[objs0_distance_to_edge > edge_mask]
        
        self.logger.info(f'{tile0}: {len(objs0)} objects')
        coords0 = SkyCoord(objs0['ra'], objs0['dec'], unit='deg')

        for tile1 in self.tiles[1:]:
            objs1 = self.objects[tile1]
            objs1['tile'] = tile1

            # remove objects from objs1 if they are >edge_mask pixels from a boundary
            detec = self.detection_image
            height1, width1 = detec.shape
            objs1_distance_to_edge = distance_to_nearest_edge(objs1['x'], objs1['y'], width1, height1)
            objs1 = objs1[objs1_distance_to_edge > edge_mask]

            self.logger.info(f'{tile1}: {len(objs1)} objects')
            coords1 = SkyCoord(objs1['ra'], objs1['dec'], unit='deg')

            # Perform matching
            idx, d2d, d3d = coords0.match_to_catalog_sky(coords1)
            match = d2d < matching_radius*u.arcsec # should have same LEN as objs0
            objs0_unique = objs0[~match]
            objs0_matched = objs0[match]
            objs1_matched = objs1[idx[match]]
            unique = np.ones(len(objs1), dtype=bool)
            unique[idx[match]] = False
            objs1_unique = objs1[unique]

            # Handle matched cases:
            assert len(objs0_matched)==len(objs1_matched)
            objs0_distance_to_edge = distance_to_nearest_edge(objs0_matched['x'], objs0_matched['y'], width0, height0)
            objs1_distance_to_edge = distance_to_nearest_edge(objs1_matched['x'], objs1_matched['y'], width1, height1)
            which = np.argmax([objs0_distance_to_edge, objs1_distance_to_edge], axis=0)

            objs1 = vstack([objs0_unique, objs0_matched[which==0], objs1_matched[which==1], objs1_unique], join_type='outer')

            tile0 = tile1
            objs0 = objs1
            coords0 = SkyCoord(objs0['ra'], objs0['dec'], unit='deg')

        self.logger.info(f'Final catalog: {len(objs1)} objects')
        objs1['id'] = np.arange(len(objs1)).astype(int)
        self.objects = objs1

    # def _collect_nsci(self, image, nrandom_pixels_per_tile, psfhom):
    #     self.logger.info(image.tile)
    #     if psfhom: 
    #         nsci = image.hom * np.sqrt(image.wht)
    #     else:
    #         nsci = image.sci * np.sqrt(image.wht)
    #     nsci[image.wht == 0] = np.nan
    #     nsci = nsci[np.isfinite(nsci)]

    #     # Select a subset of pixels to do the fitting — no need to use all of them (its slow)
    #     nsci = np.random.choice(nsci, size=nrandom_pixels_per_tile)
    #     return nsci


    def _measure_random_aperture_scaling(
        self,
        filt, 
        min_radius: float, 
        max_radius: float, 
        num_radii: int,
        num_apertures_per_sq_arcmin: int,
        output_dir: str,
        plot: bool = False,
        overwrite: bool = False,
        psfhom: bool = False,
        min_num_apertures_per_sq_arcmin: int = 30,
    ):

        if psfhom:
            output_file = os.path.join(output_dir, 'random_apertures', f'randomApertures_{filt}_psfhom_coeffs.pickle')
        else:
            output_file = os.path.join(output_dir, 'random_apertures', f'randomApertures_{filt}_coeffs.pickle')

        if os.path.exists(output_file) and not overwrite:
            self.logger.info(f'Skipping random aperture measurement for {filt}')
            return

        if psfhom:
            self.logger.info(f'Measuring random aperture scaling for {filt} (psf-homogenized)')
        else:
            self.logger.info(f'Measuring random aperture scaling for {filt} (native resolution)')

        index = 2
        aperture_diameters = np.power(np.linspace(np.power(min_radius*2, 1/index), np.power(max_radius*2, 1/index), num_radii), index)

        images = self.images.get(filter=filt)
        detec_images = self.detection_images

        # First, fit the pixel distribution of the NSCI image to get the baseline single-pixel RMS 
        self.logger.info('Fitting pixel distribution')
        nrandom_pixels_per_tile = 50000

        # with Pool(processes=20) as pool:

        #     fluxes = pool.map(partial(self._collect_nsci, nrandom_pixels_per_tile=nrandom_pixels_per_tile, psfhom=psfhom), images)
        # fluxes = np.array(fluxes)
        # _, _, rms1 = _fit_pixel_distribution(fluxes, sigma_upper=1.0, maxiters=5) 

        fluxes = np.zeros(nrandom_pixels_per_tile*len(images))
        i = 0
        for image in tqdm.tqdm(images):
            if psfhom: 
                nsci = image.hom / image.err # * np.sqrt(image.wht)
            else:
                nsci = image.sci / image.err # * np.sqrt(image.wht)
            nsci[image.wht == 0] = np.nan
            nsci = nsci[np.isfinite(nsci)]

            # Select a subset of pixels to do the fitting — no need to use all of them (its slow)
            nsci = np.random.choice(nsci, size=nrandom_pixels_per_tile)
            fluxes[i:i+nrandom_pixels_per_tile] = nsci
            i += nrandom_pixels_per_tile

        _, _, rms1 = _fit_pixel_distribution(fluxes, sigma_upper=1.0, maxiters=5) 


        self.logger.info('Getting random aperture fluxes')
        # num_apertures_total = sum([int(num_apertures_per_sq_arcmin * image.area) for image in images])
        fluxes = {i:[] for i in range(len(aperture_diameters))}
        fluxes_random = {i:[] for i in range(len(aperture_diameters))}
        rms1_random = 1

        for image in tqdm.tqdm(images):

            if psfhom: 
                nsci = image.hom / image.err #np.sqrt(image.wht)
            else:
                nsci = image.sci / image.err #np.sqrt(image.wht)
            nsci[image.wht == 0] = np.nan

            nsci_random = np.random.normal(loc=0, scale=rms1_random, size=nsci.shape)

            pixel_scale = image.pixel_scale

            n_valid_pixels = np.sum(np.isfinite(nsci))
            area = n_valid_pixels * pixel_scale**2 / 3600

            x, y = np.arange(np.shape(nsci)[1]), np.arange(np.shape(nsci)[0])
            x, y = np.meshgrid(x, y)
            x, y = x.flatten(), y.flatten()
            x = x[np.isfinite(nsci.flatten())]
            y = y[np.isfinite(nsci.flatten())]


            for i in range(num_radii):
                diameter = aperture_diameters[i]
                Nap = int(num_apertures_per_sq_arcmin * area)
                if diameter > 1.0: 
                    Nap = int(Nap * 0.25/(diameter/2)**2)
                if Nap/area < min_num_apertures_per_sq_arcmin:
                    Nap = int(min_num_apertures_per_sq_arcmin * area)
                    

                idx = np.random.randint(low=0,high=len(x),size=Nap)
                xi, yi = x[idx], y[idx]

                aperture = CircularAperture(np.array([xi,yi]).T, r=diameter/2/pixel_scale)
                tbl = aperture_photometry(nsci, aperture)
                fluxes[i].extend(list(tbl['aperture_sum']))
                
                tbl = aperture_photometry(nsci_random, aperture)
                fluxes_random[i].extend(list(tbl['aperture_sum']))
        
        self.logger.info('Fitting distribution')
        rmsN = np.zeros(num_radii)
        rmsN_err = np.zeros(num_radii)
        
        rmsN_random = np.zeros(num_radii)
        rmsN_err_random = np.zeros(num_radii)

        for i in tqdm.tqdm(range(num_radii)):
            f = np.array(fluxes[i])
            popt, perr = _fit_pixel_distribution(f, sigma_upper=1.0, maxiters=3, uncertainties=True)
            rmsN[i] = popt[2]
            rmsN_err[i] = perr[2]
            
            rmsN_random[i] = np.std(np.array(fluxes_random[i]))
            rmsN_err_random[i] = 0

        self.logger.info('Fitting curve')        
        N = np.pi*(aperture_diameters/2/pixel_scale)**2
        sqrtN = np.sqrt(N)
        conversion = images[0]._compute_unit_conv('nJy')

        # First compute the simple power law method
        def func(sqrtN, alpha, beta):
            return alpha*np.power(sqrtN, beta)
        popt, pcov = curve_fit(func, sqrtN, rmsN/rms1, p0=[1,1], maxfev=int(1e5))
        plaw = lambda x: func(x, *popt)
        
        # Then do the spline method
        from scipy.interpolate import BSpline, make_splrep
        x_all = np.append([0,1], sqrtN)
        y_all = np.append([0,1], rmsN/rms1)
        w_all = np.append([500,100], rms1/rmsN_err)

        bspline = make_splrep(
            x = x_all, 
            y = y_all, 
            w = w_all, 
            k = 3, 
            s = len(x_all)
        )

        coeffs = {
            'sigma1': float(rms1 * conversion),
            'spline': {
                'c': list(bspline.c.astype(float)),
                't': list(bspline.t.astype(float)),
                'k': int(bspline.k),
            },
            'plaw': {
                'alpha': float(popt[0]),
                'beta': float(popt[1]),
            },
            'sqrtN_max': float(np.max(sqrtN)),
        }

        with open(output_file, 'wb') as pickle_file:
            pickle.dump(coeffs, pickle_file)


        if plot:
            self.logger.info('Plotting')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3.5,3), constrained_layout=True)
            
            ax.errorbar(sqrtN, rmsN/rms1, yerr=rmsN_err/rms1, 
                linewidth=0, marker='s', mfc='none', mec='k', 
                mew=0.8, ms=5, elinewidth=0.6, ecolor='k', 
                capsize=1.5, capthick=0.6, zorder=1000)
            x = np.linspace(0, 1.1*np.max(sqrtN), 1000)
            
            y = bspline(x)
            ax.plot(x, y, color='b', label='Spline')
            
            y = plaw(x)
            ax.plot(x, y, color='m', label='Power law')

            def func(N, alpha, beta):
                return alpha*np.power(N, beta)
            popt, pcov = curve_fit(func, N, rmsN_random/rms1_random, p0=[1,1], maxfev=int(1e5))
            ax.errorbar(sqrtN, rmsN_random/rms1_random, yerr=rmsN_err_random/rms1_random, 
                linewidth=0, marker='s', mfc='none', mec='0.7', mew=0.8, ms=5, 
                elinewidth=0.6, ecolor='0.7', capsize=1.5, capthick=0.6, zorder=1000)
            x = np.linspace(0, 1.1*np.max(sqrtN), 1000)
            y = func(x**2,*popt)
            ax.plot(x, y, color='lightblue')

            ax.set_xlim(0, np.max(np.sqrt(N))*1.1)
            ax.set_ylim(-0.03*np.max(rmsN/rms1), 1.1*np.max(rmsN/rms1))
            ax.set_xlabel(r'$\sqrt{N_{\rm pix}}$' + f' ({int(pixel_scale*1000)} mas)')
            ax.set_ylabel(r'$\sigma_N/\sigma_1$')

            ax.legend(loc='upper left', title=filt.upper(), frameon=False)

            ax.annotate('Gaussian random field', (0.95,0.05), 
            xycoords='axes fraction', ha='right', va='bottom', color='0.7')

            out = os.path.join(
                output_dir, 'plots', 
                os.path.basename(output_file).replace('_coeffs.pickle', '.pdf')
            )
            self.logger.info(f'Saving to {out}')
            plt.savefig(out)
            plt.close()


    def measure_random_aperture_scaling(
        self,
        min_radius: float, 
        max_radius: float, 
        num_radii: int,
        num_apertures_per_sq_arcmin: int,
        output_dir: str,
        plot: bool = False,
        overwrite: bool = False,
        min_num_apertures_per_sq_arcmin: int = 30,
    ):

        for filt in self.filters:
            filter_hom = (self.has_aper_hom or self.has_auto)
            filter_hom = filter_hom and filt != self.psfhom_target_filter and filt not in self.psfhom_inverse_filters

            if self.has_aper_nat or not filter_hom:
                self._measure_random_aperture_scaling(
                    filt, 
                    min_radius = min_radius,
                    max_radius = max_radius,
                    num_radii = num_radii,
                    num_apertures_per_sq_arcmin = num_apertures_per_sq_arcmin,
                    plot = plot,
                    overwrite = overwrite,
                    output_dir = output_dir,
                    psfhom = False,
                )
            
            if filter_hom:
                self._measure_random_aperture_scaling(
                    filt, 
                    min_radius = min_radius,
                    max_radius = max_radius,
                    num_radii = num_radii,
                    num_apertures_per_sq_arcmin = num_apertures_per_sq_arcmin,
                    plot = plot,
                    overwrite = overwrite,
                    output_dir = output_dir,
                    psfhom = True,
                )

    @staticmethod
    def _get_random_aperture_curve(
        sqrtN: npt.ArrayLike,
        coeff_file: str
    ) -> np.ndarray:
        with open(coeff_file, 'rb') as f:
            coeffs = pickle.load(f)

        sigma1 = coeffs['sigma1']
        alpha = coeffs['plaw']['alpha']
        beta = coeffs['plaw']['beta']
        
        c = coeffs['spline']['c']
        t = coeffs['spline']['t']
        k = coeffs['spline']['k']
        sqrtN_max = coeffs['sqrtN_max']

        plaw_result = sigma1 * alpha * np.power(sqrtN, beta)

        from scipy.interpolate import BSpline
        bspline = BSpline(c=coeffs['spline']['c'], t=coeffs['spline']['t'], k=coeffs['spline']['k'])
        spline_result = sigma1 * bspline(sqrtN)

        result = np.where(sqrtN<sqrtN_max, spline_result, plaw_result)
        return result


    def apply_random_aperture_error_calibration(self, coeff_dir):
        for filt in self.filters:
            self.logger.info(f'Applying random aperture calibration for {filt}')
            pixel_scale = self.images.get(filter=filt)[0].pixel_scale

            err = self.objects[f'err_{filt}']
            
            if self.has_auto:
                sqrtN_kron = np.sqrt(self.objects['kron1_area'])
            if self.has_aper:
                sqrtN_aper = np.sqrt(np.pi)*np.array(self.aperture_diameters)/pixel_scale/2

            if self.has_aper_nat:
                # native resolution aperture photometry
                coeff_file = os.path.join(coeff_dir, 'random_apertures', f'randomApertures_{filt}_coeffs.pickle')
                result = self._get_random_aperture_curve(sqrtN_aper, coeff_file)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rmsN = np.outer(err, result)
                rmsN = (rmsN*u.nJy).to(u.Unit(self.flux_unit)).value
                self.objects.rename_column(f'e_aper_nat_{filt}', f'e_aper_nat_{filt}_uncal')
                # self.objects[f'e_aper_nat_{filt}'] = np.sqrt(self.objects[f'e_aper_nat_{filt}_uncal']**2 + rmsN**2)
                self.objects[f'e_aper_nat_{filt}'] = np.where(np.isfinite(self.objects[f'e_aper_nat_{filt}_uncal']), rmsN, np.nan)
            
            if self.has_aper_hom:
                # psf-matched aperture photometry
                if filt == self.psfhom_target_filter or filt in self.psfhom_inverse_filters:
                    coeff_file = os.path.join(coeff_dir, 'random_apertures', f'randomApertures_{filt}_coeffs.pickle')
                else:
                    coeff_file = os.path.join(coeff_dir, 'random_apertures', f'randomApertures_{filt}_psfhom_coeffs.pickle')
                result = self._get_random_aperture_curve(sqrtN_aper, coeff_file)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rmsN = np.outer(err, result)
                rmsN = (rmsN*u.nJy).to(u.Unit(self.flux_unit)).value
                self.objects.rename_column(f'e_aper_hom_{filt}', f'e_aper_hom_{filt}_uncal')
                # self.objects[f'e_aper_hom_{filt}'] = np.sqrt(self.objects[f'e_aper_hom_{filt}_uncal']**2 + rmsN**2)
                self.objects[f'e_aper_hom_{filt}'] = np.where(np.isfinite(self.objects[f'e_aper_hom_{filt}_uncal']), rmsN, np.nan)
                if filt in self.psfhom_inverse_filters:
                    self.objects[f'e_aper_hom_{filt}'] *= self.obejcts[f'psf_corr_aper_{filt}']

            if self.has_auto:
                # auto (kron) photometry
                if filt == self.psfhom_target_filter or filt in self.psfhom_inverse_filters:
                    coeff_file = os.path.join(coeff_dir, 'random_apertures', f'randomApertures_{filt}_coeffs.pickle')
                else:
                    coeff_file = os.path.join(coeff_dir, 'random_apertures', f'randomApertures_{filt}_psfhom_coeffs.pickle')
                result = self._get_random_aperture_curve(sqrtN_kron, coeff_file)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rmsN = result * err
                rmsN = (rmsN*u.nJy).to(u.Unit(self.flux_unit)).value
                self.objects.rename_column(f'e_auto_{filt}', f'e_auto_{filt}_uncal')
                # self.objects[f'e_auto_{filt}'] = np.sqrt(self.objects[f'e_auto_{filt}_uncal']**2 + rmsN**2)
                self.objects[f'e_auto_{filt}'] = np.where(np.isfinite(self.objects[f'e_auto_{filt}_uncal']), rmsN, np.nan)
                if filt in self.psfhom_inverse_filters:
                    self.objects[f'e_auto_{filt}'] *= self.obejcts[f'psf_corr_auto_{filt}']



    def compute_psf_corr_grid(
        self, 
        filt,
        output_dir, 
        plot = False, 
        overwrite = False,
    ):
            
        output_file = os.path.join(output_dir, f'{filt}_psf_corr_grid.txt')
        if os.path.exists(output_file) and not overwrite:
            self.logger.info(f'Skipping computing PSF correction for {filt}')
            return output_file

        self.logger.info(f'Computing PSF correction for {filt}')

        from photutils.aperture import EllipticalAperture, aperture_photometry

        a = np.linspace(0.05, 2.0, 50)
        q = np.linspace(0.05, 1, 50)
        
        psf = self.images.get(filter=filt)[0].psf
        pixel_scale = psf.pixel_scale
        a /= pixel_scale
        psf = psf.data


        # f444w_psf_corrs = np.zeros((len(a),len(p),len(e)))
        psf_corrs = np.zeros((len(a),len(q)))
        for i, ai in enumerate(a):
            for j,qj in enumerate(q):
                bi = qj*ai
                ap = EllipticalAperture((np.shape(psf)[0]/2, np.shape(psf)[1]/2), a=ai, b=bi, theta=0)
                tab = aperture_photometry(psf, ap)
                psf_corrs[i,j] = np.sum(psf)/float(tab['aperture_sum'])


        psf_corrs = psf_corrs.T
        a, q = np.meshgrid(a, q)
        
        np.savetxt(
            output_file, 
            np.array([a.flatten()*pixel_scale, q.flatten(), psf_corrs.flatten()]).T
        )

        if plot: 
            if not os.path.exists(os.path.join(output_dir, 'plots')):
                os.mkdir(os.path.join(output_dir, 'plots'))

            import matplotlib.pyplot as plt
            import matplotlib as mpl
            fig = plt.figure(figsize=(4.2,3.5), constrained_layout=True)
            gs = mpl.gridspec.GridSpec(nrows=1,ncols=2,width_ratios=[1,0.06], figure=fig)
            ax = fig.add_subplot(gs[0])
            im = ax.scatter(a.flatten()*pixel_scale, q.flatten(), c=psf_corrs.flatten(), vmax=3.5, marker='s', s=10)
            ax.set_xlabel('semi-major axis [arcsec]')
            ax.set_ylabel('axis ratio $q$')
            ax = fig.add_subplot(gs[1])
            fig.colorbar(im, cax=ax, label='PSF correction')
            plt.savefig(os.path.join(output_dir, 'plots', f'{filt}_psf_corr_grid.pdf'))
            plt.close()

        return output_file


    def apply_psf_corrections(
        self,
        psf_corr_file, 
    ):
        from scipy.interpolate import griddata

        a, q, c = np.loadtxt(psf_corr_file).T

        if self.has_auto:
            self.logger.info(f'Applying PSF corrections for auto photometry')
            ai = self.objects['kron2_a']
            bi = self.objects['kron2_b']
            qi = bi/ai
            kron_psf_corr = griddata(np.array([a,q]).T, c, (ai,qi), fill_value=1.0)
            for filt in self.filters:
                self.objects[f'f_auto_{filt}'] *= kron_psf_corr
                self.objects[f'e_auto_{filt}'] *= kron_psf_corr
        if self.has_aper_hom:
            self.logger.info(f'Applying PSF corrections for aper_hom photometry')
            aperture_diameters = np.array(self.aperture_diameters)
            aper_psf_corr = griddata(np.array([a,q]).T, c, (aperture_diameters/2, [1]*len(aperture_diameters)), fill_value=1.0)
            for filt in self.filters:
                self.objects[f'f_aper_hom_{filt}'] *= aper_psf_corr
                self.objects[f'e_aper_hom_{filt}'] *= aper_psf_corr


    @property
    def has_detections(self):
        return not self.objects == {}

    @property
    def has_auto(self):
        return any(['_auto' in colname for colname in self.objects.columns])
    
    @property
    def has_aper_nat(self):
        return any(['_aper_nat' in colname for colname in self.objects.columns])
    
    @property
    def has_aper_hom(self):
        return any(['_aper_hom' in colname for colname in self.objects.columns])

    @property
    def has_photometry(self):
        return self.has_auto or self.has_aper_nat or self.has_aper_hom
    
    @property
    def has_aper(self):
        return self.has_aper_nat or self.has_aper_hom
