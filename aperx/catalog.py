from .image import Image

from collections import defaultdict
from typing import List
import os, warnings, tqdm
import numpy as np
import pickle

def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5*(x-mu)**2/sigma**2)

from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats
import astropy.units as u


from astropy.wcs import WCS
from astropy.table import Table, vstack

from photutils.aperture import aperture_photometry, CircularAperture, ApertureStats

import sep
sep.set_extract_pixstack(int(1e7))
sep.set_sub_object_limit(2048)



def _fit_pixel_distribution(data, sigma_upper=1.0, maxiters=3, uncertainties=False):
    data = data.flatten()
    data = data[np.isfinite(data)]

    mean, median, std = sigma_clipped_stats(data, sigma_upper=sigma_upper, sigma_lower=10., maxiters=maxiters)

    bins = np.linspace(median-7*std, median+7*std, 100)
    y, bins = np.histogram(data, bins=bins)
    y = y/np.max(y)
    p0 = [1, median, std]
    bc = 0.5*(bins[1:]+bins[:-1])
    popt, pcov = curve_fit(gauss, bc[bc<median], y[bc<median], p0=p0)

    for i in range(maxiters):
        p0 = popt
        bins = np.linspace(popt[1]-5*popt[2], popt[1]+5*popt[2], 100)
        y, bins = np.histogram(data, bins=bins)
        y = y/np.max(y)
        bc = 0.5*(bins[1:]+bins[:-1])
        popt, pcov = curve_fit(gauss, bc[bc<median+sigma_upper*popt[2]], y[bc<median+sigma_upper*popt[2]], p0=p0)

    if uncertainties:
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    return popt

class Catalog:

    def __init__(self, 
        config, 
        filters,  
        tiles, 
        psfhom_filter, 
        flux_unit = 'uJy', 
    ):

        self.config = config
        self.filters = filters
        self.tiles = tiles
        self.flux_unit = flux_unit
        self.psfhom_filter = psfhom_filter 
        self.images = self._parse_mosaics()
        self.detection_images = {}
        self.objects = {}
        self.segmentation_maps = {}
        self.has_auto = False
        self.has_aper_native = False
        self.has_aper_psfmatched = False

    @classmethod
    def load(cls, file):
        pass

    def _parse_mosaics(self):
        images = {}
        for tile in self.tiles:
            images[tile] = {}
            for filt in self.filters:

                for mosaic_spec in self.config.mosaics:
                    # If filter is already specified, skip it 
                    if filt in images[tile]: 
                        continue

                    # If mosaic spec is not valid for this tile, skip it
                    if 'tiles' in mosaic_spec:
                        if tile not in mosaic_spec.tiles:
                            continue
                        
                    # If mosaic spec is not valid for this filter, skip it
                    if 'filters' in mosaic_spec:
                        if filt not in mosaic_spec.filters: 
                            continue
                    
                    # Otherwise, we're good to add this image
                    if 'filepath' not in mosaic_spec: 
                        raise ValueError('Missing filepath')
                    filepath = str(mosaic_spec.filepath)
                    
                    if 'filename' not in mosaic_spec: 
                        raise ValueError('Missing filename')
                    filename = str(mosaic_spec.filename)

                    filepath = os.path.join(filepath, filename)

                    if '[pixel_scale]' in filepath or '[pixel_scale]' in filename:
                        if 'pixel_scale' in mosaic_spec:
                            pixel_scale = mosaic_spec.pixel_scale
                        elif 'pixel_scale' in self.config:
                            pixel_sale = self.config.pixel_scale
                        else:
                            raise ValueError('Missing pixel_scale')

                        filepath = filepath.replace('[pixel_scale]', pixel_scale)

                    if '[version]' in filepath or '[version]' in filename:    
                        if 'version' in mosaic_spec:
                            version = mosaic_spec.version
                        elif 'version' in self.config:
                            version = self.config.version
                        else:
                            raise ValueError('missing version')

                        filepath = filepath.replace('[version]', version)

                    if '[field_name]' in filepath or '[field_name]' in filename:
                        if 'field_name' in mosaic_spec:
                            field_name = mosaic_spec.field_name
                        elif 'field_name' in self.config:
                            field_name = self.config.field_name
                        else:
                            raise ValueError('missing field_name')

                        filepath = filepath.replace('[field_name]', field_name)
                    
                    filepath = filepath.replace('[filter]', filt)

                    if 'psf_tile' in mosaic_spec:
                        psf_tile = mosaic_spec.psf_tile
                    else:
                        psf_tile = tile
                    
                    psf_tile = psf_tile.replace('[tile]', tile)
                    
                    psf_filepath = filepath.replace('[tile]', psf_tile)

                    filepath = filepath.replace('[tile]', tile)

                    if not filepath.endswith('.fits'):
                        filepath += '.fits'

                    if 'sci_ext' in mosaic_spec:
                        sci_file = filepath.replace('[ext]', mosaic_spec.sci_ext)
                    else:
                        sci_file = filepath.replace('[ext]', 'sci')

                    if 'err_ext' in mosaic_spec:
                        err_file =  filepath.replace('[ext]', mosaic_spec.err_ext)
                    else:
                        err_file = filepath.replace('[ext]', 'err')
                    
                    if 'wht_ext' in mosaic_spec:
                        wht_file = filepath.replace('[ext]', mosaic_spec.wht_ext)
                    else:
                        wht_file = filepath.replace('[ext]', 'wht')
                    
                    if 'mask_ext' in mosaic_spec:
                        mask_file = filepath.replace('[ext]', mosaic_spec.mask_ext)
                    else:
                        mask_file = filepath.replace('[ext]', 'srcmask')
                        
                    psf_file = psf_filepath.replace('[ext]', 'psf')

                    if filt == self.psfhom_filter:
                        psfmatched_file = sci_file
                    else:
                        if 'psfmatched_ext' in mosaic_spec:
                            psfmatched_file = filepath.replace('[ext]', mosaic_spec.psfmatched_ext)
                        else:
                            psfmatched_file = filepath.replace('[ext]', 'sci_psfmatched')
                        
                    if (']' in sci_file) or ('[' in sci_file):
                        raise ValueError(f'Could not parse file {sci_file}')

                    # Add to images dictionary
                    image = Image(
                        filter=filt,
                        sci_file=sci_file, 
                        err_file=err_file, 
                        wht_file=wht_file, 
                        mask_file=mask_file,
                        psf_file=psf_file, 
                        psfmatched_file=psfmatched_file
                    )

                    images[tile][filt] = image

        return images


    @property
    def images_flipped(self):
        flipped = defaultdict(dict)
        for key, val in self.images.items():
            for subkey, subval in val.items():
                flipped[subkey][key] = subval
        return flipped

    @property
    def all_images_exist(self):
        all_exist = True
        for tile in self.images:
            for filt in self.images[tile]:
                all_exist = all_exist and self.images[tile][filt].exists
        return all_exist
    
    @property
    def all_psfs_exist(self):
        all_exist = True
        for tile in self.images:
            for filt in self.images[tile]:
                all_exist = all_exist and self.images[tile][filt].has_psf
        return all_exist
    
    @property
    def all_psfmatched_images_exist(self):
        all_exist = True
        for tile in self.images:
            for filt in self.images[tile]:
                all_exist = all_exist and self.images[tile][filt].has_psfmatched_equivalent
        return all_exist

    @property
    def detection_image_exists(self):
        pass

    def detect_sources(self, 
        tile, 
        detection_scheme = 'single',
        **kwargs, 
    ):
        if 'sci' not in self.detection_images[tile] or 'err' not in self.detection_images[tile]:
            raise ValueError('Detection images not loaded')
        
        detec_sci = self.detection_images[tile]['sci']
        detec_err = self.detection_images[tile]['err']
        detec = detec_sci/detec_err
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

            print('Performing source detection')
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
            print(f"{len(objs)} detections")

        elif detection_scheme == 'hot+cold':
            assert 'cold' in kwargs, "`source_detection.cold` table not found"
            assert 'hot' in kwargs, "`source_detection.hot` table not found"

            assert 'cold_mask_scale_factor' in kwargs, "`source_detection.cold_mask_scale_factor` must be specified"
            assert 'cold_mask_min_radius' in kwargs, "`source_detection.cold_mask_min_radius` must be specified"

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

            print('Performing cold-mode source detection...')
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

            print('Performing hot-mode source detection...')
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
            
            print('Merging hot+cold detections')
            objs, segmap = self._merge_detections(
                objs_cold, segmap_cold,
                objs_hot, segmap_hot, 
                mask_scale_factor = kwargs['cold_mask_scale_factor'], 
                mask_min_radius = kwargs['cold_mask_min_radius'], 
            )

            print(f"{len(objs)} detections, {len(objs[objs['mode']=='cold'])} cold, {len(objs[objs['mode']=='hot'])} hot")

        self.objects[tile] = objs
        self.segmentation_maps[tile] = segmap

    def compute_windowed_positions(self, tile):
        print('Computing windowed positions...')
        objs = self.objects[tile]
        if 'kronrad' not in objs.columns:
            self.compute_kron_radius(tile)
        
        detec_sci = self.detection_images[tile]['sci']
        detec_err = self.detection_images[tile]['err']
        detec = detec_sci / detec_err
        mask = np.isnan(detec)
        segmap = self.segmentation_maps[tile]

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

        self.objects[tile]['xwin'] = xwin
        self.objects[tile]['ywin'] = ywin

    def add_ra_dec(self, tile, pos_key=None):
        if pos_key == 'windowed':
            x, y = self.objects[tile]['xwin'], self.objects[tile]['ywin']
        else:
            x, y = self.objects[tile]['x'], self.objects[tile]['y']
        
        wcs = self.detection_images[tile]['wcs']
        coords = wcs.pixel_to_world(x, y)
        ra = coords.ra.value
        dec = coords.dec.value
        self.objects[tile]['ra'] = ra
        self.objects[tile]['dec'] = dec

    def compute_weights(self, tile, pos_key=None):
        if pos_key == 'windowed':
            x, y = self.objects[tile]['xwin'], self.objects[tile]['ywin']
        else:
            x, y = self.objects[tile]['x'], self.objects[tile]['y']
        
        for filt in self.images[tile]:
            image = self.images[tile][filt]
            aperture = CircularAperture(np.array([x, y]).T, r=5)
            aperstats = ApertureStats(image.wht, aperture)
            self.objects[tile][f'wht_{filt}'] = aperstats.median

    def compute_kron_radius(self, tile, pos_key=None):
        print('Computing kron radius...')
        detec_sci = self.detection_images[tile]['sci']
        detec_err = self.detection_images[tile]['err']
        detec = detec_sci / detec_err
        mask = np.isnan(detec)
        objs = self.objects[tile]
        segmap = self.segmentation_maps[tile]
        if pos_key == 'windowed':
            x, y = objs['xwin'], objs['ywin']
        else:
            x, y = objs['x'], objs['y']
        a, b, theta = objs['a'], objs['b'], objs['theta']
        kronrad, krflag = sep.kron_radius(detec, x, y, a, b, theta, r=6.0, mask=mask, seg_id=objs['id'], segmap=segmap)
        self.objects[tile]['kronrad'] = kronrad
        self.objects[tile]['flag'] |= krflag

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
        mask_scale_factor, # scale factor from objs1['a'] and objs['b'] that defines the elliptical mask
        mask_min_radius, # minimum circular radius for the elliptical mask (pixels)
    ):
        '''
        Merges two lists of detections, objs1 and objs2, where all detections 
        in objs1 are included and detections in objs2 are included only if 
        they do not fall in a mask created as the union of segmap1 and an elliptical 
        mask derived from mask_scale_factor and mask_min_radius. 
        '''

        # Construct elliptical mask around sources in objs1
        m = np.zeros(segmap1.shape, dtype=bool)
        a = mask_scale_factor * objs1['a']
        b = mask_scale_factor * objs1['b']
        a[a < mask_min_radius] = mask_min_radius
        b[b < mask_min_radius] = mask_min_radius
        sep.mask_ellipse(m, objs1['x'], objs1['y'], a, b, objs1['theta'], r=1.)

        # Combine mask with segmap1 area
        mask = m | (segmap1 > 0)
        
        # Mask objects in objs2 with segmap pixels that overlap the mask
        ids_to_mask = np.unique(segmap2 * mask)[1:] 
        segmap2[np.isin(segmap2,ids_to_mask)] = 0 # set segmap to 0
        objs2 = objs2[~np.isin(objs2['id'], ids_to_mask)]

        segmap2[segmap2>0] += np.max(segmap1)
        segmap = segmap1 + segmap2

        objs2['id'] += np.max(objs1['id'])
        objs = vstack((objs1,objs2))

        return objs, segmap


    @staticmethod
    def _compute_auto_photometry(
            objs, sci, err, mask, segmap, kron_params, 
            pos_key=None,
        ):

        if pos_key == 'windowed':
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
        tile, 
        kron_params=(2.5,3.5),
        pos_key=None,
        flux_unit='uJy',
    ):
        for band in self.images[tile]:
            
            print(f'Computing auto photometry for {band}')
            self.has_auto = True

            objs = self.objects[tile]
            if 'kronrad' not in objs.columns:
                raise ValueError('kron radius must be computed before auto photometry')
            
            im = self.images[tile][band]
            im.convert_byteorder()
            sci = im.psfmatched
            err = im.err
            mask = np.isnan(sci)
            segmap = self.segmentation_maps[tile]
            
            
            flux, fluxerr, flag, a, b = self._compute_auto_photometry(
                objs, sci, err, mask, segmap, kron_params, pos_key=pos_key
            )

            # convert units
            conversion = im._compute_unit_conv(flux_unit)

            self.objects[tile][f'f_auto_{band}'] = flux * conversion
            self.objects[tile][f'e_auto_{band}'] = fluxerr * conversion
            self.objects[tile]['flag'] |= flag
    
    @staticmethod
    def _compute_aper_photometry(
            objs, sci, err, mask, segmap, aperture_diameters, 
            pos_key=None,
        ):

        if pos_key == 'windowed':
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
        tile,
        aperture_diameters, 
        psfmatched=False,
        pos_key=None,
        flux_unit='uJy',
    ):
        self.aperture_diameters = aperture_diameters

        for band in self.images[tile]:

            if psfmatched:
                print(f'Computing aper photometry for {band}, psf-matched')
                self.has_aper_native = True
            else:
                print(f'Computing aper photometry for {band}, native resolution')
                self.has_aper_psfmatched = True

            objs = self.objects[tile]
            im = self.images[tile][band]
            im.convert_byteorder()
            if psfmatched:
                sci = im.psfmatched
            else:
                sci = im.sci
            err = im.err
            mask = np.isnan(sci)
            segmap = self.segmentation_maps[tile]
            
            aperture_diameters = aperture_diameters / im.pixel_scale
            
            flux, fluxerr, flag = self._compute_aper_photometry(
                objs, sci, err, mask, segmap, aperture_diameters, pos_key=pos_key
            )

            # convert units
            conversion = im._compute_unit_conv(flux_unit)

            if psfmatched:
                self.objects[tile][f'f_aper_psfmatched_{band}'] = flux * conversion
                self.objects[tile][f'e_aper_psfmatched_{band}'] = fluxerr * conversion
            else:
                self.objects[tile][f'f_aper_native_{band}'] = flux * conversion
                self.objects[tile][f'e_aper_native_{band}'] = fluxerr * conversion



    def apply_kron_corr(
        self,
        tile,
        band, 
        thresh, 
        kron_params1,
        kron_params2,
        pos_key=None
    ):

        if band in self.images[tile]:
            im = self.images[tile][band]
            im.convert_byteorder()
            sci = im.sci
            err = im.err
            pixel_scale = im.pixel_scale
        elif band == 'detec':
            sci = self.detection_images[tile]['sci']
            err = self.detection_images[tile]['err']
            wcs = self.detection_images[tile]['wcs']
            pixel_scale = np.abs(wcs.proj_plane_pixel_scales()[0]).to(u.arcsec).value
        else:
            raise ValueError(f'Cannot compute kron correction from band {band}')

        print(f'Computing kron correction from {band}')
        mask = np.isnan(sci)
        segmap = self.segmentation_maps[tile]
        objs = self.objects[tile]
        if 'kronrad' not in objs.columns:
            raise ValueError('kron radius must be computed before auto photometry')

        flux1, fluxerr1, flag1, a1, b1 = self._compute_auto_photometry(objs, sci, err, mask, segmap, kron_params1, pos_key=pos_key)
        flux2, fluxerr2, flag2, a2, b2 = self._compute_auto_photometry(objs, sci, err, mask, segmap, kron_params2, pos_key=pos_key)
        kron_corr = flux2/flux1

        # derive KRON correction to correct from small elliptical apertures to total flux
        flag_blend = kron_corr > thresh
        kron_corr = np.where(kron_corr<1, 1, kron_corr)
        kron_corr = np.where(kron_corr>thresh, thresh, kron_corr)

        for band in self.images[tile]:
            self.objects[tile][f'f_auto_{band}'] *= kron_corr
            self.objects[tile][f'e_auto_{band}'] *= kron_corr

        self.objects[tile]['kron_corr'] = kron_corr
        self.objects[tile]['kron1_a'] = a1 * pixel_scale
        self.objects[tile]['kron1_b'] = b1 * pixel_scale
        self.objects[tile]['kron1_area'] = np.pi * a1 * b1
        self.objects[tile]['kron2_a'] = a2 * pixel_scale
        self.objects[tile]['kron2_b'] = b2 * pixel_scale
        self.objects[tile]['kron2_area'] = np.pi * a2 * b2





    def get_images(self, filt=None, tile=None):
        """
        Get a list of images corresponding to a given filter or tile (but not both, that's pointless). 
        """
        if filt is not None and tile is not None:
            raise ValueError('get_images should be used separately for filters/tiles')
        
        elif filt is not None:
            return list(self.images_flipped[filt].values())
        
        elif tile is not None:
            return list(self.images[tile].values())
        else:
            raise ValueError('get_images: must specify filter or tile')


    def generate_psfs(self, 
        fwhm_min: float = 0.75, # minimum FWHM scale for PSF generation
        fwhm_max: float = 1.75, # maximum FWHM scale for PSF generation
        max_ellip: float = 0.1, # maximum ellipticity for PSF generation
        min_snr: float = 8.0, # minimum SNR for PSF generation
        max_snr: float = 1000.0, # minimum SNR for PSF generation
        psf_size: int = 301, # size of PSF image, should be odd
        checkplots: bool = False, # whether to generate checkplots during PSF generation
        overwrite: bool = False, # whether to overwrite existing PSF files
        plot: bool = False, # plot the final PSFs for inspection
        az_average: bool = False,
    ):

        for filt in self.filters:
            images = list(self.images_flipped[filt].values())
            psf_files = [image.psf_file for image in images]

            if len(images) == 1:
                master_psf_file = None
                if images[0].has_psf and not overwrite:
                    print(f'All PSF files for {filt} already exist, skipping generation')
                    continue

            elif len(set(psf_files)) == 1:
                # Build master PSF
                master_psf_file = psf_files[0]
                if os.path.exists(master_psf_file) and not overwrite:
                    print(f'Master PSF for {filt} already exists, skipping generation')
                    continue
            else:
                master_psf_file=None
                all_exist = True
                for psf_file in psf_files:
                    all_exist = all_exist and os.path.exists(psf_file)
                if all_exist and not overwrite:
                    print(f'All PSF files for {filt} already exist, skipping generation')
                    continue

            # Build PSFs
            from .psf import PSF
            psf = PSF.build(
                images=images,
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
            )
            
            if plot:
                if len(images)==1:
                    psf.plot(save=images[0].psf_file.replace('.fits','.png'))
                elif master_psf_file:
                    psf.plot(save=master_psf_file.replace('.fits','.png'))
                    for image in images:
                        psf = PSF(image.base_file + 'psf.fits')
                        psf.plot(save=image.base_file + 'psf.png')
                # for image in images:
                #     psf.plot(save=image.psf_file.replace('.fits','.png'))

    def _build_ivw_detection_image(
        self, 
        tile,
        detection_bands: List[str], 
        sigma_upper: float = 1.0,
        maxiters: int = 3,
        psfmatched: bool = True,
    ):
        """
        Build a chi-squared detection image from a list of detection bands.

        Args:
        - detection_bands (List[str]): List of detection bands to use.
        - sigma_upper (float): Sigma clipping threshold to esimate background rms.
        - maxiters (int): Maximum number of iterations to use when fitting the pixel distribution.
        """

        print(f'Building IVW detection image for {tile}')

        # # ensure all detection bands are available in this tile
        # if not all(band in self.images[tile] for band in detection_bands):
        #     raise ValueError(f'Not all detection bands are available in tile {tile}')
        
        # Check that the detection bands are all the same shape
        shapes = []
        for band in detection_bands:
            shapes.append(self.images[tile][band].shape)
        if len(set(shapes)) > 1:
            raise ValueError(f'All detection bands must have the same shape.')
        shape = shapes[0]

        nbands = np.zeros(shape, dtype=int)
        num, den = np.zeros(shape), np.zeros(shape)
        for i, band in enumerate(detection_bands):
            # make normalized science images by multiplying by square root of the weight images
            if psfmatched:
                sci = self.images[tile][band].psfmatched
            else:
                sci = self.images[tile][band].sci
            wht = self.images[tile][band].wht # weight image

            num += sci * wht
            den += wht
            num[wht == 0] = np.nan
            nbands[np.isfinite(sci)&(wht > 0)] += 1

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mask = nbands < len(detection_bands)
            num[mask] = np.nan
            den[mask] = np.nan
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
        
        # import matplotlib.pyplot as plt
        # plt.figure()
        # h, _ = np.histogram(data, bins=bins)
        # h = h/np.max(h)
        # plt.stairs(h, bins)
        # x = np.linspace(np.min(bins), np.max(bins), 1000)
        # plt.plot(x, gauss(x, A, mu, sigma), color='r')
        # plt.savefig('test.pdf')
        # plt.close()
        
        # make signal to noise images by dividing normalized science images by rms
        # snr = (nivw - mu) / sigma
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

    def build_detection_image(
        self,
        tile: str, 
        file: str,
        type: str, # ['ivw', 'chi-mean', 'chi-sq']
        filters: List[str],
        psfmatched: bool,
        **kwargs
    ):
        if type == 'chi-mean':
            return self._build_chimean_detection_image()
        elif type == 'chi-sq':
            return self._build_chisq_detection_image()
        
        # Inverse variance weighted detection image
        elif type == 'ivw':
            if not 'sigma_upper' in kwargs:
                kwargs['sigma_upper'] = 1.0
                print('Warning: `sigma_upper` not specified, defaulting to 1.0.')
            if not 'maxiters' in kwargs:
                kwargs['maxiters'] = 3
                print('Warning: `maxiters` not specified, defaulting to 3.')

            if not all([f in self.images[tile] for f in filters]):
                raise ValueError(f'Tile {tile} does not have images for all detection bands {filters}')

            detec_sci, detec_err = self._build_ivw_detection_image(
                tile,
                detection_bands=filters, 
                sigma_upper=kwargs['sigma_upper'],
                maxiters=kwargs['maxiters'],
                psfmatched=psfmatched, 
            )
            
            # Write out the detection image to a file
            detec_sci = detec_sci.astype(self.images[tile][self.psfhom_filter].sci.dtype)
            detec_err = detec_err.astype(self.images[tile][self.psfhom_filter].sci.dtype)
            detec_wcs = self.images[tile][self.psfhom_filter].wcs
            self.detection_images[tile] = {'sci': detec_sci, 'err': detec_err, 'wcs': detec_wcs}
            
            hdr = self.images[tile][self.psfhom_filter].hdr # use the header from the psf-homogenization filter
            sci_file = file.replace('.fits', '_sci.fits')
            err_file = file.replace('.fits', '_err.fits')
            fits.writeto(sci_file, detec_sci, header=hdr, overwrite=True)
            fits.writeto(err_file, detec_err, header=hdr, overwrite=True)
        
        else:
            raise ValueError(f'Invalid detection image type: {type}. Must be one of ["ivw", "chi-mean", "chi-sq"].')
        
            
        

    def load_detection_image(self, tile, file: str):
        sci_file = file.replace('[ext]', 'sci')
        err_file = file.replace('[ext]', 'err')
        print(f'Loading existing detection image for {tile}: {os.path.basename(sci_file)}')
        detec_sci = fits.getdata(sci_file)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            detec_wcs = WCS(fits.getheader(sci_file))
        detec_err = fits.getdata(err_file)
        self.detection_images[tile] = {'sci': detec_sci, 'err': detec_err, 'wcs': detec_wcs}
            

    def measure_random_aperture_scaling(
        self,
        filt, 
        min_radius: float, 
        max_radius: float, 
        num_radii: int,
        num_apertures_per_sq_arcmin: int,
        output_dir: str,
        plot: bool = False,
        overwrite: bool = False,
        psfmatched: bool = False,
        min_num_apertures_per_sq_arcmin: int = 30,
    ):

        if psfmatched:
            output_file = os.path.join(output_dir, f'randomApertures_{filt}_psfmatched_coeffs.pickle')
        else:
            output_file = os.path.join(output_dir, f'randomApertures_{filt}_coeffs.pickle')

        if os.path.exists(output_file) and not overwrite:
            print(f'Skipping random aperture measurement for {filt}')
            return

        if psfmatched:
            print(f'Measuring random aperture scaling for {filt}, psf-matched')
        else:
            print(f'Measuring random aperture scaling for {filt}')

        index = 2
        aperture_diameters = np.power(np.linspace(np.power(min_radius*2, 1/index), np.power(max_radius*2, 1/index), num_radii), index)

        images = self.get_images(filt=filt)

        print('Fitting pixel distribution')
        nrandom_pixels_per_tile = 50000
        fluxes = np.zeros(nrandom_pixels_per_tile*len(images))
        i = 0
        for image in tqdm.tqdm(images):
            pix = image.get_background_pixels(psfmatched)
            pix = np.random.choice(pix, size=nrandom_pixels_per_tile)
            fluxes[i:i+nrandom_pixels_per_tile] = pix
            i += nrandom_pixels_per_tile

        _, _, rms1 = _fit_pixel_distribution(fluxes, sigma_upper=1.0, maxiters=5) 


        print('Getting random aperture fluxes')
        # num_apertures_total = sum([int(num_apertures_per_sq_arcmin * image.area) for image in images])
        fluxes = {i:[] for i in range(len(aperture_diameters))}
        fluxes_random = {i:[] for i in range(len(aperture_diameters))}
        rms1_random = 1

        for image in tqdm.tqdm(images):

            if psfmatched: 
                nsci = image.psfmatched * np.sqrt(image.wht)
            else:
                nsci = image.sci * np.sqrt(image.wht)

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




        print('Fitting distribution')
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


        print('Fitting curve')        
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
            print('Plotting')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(3.5,3), constrained_layout=True)
            
            ax.errorbar(sqrtN, rmsN/rms1, yerr=rmsN_err/rms1, linewidth=0, marker='s', mfc='none', mec='k', mew=0.8, ms=5, elinewidth=0.6, ecolor='k', capsize=1.5, capthick=0.6, zorder=1000)
            x = np.linspace(0, 1.1*np.max(sqrtN), 1000)
            
            y = bspline(x)
            ax.plot(x, y, color='b', label='Spline')
            
            y = plaw(x)
            ax.plot(x, y, color='m', label='Power law')

            def func(N, alpha, beta):
                return alpha*np.power(N, beta)
            popt, pcov = curve_fit(func, N, rmsN_random/rms1_random, p0=[1,1], maxfev=int(1e5))
            ax.errorbar(sqrtN, rmsN_random/rms1_random, yerr=rmsN_err_random/rms1_random, linewidth=0, marker='s', mfc='none', mec='0.7', mew=0.8, ms=5, elinewidth=0.6, ecolor='0.7', capsize=1.5, capthick=0.6, zorder=1000)
            x = np.linspace(0, 1.1*np.max(sqrtN), 1000)
            y = func(x**2,*popt)
            ax.plot(x, y, color='lightblue')


            ax.set_xlim(0, np.max(np.sqrt(N))*1.1)
            ax.set_ylim(-0.03*np.max(rmsN/rms1), 1.1*np.max(rmsN/rms1))
            ax.set_xlabel(r'$\sqrt{N_{\rm pix}}$' + f' ({int(pixel_scale*1000)} mas)')
            ax.set_ylabel(r'$\sigma_N/\sigma_1$')

            ax.legend(loc='upper left', title=filt.upper(), frameon=False)

            ax.annotate('Gaussian random field', (0.95,0.05), xycoords='axes fraction', ha='right', va='bottom', color='0.7')

            out = os.path.join(os.path.dirname(output_file), 'plots', os.path.basename(output_file).replace('_coeffs.pickle', '.pdf'))
            print(f'Saving to {out}')
            plt.savefig(out)
            plt.close()

    @staticmethod
    def _get_random_aperture_curve(sqrtN, coeff_file):
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
            print(f'Applying random aperture calibration for {filt}')
            pixel_scale = self.get_images(filt=filt)[0].pixel_scale

            wht = self.objs_final[f'wht_{filt}']
            
            if self.has_auto:
                sqrtN_kron = np.sqrt(self.objs_final['kron1_area'])
            if self.has_aper:
                sqrtN_aper = np.sqrt(np.pi)*np.array(self.aperture_diameters)/pixel_scale/2

            if self.has_aper_native:
                # native resolution aperture photometry
                coeff_file = os.path.join(coeff_dir, f'randomApertures_{filt}_coeffs.pickle')
                result = self._get_random_aperture_curve(sqrtN_aper, coeff_file)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rmsN = np.outer(1/np.sqrt(wht), result)
                rmsN = (rmsN*u.nJy).to(u.Unit(self.flux_unit)).value
                self.objs_final.rename_column(f'e_aper_native_{filt}', f'e_aper_native_{filt}_uncal')
                self.objs_final[f'e_aper_native_{filt}'] = np.sqrt(self.objs_final[f'e_aper_native_{filt}_uncal']**2 + rmsN**2)
            
            if self.has_aper_psfmatched:
                # psf-matched aperture photometry
                coeff_file = os.path.join(coeff_dir, f'randomApertures_{filt}_psfmatched_coeffs.pickle')
                result = self._get_random_aperture_curve(sqrtN_aper, coeff_file)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rmsN = np.outer(1/np.sqrt(wht), result)
                rmsN = (rmsN*u.nJy).to(u.Unit(self.flux_unit)).value
                self.objs_final.rename_column(f'e_aper_psfmatched_{filt}', f'e_aper_psfmatched_{filt}_uncal')
                self.objs_final[f'e_aper_psfmatched_{filt}'] = np.sqrt(self.objs_final[f'e_aper_psfmatched_{filt}_uncal']**2 + rmsN**2)

            if self.has_auto:
                # auto (kron) photometry
                coeff_file = os.path.join(coeff_dir, f'randomApertures_{filt}_psfmatched_coeffs.pickle')
                result = self._get_random_aperture_curve(sqrtN_kron, coeff_file)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    rmsN = result/np.sqrt(wht)
                rmsN = (rmsN*u.nJy).to(u.Unit(self.flux_unit)).value
                self.objs_final.rename_column(f'e_auto_{filt}', f'e_auto_{filt}_uncal')
                self.objs_final[f'e_auto_{filt}'] = np.sqrt(self.objs_final[f'e_auto_{filt}_uncal']**2 + rmsN**2)


    def plot_detections():
        from regions import Regions, EllipseSkyRegion
        print(datetime.now().strftime('%H:%M:%S') + ': ' + 'Plotting apertures + segmap checkplot...')
        cmap2 = make_random_cmap(len(objs)+1)
        cmap2.colors[0] = colors.to_rgba(background_color)
        fig, ax = plt.subplots(1,2,figsize=(14,10),sharex=True,sharey=True)
        ax[0].imshow(np.log10(detec_sci), vmin=0, vmax=2.5, cmap=cmap1)
        ax[1].imshow(segmap, cmap=cmap2)

        regs = []
        center_sky = detec_wcs.pixel_to_world(objs['x'], objs['y'])
        width_sky = 2.5*kronrad*objs['a']*0.03
        height_sky = 2.5*kronrad*objs['b']*0.03
        for i in range(len(objs)):
            reg = EllipseSkyRegion(center=center_sky[i], width=width_sky[i]*u.arcsec, height=height_sky[i]*u.arcsec, angle=(np.degrees(objs['theta'][i])+20)*u.deg)

            e = Ellipse(xy=(objs['x'][i], objs['y'][i]), 
                        width=2.5*kronrad[i]*objs['a'][i], height=2.5*kronrad[i]*objs['b'][i], angle=np.degrees(objs['theta'][i]))
            e.set_facecolor('none')
            e.set_linewidth(0.15)
            if objs['mode'][i]=='cold':
                e.set_edgecolor('lime')
                reg.visual['color'] = 'green'
            else:
                e.set_edgecolor('lightblue')
                reg.visual['color'] = 'lightblue'
            if objs['flag'][i]>1:
                e.set_edgecolor('r')
                reg.visual['color'] = 'red'
            ax[0].add_artist(e)
            regs.append(reg)
        ax[0].axis('off')
        ax[1].axis('off')
        plt.savefig(f'/Dropbox/research/COSMOS-Web/catalog/checkplots/checkplot_apertures+segm_hot+cold_{tile}_{version}.pdf', 
                    dpi=1000)
        plt.close()

        regs = Regions(regs)
        regs.write(f'/Dropbox/research/COSMOS-Web/catalog/checkplots/regions_{tile}_{version}.reg', format='ds9', overwrite=True)

    # fits.writeto(detection_image_filename.replace('.fits','_segm.fits'), data=segmap, header=detec_hdr, overwrite=True)


    def compute_psf_corr_grid(
        self, 
        band,
        output_dir, 
        plot = False, 
        overwrite = False,
    ):
        if band != self.psfhom_filter:
            raise ValueError('Why are we computing PSF corrections for anything other than the target filter?')
            
        output_file = os.path.join(output_dir, f'{band}_psf_corr_grid.txt')
        if os.path.exists(output_file) and not overwrite:
            print(f'Skipping computing PSF correction for {band}')
            return output_file

        print(f'Computing PSF correction for {band}')

        from photutils.aperture import EllipticalAperture, aperture_photometry



        a = np.linspace(0.05, 2.0, 50)
        q = np.linspace(0.05, 1, 50)
        
        psf = self.get_images(filt=band)[0].psf
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
            plt.savefig(os.path.join(output_dir, 'plots', f'{band}_psf_corr_grid.pdf'))
            plt.close()

        return output_file


    def apply_psf_corrections(
        self,
        psf_corr_file, 
        bands, 
    ):
        print(f'Applying PSF correction for {bands}')
        from scipy.interpolate import griddata

        for band in bands:
            if not band in self.filters:
                raise ValueError

        for tile in self.tiles:

            a, q, c = np.loadtxt(psf_corr_file).T

            if self.has_auto:
                ai = self.objects[tile]['kron2_a']
                bi = self.objects[tile]['kron2_b']
                qi = bi/ai
                kron_psf_corr = griddata(np.array([a,q]).T, c, (ai,qi), fill_value=1.0)
                for band in bands:
                    self.objects[tile][f'f_auto_{band}'] *= kron_psf_corr
                    self.objects[tile][f'e_auto_{band}'] *= kron_psf_corr
            if self.has_aper_psfmatched:
                aperture_diameters = np.array(self.aperture_diameters)
                aper_psf_corr = griddata(np.array([a,q]).T, c, (aperture_diameters/2, [1]*len(aperture_diameters)), fill_value=1.0)
                for band in bands:
                    self.objects[tile][f'f_aper_psfmatched_{band}'] *= aper_psf_corr
                    self.objects[tile][f'e_aper_psfmatched_{band}'] *= aper_psf_corr
            # if self.has_aper_native:
                # raise ValueError('Need to handle this case more carefully...')
                # aper_psf_corr = griddata(np.array([a,q]).T, c, (self.aperture_diameters/2, [1]*len(self.aperture_diameters)), fill_value=1.0)
                # for band in bands:
                #     self.objects[tile][f'f_aper_psfmatched_{band}'] *= aper_psf_corr
                #     self.objects[tile][f'e_aper_psfmatched_{band}'] *= aper_psf_corr
    

    def merge_tiles(
        self, 
        matching_radius = 0.1,
        edge_mask = 300, # pixels
    ):
        from astropy.coordinates import SkyCoord
        
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


        print(f'Merging {len(self.tiles)} tiles...')

        tile0 = self.tiles[0]
        objs0 = self.objects[tile0]
        objs0['tile'] = tile0
        
        # remove objects from objs0 if they are >edge_mask pixels from a boundary
        height0, width0 = np.shape(self.detection_images[tile0]['sci'])
        objs0_distance_to_edge = distance_to_nearest_edge(objs0['x'], objs0['y'], width0, height0)
        objs0 = objs0[objs0_distance_to_edge > edge_mask]
        
        print(tile0, len(objs0))
        coords0 = SkyCoord(objs0['ra'], objs0['dec'], unit='deg')


        for tile1 in self.tiles[1:]:
            objs1 = self.objects[tile1]
            objs1['tile'] = tile1

            # remove objects from objs1 if they are >edge_mask pixels from a boundary
            height1, width1 = np.shape(self.detection_images[tile1]['sci'])
            objs1_distance_to_edge = distance_to_nearest_edge(objs1['x'], objs1['y'], width1, height1)
            objs1 = objs1[objs1_distance_to_edge > edge_mask]

            print(tile1, len(objs1))
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

            # # Handle matched cases:
            assert len(objs0_matched)==len(objs1_matched)
            objs0_distance_to_edge = distance_to_nearest_edge(objs0_matched['x'], objs0_matched['y'], width0, height0)
            objs1_distance_to_edge = distance_to_nearest_edge(objs1_matched['x'], objs1_matched['y'], width1, height1)
            which = np.argmax([objs0_distance_to_edge, objs1_distance_to_edge], axis=0)

            objs1 = vstack([objs0_unique, objs0_matched[which==0], objs1_matched[which==1], objs1_unique], join_type='exact')

            tile0 = tile1
            objs0 = objs1
            coords0 = SkyCoord(objs0['ra'], objs0['dec'], unit='deg')

        print(len(objs1))
        objs1['id'] = np.arange(len(objs1)).astype(int)
        self.objs_final = objs1

    def write(self, 
        output_file
    ):
        self.objs_final.write(output_file, overwrite=True)


    @property
    def has_final_catalog(self):
        return hasattr(self, 'objs_final')

    @property
    def has_detections(self):
        return not self.objects == {}

    @property
    def has_photometry(self):
        return self.has_auto or self.has_aper_native or self.has_aper_psfmatched
    
    @property
    def has_aper(self):
        return self.has_aper_native or self.has_aper_psfmatched


# fig, ax = plt.subplots()
# for tile in np.unique(t['tile']):
#     ax.scatter(t['ra'][t['tile']==tile], t['dec'][t['tile']==tile], label=tile)
# p = mpl.patches.Polygon(b1v, facecolor='none', edgecolor='k')
# ax.add_patch(p)
# p = mpl.patches.Polygon(b2v, facecolor='none', edgecolor='k')
# ax.add_patch(p)
# ax.invert_xaxis()
# plt.legend()
# plt.show()
