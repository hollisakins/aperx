import os, sys
os.sched_setaffinity(0,range(48))
import multiprocess as mp
from functools import partial
from .catalog import Catalog
from . import utils
from .image import Image, Images
import argparse
# import rich
from rich.traceback import install
install()


###############################################################################################
def _parse_mosaics(config):
    tiles = config.tiles
    filters = config.filters
    
    images = Images([])
    # images = {}
    # for tile in self.tiles:
    #     images[tile] = {}
    #     for filt in self.filters:

    if 'memmap' in config:
        memmap = config.memmap
    else:
        memmap = False

    for mosaic_spec in config.mosaics:
        # If filter is already specified, skip it 
        # if filt in images[tile]: 
            # continue

        kwargs = {'memmap':memmap}

        # If mosaic spec applies only to certain tiles, use those; otherwise, use the global tile spec
        if 'tiles' in mosaic_spec:
            kwargs['tile'] = [x for x in mosaic_spec.tiles if x in config.tiles]
        else:
            kwargs['tile'] = config.tiles
            
        # If mosaic spec applies only to certain filters, use those; otherwise, use the global filter spec
        if 'filters' in mosaic_spec:
            kwargs['filter'] = [x for x in mosaic_spec.filters if x in config.filters]
        else:
            kwargs['filter'] = config.filters

        keywords = ['version', 'pixel_scale', 'sci_extension', 
                    'err_extension', 'wht_extension', 'psf_extension',
                    'psf_tile']
        for keyword in keywords:
            if keyword in mosaic_spec:
                kwargs[keyword] = getattr(mosaic_spec, keyword)

        image_patterns = mosaic_spec.patterns
        for image_pattern in image_patterns:
            if not image_pattern.endswith('.fits'):
                image_pattern += '.fits'

        images_i = Images.load(
            image_patterns,
            **kwargs
        )

        images += images_i

    return images


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()
    # TODO implement arguments overrides to certain config options?
    
    config_file = args.config
    if not os.path.exists(config_file):
        raise FileNotFoundError(f'Config file {config_file} not found')

    config = utils.parse_config_file(config_file)
    utils.config = config

    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    
    base_logger = utils.setup_logger(name='aperx')

    images = _parse_mosaics(config)
    images.pprint(filenames=False)
    

    cat = Catalog(
        images, 
        flux_unit = config.flux_unit, 
    )

    # PSF generation
    if config.psf_generation.run: 
        step_logger = utils.setup_logger(name='aperx.psf_generation')
        cat.logger = step_logger

        cat.generate_psfs(
            fwhm_min = config.psf_generation.fwhm_min,
            fwhm_max = config.psf_generation.fwhm_max,
            max_ellip = config.psf_generation.max_ellip,
            min_snr = config.psf_generation.min_snr,
            max_snr = config.psf_generation.max_snr,
            psf_size = config.psf_generation.psf_size,
            checkplots = config.psf_generation.checkplots,
            overwrite = config.psf_generation.overwrite,
            plot = config.psf_generation.plot,
            az_average = config.psf_generation.az_average,

        )
    
    def process_tile(tile):
        base_logger.info(f'========================== Tile {tile} ==========================')

        # PSF homogenization
        if config.psf_homogenization.run:
            step_logger = utils.setup_logger(name='aperx.psf_homogenization')
            cat.logger = step_logger

            target_filter = utils.validate_param(config.psf_homogenization, 'target_filter', default='f444w', message="'psf_homogenization.target_filter ' not specified, defaulting to f444w", logger=step_logger)
            inverse_filters = utils.validate_param(config.psf_homogenization, 'inverse_filters', default=[], message="'psf_homogenization.inverse_filters' not specified, defaulting to []", logger=step_logger)
            reg_fact = utils.validate_param(config.psf_homogenization, 'reg_fact', default=3e-3, message="'psf_homogenization.reg_fact' not specified, defaulting to 3e-3", logger=step_logger)
            overwrite = utils.validate_param(config.psf_homogenization, 'overwrite', default=False, message="'psf_homogenization.overwrite' not specified, defaulting to False", logger=step_logger)
            
            cat.generate_psfhomogenized_images(
                tile,
                target_filter, 
                inverse_filters=inverse_filters,
                reg_fact=reg_fact,
                overwrite=overwrite,
            )


        if config.source_detection.run: 
            step_logger = utils.setup_logger(name='aperx.source_detection')
            cat.logger = step_logger
            
            sigma_upper = utils.validate_param(config.source_detection, 'sigma_upper', default=1.0, message="'source_detection.sigma_upper' not specified, defaulting to 1.0", logger=step_logger)
            maxiters = utils.validate_param(config.source_detection, 'maxiters', default=3, message="'source_detection.maxiters' not specified, defaulting to 3", logger=step_logger)

            detec_success = cat.build_detection_image(
                tile,
                output_dir = config.source_detection.detection_image_dir,
                output_filename = config.source_detection.detection_image_filename,
                method = config.source_detection.detection_image_method,
                filters = config.source_detection.detection_image_filters,
                overwrite = config.source_detection.overwrite_detection_image, 
                psfhom = config.source_detection.detection_image_psf_homogenized,
                sigma_upper = sigma_upper, 
                maxiters = maxiters,
            )

            # Check if detection image creation was successful
            if not detec_success:
                step_logger.warning(f'Skipping catalog generation for tile {tile}')
                return

            # utils.validate_param(config.source_detection, 'detection_scheme', message="`source_detection.detection_scheme` must be specified")

            detection_scheme = config.source_detection.detection_scheme
            save_segmap = config.source_detection.save_segmap
            kwargs = dict(config.source_detection)
            del kwargs['run'], kwargs['plot'], kwargs['detection_image_dir'], kwargs['detection_image_filename'], kwargs['overwrite_detection_image'], kwargs['detection_image_method'], kwargs['detection_image_filters'], kwargs['detection_image_psf_homogenized'], kwargs['detection_scheme'], kwargs['save_segmap']

            cat.detect_sources(
                tile,
                detection_scheme = detection_scheme,
                save_segmap = save_segmap,
                **kwargs,
            )

            if config.secondary_source_detection.run: 
                pass

            # Compute the kron radius
            cat.compute_kron_radius(tile, windowed = False)

            windowed_positions = utils.validate_param(config.source_detection, 'windowed_positions', default=False, message="source_detection.windowed_positions not specified, defaulting to False", logger=step_logger)
            if windowed_positions:
                cat.compute_windowed_positions(tile)
                cat.compute_kron_radius(tile, windowed = True)
            

            cat.add_ra_dec(tile)
            cat.compute_weights(tile)

            if config.source_detection.compute_rhalf:
                cat.compute_rhalf(tile)

            if config.source_detection.plot: 
                output_dir = config.output_dir
                cat.plot_detections(tile, output_dir)

            # cat.compute_compactness()
            # cat.compute_rhalf()


        if config.photometry.run: 
            step_logger = utils.setup_logger(name='aperx.photometry')
            cat.logger = step_logger
            
            if not tile in cat.objects:
                step_logger.warning('Cannot run photometry on a catalog without detections! Did you forget to enable source_detection?')
            else:
                    
                if config.photometry.aper.run_native: 
                    aperture_diameters = config.photometry.aper.aperture_diameters

                    cat.compute_aper_photometry(
                        tile,
                        aperture_diameters = aperture_diameters, 
                        psfhom = False,
                        flux_unit = config.flux_unit,
                    )

                if config.photometry.aper.run_psfhom: 
                    aperture_diameters = config.photometry.aper.aperture_diameters

                    cat.compute_aper_photometry(
                        tile,
                        aperture_diameters = aperture_diameters, 
                        psfhom = True,
                        flux_unit = config.flux_unit,
                    )


                if config.photometry.auto.run: 

                    cat.compute_auto_photometry(
                        tile,
                        kron_params=config.photometry.auto.kron_params,
                        flux_unit = config.flux_unit,
                    )

                    if config.photometry.auto.kron_corr: 
                        cat.apply_kron_corr(
                            tile,
                            filt = config.photometry.auto.kron_corr_filter, 
                            thresh = config.photometry.auto.kron_corr_thresh, 
                            kron_params1 = config.photometry.auto.kron_params,
                            kron_params2 = config.photometry.auto.kron_corr_params,
                        )
    
    if config.parallel:
        import multiprocessing
        n_procs = int(multiprocessing.cpu_count()/2)

        with mp.Pool(n_procs) as pool:
            pool.map(process_tile, cat.tiles)
    else:
        for tile in cat.tiles:
            process_tile(tile)
        


    if config.post_processing.run: 
        step_logger = utils.setup_logger(name='aperx.post_processing')
        cat.logger = step_logger

        cat.tiles = cat.detection_images.tiles

        if config.post_processing.merge_tiles.run:
            if not cat.has_detections:
                step_logger.warning('Cannot merge tiles without detections! Did you forget to enable source_detection?')

            else:
                matching_radius = utils.validate_param(
                    config.post_processing.merge_tiles, 
                    'matching_radius', 
                    default=0.1, 
                    message="post_processing.merge_tiles.matching_radius not specified, defaulting to 0.1", 
                    logger=step_logger
                )
                edge_mask = utils.validate_param(
                    config.post_processing.merge_tiles, 
                    'edge_mask', 
                    default=300, 
                    message="post_processing.merge_tiles.matching_radius not specified, defaulting to 300", 
                    logger=step_logger
                )
                cat.merge_tiles(matching_radius=matching_radius, edge_mask=edge_mask)

        if config.post_processing.random_apertures.run: 
            cat.measure_random_aperture_scaling(
                min_radius = config.post_processing.random_apertures.min_radius,
                max_radius = config.post_processing.random_apertures.max_radius,
                num_radii = config.post_processing.random_apertures.num_radii,
                num_apertures_per_sq_arcmin = config.post_processing.random_apertures.num_apertures_per_sq_arcmin,
                plot = config.post_processing.random_apertures.checkplots,
                overwrite = config.post_processing.random_apertures.overwrite,
                output_dir = config.output_dir,
            )
            
            if cat.has_final_catalog and cat.has_photometry:
                cat.apply_random_aperture_error_calibration(coeff_dir=config.output_dir)


        if config.post_processing.psf_corrections.run: 
            
            psf_corr_file = cat.compute_psf_corr_grid(
                filt = config.psf_homogenization.target_filter, 
                output_dir = config.output_dir, 
                overwrite = config.post_processing.psf_corrections.overwrite,
                plot = config.post_processing.psf_corrections.checkplots)

            if cat.has_final_catalog and cat.has_photometry:
                cat.apply_psf_corrections(psf_corr_file)

    catalog_file = os.path.join(config.output_dir, config.output_filename)
    if not cat.has_final_catalog:
        base_logger.warning('Could not write catalog, no final merged catalog available. Something went wrong?')
    else:
        cat.write(output_file = catalog_file)

        
        

if __name__ == '__main__':
    sys.exit(main())
