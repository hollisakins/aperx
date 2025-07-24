import os, sys
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
    """
    Parse mosaic configurations and return both all images and images grouped by tile.
    
    Returns:
        tuple: (all_images, images_by_tile)
            - all_images: Images collection containing all mosaic images
            - images_by_tile: dict mapping tile names to Images collections for that tile
    """
    tiles = config.tiles
    filters = config.filters
    
    all_images = Images([])

    if 'memmap' in config:
        memmap = config.memmap
    else:
        memmap = False

    for mosaic_spec in config.mosaics:
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

        all_images += images_i

    # Create per-tile image collections
    images_by_tile = {}
    for tile in tiles:
        tile_images = all_images.get(tile=tile)
        images_by_tile[tile] = tile_images

    return all_images, images_by_tile


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

    all_images, images_by_tile = _parse_mosaics(config)
    all_images.pprint(filenames=False)
    
    # Global PSF generation (operates on all images across all tiles)
    if config.psf_generation.run: 
        from .catalog import generate_psfs
        step_logger = utils.setup_logger(name='aperx.psf_generation')

        generate_psfs(
            images=all_images,
            filters=config.filters,
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
            logger = step_logger
        )
    
    def process_tile(tile, overwrite=None, return_catalog=False):
        tile_logger = utils.setup_logger(name=f'aperx.{tile}')
        tile_logger.info(f'========================== Tile {tile} ==========================')

        catalog_file = os.path.join(config.output_dir, config.output_filename.replace('.fits',f'_{tile}.fits'))
        if overwrite is None:
            overwrite = config.overwrite
        if os.path.exists(catalog_file) and not overwrite:
            tile_logger.info(f"Loading existing catalog {catalog_file}")
            catalog = Catalog.load(catalog_file)
            if return_catalog:
                return catalog
            else:
                del catalog
                return
        
        tile_logger.info(f'Initializing catalog')
        tile_images = images_by_tile[tile]
        catalog = Catalog(tile_images, flux_unit=config.flux_unit, tile=tile)

        # PSF homogenization
        if config.psf_homogenization.run:
            step_logger = utils.setup_logger(name=f'aperx.{tile}.psf_homogenization')
            catalog.logger = step_logger

            target_filter = utils.validate_param(config.psf_homogenization, 'target_filter', default='f444w', message="'psf_homogenization.target_filter ' not specified, defaulting to f444w", logger=step_logger)
            inverse_filters = utils.validate_param(config.psf_homogenization, 'inverse_filters', default=[], message="'psf_homogenization.inverse_filters' not specified, defaulting to []", logger=step_logger)
            reg_fact = utils.validate_param(config.psf_homogenization, 'reg_fact', default=3e-3, message="'psf_homogenization.reg_fact' not specified, defaulting to 3e-3", logger=step_logger)
            overwrite = utils.validate_param(config.psf_homogenization, 'overwrite', default=False, message="'psf_homogenization.overwrite' not specified, defaulting to False", logger=step_logger)
            
            catalog.generate_psfhomogenized_images(
                target_filter, 
                inverse_filters=inverse_filters,
                reg_fact=reg_fact,
                overwrite=overwrite,
            )

        tile_logger.info(f'Starting source detection')
        if config.source_detection.run: 
            step_logger = utils.setup_logger(name=f'aperx.{tile}.source_detection')
            catalog.logger = step_logger
            
            sigma_upper = utils.validate_param(config.source_detection, 'sigma_upper', default=1.0, message="'source_detection.sigma_upper' not specified, defaulting to 1.0", logger=step_logger)
            maxiters = utils.validate_param(config.source_detection, 'maxiters', default=3, message="'source_detection.maxiters' not specified, defaulting to 3", logger=step_logger)

            detec_success = catalog.build_detection_image(
                tile = tile,
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
                return catalog

            # utils.validate_param(config.source_detection, 'detection_scheme', message="`source_detection.detection_scheme` must be specified")

            detection_scheme = config.source_detection.detection_scheme
            save_segmap = config.source_detection.save_segmap
            kwargs = dict(config.source_detection)
            del kwargs['run'], kwargs['plot'], kwargs['detection_image_dir'], kwargs['detection_image_filename'], kwargs['overwrite_detection_image'], kwargs['detection_image_method'], kwargs['detection_image_filters'], kwargs['detection_image_psf_homogenized'], kwargs['detection_scheme'], kwargs['save_segmap']

            catalog.detect_sources(
                detection_scheme = detection_scheme,
                save_segmap = save_segmap,
                **kwargs,
            )

            if config.secondary_source_detection.run: 
                pass

            # Compute the kron radius
            catalog.compute_kron_radius(windowed = False)

            windowed_positions = utils.validate_param(config.source_detection, 'windowed_positions', default=False, message="source_detection.windowed_positions not specified, defaulting to False", logger=step_logger)
            if windowed_positions:
                catalog.compute_windowed_positions()
                catalog.compute_kron_radius(windowed = True)
            

            catalog.add_ra_dec()
            catalog.compute_weights()

            if config.source_detection.compute_rhalf:
                catalog.compute_rhalf()

            if config.source_detection.plot: 
                output_dir = config.output_dir
                catalog.plot_detections(output_dir)

            # cat.compute_compactness()
            # cat.compute_rhalf()


        if config.photometry.run: 
            step_logger = utils.setup_logger(name=f'aperx.{tile}.photometry')
            catalog.logger = step_logger
            
            if catalog.objects is None:
                step_logger.warning('Cannot run photometry on a catalog without detections! Did you forget to enable source_detection?')
            else:
                    
                if config.photometry.aper.run_native: 
                    aperture_diameters = config.photometry.aper.aperture_diameters

                    catalog.compute_aper_photometry(
                        aperture_diameters = aperture_diameters, 
                        psfhom = False,
                        flux_unit = config.flux_unit,
                    )

                if config.photometry.aper.run_psfhom: 
                    aperture_diameters = config.photometry.aper.aperture_diameters

                    catalog.compute_aper_photometry(
                        aperture_diameters = aperture_diameters, 
                        psfhom = True,
                        flux_unit = config.flux_unit,
                    )


                if config.photometry.auto.run: 

                    catalog.compute_auto_photometry(
                        kron_params=config.photometry.auto.kron_params,
                        flux_unit = config.flux_unit,
                    )

                    if config.photometry.auto.kron_corr: 
                        catalog.apply_kron_corr(
                            filt = config.photometry.auto.kron_corr_filter, 
                            thresh = config.photometry.auto.kron_corr_thresh, 
                            kron_params1 = config.photometry.auto.kron_params,
                            kron_params2 = config.photometry.auto.kron_corr_params,
                        )
        
        # Write output to file
        catalog.write(output_file = catalog_file)

        if return_catalog:
            return catalog
        else:
            del catalog
            return
    
    # Process each tile
    if config.parallel:
        import multiprocessing
        n_procs = int(multiprocessing.cpu_count()/2)
        
        with mp.Pool(n_procs) as pool:
            for tile in config.tiles:
                pool.apply_async(process_tile, args = (tile, ))
            pool.close()
            pool.join()
    else:
        for tile in config.tiles:
            process_tile(tile)

        
    # Reload tiles from FITS files
    catalogs = {}
    for tile in config.tiles:
        catalogs[tile] = process_tile(tile, overwrite=False, return_catalog=True)
        


    # Post-processing with merged catalog
    merged_catalog = None
    if config.post_processing.run: 
        step_logger = utils.setup_logger(name='aperx.post_processing')

        if config.post_processing.merge_tiles.run:
            # Check if catalogs have detections
            has_detections = all(catalog.has_detections for catalog in catalogs.values())
            if not has_detections:
                step_logger.warning('Cannot merge tiles without detections! Did you forget to enable source_detection?')
            else:
                from .catalog import merge_catalogs
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
                merged_catalog = merge_catalogs(
                    catalogs, 
                    matching_radius=matching_radius, 
                    edge_mask=edge_mask,
                    logger=step_logger
                )
                del catalogs

        if config.post_processing.random_apertures.run and merged_catalog is not None: 
            merged_catalog.logger = step_logger
            merged_catalog.measure_random_aperture_scaling(
                min_radius = config.post_processing.random_apertures.min_radius,
                max_radius = config.post_processing.random_apertures.max_radius,
                num_radii = config.post_processing.random_apertures.num_radii,
                num_apertures_per_sq_arcmin = config.post_processing.random_apertures.num_apertures_per_sq_arcmin,
                plot = config.post_processing.random_apertures.checkplots,
                overwrite = config.post_processing.random_apertures.overwrite,
                output_dir = config.output_dir,
            )
            
            if merged_catalog.has_photometry:
                merged_catalog.apply_random_aperture_error_calibration(coeff_dir=config.output_dir)


        if config.post_processing.psf_corrections.run and merged_catalog is not None: 
            merged_catalog.logger = step_logger
            psf_corr_file = merged_catalog.compute_psf_corr_grid(
                filt = config.psf_homogenization.target_filter, 
                output_dir = config.output_dir, 
                overwrite = config.post_processing.psf_corrections.overwrite,
                plot = config.post_processing.psf_corrections.checkplots)

            if merged_catalog.has_photometry:
                merged_catalog.apply_psf_corrections(psf_corr_file)

    merged_catalog.set_missing_to_nan()

    # Write final catalog
    catalog_file = os.path.join(config.output_dir, config.output_filename)
    if merged_catalog is None:
        base_logger.warning('Could not write catalog, no final merged catalog available. Something went wrong?')
    else:
        merged_catalog.write(output_file = catalog_file)

        
        

if __name__ == '__main__':
    sys.exit(main())
