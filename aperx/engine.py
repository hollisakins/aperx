import os
import multiprocess as mp
from functools import partial
from .catalog import Catalog
from . import utils
import argparse
# import rich
from rich.traceback import install
install()


###############################################################################################

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
    

    cat = Catalog(config, config.filters, config.tiles, config.psf_homogenization.target_filter, flux_unit=config.flux_unit)
    cat.pprint()

    base_logger = utils.setup_logger(name='aperx')

    if config.psf_generation.run: 
        step_logger = utils.setup_logger(name='aperx.psf_generation')

        if cat.all_psfs_exist and not config.psf_generation.overwrite:
            print('All necessary PSFs exist already, no need to generate?')
        else:
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

    if config.psf_homogenization.run:
        # homogenize PSFs
        target_filter = config.psf_homogenization.target_filter 
        overwrite = config.psf_homogenization.overwrite 
        reg_fact = config.psf_homogenization.reg_fact

        if target_filter not in config.filters:
            raise ValueError(f'PSF homogenization: target filter must be in config.filters = {config.filters}')

        for tile in cat.images:
            target_image = cat.images[tile][target_filter]
            images = [cat.images[tile][f] for f in cat.images[tile] if f != target_filter]
            
            for image in images:
                image.generate_psfmatched_image(
                    target_filter = target_filter,
                    target_psf_file = target_image.psf_file,
                    reg_fact = reg_fact, 
                    overwrite = overwrite
                )

    if config.source_detection.run: 
        
        for tile in cat.tiles:
            detec_file = config.source_detection.detection_image.replace('[tile]', tile)
            detec_sci_file = detec_file.replace('[ext]', 'sci')
            detec_err_file = detec_file.replace('[ext]', 'err')
            both_files_exist = os.path.exists(detec_sci_file) and os.path.exists(detec_err_file)

            if not both_files_exist or config.source_detection.overwrite_detection_image:
                cat.build_detection_image(
                    tile = tile, 
                    file = detec_file,
                    type = config.source_detection.detection_image_type,
                    filters = config.source_detection.detection_image_filters,
                    psfmatched = config.source_detection.detection_image_psfmatched,
                )
            else:
                cat.load_detection_image(tile, detec_file)

        assert 'detection_scheme' in config.source_detection, "`source_detection.detection_scheme` must be specified"
        detection_scheme = config.source_detection.detection_scheme
        kwargs = dict(config.source_detection)
        del kwargs['run'], kwargs['detection_image'], kwargs['overwrite_detection_image'], kwargs['detection_image_type'], kwargs['detection_image_filters'], kwargs['detection_image_psfmatched'], kwargs['detection_scheme']

        for tile in cat.tiles:
            cat.detect_sources(
                tile,
                detection_scheme = detection_scheme,
                **kwargs,
            )
                
            pos_key = None
            cat.compute_kron_radius(tile, pos_key=pos_key)

            if config.source_detection.windowed_positions:
                cat.compute_windowed_positions(tile)
                pos_key = 'windowed'
                cat.compute_kron_radius(tile, pos_key=pos_key)

            cat.add_ra_dec(tile, pos_key=pos_key)
            cat.compute_weights(tile, pos_key=pos_key)

        if config.secondary_source_detection.run: 
            pass

    if config.photometry.run: 
        if not cat.has_detections:
            raise ValueError('Cannot run photometry on a catalog without detections! Did you forget to enable source_detection?')
            
        for tile in cat.tiles:

            if config.photometry.aper.run_native: 
                aperture_diameters = config.photometry.aper.aperture_diameters

                cat.compute_aper_photometry(
                    tile, 
                    aperture_diameters = aperture_diameters, 
                    psfmatched = False,
                    pos_key = pos_key,
                    flux_unit = config.flux_unit,
                )

            if config.photometry.aper.run_psfmatched: 
                aperture_diameters = config.photometry.aper.aperture_diameters

                cat.compute_aper_photometry(
                    tile, 
                    aperture_diameters = aperture_diameters, 
                    psfmatched = True,
                    pos_key = pos_key,
                    flux_unit = config.flux_unit,
                )


            if config.photometry.auto.run: 

                cat.compute_auto_photometry(
                    tile, 
                    kron_params=config.photometry.auto.kron_params,
                    pos_key = pos_key,
                    flux_unit = config.flux_unit,
                )

                if config.photometry.auto.kron_corr: 
                    cat.apply_kron_corr(
                        tile, 
                        band = config.photometry.auto.kron_corr_band, 
                        thresh = config.photometry.auto.kron_corr_thresh, 
                        kron_params1 = config.photometry.auto.kron_params,
                        kron_params2 = config.photometry.auto.kron_corr_params,
                        pos_key = pos_key,
                    )
        




    if config.post_processing.run: 

        if config.post_processing.merge_tiles.run:
            cat.merge_tiles()

        if config.post_processing.random_apertures.run: 

            for filt in cat.filters:
                if config.photometry.aper.run_native:
                    cat.measure_random_aperture_scaling(
                        filt, 
                        min_radius = config.post_processing.random_apertures.min_radius,
                        max_radius = config.post_processing.random_apertures.max_radius,
                        num_radii = config.post_processing.random_apertures.num_radii,
                        num_apertures_per_sq_arcmin = config.post_processing.random_apertures.num_apertures_per_sq_arcmin,
                        plot = config.post_processing.random_apertures.checkplots,
                        overwrite = config.post_processing.random_apertures.overwrite,
                        output_dir = config.output_dir,
                        psfmatched = False,
                    )
                if config.photometry.aper.run_psfmatched or config.photometry.auto.run:
                    cat.measure_random_aperture_scaling(
                        filt, 
                        min_radius = config.post_processing.random_apertures.min_radius,
                        max_radius = config.post_processing.random_apertures.max_radius,
                        num_radii = config.post_processing.random_apertures.num_radii,
                        num_apertures_per_sq_arcmin = config.post_processing.random_apertures.num_apertures_per_sq_arcmin,
                        plot = config.post_processing.random_apertures.checkplots,
                        overwrite = config.post_processing.random_apertures.overwrite,
                        output_dir = config.output_dir,
                        psfmatched = True,
                    )
            
            if cat.has_final_catalog and cat.has_photometry:
                cat.apply_random_aperture_error_calibration(coeff_dir=config.output_dir)


        if config.post_processing.psf_corrections.run: 
            
            band = config.psf_homogenization.target_filter
            output_dir = config.output_dir
            overwrite = config.post_processing.psf_corrections.overwrite
            plot = config.post_processing.psf_corrections.checkplots

            psf_corr_file = cat.compute_psf_corr_grid(
                band, 
                output_dir, 
                overwrite=overwrite,
                plot=plot)

            if cat.has_final_catalog and cat.has_photometry:
                cat.apply_psf_corrections(
                    psf_corr_file, 
                    bands = config.filters, 
                )

    if not cat.has_final_catalog:
        print('Could not write catalog, no final merged catalog available. Something went wrong?')
    else:
        cat.write(output_file = os.path.join(config.output_dir, config.output_filename))
        

if __name__ == '__main__':
    sys.exit(main())
