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
    

    cat = Catalog(config)
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
                psf_size = config.psf_generation.psf_size,
                checkplots = config.psf_generation.checkplots,
                overwrite = config.psf_generation.overwrite,
                plot = config.psf_generation.plot,
            )

    if config.psf_homogenization.run:
        # homogenize PSFs
        target_filter = config.psf_homogenization.target_filter 
        overwrite = config.psf_homogenization.overwrite 

        if target_filter not in config.filters:
            raise ValueError(f'PSF homogenization: target filter must be in config.filters = {config.filters}')

        for tile in cat.images:
            target_image = cat.images[tile][target_filter]
            images = [cat.images[tile][f] for f in cat.images[tile] if f != target_filter]
            
            for image in images:
                image.generate_psfmatched_image(
                    target_filter = target_filter,
                    target_psf_file = target_image.psf_file,
                    overwrite = overwrite
                )

    if config.source_detection.run: 
        
        detec_file = config.source_detection.detection_image
        if not os.path.exists(detec_file) or config.source_detection.overwrite_detection_image:
            catalog.build_detection_image(
                file = detec_file,
                type = config.source_detection.detection_image_type,
                filters = config.source_detection.detection_image_filters,
                psfmatched = config.source_detection.detection_image_psfmatched,
            )
        else:
            catalog.load_detection_image(detec_file)

        catalog.source_detection(config.source_detection)
    
        if config.secondary_source_detection.run: 
            pass

    if config.photometry.run: 
        pass

    if config.post_processing.run: 

        if config.post_processing.random_apertures.run: 
            pass

        if config.post_processing.psf_corrections.run: 
            pass
        

if __name__ == '__main__':
    sys.exit(main())
