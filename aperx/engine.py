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


    base_logger = utils.setup_logger(name='aperx')

    if config.psf_generation.run: 
        step_logger = utils.setup_logger(name='aperx.psf_generation')
        if cat.all_psfs_exist:
            logger.warning('All necessary PSFs exist already, no need to generate?')
        
        psf_size = config.psf_generation.psf_size
        overwrite = config.psf_generation.overwrite
        
        cat.generate_psfs(
            psf_size=psf_size, 
            overwrite=overwrite)

    if config.psf_homogenization.run:
        # homogenize PSFs
        target_filter = config.psf_homogenization.target_filter 
        overwrite = config.psf_homogenization.overwrite 

        if target_filter not in config.filters:
            raise ValueError(f'PSF homogenization: target filter must be in {config.fitlers}')

        target_images = cat.get_images(filt=target_filter)
        psf_files = [image.psf_file for image in target_images]
        if len(set(psf_files)) != 1:
            raise ValueError('Target filter has differnt PSF files, something went awry')
        target_psf_file = psf_files[0]

        
        for image in cat.images:
            image.generate_psfmatched_image(
                target_filter = target_filter,
                target_psf_file = target_psf_file, 
                overwrite = overwrite
            )

    if config.source_detection.run: 
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
