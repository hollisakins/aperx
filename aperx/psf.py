from .image import Image
import os
from typing import List
import numpy as np
import subprocess

nominal_psf_fwhms = {
    'f070w': 0.023,
    'f090w': 0.030,
    'f115w': 0.037,
    'f140m': 0.046,
    'f150w': 0.049,
    'f162m': 0.053,
    'f164n': 0.054,
    'f150w2': 0.045,
    'f182m': 0.060,
    'f187n': 0.061,
    'f200w': 0.064,
    'f210m': 0.068,
    'f212n': 0.069,
    'f250m': 0.082,
    'f277w': 0.088,
    'f300m': 0.097,
    'f322w2': 0.096,
    'f323n': 0.106,
    'f335m': 0.109,
    'f356w': 0.114,
    'f360m': 0.118,
    'f405n': 0.132,
    'f410m': 0.133,
    'f430m': 0.139,
    'f444w': 0.140,
    'f460m': 0.151,
    'f466n': 0.152,
    'f470n': 0.154,
    'f480m': 0.157,
}

def _run_se(detec_image, weight_image, output_cat, vignet_size, sex_install=None):
    param_file = output_cat.replace('.fits', f'.param')
    se_run_script = output_cat.replace('.fits', f'.sh')

    with open(param_file, 'w') as param:
        param.write('NUMBER\n')
        param.write('FLAGS\n')
        param.write('X_IMAGE\n')
        param.write('Y_IMAGE\n')
        param.write('X2_IMAGE\n')
        param.write('Y2_IMAGE\n')
        param.write('XY_IMAGE\n')
        param.write('ERRX2_IMAGE\n')
        param.write('ERRY2_IMAGE\n')
        param.write('ERRXY_IMAGE\n')
        param.write('X_WORLD\n')
        param.write('Y_WORLD\n')
        param.write('X2_WORLD\n')
        param.write('Y2_WORLD\n')
        param.write('XY_WORLD\n')
        param.write('ERRX2_WORLD\n')
        param.write('ERRY2_WORLD\n')
        param.write('ERRXY_WORLD\n')
        param.write('ALPHA_J2000\n')
        param.write('DELTA_J2000\n')
        param.write('A_IMAGE\n')
        param.write('B_IMAGE\n')
        param.write('ERRA_IMAGE\n')
        param.write('ERRB_IMAGE\n')
        param.write('A_WORLD\n')
        param.write('B_WORLD\n')
        param.write('ERRA_WORLD\n')
        param.write('ERRB_WORLD\n')
        param.write('THETA_IMAGE\n')
        param.write('ERRTHETA_IMAGE\n')
        param.write('THETA_WORLD\n')
        param.write('ERRTHETA_WORLD\n')
        param.write('THETA_J2000\n')
        param.write('ERRTHETA_J2000\n')
        param.write('ELONGATION\n')
        param.write('ELLIPTICITY\n')
        param.write('FLUX_APER\n')
        param.write('FLUXERR_APER\n')
        param.write('MAG_APER\n')
        param.write('MAGERR_APER\n')
        param.write('FLUX_AUTO\n')
        param.write('FLUXERR_AUTO\n')
        param.write('MAG_AUTO\n')
        param.write('MAGERR_AUTO\n')
        param.write('FLUX_RADIUS\n')
        param.write('KRON_RADIUS\n')
        param.write('MU_MAX\n')
        param.write('CLASS_STAR\n')
        param.write('FWHM_WORLD\n')
        param.write('SNR_WIN\n')
        param.write(f'VIGNET({vignet_size},{vignet_size})\n')
    
    if sex_install is None:
        sex_install = 'sex'
    
    print(f'SE run script {se_run_script}')
    
    with open(se_run_script, 'w') as script:
        script.write('') #...
        script.write(f"{sex_install} '{detec_image}' ")
        script.write(f"-CATALOG_NAME '{output_cat}' ")
        script.write(f"-CATALOG_TYPE FITS_LDAC ")
        script.write(f"-PARAMETERS_NAME '{param_file}' ")
        script.write(f"-DETECT_MINAREA 8 ")
        script.write(f"-DETECT_THRESH 3.0 ")
        script.write(f"-ANALYSIS_THRESH 2.5 ")
        script.write(f"-FILTER Y ")
        script.write(f"-FILTER_NAME '/home/hakins/COSMOS-Web/se++/conv/gauss_5.0_9x9.conv' ")
        script.write(f"-DEBLEND_NTHRESH 32 ")
        script.write(f"-DEBLEND_MINCONT 0.001 ")
        script.write(f"-CLEAN Y ")
        script.write(f"-CLEAN_PARAM 1.0 ")
        script.write(f"-MASK_TYPE CORRECT ")
        script.write(f"-WEIGHT_IMAGE '{weight_image}' ")
        script.write(f"-WEIGHT_TYPE MAP_WEIGHT ")
        script.write(f"-RESCALE_WEIGHTS Y ")
        script.write(f"-WEIGHT_GAIN N ")
        script.write(f"-PHOT_APERTURES 15 ")
        script.write(f"-PHOT_AUTOPARAMS 2.5,1.0 ")
        script.write(f"-PHOT_PETROPARAMS 2.0,1.0 ")
        script.write(f"-PHOT_AUTOAPERS 0.1,0.1 ")
        script.write(f"-PHOT_FLUXFRAC 0.5 ")
        script.write(f"-MAG_ZEROPOINT 28.9 ")
        script.write(f"-GAIN 1.0 ")
        script.write(f"-PIXEL_SCALE 0 ")
        script.write(f"-SEEING_FWHM 0.1 ")
        script.write(f"-BACK_TYPE AUTO ")
        script.write(f"-BACK_SIZE 256 ")
        script.write(f"-BACK_FILTERSIZE 3 ")
        script.write(f"-BACKPHOTO_TYPE LOCAL ")
        script.write(f"-BACKPHOTO_THICK 64 ")
        script.write(f"-BACK_FILTTHRESH 0.0 ")

    subprocess.run(['bash', se_run_script])

    # os.remove(param_file)
    # os.remove(se_run_script)


def _run_psfex(
        input_catalog, 
        output_filename, 
        fwhm_min=1.5, 
        fwhm_max=4.0,
        max_ellip=0.1,
        min_snr=8.0,
        psf_size=301,
        checkplots=False,
    ):
    """
    Run PSFEx on a catalog.
    """
    randn = np.random.randint(0, 10000)
    # psfex_run_script = os.path.join(config.temp_path, f'psfex_run_script_{randn}.sh')

    with open(psfex_run_script_name, 'w') as script:
        script.write(f'psfex {input_catalog}') # /softs/astromatic/psfex/3.22.1/bin/psfex $SEcat
        script.write(f'      -c /home/hakins/PSFEx/default.psfex')
        script.write(f'      -PSF_SIZE {psf_size},{psf_size}')
        script.write(f'      -PSFVAR_DEGREES 0')
        script.write(f'      -PSFVAR_NSNAP 1')
        script.write(f'      -PSF_SAMPLING 1.0')
        script.write(f'      -PSF_RECENTER Y')
        script.write(f'      -BASIS_TYPE PIXEL_AUTO')
        script.write(f'      -PSF_SUFFIX ".psf"')
        script.write(f'      -PSF_ACCURACY 0.01')
        script.write(f'      -SAMPLEVAR_TYPE NONE')
        script.write(f'      -SAMPLE_VARIABILITY 0.3')
        script.write(f'      -SAMPLE_AUTOSELECT YES')
        script.write(f'      -SAMPLE_FWHMRANGE {fwhm_min},{fwhm_max}')
        script.write(f'      -SAMPLE_MINSN {min_snr}')
        script.write(f'      -SAMPLE_MAXELLIP {max_ellip}')
        script.write(f'      -BASIS_NUMBER 20')
        script.write(f'      -PHOTFLUX_KEY "FLUX_APER(1)"')
        script.write(f'      -PHOTFLUXERR_KEY "FLUXERR_APER(1)"')
        if checkplots:
            script.write(f'      -CHECKPLOT_TYPE SELECTION_FWHM')
            script.write(f'      -CHECKPLOT_NAME {input_catalog}')
        script.write(f'      -OUTCAT_TYPE FITS_LDAC')
        script.write(f'      -OUTCAT_NAME {output_filename}.cat')


    subprocess.run(['bash', psfex_run_script])

    os.remove(input_catalog)
    os.remove(outcat)




class PSF:

    def __init__(self, filepath):

        if not filepath.endswith('.fits'):
            raise ValueError('PSF file must be a FITS file.')

        self.data = fits.getdata(filepath)


    def derive_aperture_correction(self, aperture_radius: float):
        """
        Derive the aperture correction for the PSF.
        """
        pass

    def plot(self):
        """
        Plot the PSF.
        """
        pass

    @classmethod
    def build(cls, 
        images: List[Image],
        output_file: str,  
        fwhm_min_scale: float,
        fwhm_max_scale: float,
        max_ellip: float, 
        min_snr: float,
        checkplots: bool,
        psf_size: int,
    ):
        """
        Build a PSF model from a list of images.
        """
        
        if not psf_size % 2:
            raise ValueError('PSF size must be odd.')
        
        final_catalog = output_file.replace('.fits', '_secat.fits')
        output_catalogs = []
        for image in images:
            print(image)
            if not image.exists:
                raise FileNotFoundError(f'{image} does not exist.')

            output_cat = image.base_file + 'psf_secat.fits' 
            
            # Run SExtractor on the image
            _run_se(image.sci_file, image.wht_file, output_cat, psf_size, sex_install=None)

            output_catalogs.append(output_cat)

        quit()

        if len(output_catalogs) > 1:
            # Merge catalogs, if more than one
            # final_catalog = ...
            pass
        else:
            os.rename(output_catalogs[0], final_catalog)

        # Determine the image filter, relevant for setting fwhm_min, fwhm_max
        filters = []
        for image in images:
            filters.append(image.filter)
        if len(set(filters)) != 1:
            raise ValueError('All images must be in the same filter!')
        filt = filters[0]

        # Determine the pixel scale for the image, relevant for setting fwhm_min, fwhm_max
        pixel_scales = []
        for image in images:
            pixel_scales.append(image.pixel_scale)
        if len(set(pixel_scales)) != 1:
            raise ValueError('All images must have same pixel scale!')
        pixel_scale = pixel_scales[0] # in arcsec

        fwhm_min = nominal_psf_fwhms[filt] / pixel_scale * fwhm_min_scale
        fwhm_max = nominal_psf_fwhms[filt] / pixel_scale * fwhm_max_scale

        # Run PSFEx on the final catalog
        _run_psfex(
            final_catalog, 
            output_file, 
            fwhm_min=fwhm_min,
            fwhm_max=fwhm_max,
            max_ellip=max_ellip,
            min_snr=min_snr,
            checkplot_fwhm=checkplot_fwhm,
            psf_size=psf_size
        )

        psf = fits.open(output_psf)[1].data['PSF_MASK'][0][0]
        size = np.shape(psf)[0]

        wcs = WCS(fits.getheader(images[0]))

        new_wcs = WCS(naxis=2)
        new_wcs.wcs.crpix = [size/2,size/2]
        ps = np.abs(wcs.proj_plane_pixel_scales()[0].value)
        new_wcs.wcs.cdelt = [-ps, ps]
        new_wcs.wcs.crval = wcs.wcs.crval
        new_wcs.wcs.ctype = wcs.wcs.ctype

        fits.writeto(output_psf.replace('.psf','.fits'), data=psf, header=new_wcs.to_header(), overwrite=True)

        # Remove extraneous files



        return cls(...)