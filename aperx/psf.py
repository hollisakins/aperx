
def _run_se(detec_image, weight_image, output_cat, vignet_size, sex_install=None):
    randn = np.random.randint(0, 10000)
    param_file = os.path.join(config.temp_path, f'psfex_cat_{randn}.param')
    se_run_script = os.path.join(config.temp_path, f'se_run_script_{randn}.sh')

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
        param.write('FLUX_APER(1)\n')
        param.write('FLUXERR_APER(1)\n')
        param.write('MAG_APER(1)\n')
        param.write('MAGERR_APER(1)\n')
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
    
    with open(se_run_script, 'w') as script:
        script.write('') #...
        script.write(f'{sex_install} {detec_image},{detec_image}\ \n')
        script.write(f'    -CATALOG_NAME {output_cat}\ \n')
        script.write(f'    -CATALOG_TYPE FITS_LDAC\ \n')
        script.write(f'    -PARAMETERS_NAME {param_file}\ \n')
        script.write(f'    -DETECT_MINAREA 8\ \n')
        script.write(f'    -DETECT_THRESH 3.0\ \n')
        script.write(f'    -ANALYSIS_THRESH 2.5\ \n')
        script.write(f'    -FILTER Y\ \n')
        script.write(f'    -FILTER_NAME /home/hakins/COSMOS-Web/se++/conv/gauss_5.0_9x9.conv\ \n')
        script.write(f'    -DEBLEND_NTHRESH 32\ \n')
        script.write(f'    -DEBLEND_MINCONT 0.001\ \n')
        script.write(f'    -CLEAN Y\ \n')
        script.write(f'    -CLEAN_PARAM 1.0\ \n')
        script.write(f'    -MASK_TYPE CORRECT\ \n')
        script.write(f'    -WEIGHT_IMAGE {weight_image},{weight_image}\ \n')
        script.write(f'    -WEIGHT_TYPE MAP_WEIGHT,MAP_WEIGHT\ \n')
        script.write(f'    -RESCALE_WEIGHTS Y\ \n')
        script.write(f'    -WEIGHT_GAIN N\ \n')
        script.write(f'    -PHOT_APERTURES 150\ \n')
        script.write(f'    -PHOT_AUTOPARAMS 2.5, 1.0\ \n')
        script.write(f'    -PHOT_PETROPARAMS 2.0, 1.0\ \n')
        script.write(f'    -PHOT_AUTOAPERS 0.1, 0.1\ \n')
        script.write(f'    -PHOT_FLUXFRAC 0.5\ \n')
        script.write(f'    -MAG_ZEROPOINT 28.9\ \n')
        script.write(f'    -GAIN 1.0\ \n')
        script.write(f'    -PIXEL_SCALE 0\ \n')
        script.write(f'    -SEEING_FWHM 0.1\ \n')
        scripts.write(f'   -BACK_TYPE AUTO\ \n')
        scripts.write(f'   -BACK_SIZE 256\ \n')
        scripts.write(f'   -BACK_FILTERSIZE 3\ \n')
        scripts.write(f'   -BACKPHOTO_TYPE LOCAL\ \n')
        scripts.write(f'   -BACKPHOTO_THICK 64\ \n')
        scripts.write(f'   -BACK_FILTTHRESH 0.0\ \n')

    subprocess.run(se_run_script)

    os.remove(param_file)
    os.remove(se_run_script)


def _run_psfex(
        input_catalog, 
        output_filename, 
        fwhm_min=1.5, 
        fwhm_max=4.0,
        max_ellip=0.1,
        psf_size=301,
        checkplot_fwhm=False,
    ):
    """
    Run PSFEx on a catalog.
    """
    randn = np.random.randint(0, 10000)
    psfex_run_script = os.path.join(config.temp_path, f'psfex_run_script_{randn}.sh')

    with open(psfex_run_script_name, 'w') as script:
        script.write(f'psfex {input_catalog}\ \n') # /softs/astromatic/psfex/3.22.1/bin/psfex $SEcat
        script.write(f'      -c /home/hakins/PSFEx/default.psfex\ \n')
        script.write(f'      -PSF_SIZE {psf_size},{psf_size}\ \n')
        script.write(f'      -PSFVAR_DEGREES 0\ \n')
        script.write(f'      -PSFVAR_NSNAP 1\ \n')
        script.write(f'      -PSF_SAMPLING 1.0\ \n')
        script.write(f'      -PSF_RECENTER Y\ \n')
        script.write(f'      -BASIS_TYPE PIXEL_AUTO\ \n')
        script.write(f'      -PSF_SUFFIX ".psf"\ \n')
        script.write(f'      -PSF_ACCURACY 0.01\ \n')
        script.write(f'      -SAMPLEVAR_TYPE NONE\ \n')
        script.write(f'      -SAMPLE_VARIABILITY 0.3\ \n')
        script.write(f'      -SAMPLE_AUTOSELECT YES\ \n')
        script.write(f'      -SAMPLE_FWHMRANGE {fwhm_min},{fwhm_max}\ \n')
        script.write(f'      -SAMPLE_MINSN {min_snr}\ \n')
        script.write(f'      -SAMPLE_MAXELLIP {max_ellip}\ \n')
        script.write(f'      -BASIS_NUMBER 20\ \n')
        script.write(f'      -PHOTFLUX_KEY "FLUX_APER(1)"\ \n')
        script.write(f'      -PHOTFLUXERR_KEY "FLUXERR_APER(1)"\ \n')
        if checkplot_fwhm:
            script.write(f'      -CHECKPLOT_TYPE SELECTION_FWHM\ \n')
            script.write(f'      -CHECKPLOT_NAME {}\ \n')
        script.write(f'      -OUTCAT_TYPE FITS_LDAC\ \n')
        script.write(f'      -OUTCAT_NAME {output_filename}.cat\ \n')


    subprocess.run(psfex_run_script)

    os.remove(input_catalog)
    os.remove(outcat)




class PSF:

    @classmethod
    def load(cls, filepath):
        psf = fits.open(filepath)[1].data['PSF_MASK'][0][0]
        return cls(psf)

    @classmethod
    def build(cls, 
        images: List[str], 
        psf_size: int = 301,
    ):
        """
        Build a PSF model from a list of images.
        """
        
        if not psf_size % 2:
            raise ValueError('PSF size must be odd.')
        
        output_catalogs = []
        for image in images:
            if not os.path.exists(image):
                raise FileNotFoundError(f'{image} does not exist.')
            
            if '_i2d.fits' in image:
                raise ValueError('Cannot build PSFs from i2d files (atm).')

            elif '_sci.fits' in image:
                weight_image = image.replace('_sci.fits', '_wht.fits')

            output_cat = image.replace('_sci.fits', '_psfex_secat.fits')
            
            # Run SExtractor on the image
            _run_se(image, weight_image, output_cat, psf_size, sex_install=None):

            output_catalogs.append(output_cat)

        if len(output_catalogs) > 1:
            # Merge catalogs, if more than one
            # final_catalog = ...
            # rename to remove tiling? 
            pass
        else:
            final_catalog = output_catalogs[0]

        # Run PSFEx on the final catalog
        output_psf = final_catalog.replace('.fits', '_psf.fits')

        _run_psfex(final_catalog, 
            output_psf, 
            fwhm_min=fwhm_min,
            fwhm_max=fwhm_max,
            max_ellip=max_ellip,
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