

class PSF:

    @classmethod
    def build(cls, image):
        param_file = os.path.join(config.aperx_data_path, 'psfex_cat.param')
        


        with open(se_run_script_name, 'w') as script:
            script.write('') #...
            script.write(f'sex {detec_image},{detec_image}\ \n')
            script.write(f'    -c {config_file}\ \n')
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

            # script.write(f'    -CHECKIMAGE_TYPE SEGMENTATION,APERTURES\ \n')
            # script.write(f'    -CHECKIMAGE_NAME SEGMENTATION,APERTURES\ \n')

        subprocess.run(os.path.join(config.temp_path, se_run_script_name))


        fwhm_min=1.5
        fwhm_max=4.0 #7.0 for LW
        min_snr=8.0
        psf_size = 301

        with open(psfex_run_script_name, 'w') as script:
            script.write(f'psfex {catalog}\ \n') # /softs/astromatic/psfex/3.22.1/bin/psfex $SEcat
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
            script.write(f'      -SAMPLE_MAXELLIP 0.1\ \n')
            script.write(f'      -BASIS_NUMBER 20\ \n')
            script.write(f'      -PHOTFLUX_KEY "FLUX_APER(1)"\ \n')
            script.write(f'      -PHOTFLUXERR_KEY "FLUXERR_APER(1)"\ \n')
            # script.write(f'      -CHECKPLOT_TYPE SELECTION_FWHM\ \n')
            # script.write(f'      -CHECKPLOT_NAME $dir_psfs'checkplot_fwhm'\ \n')
            script.write(f'      -OUTCAT_TYPE FITS_LDAC\ \n')
            script.write(f'      -OUTCAT_NAME {outcat}\ \n')


        subprocess.run(os.path.join(config.temp_path, psfex_run_script_name))


        psf = fits.open(psfpath)[1].data['PSF_MASK'][0][0]
        size = np.shape(psf)[0]

        hdr = fits.open(imgpath)[0].header
        wcs = WCS(hdr)

        new_wcs = WCS(naxis=2)
        new_wcs.wcs.crpix = [size/2,size/2]
        ps = np.abs(wcs.proj_plane_pixel_scales()[0].value)
        new_wcs.wcs.cdelt = [-ps, ps]
        new_wcs.wcs.crval = wcs.wcs.crval
        new_wcs.wcs.ctype = wcs.wcs.ctype

        fits.writeto(psfpath.replace('.psf','.fits'), data=psf, header=new_wcs.to_header(), overwrite=True)


        return cls(...)