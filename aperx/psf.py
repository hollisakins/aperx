from .utils import Gaussian
import os, warnings, subprocess
from typing import List
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.wcs import WCS
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.nddata.utils import NoOverlapError, PartialOverlapError

nominal_psf_fwhms = {
    'f435w': 0.045,
    'f606w': 0.075,
    'f814w': 0.100,
    'f098m': 0.210,
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

def _run_se(
        detec_image, 
        weight_image, 
        output_cat, 
        vignet_size, 
        seeing_fwhm: float = 0.1,  # in arcseconds, default seeing FWHM
        sex_install: str = 'sex',
        overwrite: bool = False,
    ):
    if os.path.exists(output_cat):
        return 

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
        param.write('XWIN_IMAGE\n')
        param.write('YWIN_IMAGE\n')
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
        script.write(f"-FILTER_NAME '/home/hakins/COSMOS-Web/se++/conv/gauss_1.5_3x3.conv' ")
        script.write(f"-DEBLEND_NTHRESH 32 ")
        script.write(f"-DEBLEND_MINCONT 0.005 ")
        script.write(f"-CLEAN Y ")
        script.write(f"-CLEAN_PARAM 1.0 ")
        script.write(f"-MASK_TYPE CORRECT ")
        script.write(f"-WEIGHT_IMAGE '{weight_image}' ")
        script.write(f"-WEIGHT_TYPE MAP_WEIGHT ")
        script.write(f"-RESCALE_WEIGHTS Y ")
        script.write(f"-WEIGHT_GAIN Y ")
        script.write(f"-WEIGHT_THRESH 0 ")
        script.write(f"-PHOT_APERTURES 31 ")
        script.write(f"-PHOT_AUTOPARAMS 2.5,1.0 ")
        script.write(f"-PHOT_PETROPARAMS 2.0,1.0 ")
        script.write(f"-PHOT_AUTOAPERS 0.1,0.1 ")
        script.write(f"-PHOT_FLUXFRAC 0.5 ")
        script.write(f"-MAG_ZEROPOINT 28.9 ")
        script.write(f"-GAIN 1.0 ")
        script.write(f"-PIXEL_SCALE 0 ")
        script.write(f"-BACK_TYPE AUTO ")
        script.write(f"-BACK_SIZE 128 ")
        script.write(f"-BACK_FILTERSIZE 3 ")
        script.write(f"-BACKPHOTO_TYPE LOCAL ")
        script.write(f"-BACKPHOTO_THICK 24 ")
        script.write(f"-BACK_FILTTHRESH 0.0 ")
        script.write(f"-SEEING_FWHM {seeing_fwhm} ")
        script.write(f"-STARNNW_NAME /home/hakins/PSFEx/default.nnw ")

    subprocess.run(['bash', se_run_script], check=True)

    os.remove(param_file)
    os.remove(se_run_script)


def _run_psfex(
        input_catalog, 
        output_file, 
        fwhm_min=1.5, 
        fwhm_max=4.0,
        max_ellip=0.1,
        min_snr=8.0,
        psf_size=301,
        badpixel_nmax=150,
        checkplots=False,
        psfex_install='psfex',
    ):
    """
    Run PSFEx on a catalog.
    """
    randn = np.random.randint(0, 10000)
    psfex_run_script = output_file.replace('.fits', f'_psfex.sh')

    with open(psfex_run_script, 'w') as script:
        script.write(f'{psfex_install} {input_catalog} ') 
        script.write(f'-c /home/hakins/PSFEx/default.psfex ')
        script.write(f'-PSF_SIZE {psf_size},{psf_size} ')
        script.write(f'-PSFVAR_DEGREES 0 ')
        script.write(f'-PSFVAR_NSNAP 1 ')
        script.write(f'-PSF_SAMPLING 0 ')
        script.write(f'-PSF_RECENTER Y ')
        script.write(f'-BASIS_TYPE PIXEL_AUTO ')
        script.write(f'-PSF_SUFFIX ".psf" ')
        script.write(f'-PSF_ACCURACY 0.01 ')
        script.write(f'-BASIS_NUMBER 30 ')
        script.write(f'-SAMPLEVAR_TYPE NONE ')
        script.write(f'-SAMPLE_VARIABILITY 0.3 ')
        script.write(f'-SAMPLE_AUTOSELECT YES ')
        script.write(f'-SAMPLE_FWHMRANGE {fwhm_min},{fwhm_max} ')
        script.write(f'-SAMPLE_MINSN {min_snr} ')
        script.write(f'-SAMPLE_MAXELLIP {max_ellip} ')
        script.write(f'-CENTER_KEYS XWIN_IMAGE,YWIN_IMAGE ')
        script.write(f'-PHOTFLUX_KEY "FLUX_APER(1)" ')
        script.write(f'-PHOTFLUXERR_KEY "FLUXERR_APER(1)" ')
        script.write(f'-BADPIXEL_FILTER Y ')
        script.write(f'-BADPIXEL_NMAX {badpixel_nmax} ')
        if checkplots:
            script.write(f'-CHECKPLOT_TYPE SELECTION_FWHM ')
            script.write(f'-CHECKPLOT_NAME fwhm ')
        script.write(f'-OUTCAT_TYPE FITS_LDAC ')
        script.write(f'-OUTCAT_NAME {output_file} ')

    subprocess.run(['bash', psfex_run_script], check=True)

    os.remove(psfex_run_script)




class PSF:

    def __init__(self, filepath):

        if not filepath.endswith('.fits'):
            raise ValueError('PSF file must be a FITS file.')

        self.name = os.path.basename(filepath)
        self.data = fits.getdata(filepath)
        self.wcs = WCS(fits.getheader(filepath))
        self.pixel_scale = np.abs(self.wcs.proj_plane_pixel_scales()[0]).to(u.arcsec).value

    def derive_aperture_correction(self, aperture_radius: float):
        """
        Derive the aperture correction for the PSF.
        """
        pass

    def plot(self, 
            save: bool | str = False
        ):
        """
        Plot the PSF.
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        cmap = plt.colormaps['inferno']
        cmap.set_bad('k')
        norm = mpl.colors.LogNorm(vmin=1e-5, vmax=1)

        fig, ax = plt.subplots(figsize=(6,5),constrained_layout=True)
        im = ax.imshow(self.data/np.max(self.data), cmap=cmap, norm=norm)
        cbar = fig.colorbar(mappable=im)
        ax.set_title(self.name, fontsize=8)

        ax.axis('off')
        if save:
            print(f'Saving PSF plot to {save}')
            plt.savefig(save, dpi=300)
        plt.close()
        

    # @classmethod
    # def build(cls, 
    #     images: Image | List[Image], 
    #     fwhm_min_scale: float,
    #     fwhm_max_scale: float,
    #     max_ellip: float, 
    #     min_snr: float,
    #     badpixel_nmax: int,
    #     checkplots: bool,
    #     psf_size: int,
    #     psfex_install: str = 'psfex',
    #     sex_install: str = 'sex',
    #     overwrite: bool = False,
    #     master_psf_file = None,
    # ):
    #     """
    #     Build a PSF model from a list of images.
    #     images can be a list of multiple images or a single image
    #     """
        
    #     if not psf_size % 2:
    #         raise ValueError('PSF size must be odd.')

    #     # Determine the image filter, relevant for setting fwhm_min, fwhm_max
    #     if isinstance(images, Image):
    #         filt = images.filter
    #         images = [images]
    #     elif len(images)==1:
    #         filt = images[0].filter
    #     else:
    #         filters = []
    #         for image in images:
    #             filters.append(image.filter)
    #         if len(set(filters)) != 1:
    #             raise ValueError('If providing a list of images, all images must be in the same filter!')
    #         filt = filters[0]

    #     se_cats = []
    #     psfex_se_cats = []
    #     for image in images:
    #         print(f'Generating PSF for {image}')
    #         if not image.exists:
    #             raise FileNotFoundError(f'{image} does not exist.')

    #         se_cat = image.base_file + 'psf_secat.fits' 
    #         psfex_se_cat = image.base_file + 'psf_secat_psfex.fits'
            
    #         # Run SExtractor on the image
    #         _run_se(
    #             image.sci_file, 
    #             image.wht_file, 
    #             se_cat, 
    #             psf_size, 
    #             seeing_fwhm = nominal_psf_fwhms[filt],  # Use the nominal PSF FWHM for the filter
    #             sex_install=sex_install,
    #             overwrite=overwrite,
    #         )

    #         # Determine the pixel scale for the image, relevant for setting fwhm_min, fwhm_max
    #         pixel_scale = image.pixel_scale
    #         fwhm_min = nominal_psf_fwhms[filt] / pixel_scale * fwhm_min_scale
    #         fwhm_max = nominal_psf_fwhms[filt] / pixel_scale * fwhm_max_scale

    #         # Run PSFEx on the final catalog
    #         _run_psfex(
    #             se_cat, 
    #             psfex_se_cat, 
    #             fwhm_min=fwhm_min,
    #             fwhm_max=fwhm_max,
    #             max_ellip=max_ellip,
    #             min_snr=min_snr,
    #             psf_size=psf_size,
    #             checkplots=checkplots,
    #             badpixel_nmax=badpixel_nmax,
    #             psfex_install=psfex_install,
    #         )

    #         psf_file = se_cat.replace('.fits','.psf')
    #         final_psf_file = psf_file.replace('_secat.psf','.fits')

    #         # Clean up extraneous files
    #         base = os.path.basename(se_cat)
    #         os.remove('psfex.xml')
    #         os.remove(f'chi_{base}')
    #         os.remove(f'moffat_{base}')
    #         os.remove(f'proto_{base}')
    #         os.remove(f'resi_{base}')
    #         os.remove(f'samp_{base}')
    #         os.remove(f'snap_{base}')
    #         os.remove(f'submoffat_{base}')
    #         os.remove(f'subsym_{base}')

    #         if checkplots:
    #             os.rename(f"fwhm_{base.replace('.fits','.png')}", psf_file.replace('.psf','_fwhm.png'))
        
    #         psf = fits.open(psf_file)[1].data['PSF_MASK'][0][0]
    #         size = np.shape(psf)[0]

    #         wcs = images[0].wcs

    #         new_wcs = WCS(naxis=2)
    #         new_wcs.wcs.crpix = [size/2,size/2]
    #         ps = np.abs(wcs.proj_plane_pixel_scales()[0].value)
    #         new_wcs.wcs.cdelt = [-ps, ps]
    #         new_wcs.wcs.crval = wcs.wcs.crval
    #         new_wcs.wcs.ctype = wcs.wcs.ctype

    #         fits.writeto(final_psf_file, data=psf, header=new_wcs.to_header(), overwrite=True)

    #         os.remove(psf_file)

    #         if master_psf_file is not None:
    #             se_cats.append(se_cat)
    #             psfex_se_cats.append(psfex_se_cat)
    #         else:
    #             os.remove(se_cat)
    #             os.remove(psfex_se_cat)


    #         #quit()


    #     # If we're handling a list of input images, then we also create a master PSF. 
    #     if master_psf_file is not None:
    #         merged_catalog = master_psf_file.replace('.fits','_secat.fits')

    #         print(f'Merging & filtering {len(se_cats)} SE catalogs...')
    #         print('1', os.path.basename(se_cats[0]))
    #         se_cat1 = fits.open(se_cats[0])
    #         # wcs_main = images[0].wcs
    #         hdu0 = se_cat1[0]
    #         hdu1 = se_cat1[1]
    #         hdu2 = se_cat1[2]
    #         hdr = hdu2.header
    #         data1 = hdu2.data
    #         tab1 = Table(data1)

    #         psfex_se_cat1 = fits.open(psfex_se_cats[0])
    #         condition = psfex_se_cat1[2].data['FLAGS_PSF'] == 0
    #         tab1 = tab1[condition]

    #         for i,se_cat in enumerate(se_cats[1:]):
    #             print(i+2, os.path.basename(se_cat))

    #             tab2 = Table.read(se_cat, hdu=2)
    #             psfex_se_cat2 = fits.open(psfex_se_cats[i+1])
    #             condition = psfex_se_cat2[2].data['FLAGS_PSF'] == 0 
    #             tab2 = tab2[condition]
                
    #             # coordinate matching

    #             tab1 = vstack([tab1,tab2], join_type='exact')
    #             del tab2
            
    #         print(f'Saving to {merged_catalog}')
    #         hdu2 = fits.BinTableHDU(data=tab1.as_array(), header=hdr)
    #         hdul = fits.HDUList([hdu0, hdu1, hdu2])
    #         hdul.writeto(merged_catalog, overwrite=True)


    #         # Determine the pixel scale for the image, relevant for setting fwhm_min, fwhm_max
    #         pixel_scales = []
    #         for image in images:
    #             pixel_scales.append(image.pixel_scale)
    #         if len(set(pixel_scales)) != 1:
    #             raise ValueError('all images must have same pixel scale to generate master PSF')
    #         pixel_scale = pixel_scales[0]

    #         fwhm_min = nominal_psf_fwhms[filt] / pixel_scale * fwhm_min_scale
    #         fwhm_max = nominal_psf_fwhms[filt] / pixel_scale * fwhm_max_scale

    #         # Run PSFEx on the final catalog
    #         _run_psfex(
    #             merged_catalog, 
    #             merged_catalog.replace('_secat.fits','_secat_psfex.fits'), 
    #             fwhm_min=fwhm_min,
    #             fwhm_max=fwhm_max,
    #             max_ellip=max_ellip,
    #             min_snr=min_snr,
    #             psf_size=psf_size,
    #             checkplots=checkplots,
    #             psfex_install=psfex_install,
    #         )

    #         psf_file = merged_catalog.replace('.fits','.psf')
            
    #         # Clean up extraneous files
    #         base = os.path.basename(merged_catalog)
    #         os.remove('psfex.xml')
    #         os.remove(f'chi_{base}')
    #         os.remove(f'moffat_{base}')
    #         os.remove(f'proto_{base}')
    #         os.remove(f'resi_{base}')
    #         os.remove(f'samp_{base}')
    #         os.remove(f'snap_{base}')
    #         os.remove(f'submoffat_{base}')
    #         os.remove(f'subsym_{base}')

    #         if checkplots:
    #             os.rename(f"fwhm_{base.replace('.fits','.png')}", psf_file.replace('.psf','_fwhm.png'))
        
    #         psf = fits.open(psf_file)[1].data['PSF_MASK'][0][0]
    #         size = np.shape(psf)[0]

    #         wcs = images[0].wcs

    #         new_wcs = WCS(naxis=2)
    #         new_wcs.wcs.crpix = [size/2,size/2]
    #         ps = np.abs(wcs.proj_plane_pixel_scales()[0].value)
    #         new_wcs.wcs.cdelt = [-ps, ps]
    #         new_wcs.wcs.crval = wcs.wcs.crval
    #         new_wcs.wcs.ctype = wcs.wcs.ctype

    #         final_psf_file = psf_file.replace('_secat.psf','.fits')
    #         fits.writeto(final_psf_file, data=psf, header=new_wcs.to_header(), overwrite=True)

    #         os.remove(psf_file)

    #         for i in range(len(se_cats)):
    #             os.remove(se_cats[i])
    #             os.remove(psfex_se_cats[i])

    #     return cls(final_psf_file)
    
    @classmethod
    def build(cls, 
        images, 
        fwhm_min: float,
        fwhm_max: float,
        max_ellip: float, 
        min_snr: float,
        max_snr: float,
        checkplots: bool,
        psf_size: int,
        overwrite: bool = False,
        master_psf_file = None,
        az_average: bool = False,
    ):
        """
        Build a PSF model from a list of images.
        images can be a list of multiple images or a single image
        """
        
        if not psf_size % 2:
            raise ValueError('PSF size must be odd.')

        # Determine the image filter, relevant for setting fwhm_min, fwhm_max
        if not type(images) == list:
            filt = images.filter
            images = [images]
        elif len(images)==1:
            filt = images[0].filter
        else:
            filters = []
            for image in images:
                filters.append(image.filter)
            if len(set(filters)) != 1:
                raise ValueError('If providing a list of images, all images must be in the same filter!')
            filt = filters[0]

        # se_cats = []
        # psfex_se_cats = []


        # import sep, set some options 
        import sep
        sep.set_extract_pixstack(int(1e7))
        sep.set_sub_object_limit(2048)

        if master_psf_file is not None:
            x_all = np.array([])
            y_all = np.array([])
            which_img = np.array([])

        for image_i, image in enumerate(images):
            print(f'Generating PSF for {image}')
            if not image.exists:
                raise FileNotFoundError(f'{image} does not exist.')

            from photutils.segmentation import make_2dgaussian_kernel
            kernel = make_2dgaussian_kernel(fwhm=1.5, size=5).array

            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore')
            #     invvar = 1/image.wht

            # Detect sources in the image to identify stars
            print(f'\t Detecting sources')

            image.convert_byteorder()
            mask = np.isnan(image.sci)

            objs, segmap = sep.extract(
                image.sci, 
                err = image.err, # np.sqrt(invvar),
                thresh = 2.5,
                minarea = 8, 
                deblend_nthresh = 32, 
                deblend_cont = 0.005,
                mask = mask,
                filter_type = 'matched',
                filter_kernel = kernel,
                clean = True,
                clean_param = 1.0,
                segmentation_map=True)
            
            seg_id = np.arange(1, len(objs)+1, dtype=np.int32)
            objs['theta'][objs['theta']>np.pi/2] -= np.pi
            objs['b'][np.isnan(objs['b'])] = objs['a'][np.isnan(objs['b'])]
            
            # Compute fluxes
            print('\t Computing kron radius (iteration 1)')
            kronrad, krflag = sep.kron_radius(
                image.sci, objs['x'], objs['y'], 
                objs['a'], objs['b'], objs['theta'], 
                6.0, mask=mask, seg_id=seg_id, segmap=segmap,
            )

            # print('\t', objs['x'].min(), objs['x'].max())
            # print('\t', objs['y'].min(), objs['y'].max())
            # print('\t', objs['a'].min(), objs['a'].max())
            # print('\t', objs['b'].min(), objs['b'].max())
            # print('\t', objs['theta'].min(), objs['theta'].max())
            # print('\t', kronrad.min(), kronrad.max())

            print('\t Computing fluxes (iteration 1)')
            flux, fluxerr, flag = sep.sum_ellipse(
                image.sci, objs['x'], objs['y'], 
                objs['a'], objs['b'], objs['theta'], 
                2.5*kronrad, subpix=1, mask=mask,
                seg_id=seg_id, segmap=segmap,
            )
            flag |= krflag  # combine flags into 'flag'

            print('\t Computing FWHM (iteration 1)')
            rhalf, rflag = sep.flux_radius(
                image.sci, objs['x'], objs['y'], 
                6.*objs['a'], 0.5, 
                seg_id=seg_id, segmap=segmap,
                mask=mask, normflux=flux, subpix=5
            )

            print('\t Computing windowed positions')
            sig = 2. / 2.35 * rhalf  
            xwin, ywin, flag = sep.winpos(
                image.sci, objs['x'], objs['y'], sig, 
                mask=mask)

            # Compute fluxes
            print('\t Computing kron radius (iteration 2)')
            kronrad, krflag = sep.kron_radius(
                image.sci, xwin, ywin,
                objs['a'], objs['b'], objs['theta'], 
                6.0, mask=mask, seg_id=seg_id, segmap=segmap,
            )
            
            print('\t Computing fluxes (iteration 2)')
            flux, fluxerr, flag = sep.sum_ellipse(
                image.sci, xwin, ywin,
                objs['a'], objs['b'], objs['theta'], 
                2.5*kronrad, subpix=1, mask=mask,
                seg_id=seg_id, segmap=segmap,
            )
            flag |= krflag  # combine flags into 'flag'

            print('\t Computing FWHM (iteration 2)')
            rhalf, rflag = sep.flux_radius(
                image.sci, xwin, ywin, 
                6.*objs['a'], 0.5, 
                seg_id=seg_id, segmap=segmap,
                mask=mask, normflux=flux, subpix=5
            )
            flag |= rflag
            fwhm = 2 * rhalf

            aperture_radius = 3*nominal_psf_fwhms[filt]/image.pixel_scale
            flux, fluxerr, cflag = sep.sum_circle(
                image.sci, xwin, ywin, 
                aperture_radius, err=image.err
            )
            flag |= cflag
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                snr = flux/fluxerr
            ellip = (objs['a']-objs['b'])/(objs['a']+objs['b'])
            cond = np.logical_and.reduce((
                fwhm > fwhm_min, 
                fwhm < fwhm_max,
                snr > min_snr, 
                snr < max_snr, 
                ellip < max_ellip,
            ))
            print(f'\t Identified {len(cond[cond])} PSF stars')

            psf_file = image.base_file + 'psf.fits'

            if checkplots:
                fig, ax = plt.subplots(constrained_layout=True)
                ax.scatter(fwhm, snr, linewidths=0, s=1, color='k', marker='o')
                ax.scatter(fwhm[cond], snr[cond], linewidths=0, s=2, color='tab:red', marker='o')
                ax.set_xlabel('FWHM [pix]')
                ax.set_xlim(0.7, 100)
                ax.set_ylabel('SNR')
                ax.set_ylim(3, 1e5)
                ax.loglog()
                ax.plot([fwhm_min, fwhm_min], [min_snr, max_snr], linewidth=1, color='r', linestyle='--')
                ax.plot([fwhm_max, fwhm_max], [min_snr, max_snr], linewidth=1, color='r', linestyle='--')
                ax.plot([fwhm_min, fwhm_max], [min_snr, min_snr], linewidth=1, color='r', linestyle='--')
                ax.plot([fwhm_min, fwhm_max], [max_snr, max_snr], linewidth=1, color='r', linestyle='--')
                ax.axvline(nominal_psf_fwhms[filt]/image.pixel_scale, linewidth=1, color='b', linestyle=':')
                plt.savefig(psf_file.replace('.fits','_fwhm.pdf'))
                plt.close()

            print('\t Computing PSF...')
            x = xwin[cond]
            y = ywin[cond]
            from astropy.nddata import Cutout2D
            # cutouts = np.zeros((len(x), psf_size, psf_size))
            x_grid, y_grid = np.zeros(psf_size**2 * len(x)), np.zeros(psf_size**2 * len(y))
            z_grid = np.zeros(psf_size**2 * len(y))
            for i in range(len(x)):
                xi, yi = np.arange(psf_size)+0.5, np.arange(psf_size)+0.5
                xi, yi = np.meshgrid(xi, yi)
                xi, yi = xi.flatten(), yi.flatten()
                try:
                    cutout = Cutout2D(image.sci, position=(x[i],y[i]), size=psf_size, mode='strict')
                    # Compare input_position_cutout and input_position_original, correct for sub-pixel offset in the windowed position 
                    dx = cutout.input_position_cutout[0] - psf_size//2
                    dy = cutout.input_position_cutout[1] - psf_size//2
                    xi += dx
                    yi += dy
                    zi = cutout.data.flatten()
                except (NoOverlapError, PartialOverlapError):
                    zi = np.full(psf_size**2, np.nan)

                if az_average:
                    theta = np.random.uniform(0, np.pi)
                    xp = (xi-psf_size/2) * np.cos(theta) - (yi-psf_size/2) * np.sin(theta) + psf_size/2
                    yp = (xi-psf_size/2) * np.sin(theta) + (yi-psf_size/2) * np.cos(theta) + psf_size/2
                    xi = xp
                    yi = yp

                x_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = xi
                y_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = yi
                z_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = zi

            x_bins = np.arange(psf_size)
            y_bins = np.arange(psf_size) 
            x_bins = np.append(x_bins, x_bins[-1]+1)
            y_bins = np.append(y_bins, y_bins[-1]+1)

            from scipy.stats import binned_statistic_2d
            psf, _, _, _ = binned_statistic_2d(
                x_grid, y_grid, z_grid, 
                bins=(x_bins, y_bins), 
                statistic=np.nanmedian
            )
            psf = psf.T

            # Subtract the background from the PSF
            print('\t Background-subtracting PSF')
            from astropy.stats import sigma_clipped_stats
            from scipy.optimize import curve_fit
            mean, median, std = sigma_clipped_stats(psf)
            bins = np.linspace(-3*std+median, 5*std+median, 80)
            bc = 0.5*(bins[1:]+bins[:-1])
            h, _ = np.histogram(psf.flatten(), bins)
            h = h / np.max(h)
            p0 = [1, bc[np.argmax(h)], std]
            
            try:
                popt, pcov = curve_fit(Gaussian, bc[bc<2*std+median], h[bc<2*std+median], p0=p0)
                popt, pcov = curve_fit(Gaussian, bc[bc<1*std+median], h[bc<1*std+median], p0=popt)
                bkg = popt[1]
            except:
                bkg = median
            
            psf = psf - bkg
            
            psf[psf < 0] = 0
            xg, yg = np.arange(psf_size), np.arange(psf_size)
            xg, yg = np.meshgrid(xg, yg)
            dist = np.sqrt((xg-psf_size/2)**2 + (yg-psf_size/2)**2)
            psf[dist > psf_size/2] = 0



            print(f'\t Writing to {psf_file}')
            wcs = images[0].wcs
            new_wcs = WCS(naxis=2)
            new_wcs.wcs.crpix = [psf_size/2,psf_size/2]
            ps = np.abs(wcs.proj_plane_pixel_scales()[0].value)
            new_wcs.wcs.cdelt = [-ps, ps]
            new_wcs.wcs.crval = wcs.wcs.crval
            new_wcs.wcs.ctype = wcs.wcs.ctype

            fits.writeto(psf_file, data=psf, header=new_wcs.to_header(), overwrite=True)

            if master_psf_file is not None:
                x_all = np.append(x_all, x)
                y_all = np.append(y_all, y)
                which_img = np.append(which_img, [image_i]*len(x))

        if master_psf_file is not None:
            which_img = which_img.astype(int)
            print(f'Combining {len(x_all)} stars from {len(images)} images into master PSF')

            print('\t Computing PSF...')
            # cutouts = np.zeros((len(x_all), psf_size, psf_size))
            # for i in range(len(x_all)):
            #     xi = x_all[i]
            #     yi = y_all[i]
            #     image = images[which_img[i]]
            #     try:
            #         cutout = Cutout2D(image.sci, position=(xi,yi), size=psf_size, mode='strict')
            #         cutouts[i] = cutout.data
            #     except:
            #         cutouts[i] = np.full((psf_size,psf_size),np.nan)
                
            x_grid, y_grid = np.zeros(psf_size**2 * len(x_all)), np.zeros(psf_size**2 * len(y_all))
            z_grid = np.zeros(psf_size**2 * len(y_all))
            for i in range(len(x_all)):
                xi, yi = np.arange(psf_size)+0.5, np.arange(psf_size)+0.5
                xi, yi = np.meshgrid(xi, yi)
                xi, yi = xi.flatten(), yi.flatten()
                image = images[which_img[i]]
                try:
                    cutout = Cutout2D(image.sci, position=(x_all[i],y_all[i]), size=psf_size, mode='strict')
                    # Compare input_position_cutout and input_position_original, correct for sub-pixel offset in the windowed position 
                    dx = cutout.input_position_cutout[0] - psf_size//2
                    dy = cutout.input_position_cutout[1] - psf_size//2
                    xi += dx
                    yi += dy
                    zi = cutout.data.flatten()
                except (NoOverlapError, PartialOverlapError):
                    zi = np.full(psf_size**2, np.nan)

                if az_average:
                    theta = np.random.uniform(0, np.pi)
                    xp = (xi-psf_size/2) * np.cos(theta) - (yi-psf_size/2) * np.sin(theta) + psf_size/2
                    yp = (xi-psf_size/2) * np.sin(theta) + (yi-psf_size/2) * np.cos(theta) + psf_size/2
                    xi = xp
                    yi = yp

                x_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = xi
                y_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = yi
                z_grid[i*(psf_size**2):(i+1)*(psf_size)**2] = zi

            x_bins = np.arange(psf_size)
            y_bins = np.arange(psf_size) 
            x_bins = np.append(x_bins, x_bins[-1]+1)
            y_bins = np.append(y_bins, y_bins[-1]+1)

            from scipy.stats import binned_statistic_2d
            psf, _, _, _ = binned_statistic_2d(
                x_grid, y_grid, z_grid, 
                bins=(x_bins, y_bins), 
                statistic=np.nanmedian
            )
            psf = psf.T

            # Subtract the background from the PSF
            print('\t Background-subtracting PSF')
            from astropy.stats import sigma_clipped_stats
            from scipy.optimize import curve_fit
            mean, median, std = sigma_clipped_stats(psf)
            bins = np.linspace(-3*std+median, 5*std+median, 80)
            bc = 0.5*(bins[1:]+bins[:-1])
            h, _ = np.histogram(psf.flatten(), bins)
            h = h / np.max(h)
            p0 = [1, bc[np.argmax(h)], std]
            
            try:
                popt, pcov = curve_fit(Gaussian, bc[bc<2*std+median], h[bc<2*std+median], p0=p0)
                popt, pcov = curve_fit(Gaussian, bc[bc<1*std+median], h[bc<1*std+median], p0=popt)
                bkg = popt[1]
            except:
                bkg = median
            
            psf = psf - bkg
            psf[psf < 0] = 0
            xg, yg = np.arange(psf_size), np.arange(psf_size)
            xg, yg = np.meshgrid(xg, yg)
            dist = np.sqrt((xg-psf_size/2)**2 + (yg-psf_size/2)**2)
            psf[dist > psf_size/2] = 0

            psf_file = master_psf_file
            print(f'\t Writing to {psf_file}')

            wcs = images[0].wcs
            new_wcs = WCS(naxis=2)
            new_wcs.wcs.crpix = [psf_size/2,psf_size/2]
            ps = np.abs(wcs.proj_plane_pixel_scales()[0].value)
            new_wcs.wcs.cdelt = [-ps, ps]
            new_wcs.wcs.crval = wcs.wcs.crval
            new_wcs.wcs.ctype = wcs.wcs.ctype

            fits.writeto(psf_file, data=psf, header=new_wcs.to_header(), overwrite=True)

        return cls(psf_file)
