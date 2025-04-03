from .image import Image

from collections import defaultdict
import os
from typing import List

def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5*(x-mu)**2/sigma**2)

from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats

from rich.table import Table
from rich.console import Console
console = Console()
console._log_render.omit_repeated_times = False

# def rich_str(x):
#     with console.capture() as capture:
#         console.print(x, end="")
#     return capture.get()


def _fit_pixel_distribution(data, sigma_upper=1.0, maxiters=3):
    data = data.flatten()
    data = data[np.isfinite(data)]

    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=maxiters)

    bins = np.linspace(-7*std, 7*std, 100)
    y, bins = np.histogram(data, bins=bins)
    y = y/np.max(y)
    p0 = [1, median, std]
    bc = 0.5*(bins[1:]+bins[:-1])
    popt, pcov = curve_fit(gauss, bc[bc<0], y[bc<0], p0=p0)

    for i in range(maxiters):
        p0 = popt
        bins = np.linspace(-4*popt[2], 4*popt[2], 100)
        y, bins = np.histogram(data, bins=bins)
        y = y/np.max(y)
        bc = 0.5*(bins[1:]+bins[:-1])
        popt, pcov = curve_fit(gauss, bc[bc<sigma_upper*popt[2]], y[bc<sigma_upper*popt[2]], p0=p0)

    mu = popt[1]
    sigma = popt[2]
    return mu, sigma

class Catalog:

    def __init__(self, config):
        self.config = config
        self.filters = config.filters
        self.tiles = config.tiles
        self.psfhom_filter = config.psf_homogenization.target_filter

        self.images = self._parse_mosaics()

        # # Validate things
        # if not self.all_images_exist:
        #     raise FileNotFoundError('Not all images exist!')

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
                    images[tile][filt] = Image(
                        filter=filt,
                        sci_file=sci_file, 
                        err_file=err_file, 
                        wht_file=wht_file, 
                        psf_file=psf_file, 
                        psfmatched_file=psfmatched_file
                    )

        return images

    def pprint(self, filenames=False):
        from rich.table import Table
        for tile in self.images:
            table = Table()
            table.add_column(tile)

            for filt in self.images[tile]:
                table.add_column(filt, justify="center", style="cyan", no_wrap=True)
            
            def label(file):
                l = ''
                if os.path.exists(file):
                    l += ':white_check_mark:'
                else:
                    l += ':x:'
                if filenames:
                    l += ' ' + os.path.basename(file)
                return l

            table.add_row('SCI', *[label(im.sci_file) for im in self.images[tile].values()])
            table.add_row('ERR', *[label(im.err_file) for im in self.images[tile].values()])
            table.add_row('WHT', *[label(im.wht_file) for im in self.images[tile].values()])
            table.add_row('PSF', *[label(im.psf_file) for im in self.images[tile].values()])
            table.add_row('PSFmatched', *[label(im.psfmatched_file) for im in self.images[tile].values()])

            console.print(table)

        #     # for filt in self.images[tile]:

        # 'Exists: X'
        # 'PSF: X'
        # 'PSFmatched: X'
        # pass


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

    def detect_sources(self):
        pass

    def get_auto_photometry(self):
        pass

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
        psf_size: int = 301, # size of PSF image, should be odd
        checkplots: bool = False, # whether to generate checkplots during PSF generation
        overwrite: bool = False, # whether to overwrite existing PSF files
        plot: bool = False, # plot the final PSFs for inspection
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
                checkplots=checkplots,
                psf_size=psf_size,
                overwrite=overwrite,
                master_psf_file=master_psf_file,
            )
            
            if plot:
                if len(images)==1:
                    psf.plot(save=images[0].psf_file.replace('.fits','.png'))
                elif master_psf_file:
                    psf.plot(save=master_psf_file.replace('.fits','.png'))
                    for image in images:
                        psf = PSF(image.sci_file.replace('_sci.fits','_psf.fits'))
                        psf.plot(save=image.sci_file.replace('_sci.fits', '_psf.png'))
                # for image in images:
                #     psf.plot(save=image.psf_file.replace('.fits','.png'))

    def _build_ivw_detection_image(
        self, 
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
        - truncate (bool): Whether to truncate individual SNR maps at 0.
        """

        self.detection_images = {}
        for tile in self.images:

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

            nivw = num/den * np.sqrt(1/den) # normalized ivw
                            
            # fit pixel distribution of normalized science images to a Gaussian to get the rms
            _, rms = _fit_pixel_distribution(
                nivw, 
                sigma_upper = sigma_upper, 
                maxiters = maxiters
            )
            
            # make signal to noise images by dividing normalized science images by rms
            snr = nivw / rms

            self.detection_images[tile] = snr # store the SNR image for this tile

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
        elif type == 'ivw':
            if not 'sigma_upper' in kwargs:
                kwargs['sigma_upper'] = 1.0
                print('Warning: `sigma_upper` not specified, defaulting to 1.0.')
            if not 'maxiters' in kwargs:
                kwargs['maxiters'] = 3
                print('Warning: `maxiters` not specified, defaulting to 3.')
            self._build_ivw_detection_image(
                detection_bands=filters, 
                sigma_upper=kwargs['sigma_upper'],
                maxiters=kwargs['maxiters'],
                psfmatched=psfmatched, 
            )
        
        else:
            raise ValueError(f'Invalid detection image type: {type}. Must be one of ["ivw", "chi-mean", "chi-sq"].')
        
        # Write out the detection image(s) to a file
        for tile in self.detection_images:
            f = file.replace('[tile]', tile)
            wcs = self.images[tile][self.psfhom_filter].wcs # use the WCS of the psf-homogenization filter, or any filter in the tile
            fits.writeto(f, self.detection_images[tile], overwrite=True)
        

    def load_detection_image(self, file: str):
        self.detection_images = {}
        for tile in self.images:
            f = file.replace('[tile]', tile)
            self.detection_images[tile] = fits.getdata(f)
            

    def measure_random_aperture_scaling(filt, 
        min_radius: float, 
        max_radius: float, 
        num_radii: int,
        num_apertures_per_sq_arcmin: int,
        plot: bool = False,
    ):
        aperture_diameters = np.linspace(min_radius, max_radius, num_radii) * 2 # in arcsec

        images = self.get_images(filt=filt)
        num_apertures_total = [int(num_apertures_per_sq_arcmin * image.area) for image in images].sum()
        fluxes = np.zeros((num_apertures_total, num_radii))

        i = 0
        for image in images:
            flux = image.get_random_aperture_fluxes(
                aperture_diameters,
                num_apertures_per_sq_arcmin,
                source_mask=image.srcmask,
            )
            N = np.shape(flux)[0]
            fluxes[i:i+N, :] = flux
            i += N


        for i in tqdm.tqdm(range(num_radii)):
            f = fluxes[:,i]
            mu, sigma = _fit_pixel_distribution(f, sigma_upper=1.0, maxiters=3)
            rmsN[i] = sigma

        N = np.pi*(aperture_diameters/2/0.03)**2

        def func(x, alpha, beta):
            return alpha*np.power(x,beta) 
        popt, pcov = curve_fit(func, np.sqrt(N), rmsN/rms1, p0=[1,1.5], maxfev=int(1e5))
        alpha,beta = popt


        fig, ax = plt.subplots(figsize=(3.5,3))
        ax.scatter(np.sqrt(N), rmsN/rms1, color='k', linewidths=1, marker='x', s=35, zorder=1000)
        ax.scatter(np.sqrt(N), rmsN/rms1, color='k', linewidths=1, marker='+', s=50, zorder=1000)
        x = np.linspace(0, 30, 1000)
        y = func(x,*popt)
        ax.plot(x, y, color='b')
        ax.plot(x, alpha*x, color='k', linestyle='--')
        ax.plot(x, alpha*x**2, color='k', linestyle='-.')


        ax.set_ylim(0, 1.3*np.max(rmsN/rms1))
        ax.set_xlim(0, 30)
        ax.set_xlabel(r'$\sqrt{N_{\rm pix}}$')
        ax.set_ylabel(r'$\sigma_N/\sigma_1$')


        if band=='f814w':
            photflam, photplam = 6.99715969242424E-20, 8047.468423484849
            conversion = 3.33564e13 * (photplam)**2 * photflam
        else:
            conversion = 1e15*((0.03*u.arcsec)**2).to(u.sr).value # MJy/sr to nJy/pix
            
        ax.annotate(f'A1-B10 {band.upper()}' + '\n' + fr'$\sigma_1/$nJy  $= {rms1*conversion:.2f}$' + '\n' + fr'$\alpha = {alpha:.3f}$' + '\n' + fr'$\beta = {beta:.3f}$', (0.05,0.95),xycoords='axes fraction', ha='left', va='top')
        # ax.annotate(f'{tile} {band.upper()}' + '\n' + fr'$\alpha = {alpha:.3f}$' + '\n' + fr'$\beta = {beta:.3f}$'+ '\n' + fr'$\gamma = {gamma:.3f}$' + '\n' + fr'$\delta = {delta:.3f}$', (0.05,0.95),xycoords='axes fraction', ha='left', va='top')

        if not psfMatched or band in ['f444w','f770w']:
            plt.savefig(f'/Dropbox/research/COSMOS-Web/catalog/checkplots/randomApertures_{band}.pdf')
        else:
            plt.savefig(f'/Dropbox/research/COSMOS-Web/catalog/checkplots/randomApertures_{band}_psfMatched.pdf')
        plt.close()
