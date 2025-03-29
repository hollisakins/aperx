from .image import Image

from collections import defaultdict
import os
from typing import List

def gauss(x, A, mu, sigma):
    return A * np.exp(-0.5*(x-mu)**2/sigma**2)

from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats

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

        self.images = self._parse_mosaics()

        # Validate things
        if not self.all_images_exist:
            raise FileNotFoundError('Not all images exist!')

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

                    psf_filepath = filepath.replace('[tile]', 'master') # TODO could be more general...

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
        psfex_install: str = 'psfex', # path to PSFEx installation
        fwhm_min_scale: float = 0.75, # minimum FWHM scale for PSF generation
        fwhm_max_scale: float = 1.75, # maximum FWHM scale for PSF generation
        max_ellip: float = 0.1, # maximum ellipticity for PSF generation
        min_snr: float = 8.0, # minimum SNR for PSF generation
        psf_size: int = 301, # size of PSF image, should be odd
        checkplots: bool = False, # whether to generate checkplots during PSF generation
        overwrite: bool = False, # whether to overwrite existing PSF files
    ):

        for filt in self.filters:
            images = list(self.images_flipped[filt].values())
            psf_files = [image.psf_file for image in images]

            if len(set(psf_files)) != 1:
                raise ValueError('All images for a given filter should have the same PSF file')
            
            psf_file = psf_files[0]

            if os.path.exists(psf_file) and not overwrite:
                print(f'PSF for {filt} already exists, skipping generation')
                continue
            
            # Build PSFs
            from .psf import PSF
            PSF.build(
                images,
                psf_file,
                fwhm_min_scale,
                fwhm_max_scale,
                max_ellip, 
                min_snr,
                checkplots,
                psf_size,
            )


    def _build_chisq_detection_image(
        self, 
        detection_bands: List[str], 
        sigma_upper: float = 1.0,
        maxiters: int = 3,
        truncate: bool = True,
    ):
        """
        Build a chi-squared detection image from a list of detection bands.

        Args:
        - detection_bands (List[str]): List of detection bands to use.
        - sigma_upper (float): Sigma clipping threshold to esimate background rms.
        - maxiters (int): Maximum number of iterations to use when fitting the pixel distribution.
        - truncate (bool): Whether to truncate individual SNR maps at 0.
        """

        shapes = []
        for band in detection_bands:
            shapes.append(self.get_sci(band).shape)
            # check that the detection bands are psf matched

        if len(set(shapes)) > 1:
            raise ValueError('All detection bands must have the same shape.')
        shape = shapes[0]

        snr_images = np.zeros((len(detection_bands), *shape))
        chisq = np.zeros(shape)
        for i, band in enumerate(detection_bands):
            # make normalized science images by multiplying by square root of the weight images
            normalized_image = self.get_sci(band) * np.sqrt(self.get_wht(band))
            
            # fit pixel distribution of normalized science images to a Gaussian to get the rms
            _, rms = _fit_pixel_distribution(
                normalized_image, 
                sigma_upper = sigma_upper, 
                maxiters = maxiters
            )
            
            # make signal to noise images by dividing normalized science images by rms
            snr = normalized_image / rms
            if truncate:
                snr[snr < 0] = 0
            snr_images[i] = snr
            
        # make chi-sq images by summing in quadrature the signal to noise images
        chisq = np.sum(np.power(snr_images, 2.))

        return chisq

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
