
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


    def __init__(self, _):

        # Validate things


        # load science images and weight images
        

        pass

    def get_sci(self, band):
        """
        Load the science image for the given band using memory mapping.

        Args:
        - band (str): The band for which to load the science image.

        Returns:
        - np.ndarray: The loaded science image.
        """
        file_path = self.science_images[band]
        with fits.open(file_path, memmap=True) as hdul:
            sci_image = hdul[0].data
        return sci_image

    @property
    def all_images_exist(self):
        pass
    
    @property
    def all_psfs_exist(self):
        pass
    
    @property
    def all_psfmatched_images_exist(self):
        pass

    @property
    def detection_image_exists(self):
        pass

    def detect_sources(self):
        pass

    def get_auto_photometry(self):
        pass



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

    