import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
from cellment import background

from img_manager import corrector as corr


class VariableBackgroundCorrector(corr.GeneralCorrector):
    """An image or stack image corrector for variable background correction.
    Parameters for correction can be modified and fit accordingly.

    Attributes
    ----------
    bkg_model : lmfit.Model class
        The exponential model used to fit the dark noise images
    bkg_params : lmfit.Parameters class
        The parameters for the background correction
    time_step : float, scalar
        Time step between subsequent images

    Methods
    -------
    bkg_exponential(x, amplitude, characteristic_time, constant)
        Exponential function to be used in  background fitting
    subtract_background(self, stack, time_step)
        Subtracts background using the saved parameters from a stack of images
    find_bkg_correction(self, stack, time_step, pp=None)
        Loads a dark image stack and fits and saves the background correction
        parameters
    correct(stack)
        This function subtracts background from stack. time_step
        attribute is used.
    to_dict()
        Returns an OrderedDict with the parameters.
    load_from_dict(path)
        Loads the parameters from a saved OrderedDict
    """

    def __init__(self):
        """Return an image corrector with some initial parameters for
        background, bleaching and blleding correction."""
        # background
        self.corrector_species = 'VariableBackgroundCorrector'

        self.bkg_model = lm.Model(self.bkg_exponential, independent_vars=['x'])
        self.bkg_params = self.bkg_model.make_params(amplitude=9.30909784,
                                                     characteristic_time=152.75328323,
                                                     constant=43.32958973)
        self.bkg_params['amplitude'].set(min=0)
        self.bkg_params['characteristic_time'].set(min=0)
        self.time_step = None

    # Background Correction
    #######################
    @staticmethod
    def bkg_exponential(x, amplitude, characteristic_time, constant):
        """Exponential function to fit background increment in time series.

        Parameters
        ----------
        x : list, numpy.array
            timepoints of function evaluation
        amplitude : float
            Amplitude of background noise variation through time
        characteristic_time : float
            Characteristic time of background variation
        constant : float
            Background noise stable value for long measurements

        Returns
        -------
        Depending on input, returns value, list or array of background noise
        for the input timepoints
        """
        return -amplitude * np.exp(-x / characteristic_time) + constant

    def subtract_background(self, stack, time_step):
        """Subtracts background from a stack, considering time steps between
        images and using saved background parameters.

        Parameters
        ----------
        stack : numpy.array
            Time series of images to be corrected
        time_step
            Time step between acquired images

        Returns
        -------
            Returns the corrected stack
        """
        times = np.arange(0, time_step * len(stack), time_step)
        stack_corrected = stack.copy()
        background = self.bkg_model.eval(self.bkg_params, x=times)
        for ind, frame in enumerate(stack_corrected):
            stack_corrected[ind] = frame - background[ind]

        return stack_corrected

    def find_bkg_correction(self, stack, time_step, pp=None):
        """Given a dark noise stack, it calculates mean per timepoint, fits the
         parameters and saves them. If pp is given a PdfPages, images of fit
         are saved.

        If fit is not converging, you can manually modify parameters with:
            >>> corrector.bkg_params['constant'].set(value=40)
            >>> corrector.bkg_params['amplitude'].set(value=10)
            >>> corrector.bkg_params['characteristic_time'].set(value=20)

        Parameters
        ----------
        stack : numpy.array
            stack of dark noise images to find background (can be masked images)
        time_step : float
            Time step between acquired images
        pp : PdfPages
            Images of fitting are saved in this pdf
        """
        times = np.arange(0, time_step * len(stack), time_step)

        means = []
        for frame in stack:
            means.append(np.nanmean(frame.flatten()))

        result = self.bkg_model.fit(means, params=self.bkg_params, x=times)

        for key in result.best_values.keys():
            self.bkg_params[key].set(value=result.best_values[key])

            # Save a pdf of applied correction
        if pp is not None:
            plt.plot(times, means, 'ob')
            plt.plot(times, self.bkg_model.eval(self.bkg_params, x=times))
            plt.xlabel('time (s)')
            plt.ylabel('mean intensity (a.u.)')
            pp.savefig()
            plt.close()

            corrected = self.subtract_background(stack, time_step)

            means = []
            for frame in corrected:
                means.append(np.mean(frame.flatten()))

            plt.plot(times, means, 'ob')
            plt.xlabel('time (s)')
            plt.ylabel('mean intensity (a.u.)')
            pp.savefig()
            plt.close()

    def correct(self, stack):
        """This function subtracts background from stack. time_step attribute
        is used."""
        return self.subtract_background(stack, self.time_step)

    def to_dict(self):
        """Returns an OrderedDict with the parameters."""
        # TODO: test
        return {'corrector_species': self.corrector_species,
                'params': self.bkg_params.valuesdict()}

    def load_from_dict(self, valuesdict):
        """Loads the parameters from a saved OrderedDict"""
        # TODO: test
        self.bkg_params = self.bkg_model.make_params(valuesdict['params'])


class ConstantBackgroundCorrector(corr.GeneralCorrector):
    """Constant background corrector. Subtracts a constant value from images.
    If bkg_value is an array, it subtracts a different value from each frame.

    Attributes
    ----------
    bkg_value : int, float or numpy.ndarray
        value(s) to subtract from every (each) frame

    Methods
    -------
    find_background(masked_stack, percentile=50)

    """

    def __init__(self, bkg_val=None):
        self.corrector_species = 'ConstantBackgroundCorrector'

        self.bkg_value = bkg_val

    def find_background(self, masked_stack, percentile=50):
        """Finds the background using the percentile of the masked stack.

        Parameters
        ----------
        masked_stack : numpy.ndarray
            stack of images that have NaNs where signal is present.
        percentile : 0 to 100 int (default=50)
            Percentile to be used as background value.
        """
        self.bkg_value = np.nanpercentile(masked_stack, percentile)

    def correct(self, stack):
        """Subtracts the background from the stack using bkg_value. If
        bkg_value is an array, it must have the same length as stack. Results
        are clipped so that no value is below 0.

        Parameters
        ----------
        stack : numpy.ndarray
            Stack to be corrected

        Returns
        -------
        stack : numpy.ndarray
            Corrected stack
        """
        if isinstance(self.bkg_value, (int, float)):
            return np.clip(stack - self.bkg_value, 0, np.inf)
        if len(self.bkg_value.shape) == 1:
            for n, (this_img, this_bkg) in enumerate(zip(stack, self.bkg_value)):
                stack[n] = np.clip(this_img - this_bkg, 0, np.inf)

        return stack

    def to_dict(self):
        """Generates a dictionary with the attributes of the corrector"""
        return {'corrector_species': self.corrector_species,
                'bkg_value': self.bkg_value}

    def load_from_dict(self, parameter_dict):
        """Loads parameter from dictionary for creation

        Parameters
        ----------
        parameter_dict : dictionary
            Dictionary with bkg_value attribute to load
        """
        self.bkg_value = parameter_dict['bkg_value']


class SMOBackgroundCorrector(corr.GeneralCorrector):
    """Uses the SMOperator to estimate background and then subtracts it from
    images. Results are clipped so that no value is below 0.

    Attributes
    ----------
    corrector_species : str
        Name of the species to rebuild it.
    bkg_value : float, int, numpy.ndarray
        Value(s) of background to subtract from stack
    sigma : scalar (default=2)
        width of gaussian to use for the SMOperator
    size : int (default=51)
        size of window to use for the SMOperator
    percentile : float (default=0.5)
        Percentile to use as background from the background distribution
        (between 0 and 1)

    Methods
    -------
    find_background(stack)
        Uses SMO and parameters to find the background in the given stack.
    correct(stack)
        Subtracts background from the given stack.
    load_from_dict(parameter_dict)
        Loads parameters from dictionary.
    """

    def __init__(self, bkg_val=None, sigma=2, size=51, percentile=0.5):

        self.corrector_species = 'SMOBackgroundCorrector'

        self.bkg_value = bkg_val
        self.sigma = sigma
        self.size = size
        self.percentile = percentile

    def find_background(self, stack):
        """Uses SMO and parameters to find the background in the given stack."""
        self.bkg_value = self._find_background(stack)

    def _find_background(self, stack):
        """Recursively finds background in each frame."""
        if len(stack.shape) > 2:
            bkg = np.asarray([
                self._find_background(this)
                for this in stack])

        else:
            bkg = background.bg_rv(stack,
                                   self.sigma, self.size).ppf(self.percentile)

        return bkg

    def correct(self, stack):
        """Subtracts background from the given stack."""
        if isinstance(self.bkg_value, (int, float)):
            return np.clip(stack - self.bkg_value, 0, np.inf)
        if len(self.bkg_value.shape) == 1:
            for n, (this_img, this_bkg) in enumerate(zip(stack, self.bkg_value)):
                stack[n] = np.clip(this_img - this_bkg, 0, np.inf)

        return stack

    def to_dict(self):
        """Generates a dictionary with the attributes."""
        return {'corrector_species': self.corrector_species,
                'bkg_value': self.bkg_value,
                'sigma': self.sigma,
                'size': self.size,
                'percentile': self.percentile}

    def load_from_dict(self, parameter_dict):
        """Loads parameters from dictionary."""
        self.bkg_value = parameter_dict['bkg_value']
        self.sigma = parameter_dict['sigma']
        self.size = parameter_dict['size']
        self.percentile = parameter_dict['percentile']


class IlluminationCorrector(corr.GeneralCorrector):
    """Corrects for inhomogeneous illumination by division.

    Attributes
    ----------
    illumination : numpy.ndarray
        Image of homogeneous sample.

    Methods
    -------
    correct(stack)
        Corrects stack for inhomogeneous illumination.
    to_dict()
        Returns a dictionary with the image used for correction.
    load_from_dict(parameter_dict)
        Loads the parameters from a dictionary.
    """

    def __init__(self, illumination=None):

        self.corrector_species = 'IlluminationCorrector'

        self.illumination = illumination

    def correct(self, stack):
        """Corrects stack for inhomogeneous illumination."""
        if len(self.illumination.shape) > 1:
            # todo: only works if illumination is an array with shape attribute
            if len(stack.shape) == 2:
                stack = stack[np.newaxis, :]

            for n, this_img in enumerate(stack):
                stack[n] = this_img / self.illumination

        else:
            stack = stack / self.illumination

        # Normalize by illumination
        return stack

    def to_dict(self):
        """Returns a dictionary with the image used for correction."""
        # TODO: should be saving the path to the image or sth else.
        return {'corrector_species': self.corrector_species,
                'illumination': self.illumination}

    def load_from_dict(self, parameter_dict):
        """Loads the parameters from a dictionary."""
        self.illumination = parameter_dict['illumination']


class ExposureCorrector(corr.GeneralCorrector):
    """Masks as NaNs over and underexposed pixels.

    Attributes
    ----------
    bit : int
        Camera bit to estimate maximum value. Default is 2
    max_value : int
        Saturation value for image. Is estimated from bit
    min_value : int
        Minimum value of image. default is 0
    replace_value : float
        Value to use as replacement. Default is np.nan

    Methods
    -------
    correct(stack)
        Replaces over and under exposed pixels with replace_value
    to_dict()
        Returns a dictionary with parameters used for correction.
    load_from_dict(parameter_dict)
        Loads the parameters from a dictionary.
    """

    def __init__(self, bit=12):
        self.bit = bit
        self.max_value = 2 ** self.bit - 1
        self.min_value = 0
        self.replace_value = np.nan

    def correct(self, stack):
        """Replaces over and under exposed pixels with replace_value"""
        stack[stack == self.max_value] = self.replace_value
        stack[stack < self.min_value] = self.replace_value

        return stack

    def to_dict(self):
        """Returns a dictionary with parameters used for correction."""
        return {'corrector_species': self.corrector_species,
                'bit': self.bit,
                'max_value': self.max_value,
                'min_value': self.min_value,
                'replace_value': self.replace_value}

    def load_from_dict(self, parameter_dict):
        """Loads the parameters from a dictionary."""
        self.bit = parameter_dict['bit']
        self.max_value = parameter_dict['max_value']
        self.min_value = parameter_dict['min_value']
        self.replace_value = parameter_dict['replace_value']
