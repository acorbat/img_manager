import json
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

    Methods
    -------
    bkg_exponential(x, amplitude, characteristic_time, constant)
        Exponential function to be used in  background fitting
    subtract_background(self, stack, time_step)
        Subtracts background using the saved parameters from a stack of images
    find_bkg_correction(self, stack, time_step, pp=None)
        Loads a dark image stack and fits and saves the background correction
        parameters
    """

    def __init__(self):
        """Return an image corrector with some initial parameters for
        background, bleaching and blleding correction."""
        # background
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
        """This function subtracts background from stack. time_step attribute is used."""
        return self.subtract_background(stack, self.time_step)

    def to_dict(self):
        """Returns an OrderedDict with the parameters."""
        # TODO: test
        return self.bkg_params.valuesdict()

    def load_from_dict(self, valuesdict):
        """Loads the parameters from a saved OrderedDict"""
        # TODO: test
        self.bkg_params = self.bkg_model.make_params(valuesdict)


class ConstantBackgroundCorrector(corr.GeneralCorrector):

    def __init__(self, bkg_val=None):

        self.bkg_value = bkg_val

    def find_background(self, masked_stack, percentile=50):
        self.bkg_value = np.nanpercentile(masked_stack, percentile)

    def correct(self, stack):
        if isinstance(self.bkg_value, (int, float)):
            return np.clip(stack - self.bkg_value, 0, np.inf)
        if len(self.bkg_value.shape) == 1:
            for n, (this_img, this_bkg) in enumerate(zip(stack, self.bkg_value)):
                stack[n] = np.clip(this_img - this_bkg, 0, np.inf)

        return stack

    def to_dict(self):
        return {'bkg_value': self.bkg_value}

    def load_from_dict(self, parameter_dict):
        self.bkg_value = parameter_dict['bkg_value']


class SMOBackgroundCorrector(corr.GeneralCorrector):

    def __init__(self, bkg_val=None, sigma=2, size=51, percentile=0.5):
        super().__init__()
        self.bkg_value = bkg_val
        self.sigma = sigma
        self.size = size
        self.percentile = percentile

    def find_background(self, stack):
        self.bkg_value = self._find_background(stack)

    def _find_background(self, stack):
        if len(stack.shape) > 2:
            bkg = np.asarray([
                self._find_background(this)
                for this in stack])

        else:
            bkg = background.bg_rv(stack,
                                   self.sigma, self.size).ppf(self.percentile)

        return bkg

    def correct(self, stack):
        if isinstance(self.bkg_value, (int, float)):
            return np.clip(stack - self.bkg_value, 0, np.inf)
        if len(self.bkg_value.shape) == 1:
            for n, (this_img, this_bkg) in enumerate(zip(stack, self.bkg_value)):
                stack[n] = np.clip(this_img - this_bkg, 0, np.inf)

        return stack

    def to_dict(self):
        return {'bkg_value': self.bkg_value}

    def load_from_dict(self, parameter_dict):
        self.bkg_value = parameter_dict['bkg_value']


class IlluminationCorrector(corr.GeneralCorrector):

    def __init__(self, illumination=None, replace_exposure=True):

        self.illumination = illumination
        self.replace_exposure = replace_exposure
        self.overexposed_value = 4095
        self.underexposed_value = 0

    def correct(self, stack):

        # ignore over or underexposed pixels
        if self.replace_exposure:
            stack[stack == self.overexposed_value] = np.nan
            stack[stack == self.underexposed_value] = np.nan

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
        return {'illumination': self.illumination}

    def load_from_dict(self, parameter_dict):
        self.illumination = parameter_dict['illumination']


class ExposureCorrector(corr.GeneralCorrector):

    def __init__(self, bit=12):
        self.bit = bit
        self.max_value = 2 ** self.bit - 1
        self.min_value = 0
        self.replace_value = np.nan

    def correct(self, stack):
        stack[stack == self.max_value] = self.replace_value
        stack[stack == self.min_value] = self.replace_value

        return stack

    def to_dict(self):
        return {'bit': self.bit,
                'max_value': self.max_value,
                'min_value': self.min_value,
                'replace_value': self.replace_value}

    def load_from_dict(self, parameter_dict):
        self.bit = parameter_dict['bit']
        self.max_value = parameter_dict['max_value']
        self.min_value = parameter_dict['min_value']
        self.replace_value = parameter_dict['replace_value']
