import json
import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt

from . import correctors


class CorrectorArmy(object):
    """An image or stack image corrector for background, bleaching and bleeding. Parameters for correction can be modified
    and fit accordingly.

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
        Loads a dark image stack and fits and saves the background correction parameters
    bleaching_exponential(x, amplitude, characteristic_time, constant)
        Exponential function to fit bleaching in time series
    correct_bleaching(self, stack, time_step)
        Corrects bleaching effects from a stack, considering time steps between images and using saved bleaching
        parameters
    find_bleaching(self, stack, time_step, pp=None)
        Given a stack of images, it calculates sum of intensity per timepoint, fits the bleaching exponential curve,
        finds the ideal parameters and saves them. If pp is given a PdfPages, images of fit are saved.
    """

    def __init__(self):
        """Return an image corrector with some initial parameters for background, bleaching and bleeding correction."""

        self.channels = []

    # General
    #########
    def save(self, path):
        """Save actual state of corrector to json.

        Parameters
        ----------
        path : path
            path to file where corrector is to be saved
        """
        with open(str(path), 'w') as fp:
            data = {'bkg': self.bkg_params.dumps(),
                    'bleach': self.bleach_params.dumps(),
                    'bleed': [self.bleed_mean, self.bleed_error]}
            json.dump(data, fp)

    def load(self, path):
        """Load parameters from json file to corrector.

        Parameters
        ----------
        path : path
            path to file where corrector is to be saved
        """
        with open(str(path), 'r') as fp:
            data = json.load(fp)

            self.bkg_params.loads(data['bkg'])
            self.bleach_params.loads(data['bleach'])
            self.bleed_mean, self.bleed_error = data['bleed']

    def subtract_and_normalize(self, stack, time_step):
        """Apply consecutively a background subtraction and a bleaching normalization.

        Parameters
        ----------
        stack : numpy.array
            Time series of images to be corrected
        time_step
            Time step between acquired images

        Returns
        -------
        stack_corrected : numpy.array
            Returns the corrected numpy.array for background and bleaching
        """
        stack_corrected = self.subtract_background(stack, time_step)

        stack_corrected = self.correct_bleaching(stack_corrected, time_step)

        return stack_corrected


class Channel(object):

    def __init__(self, name):
        self.name = name
        self.background_correctors = []
        self.bleaching_correctors = []
        self.shift_correctors = []
        self.bleeding_correctors = []

        self.stack = [None]
        self.stack_state = []  # attribute to append procedures applied to the stack

    def load_stack(self, stack):
        """Load a stack to the channel. It restarts state variables."""
        if not len(stack.shape) > 2:
            stack = stack[np.newaxis, ...]

        self.stack = stack
        self.stack_state = []
        self.stack_state.append('loaded')

    def add_background_corrector(self, background_corrector):
        if not issubclass(background_corrector, GeneralCorrector):
            raise TypeError('Not a child of GeneralCorrector')

        self.background_correctors.append(background_corrector)

    def add_bleaching_corrector(self, bleaching_corrector):
        if not issubclass(bleaching_corrector, GeneralCorrector):
            raise TypeError('Not a child of GeneralCorrector')

        self.bleaching_correctors.append(bleaching_corrector)

    def add_shift_corrector(self, shift_corrector):
        if not issubclass(shift_corrector, GeneralCorrector):
            raise TypeError('Not a child of GeneralCorrector')

        self.bleaching_correctors.append(shift_corrector)

    def add_bleeding_corrector(self, bleeding_corrector):
        if not issubclass(bleeding_corrector, GeneralCorrector):
            raise TypeError('Not a child of GeneralCorrector')

        self.bleaching_correctors.append(bleeding_corrector)


class GeneralCorrector(object):
    """Example of the mandatory methods required by any corrector."""

    def __init__(self):
        self.corrector_species = 'General'

    def correct(self, stack):
        """This function should only receive a stack and return the stack with the correction."""
        raise NotImplementedError('This corrector has not implemented the correct method')

    def to_dict(self):
        """This method should be implemented to be able to save the variables of the corrector as a dictionary."""
        raise NotImplementedError('This corrector has not implemented the to_dict method')

    def load_from_dict(self, parameter_dict):
        """This method should be implemented to be able to load the variables of the corrector from a dictionary."""
        raise NotImplementedError('This corrector has not implemented the load_from_dict method')
