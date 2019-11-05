import json
import numpy as np
import lmfit as lm
import imreg_dft as ird
import matplotlib.pyplot as plt

from img_manager import corrector as corr


class BleedingCorrector(corr.GeneralCorrector):
    """This bleeding corrector can find the bleed factor between two channels
    and correct for it.

    Attributes
    ----------
    bleed_mean : float
        Bleeding factor between channels
    bleed_error : float
        Standard error of the mean of the estimated bleeding factor

    Methods
    -------
    correct_bleeding(channel_1, channel_2)
        Corrects bleeding from channel_1 into channel_2.
    find_bleeding(list_channel_1, list_channel_2, pp=None)
        Finds bleeding factor from two lists corresponding to values from
        each channel.
    correct(stack_source, stack_to_correct)
        This function corrects bleeding from one stack source into another
        stack_to_correct.
    to_dict()
        Returns a dictionary with the parameters.
    load_from_dict(valuesdict)
        Loads the parameters from a saved dictionary.
    """

    def __init__(self):
        self.corrector_species = 'BleedingCorrector'

        # bleeding
        self.bleed_mean = 0
        self.bleed_error = 0

    # Bleeding Correction
    #####################
    def correct_bleeding(self, channel_1, channel_2):
        """Corrects bleeding from channel_1 into channel_2.

        Parameters
        ----------
        channel_1 : numpy.array
            Stack of images from channel_1 one that is bleeding into channel_2
        channel_2 : numpy.array
            Stack of images from channel_2 that is being contaminated by channel_1

        Returns
        -------
        Corrected stacks of channel_2
        """
        return channel_2 - self.bleed_mean * channel_1

    def find_bleeding(self, list_channel_1, list_channel_2, pp=None):
        """Finds bleeding factor from two lists corresponding to values from
        each channel. Mean of ratios is saved as bleeding factor, while error
        of the mean is saved as its error. If pp is given a PdfPages, the a
        histogram of ratios is saved.

        Lists of values may proceed from mean of labeled regions or lists of
        flattened pixels, etc.

        Parameters
        ----------
        list_channel_1 : list, numpy.array
            one dimensional array or list of values corresponding to
            intensities from channel_1
        list_channel_2 : list, numpy.array
            one dimensional array or list of values corresponding to
            intensities from channel_2
        pp : PdfPages
            Images of histograms are saved in this pdf
        """
        assert len(list_channel_1) == len(list_channel_2), 'Lists are not the same length.'

        list_channel_1 = np.asarray(list_channel_1)
        list_channel_2 = np.asarray(list_channel_2)
        ratios = list_channel_2 / list_channel_1

        self.bleed_mean = np.nanmean(ratios)
        self.bleed_error = np.nanstd(ratios) / len(ratios)

        if pp is not None:
            plt.hist(ratios, bins=30)
            plt.xlabel('intensity ratio')
            plt.ylabel('frequency')
            pp.savefig()
            plt.close()

    def correct(self, stack_source, stack_to_correct):
        """This function corrects bleeding from one stack source into another
        stack_to_correct."""
        return self.correct_bleeding(stack_source, stack_to_correct)

    def to_dict(self):
        """Returns a dictionary with the parameters."""
        return {'corrector_species': 'BleedingCorrector',
                'bleed_mean': self.bleed_mean,
                'bleed_error': self.bleed_error}

    def load_from_dict(self, valuesdict):
        """Loads the parameters from a saved dictionary."""
        self.bleed_mean = valuesdict['bleed_mean']
        self.bleed_error = valuesdict['bleed_error']


class ShiftCorrector(corr.GeneralCorrector):
    """Estimates the shift between two channels using the image registration
    package (correlation) and corrects the shift of channels.

    Attributes
    ----------
    tvec : tuple
        vector to use as shift between channels

    Methods
    -------
    find_shift(master_stack, stack_to_move)
        Finds the necessary shift to correct shift_to_move in order to match
        master stack.
    correct(stack)
        Shifts the given stack of images according to tvec.
    to_dict()
        Returns a dictionary with the parameters.
    load_from_dict(parameter_dict)
        Loads the parameters from a saved dictionary.
    """
    def __init__(self, tvec=None):
        self.corrector_species = 'ShiftCorrector'
        self.tvec = tvec

    def find_shift(self, master_stack, stack_to_move):
        """Finds the necessary shift to correct shift_to_move in order to match
        master stack.

        Parameters
        ----------
        master_stack : numpy.ndarray 2D
            Image to use as reference
        stack_to_move : numpy.ndarray 2D
            Image that is to be shifted to match the reference
        """
        result = ird.translation(master_stack, stack_to_move)
        self.tvec = [int(this) for this in result['tvec']]

    def correct(self, stack):
        """Shifts the given stack of images according to tvec.

        Parameters
        ----------
        stack : numpy.ndarray
            stack of images to be shifted

        Returns
        -------
        stack : numpy.ndarray
            Shifted stack of images
        """
        if len(stack.shape) == 2:
            stack = stack[np.newaxis, :]

        for n, this_img in enumerate(stack):
            stack[n] = ird.imreg.transform_img(this_img, tvec=self.tvec, mode='nearest')

        return stack

    def to_dict(self):
        """Returns a dictionary with the parameters."""
        return {'corrector_species': 'ShiftCorrector',
                'tvec': self.tvec}

    def load_from_dict(self, parameter_dict):
        """Loads the parameters from a saved dictionary."""
        self.tvec = parameter_dict['tvec']
