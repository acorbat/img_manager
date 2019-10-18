import json
import numpy as np
import lmfit as lm
import imreg_dft as ird
import matplotlib.pyplot as plt

from img_manager import corrector as corr


class BleedingCorrector(corr.GeneralCorrector):

    def __init__(self):
        # bleeding
        self.bleed_mean = 0
        self.bleed_error = 0

    # Bleeding Correction
    #####################
    def correct_bleeding(self, channel_1, channel_2):
        """Correct bleeding in channel_1 into channel_2.

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
        """Finds bleeding factor from two lists corresponding to values from each channel. Mean of ratios is saved as
        bleeding factor, while error of the mean is saved as its error. If pp is given a PdfPages, the a histogram of
        ratios is saved.

        Lists of values may proceed from mean of labeled regions or lists of flattened pixels, etc.

        Parameters
        ----------
        list_channel_1 : list, numpy.array
            one dimensional array or list of values corresponding to intensities from channel_1
        list_channel_2 : list, numpy.array
            one dimensional array or list of values corresponding to intensities from channel_2
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
        """This function subtracts background from stack. time_step attribute is used."""
        return self.correct_bleeding(stack_source, stack_to_correct)

    def to_dict(self):
        """Returns an OrderedDict with the parameters."""
        # TODO: test
        return {'bleed_mean': self.bleed_mean,
                'bleed_error': self.bleed_error}

    def load_from_dict(self, valuesdict):
        """Loads the parameters from a saved OrderedDict"""
        # TODO: test
        self.bleed_mean = valuesdict['bleed_mean']
        self.bleed_error = valuesdict['bleed_error']


class ShiftCorrector(corr.GeneralCorrector):

    def __init__(self, tvec=None):
        self.tvec = tvec

    def find_shift(self, master_stack, stack_to_move):
        result = ird.translation(master_stack, stack_to_move)
        self.tvec = [int(this) for this in result['tvec']]

    def correct(self, stack):
        if len(stack.shape) == 2:
            stack = stack[np.newaxis, :]

        for n, this_img in enumerate(stack):
            stack[n] = ird.imreg.transform_img(this_img, tvec=self.tvec, mode='nearest')

        return stack

    def to_dict(self):
        return {'tvec': self.tvec}

    def load_from_dict(self, parameter_dict):
        self.tvec = parameter_dict['tvec']
