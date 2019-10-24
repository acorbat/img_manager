import json
import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt

from img_manager import corrector as corr

class BleachingCorrector(corr.GeneralCorrector):
    """Bleaching Corrector estimates bleaching by fitting an exponential
    function to data and then divides a stack by this values to correct for
    bleaching.

    Attributes
    ----------
    bleach_model : lmfit.Model class
        The exponential model used to fit the dark noise images
    bleach_params : lmfit.Parameters class
        The parameters for the background correction

    Methods
    -------
    bleaching_exponential(x, amplitude, characteristic_time, constant)
        Exponential function to be used in bleaching fitting
    correct_bleaching(self, stack)
        Corrects bleaching effects from a stack, considering time steps
        between images and using saved bleaching parameters.
    find_bleaching(self, stack, pp=None)
        Given a stack of images, it calculates sum of intensity per
        timepoint, fits the bleaching exponential curve, finds the ideal
        parameters and saves them. If pp is given a PdfPages, images of fit are
        saved.
    correct(stack)
        This function corrects bleaching from stack.
    to_dict()
        Returns an OrderedDict with the parameters.
    load_from_dict(path)
        Loads the parameters from a saved OrderedDict
    """

    def __init__(self):

        self.corrector_species = 'BleachingCorrector'

        # bleaching
        self.bleach_model = lm.Model(self.bleaching_exponential, independent_vars=['x'])
        self.bleach_params = self.bleach_model.make_params(amplitude=1.23445829e+06,
                                                           characteristic_time=5.91511605e+02,
                                                           constant=1.33089950e-04)

    # Bleaching Correction
    ######################
    @staticmethod
    def bleaching_exponential(x, amplitude, characteristic_time, constant):
        """Exponential function to fit bleaching in time series.

        Parameters
        ----------
        x : list, numpy.array
            timepoints of function evaluation
        amplitude : float
            Amplitude of bleaching decay
        characteristic_time : float
            Characteristic time of bleaching
        constant : float
            Constant value reached after long exposure

        Returns
        -------
        Depending on input, returns value, list or array of intensity values
        for the input timepoints
        """
        return amplitude * np.exp(-x / characteristic_time) + constant

    def correct_bleaching(self, stack):
        """Corrects bleaching effects from a stack, considering time steps
        between images and using saved bleaching parameters.

        Parameters
        ----------
        stack : numpy.array
            Time series of images to be corrected

        Returns
        -------
            Returns the corrected stack
        """
        stack_corrected = stack.copy()
        times = np.arange(0, len(stack))
        bleached_intensity = self.bleach_model.eval(self.bleach_params, x=times)
        for ind, frame in enumerate(stack_corrected):
            stack_corrected[ind] = frame / bleached_intensity[ind]

        return stack_corrected

    def find_bleaching(self, stack, pp=None):
        """Given a stack of images, it calculates sum of intensity per
        timepoint, fits the bleaching exponential curve, finds the ideal
        parameters and saves them. If pp is given a PdfPages, images of fit are
         saved.

        The set of images given should already be background subtracted.

        If fit is not converging, you can amnually modify parameters with:
            >>> corrector.bleach_params['constant'].set(value=40)
            >>> corrector.bleach_params['amplitude'].set(value=10)
            >>> corrector.bleach_params['characteristic_time'].set(value=20)

        Parameters
        ----------
        stack : numpy.array
            stack of images to find intensity bleaching (can be masked with nan images). Background should be subtracted
            before
        time_step : float
            Time step between acquired images
        pp : PdfPages
            Images of fitting are saved in this pdf
        """
        times = np.arange(0, (len(stack) + 1))[0:len(stack)]

        bleach_stack = stack.copy()
        total_intensity = [np.nansum(this_stack) for this_stack in bleach_stack]

        result = self.bleach_model.fit(total_intensity,
                                       params=self.bleach_params, x=times)

        for key in result.best_values.keys():
            self.bleach_params[key].set(value=result.best_values[key])

        # Save a pdf of applied correction
        if pp is not None:
            plt.plot(times, total_intensity, 'ob')
            plt.plot(times, self.bleach_model.eval(self.bleach_params, x=times))
            plt.xlabel('time (s)')
            plt.ylabel('summed intensity (a.u.)')
            pp.savefig()
            plt.close()

            corrected = self.correct_bleaching(stack, time_step)

            total_intensity = [np.nansum(this_stack) for this_stack in corrected]

            plt.plot(times, total_intensity, 'ob')
            plt.xlabel('time (s)')
            plt.ylabel('mean intensity (a.u.)')
            pp.savefig()
            plt.close()

    def correct(self, stack):
        """This function corrects bleaching from stack."""
        return self.correct_bleaching(stack)

    def to_dict(self):
        """Returns an OrderedDict with the parameters."""
        # TODO: test
        return {'corrector_species': self.corrector_species,
                'params': self.bleach_params.valuesdict()}

    def load_from_dict(self, valuesdict):
        """Loads the parameters from a saved OrderedDict"""
        # TODO: test
        self.bleach_params = self.bleach_params.make_params(valuesdict['params'])
