import json
import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt


class Corrector(object):
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
        Loads a dark image stack and fits and saves the background correction paramaters
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
        """Return an image corrector with some initial parameters for background, bleaching and blleding correction."""
        # background
        self.bkg_model = lm.Model(self.bkg_exponential, independent_vars=['x'])
        self.bkg_params = self.bkg_model.make_params(amplitude=9.30909784,
                                                     characteristic_time=152.75328323,
                                                     constant=43.32958973)
        self.bkg_params['amplitude'].set(min=0)
        self.bkg_params['characteristic_time'].set(min=0)

        # bleaching
        self.bleach_model = lm.Model(self.bleaching_exponential, independent_vars=['x'])
        self.bleach_params = self.bleach_model.make_params(amplitude=1.23445829e+06,
                                                           characteristic_time=5.91511605e+02,
                                                           constant=1.33089950e-04)

        # bleeding
        self.bleed_mean = 0
        self.bleed_error = 0

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
        Depending on input, returns value, list or array of background noise for the input timepoints
        """
        return -amplitude * np.exp(-x / characteristic_time) + constant

    def subtract_background(self, stack, time_step):
        """Subtracts background from a stack, considering time steps between images and using saved background
        parameters.

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
        """Given a dark noise stack, it calculates mean per timepoint, fits the parameters and saves them. If pp is
        given a PdfPages, images of fit are saved.

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
        Depending on input, returns value, list or array of intensity values for the input timepoints
        """
        return amplitude * np.exp(-x / characteristic_time) + constant

    def correct_bleaching(self, stack, time_step):
        """Corrects bleaching effects from a stack, considering time steps between images and using saved bleaching
        parameters.

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
        stack_corrected = stack.copy()
        times = np.arange(0, time_step * len(stack), time_step)
        bleached_intensity = self.bleach_model.eval(self.bleach_params, x=times)
        for ind, frame in enumerate(stack_corrected):
            stack_corrected[ind] = frame / bleached_intensity[ind]

        return stack_corrected

    def find_bleaching(self, stack, time_step, pp=None):
        """Given a stack of images, it calculates sum of intensity per timepoint, fits the bleaching exponential curve,
        finds the ideal parameters and saves them. If pp is given a PdfPages, images of fit are saved.

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
        times = np.arange(0, time_step * len(stack), time_step)

        bleach_stack = stack.copy()
        total_intensity = [np.nansum(this_stack) for this_stack in bleach_stack]

        result = self.bleach_model.fit(total_intensity, params=self.bleach_params, x=times)

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
