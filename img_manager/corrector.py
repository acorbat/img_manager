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
        """Given a dark noise path, it loads, calculates mean per timepoint, fits the parameters and saves them. If pp
        is given a PdfPages, images of fit are saved.

        If fit is not converging, you can amnually modify parameters with:
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


    ## Bleaching Correction

    ## Bleeding Correction