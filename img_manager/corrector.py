import numpy as np
import lmfit as lm

from img_manager import oiffile as oif
from img_manager import tifffile as tif

from foci_finder import pipelines as pipe

class Corrector(object):
    """An image or stack image corrector for background and bleaching. Parameters for correction can be modified
    accordingly."""

    def __init__(self):
        """Return an image corrector with some initial parameters for background and bleaching correction."""
        # background
        self.bkg_model = lm.Model(self.bkg_exponential, independent_vars=['x'])
        self.bkg_params = self.bkg_model.make_params(amplitude=9.30909784,
                                                     characteristic_time=152.75328323,
                                                     constant=43.32958973)
        self.bkg_params['amplitude'].set(min=0)
        self.bkg_params['characteristic_time'].set(min=0)

    ## Background Correction
    @staticmethod
    def bkg_exponential(x, amplitude, characteristic_time, constant):
        return -amplitude * np.exp(-x / characteristic_time) + constant


    def subtract_background(self, stack, time_step):
        times = np.arange(0, time_step * len(stack), time_step)
        stack_corrected = stack.copy()
        background = self.bkg_model.eval(self.bkg_params, x=times)
        for ind, frame in enumerate(stack_corrected):
            stack_corrected[ind] = frame - background[ind]

        return stack_corrected


    def find_bkg_correction(self, dark_img_dir):
        dark_img = oif.OifFile(str(dark_img_dir))

        stack = dark_img.asarray()[0].astype(float)
        stack = tif.transpose_axes(stack, 'ZTYX', asaxes='TZYX')

        time_step = pipe.get_t_step(dark_img)
        times = np.arange(0, time_step * len(stack), time_step)

        means = []
        for frame in stack:
            means.append(np.mean(frame.flatten()))

        result = self.bkg_model.fit(means, params=self.bkg_params, x=times)
        self.bkg_params = result.best_values

    ## Bleaching Correction

    ## Bleeding Correction