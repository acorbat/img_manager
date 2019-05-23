import json
import inspect
import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt

from collections import OrderedDict
from . import correctors


class CorrectorArmy(object):
    """An image or stack image corrector for background, bleaching and bleeding. Parameters for correction can be modified
    and fit accordingly.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self):
        """Return an image corrector with some initial parameters for background, bleaching and bleeding correction."""

        self.channels = OrderedDict()

    def add_channel(self, channel):
        if isinstance(channel, str):
            channel = Channel(channel)

        self.channels.update({channel.name: channel})

    def to_dict(self):
        channel_dict = {chan: self.channels[chan].to_dict() for chan in self.channels}

        return channel_dict

    def save(self, path):
        """Save actual state of corrector to json.

        Parameters
        ----------
        path : path
            path to file where corrector is to be saved
        """
        with open(str(path), 'w') as fp:
            json.dump(self.to_dict(), fp)

    def load(self, path):
        """Load parameters from json file to corrector.

        Parameters
        ----------
        path : path
            path to file where corrector is to be saved
        """
        # TODO: Not implemented
        with open(str(path), 'r') as fp:
            data = json.load(fp)

            self.bkg_params.loads(data['bkg'])
            self.bleach_params.loads(data['bleach'])
            self.bleed_mean, self.bleed_error = data['bleed']

    def run_correctors(self, stack, time_step):
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
        for channel in self.channels:
            channel.correct_background()
            channel.correct_bleaching()
            channel.correct_shift()

        # Implement bleeding correction should be separate as it happens after every channel is corrected for the rest.
        # for channel in self.channels:
        #     for source_channel, corrector in channel.bleeding_correctors:
        #         corrector.correct(self.channels[source_channel].stack, channel.stack)

    def __getitem__(self, item):
        return self.channels[item]


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
        if GeneralCorrector.__name__ not in [c.__name__ for c in inspect.getmro(type(background_corrector))]:
            raise TypeError('Not a child of GeneralCorrector')

        self.background_correctors.append(background_corrector)

    def add_bleaching_corrector(self, bleaching_corrector):
        if GeneralCorrector.__name__ not in [c.__name__ for c in inspect.getmro(type(bleaching_corrector))]:
            raise TypeError('Not a child of GeneralCorrector')

        self.bleaching_correctors.append(bleaching_corrector)

    def add_shift_corrector(self, shift_corrector):
        if GeneralCorrector.__name__ not in [c.__name__ for c in inspect.getmro(type(shift_corrector))]:
            raise TypeError('Not a child of GeneralCorrector')

        self.shift_correctors.append(shift_corrector)

    def add_bleeding_corrector(self, bleeding_corrector, source_channel):
        if GeneralCorrector.__name__ not in [c.__name__ for c in inspect.getmro(type(bleeding_corrector))]:
            raise TypeError('Not a child of GeneralCorrector')

        self.bleeding_correctors.append((source_channel, bleeding_corrector))

    def correct_background(self):
        for background_corrector in self.background_correctors:
            self.stack = background_corrector.correct(self.stack)

        self.stack_state.append('background corrected')

    def correct_bleaching(self):
        for bleaching_corrector in self.bleaching_correctors:
            self.stack = bleaching_corrector.correct(self.stack)

        self.stack_state.append('bleaching corrected')

    def correct_shift(self):
        for shift_corrector in self.shift_correctors:
            self.stack = shift_corrector.correct(self.stack)

        self.stack_state.append('shift corrected')

    def correct_bleeding(self, source_stack):
        # TODO: not well implemented
        for bleed_corrector in self.bleeding_correctors:
            self.stack = bleed_corrector.correct(source_stack, self.stack)

        self.stack_state.append('bleeding corrected')

    def to_dict(self):
        background_dict = {n: this_corr.to_dict() for n, this_corr in enumerate(self.background_correctors)}
        bleaching_dict = {n: this_corr.to_dict() for n, this_corr in enumerate(self.bleaching_correctors)}
        shift_dict = {n: this_corr.to_dict() for n, this_corr in enumerate(self.shift_correctors)}
        bleed_dict = {source_channel: this_corr.to_dict() for source_channel, this_corr in self.bleeding_correctors}

        channel_dictionary = {'background': background_dict,
                              'bleaching': bleaching_dict,
                              'shift': shift_dict,
                              'bleeding': bleed_dict}

        return channel_dictionary

    def load_from_dict(self):
        # TODO: implement a load. How do we recognize which corrector load
        pass


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
