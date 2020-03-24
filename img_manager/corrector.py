import inspect
import numpy as np

from collections import OrderedDict
from serialize import dump, load


class CorrectorArmy(object):
    """CorrectorArmy class allows to keep track of all the channels in an
    experiment, where each channel keeps track of all the corrections applied
    to each one including the relation between them, such as shift or bleeding.

    Attributes
    ----------
    channels : OrderedDict
        An ordered dictionary containing all the channels

    Methods
    -------
    add_channel(Channel Class)
        adds a channel to the army
    to_dict()
        returns a dictionary with the information of every channel
    save(path)
        Saves the army to file
    load(path)
        Loads an army from a file (not implemented)
    run_correctors()
        runs all the correctors in every channel
    """

    def __init__(self):
        """Return an image corrector with some initial parameters for
        background, bleaching and bleeding correction."""

        self.channels = OrderedDict()

    def add_channel(self, channel):
        """Adds a channel to the corrector army.

        Parameters
        ----------
        channel : Channel, str
            channel to be added or created if not Channel class
        """
        if isinstance(channel, str):
            channel = Channel(channel)

        self.channels.update({channel.name: channel})

    def to_dict(self):
        """Returns a dictionary containing all the information of the different
         channels and their correctors.

        Returns
        -------
        A dictionary containing all the channels and their correctors
        """
        channel_dict = {chan: self.channels[chan].to_dict() for chan in self.channels}

        return channel_dict

    def save(self, path):
        """Save actual state of corrector to yaml.

        Parameters
        ----------
        path : path
            path to file where army is to be saved
        """
        dump(self.to_dict(), path)

    def load(self, path):
        """Load parameters from yaml file to CorrectorArmy.

        Parameters
        ----------
        path : path
            path to file where army is to be saved
        """
        # TODO: Not implemented
        data = load(path)

        self.bkg_params.loads(data['bkg'])
        self.bleach_params.loads(data['bleach'])
        self.bleed_mean, self.bleed_error = data['bleed']

    def run_correctors(self):
        """Runs every corrector on every channel so that stacks are corrected.
        """
        for channel in self.channels:
            self.channels[channel].correct_background()
            self.channels[channel].correct_bleaching()
            self.channels[channel].correct_shift()

        # Implement bleeding correction should be separate as it happens after every channel is corrected for the rest.
        # for channel in self.channels:
        #     for source_channel, corrector in channel.bleeding_correctors:
        #         corrector.correct(self.channels[source_channel].stack, channel.stack)

    def __getitem__(self, item):
        return self.channels[item]


class Channel(object):
    """Channel class contains all the information pertaining a specific channel
    of the experiment to be analyzed. It has all the correctors that are to be
    applied to the stacks. This class also contains the stack of images to be
    corrected.

    Attributes
    ----------
    name : str
        name of the channel
    background_correctors : list
        List of background corrector to be subsequently applied to the stack
    bleaching_correctors : list
        List of bleaching corrector to be subsequently applied to the stack
    shift_correctors : list
        List of shift corrector to be subsequently applied to the stack
    bleeding_correctors : list
        List of bleeding corrector to be subsequently applied to the stack
    stack : numpy.ndarray
        Stack of images to be corrected (or already corrected)
    stack_state : list
        list of applied procedures on stack

    Methods
    -------
    load_stack(stack)
        Load a stack to the channel. It restarts state variables.
    add_background_corrector(background_corrector)
        Adds a background corrector only if it is a child of GeneralCorrector
    add_bleaching_corrector(background_corrector)
        Adds a background corrector only if it is a child of GeneralCorrector
    add_shift_corrector(shift_corrector)
        Adds a shift corrector only if it is a child of GeneralCorrector
    add_bleeding_corrector(bleeding_corrector, source_channel)
        Adds a bleeding corrector only if it is a child of GeneralCorrector
    correct_background()
        Runs all the background correctors in channel.
    correct_bleaching()
        Runs all the bleaching correctors in channel.
    correct_shift()
        Runs all the shift correctors in channel.
    correct_bleeding()
        Runs all the bleeding correctors in channel.
    to_dict()
        Creates a dictionary enumerating every corrector of each kind and where
         each value is the dictionary obtained from the corrector.
    load_from_dict(path)
        Not Implemented Yet
    """

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
        """Adds a background corrector only if it is a child of
        GeneralCorrector"""
        if GeneralCorrector.__name__ not in [c.__name__ for c in inspect.getmro(type(background_corrector))]:
            raise TypeError('Not a child of GeneralCorrector')

        self.background_correctors.append(background_corrector)

    def add_bleaching_corrector(self, bleaching_corrector):
        """Adds a bleaching corrector only if it is a child of
        GeneralCorrector"""
        if GeneralCorrector.__name__ not in [c.__name__ for c in inspect.getmro(type(bleaching_corrector))]:
            raise TypeError('Not a child of GeneralCorrector')

        self.bleaching_correctors.append(bleaching_corrector)

    def add_shift_corrector(self, shift_corrector):
        """Adds a shift corrector only if it is a child of GeneralCorrector"""
        if GeneralCorrector.__name__ not in [c.__name__ for c in inspect.getmro(type(shift_corrector))]:
            raise TypeError('Not a child of GeneralCorrector')

        self.shift_correctors.append(shift_corrector)

    def add_bleeding_corrector(self, bleeding_corrector, source_channel):
        """Adds a bleeding corrector only if it is a child of
        GeneralCorrector

        Parameters
        ----------
        bleeding_corrector : BleedingCorrector
            Bleeding Corrector to use
        source_channel : str
            Name of the channel to use as source of bleeding"""
        if GeneralCorrector.__name__ not in [c.__name__ for c in inspect.getmro(type(bleeding_corrector))]:
            raise TypeError('Not a child of GeneralCorrector')

        self.bleeding_correctors.append((source_channel, bleeding_corrector))

    def correct_background(self):
        """Runs all the background correctors in channel."""
        for background_corrector in self.background_correctors:
            self.stack = background_corrector.correct(self.stack)

        self.stack_state.append('background corrected')

    def correct_bleaching(self):
        """Runs all the bleaching correctors in channel."""
        for bleaching_corrector in self.bleaching_correctors:
            self.stack = bleaching_corrector.correct(self.stack)

        self.stack_state.append('bleaching corrected')

    def correct_shift(self):
        """Runs all the shift correctors in channel."""
        for shift_corrector in self.shift_correctors:
            self.stack = shift_corrector.correct(self.stack)

        self.stack_state.append('shift corrected')

    def correct_bleeding(self, source_stack):
        """Runs all the bleeding correctors in channel."""
        # TODO: not well implemented
        for bleed_corrector in self.bleeding_correctors:
            self.stack = bleed_corrector.correct(source_stack, self.stack)

        self.stack_state.append('bleeding corrected')

    def to_dict(self):
        """Creates a dictionary enumerating every corrector of each kind and
        where each value is the dictionary obtained from the corrector."""
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
