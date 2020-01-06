import pathlib

from datetime import datetime

import oiffile as oif
import tifffile as tif


class FV1000(oif.OifFile):
    """FV1000 saves stacks and images in OifFile format and some of the most
    used functionalities need to be added to its methods.

    Methods
    -------
    get_x_step(self)
        Get the pixel size in x direction in microns
    get_y_step(self)
        Get the pixel size in y direction in microns
    get_z_step(self)
        Get the pixel size in z direction in microns
    get_t_step(self)
        Get time step in seconds
    get_scales(self)
        Get dictionary with scales
    get_axes(self)
        Gets a string of the axis in order
    get_clip_bbox
        Gets the bounded box of the clipped image
    get_axes_info(self, axes)
        Gets dictionary with all the information concerning specified axes
    transpose_axes(self)
        Loads the stack in the order specified by axes
    get_next_path(self)
        If it's a Time Controller automatically generated set, it gets the path
        for the next set of images
    get_last_path(self)
        It gets the last path for the existing images generated automatically
        by Time Controller
    get_other_path(self, kind='ble')
        For bleaching experiments, it changes the second to last underscored
        separated part for kind and gets the last path for that image
    """

    # Usually used getters for stack parameters
    ###########################################
    def get_x_step(self):
        """Get the pixel size in x direction in microns"""
        return self.mainfile['Reference Image Parameter']['WidthConvertValue']

    def get_y_step(self):
        """Get the pixel size in y direction in microns"""
        return self.mainfile['Reference Image Parameter']['HeightConvertValue']

    def get_z_step(self):
        """Get the pixel size in z direction in microns"""
        # It was in nanometers
        return self.mainfile['Axis 3 Parameters Common']['Interval'] / 1000

    def get_t_step(self):
        """Get time step in seconds"""
        start = self.mainfile['Axis 4 Parameters Common']['StartPosition']
        end = self.mainfile['Axis 4 Parameters Common']['EndPosition']
        size_t = self.mainfile['Axis 4 Parameters Common']['MaxSize']
        return (end - start) / ((size_t - 1) * 1000)

    def get_scales(self):
        """Get dictionary with scales"""
        scales = {'X': self.get_x_step(),
                  'Y': self.get_y_step(),
                  'Z': self.get_z_step(),
                  'T': self.get_t_step()}
        return scales

    def get_axes(self):
        """Gets a string of the axis in order"""
        axes = self.mainfile['Axis Parameter Common']['AxisOrder']
        axes = axes[2:] + 'YX'
        return axes

    def get_clip_bbox(self):
        """Gets the bounded box of the clipped image"""
        x_start = self.mainfile['Axis 0 Parameters Common']['ClipPosition']
        y_start = self.mainfile['Axis 1 Parameters Common']['ClipPosition']
        x_size = self.mainfile['Axis 0 Parameters Common']['MaxSize']
        y_size = self.mainfile['Axis 1 Parameters Common']['MaxSize']
        return (y_start, y_start + y_size, x_start, x_start + x_size)

    def get_acquisition_time(self):
        """Gets acquistion time of stack in datetime format"""
        time_format = "%Y-%m-%d %H:%M:%S %f"
        time = self.mainfile['Acquisition Parameters Common']['ImageCaputreDate'] + ' ' +\
               str(self.mainfile['Acquisition Parameters Common']['ImageCaputreDate+MilliSec'])

        return datetime.strptime(time, time_format)

    def get_axes_info(self, axes):
        """Gets dictionary with all the information concerning specified axes"""
        if isinstance(axes, str):
            axes_dict = {ax: str(i) for i, ax in enumerate(['X', 'Y', 'C', 'Z', 'T', 'A', 'L', 'P', 'Q'])}
            axes = axes_dict[axes]

        return self.mainfile['Axis ' + str(axes) + ' Parameters Common']

    def transpose_axes(self, axes, dtype='float'):
        """Loads the stack in the order specified by axes"""
        stack = self.asarray().astype(dtype)
        actual_axes = self.get_axes()
        return tif.transpose_axes(stack, actual_axes, asaxes=axes)

    # Specific Function for automatic file generation from Time Controller
    ######################################################################
    def get_next_path(self, path=None):
        """If it's a Time Controller automatically generated set, it gets the path for the next set of images"""
        if path is None:
            path = pathlib.Path(self._fname)
        name_parts = path.stem.split('_')
        num = int(name_parts[-1]) + 1
        next_name = '_'.join(name_parts[:-1]) + '_%02d.oif' % num
        return path.with_name(next_name)

    def get_last_path(self, path=None):
        """It gets the last path for the existing images generated automatically by Time Controller"""
        if path is None:
            path = pathlib.Path(self._fname)
        next_path = self.get_next_path(path=path)
        while next_path.exists():
            path = next_path
            next_path = self.get_next_path(path=path)

        return path

    def get_other_path(self, kind='ble'):
        """For bleaching experiments, it changes the second to last underscored separated part for kind and gets the
        last path for that image"""
        path = pathlib.Path(self._fname)
        name_parts = path.stem.split('_')
        next_name = '_'.join(name_parts[:-2]) + '_' + '_'.join([kind, '01.oif'])
        path = path.with_name(next_name)
        path = self.get_last_path(path=path)

        return path
