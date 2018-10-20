import pathlib

from datetime import datetime

from img_manager import oiffile as oif

class FV1000(oif.OifFile):
    """FV1000 saves stacks and images in OifFile format and some of the most used functionalities need to be added to
    its methods.

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
    get_axis(self)
        Gets a string of the axis in order
    get_clip_bbox
        Gets the bounded box of the clipped image
    get_next_path(self)
        If it's a Time Controller automatically generated set, it gets the path for the next set of images
    get_last_path(self)
        It gets the last path for the existing images generated automatically by Time Controller
    get_other_path(self, kind='ble')
        For bleaching experiments, it changes the second to last underscored separated part for kind and gets the last
        path for that image
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
        return self.mainfile['Axis 3 Parameters Common']['Interval'] / 1000  # It was in nanometers

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

    def get_axis(self):
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
        time = self.mainfile['General']['ImageCaputreDate'][1:-1] + ' ' + self.mainfile['General']['ImageCaputreDate+MilliSec']
        return datetime.strptime(time, time_format)

    # Specific Function for automatic file generation from Time Controller
    ######################################################################
    def get_next_path(self):
        """If it's a Time Controller automatically generated set, it gets the path for the next set of images"""
        path = pathlib.Path(self._fname)
        name_parts = path.stem.split('_')
        num = int(name_parts[-1]) + 1
        next_name = '_'.join(name_parts[:-1]) + '_%02d.oif' % num
        return path.with_name(next_name)

    def get_last_path(self):
        """It gets the last path for the existing images generated automatically by Time Controller"""
        path = pathlib.Path(self._fname)
        next_path = self.get_next_path()
        while next_path.exists():
            path = next_path
            next_path = self.get_next_path()

        return path

    def get_other_path(self, kind='ble'):
        """For bleaching experiments, it changes the second to last underscored separated part for kind and gets the
        last path for that image"""
        path = pathlib.Path(self._fname)
        name_parts = path.stem.split('_')
        next_name = '_'.join(name_parts[:-2]) + '_' + '_'.join([kind, '01.oif'])
        path = path.with_name(next_name)
        other_img = FV1000(str(path))
        return other_img.get_last_path()
