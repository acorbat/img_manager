import numpy as np
import xarray as xr
from itertools import islice
from collections.abc import Iterable

from czifile import CziFile


class LSM880(CziFile):
    """LSM880 inherits from CziFile and adds some functionality to make it
    easier to manipulate data.

    Attributes
    ----------
    key_index : xarray
        Xarray where dimensions are dimensions present in data, coordinates are
        integers and keys to blocks are data.

    Methods
    -------
    get_dim_dict(self)
        Generate a dictionary with the size of each dimension.
    get_key_index(self):
        Generate a matrix with the shape of the data and the keys for each
        image.
    block(self, block_index)
        Returns a block given an index.
    groupby(self, dim)
        Iterator over the selected dimensions.
    """

    def __init__(self, arg):
        super().__init__(arg)
        self.key_index = self.get_key_index()

    def get_dim_dict(self):
        """Generate a dictionary with the size of each dimension."""
        return {this: that for this, that in zip(self.axes, self.shape)}

    def get_key_index(self):
        """Generate a matrix with the shape of the data and the keys for each
        image."""
        dim_dict = self.get_dim_dict()
        axes = [ax for ax in self.axes
                if dim_dict[ax] != 1 and ax not in ['X', 'Y']]

        key_index = np.zeros(tuple(dim_dict[dim] for dim in axes), dtype=int)
        dim_dict_inds = {dim: range(dim_dict[dim]) for dim in axes}
        key_index = xr.DataArray(key_index, coords=dim_dict_inds, dims=axes)

        for ind, block in enumerate(self.filtered_subblock_directory):
            block_dict = {this.dimension: this.start for this in
                          block.dimension_entries}
            dim_ind = tuple(block_dict[dim] for dim in axes)
            key_index[dim_ind] = ind

        return key_index

    def block(self, block_index):
        """
        Returns a block given an index. In case that the
        block does not exist it raises an Exception.

        Parameters
        ----------
        block_index : int, iterable
            index of block or blocks to retrieve
        Returns
        -------
        blocks of an image.
        """
        if np.iterable(block_index):
            block = []
            for this_index in block_index:
                block.append(self.block(this_index))
            block = np.asarray(block)

        else:
            block_iterator = self.filtered_subblock_directory
            slice_ = islice(block_iterator, block_index, None)
            block = next(slice_, None)
            block = block.data_segment().data()
            block = np.squeeze(block)

        return block

    def groupby(self, dim):
        """Iterator over the selected dimensions."""
        for this in self.key_index.groupby(dim):
            yield this

    def get_times(self):
        """Gets time from timestamps of each Series."""
        atachs = self.attachments()
        for attach in atachs:
            if 'TimeStamps' in str(attach):
                break

        times = np.diff(attach.data())
        return np.concatenate(([0], times))

    def __getitem__(self, key):
        """Load image according to given keys. If a dictionary, load the
        according images in the correct shape. If it is a int or list of ints,
        then return list of images."""
        if isinstance(key, dict):
            key_index = self.key_index[key]

            dim_dict = self.get_dim_dict()
            out_data = np.zeros(key_index.shape + (dim_dict['Y'], dim_dict['X']),
                                dtype='uint16')

            out_coords = dict(key_index.coords)
            out_coords.update({ax: range(dim_dict[ax]) for ax in ['Y', 'X']})

            out_dims = key_index.dims + ('Y', 'X')

            out = xr.DataArray(out_data, coords=out_coords, dims=out_dims)

            my_keys = key_index.stack(lin=key_index.dims)
            for this_vars, this_gp in my_keys.groupby('lin'):
                out.loc[{ax: var for ax, var in
                     zip(key_index.dims, this_vars)}] = self.block(
                    this_gp.values)

            return out

        elif isinstance(key, (Iterable, int)):
            return self.block(key)

        else:
            raise TypeError('Can not handle that type of key')
