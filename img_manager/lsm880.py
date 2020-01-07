import numpy as np
from itertools import islice

from czifile import CziFile


class LSM880(CziFile):

    def __init__(self, arg):
        super().__init__(arg)
        self.ind_axes, self.key_index = self.get_key_index()

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

        for ind, block in enumerate(self.subblock_directory):
            block_dict = {this.dimension: this.start for this in
                          block.dimension_entries}
            dim_ind = tuple(block_dict[dim] for dim in axes)
            key_index[dim_ind] = ind

        return axes, key_index

    def block(self, block_index):
        """
        Returns a block given an index. In case that the
        block does not exist it raises an Exception.

        Parameters
        ----------
        path : string, pathlib.Path
            path to the file
        block_index : int
            index of block to retrieve
        Returns
        -------
        blocks of an image.
        """
        block_iterator = self.filtered_subblock_directory
        slice_ = islice(block_iterator, block_index, None)
        block = next(slice_, None)
        block = block.data_segment().data()
        block = np.squeeze(block)

        return block
