# img_manager
Aimed to managing image files ordered in folders.

## Corrector
Corrector contains the corrector class which aims at finding and applying background, bleaching and bleeding parameters for correction each channel.

For the time being, there should be one Corrector for each channel with its own parameters. Bleeding can only be applied between two channels and this should be fixed.

## FV1000
The FV1000 class was made to save usual getters for parameters that are repeteadly needed between different analytical pipelines.

There has been some added functionality for handling paths of automatically generated filenames by Time Controller and specific ways of generating names in bleaching experiments. This functionality is also too specific. 

## Acknowledgements
I am grateful for Christoph Gohlke for providing the backbone for reading Tiff and Oif files (https://www.lfd.uci.edu/~gohlke/)
