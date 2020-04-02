from setuptools import setup, find_packages

setup(
    name='img_manager',
    version='0.2.0',
    packages=find_packages(),
    url='https://github.com/acorbat/img_manager/tree/master/img_manager',
    license='MIT',
    author='Agustin Corbat',
    author_email='acorbat@df.uba.ar',
    description='Image IO and correction.',
    install_requires=['numpy', 'matplotlib', 'lmfit', 'imreg_dft',
                      'tifffile', 'oiffile', 'czifile', 'xarray',
                      'scikit-image',
                      'serialize @ git+https://github.com/hgrecco/serialize',
                      'cellment @ git+https://github.com/maurosilber/cellment']
)
