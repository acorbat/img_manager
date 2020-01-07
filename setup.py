from setuptools import setup

setup(
    name='img_manager',
    version='0.2.0',
    packages=['img_manager'],
    url='https://github.com/acorbat/img_manager/tree/master/img_manager',
    license='MIT',
    author='Agustin Corbat',
    author_email='acorbat@df.uba.ar',
    description='Image IO and correction.',
    install_requires=['numpy', 'datetime', 'matplotlib', 'lmfit', 'imreg_dft',
                      'tifffile', 'oiffile'],
    dependency_links=['https://github.com/maurosilber/cellment.git',
                      'https://github.com/hgrecco/serialize.git']
)