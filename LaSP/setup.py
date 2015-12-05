from distutils.core import setup

setup(
    name='LaSP',
    version='0.5',
    packages=['lasp',],
    license='',
    long_description=open('README.md').read(),
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'tables',
                      'h5py',
                      'nitime',
                      'spams',
                      'brian',
                      'pandas'
        ]
)
