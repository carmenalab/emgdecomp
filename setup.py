from setuptools import setup, find_packages

setup(
    name = 'emgdecomp',
    version = '0.1.0',
    description = """Package for decomposing EMG signals into motor unit firings.""",
    author = "Emanuele Formento, Paul Botros",
    author_email = "pbotros@berkeley.edu",
    url = 'https://github.com/carmenalab/emgdecomp',
    packages = find_packages(include=['emgdecomp', 'emgdecomp.*']),
    include_package_data=True,
    setup_requires=['pytest-runner'],
    install_requires=[
        'matplotlib',
        'scipy',
        'numpy',
        'numba',
    ],
    tests_require=[
        'pytest',
        'pytest-mock',
    ],
    extras_require={
        'dask': ['dask[array,distributed]'],
        'cuda': ['cupy'],
    }
)

