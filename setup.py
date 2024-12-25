# setup.py
from setuptools import setup, find_packages

setup(
    name="quantum_gravity",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'mpi4py',
        'matplotlib'
    ]
)