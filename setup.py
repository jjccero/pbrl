from setuptools import find_packages, setup

from pbrl import __version__

setup(
    name='pbrl',
    version=__version__,
    description='PBRL: A General Reinforcement Learning Library based on PyTorch',
    author='jjccero',
    url='https://github.com/jjccero/pbrl',
    license='MIT',
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=[
        'pyglet',
        'gym==0.19',
        'numpy>=1.19',
        'cloudpickle',
        'tensorboard',
        'tensorboardx',
        'torch>=1.7'
    ]
)
