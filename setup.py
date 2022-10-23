import imp
import os
from glob import glob

import setuptools

extra_files = glob(
    "colosseum" + os.sep + "benchmark" + os.sep + "**" + os.sep + "*.gin",
    recursive=True,
) + glob(
    "colosseum" + os.sep + "benchmark" + os.sep + "**" + os.sep + "*.yml",
    recursive=True,
)

setuptools.setup(
    name="rl-colosseum",
    description=(
        "A pioneering Python package that creates a bridge between theory and practice in tabular reinforcement learning."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MichelangeloConserva/Colosseum",
    author="Michelangelo Conserva",
    author_email="mi.conserva@gmail.com",
    license="Apache License, Version 2.0",
    version=imp.load_source("_metadata", "colosseum/_metadata.py").__version__,
    keywords="reinforcement-learning python machine-learning",
    packages=setuptools.find_packages(),
    package_data={"rl-colosseum": extra_files},
    install_requires=[
        "adjustText",
        "dm-env",
        "gin-config",
        "gym",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "pydtmc",
        "pygame",
        "pyyaml",
        "ray[tune]==1.9.1",
        "scipy>=1.7.3",
        "seaborn>=0.10",
        "sparse",
        "toolz",
        "tqdm",
        "PyYAML>=6.0",
        "pygraphviz",
        "wrapt-timeout-decorator",
        "bsuite[baseline]",
        "dm-sonnet",
        "dm-tree",
        "tensorflow",
        "tensorflow_probability",
        "trfl",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
