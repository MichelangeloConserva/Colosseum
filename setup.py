import os
import pathlib

import setuptools

HERE = pathlib.Path(__file__).parent

VERSION = "1.2.0"
DESCRIPTION = "Research-oriented library for reinforcement learning."
LICENSE = "MIT License"
PACKAGE_NAME = "colosseum"
KEYWORDS = "reinforcement-learning python"

AUTHOR = "Anonymous"
AUTHOR_EMAIL = "Anonymous"
URL = "Anonymous"

libraries = []
if os.name == "posix":
    libraries.append("m")


setuptools.setup(
    name=PACKAGE_NAME,
    description=DESCRIPTION,
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="Anonymous",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    version=VERSION,
    keywords=KEYWORDS,
    packages=setuptools.find_packages(),
    install_requires=[
        "adjustText",
        "bsuite",
        "dm-env",
        "dm-sonnet",
        "frozendict",
        "gin-config",
        "gym",
        "matplotlib",
        "networkx",
        "numpy",
        "pandas",
        "pebble",
        "pygraphviz",
        "pydtmc",
        "pygame",
        "pyyaml",
        "ray[tune]",
        "scipy>=1.7.3",
        "seaborn>=0.12",
        "sparse",
        "tensorflow",
        "timeout-decorator",
        "toolz",
        "tqdm",
        "PyYAML>=6.0",
        "wrapt_timeout_decorator",
    ],
    # scripts=glob("bin" + os.sep + "*"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        # 'License :: OSI Approved :: Apache Software License',
        "Operating System :: POSIX :: Linux",
        # 'Operating System :: Microsoft :: Windows',
        # 'Operating System :: MacOS :: MacOS X',
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # zip_safe=F,  # Doesn't create an egg - easier to debug and hack on
)

