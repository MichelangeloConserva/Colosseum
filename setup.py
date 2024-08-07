import importlib
import setuptools

spec = importlib.util.spec_from_file_location("_metadata", "colosseum/_metadata.py")
_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_metadata)

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
    version=_metadata.__version__,
    keywords="reinforcement-learning python machine-learning",
    packages=setuptools.find_packages(),
    include_package_data=True,
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
        "ray[tune]>=1.9.1",
        "scipy>=1.7.3",
        "seaborn>=0.10",
        "sparse",
        "toolz",
        "tqdm",
        "PyYAML>=6.0",
        "graphviz",
        "pygraphviz",
        "wrapt-timeout-decorator",
        "bsuite[baselines]",
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
