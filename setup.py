import os

from setuptools import find_packages, setup


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), encoding="utf-8") as fp:
        return fp.read()


# Package meta-data.
NAME = "abl"
DESCRIPTION = "abductive learning package project"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = None


# BEFORE importing setuptools, remove MANIFEST. Otherwise it may not be
# properly updated when the contents of directories change (true for distutils,
# not sure about setuptools).
if os.path.exists("MANIFEST"):
    os.remove("MANIFEST")

here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except FileNotFoundError:
    # Handle the case where the file does not exist
    print("requirements.txt file not found.")
    REQUIRED = []
except Exception as e:
    # Handle other possible exceptions
    print(f"An error occurred: {e}")
    REQUIRED = []

EXTRAS = {
    "test": [
        "pytest-cov",
        "black==22.10.0",
    ],
}

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


if __name__ == "__main__":
    setup(
        name=NAME,
        version=about["__version__"],
        license="MIT Licence",
        url="https://github.com/AbductiveLearning/ABL-Package",
        packages=find_packages(),
        include_package_data=True,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires=REQUIRES_PYTHON,
        install_requires=REQUIRED,
        extras_require=EXTRAS,
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Programming Language :: Python",
            "Topic :: Software Development",
            "Topic :: Scientific/Engineering",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.8",
        ],
    )
