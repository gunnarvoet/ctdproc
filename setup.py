from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

setup(
    # Project information
    name="ctdproc",
    version="0.1.2",
    author="Gunnar Voet",
    author_email="gvoet@ucsd.edu",
    url="https://github.com/gunnarvoet/ctdproc",
    license="MIT License",

    # Description
    description="Library for CTD data processing",
    long_description=f"{readme}\n\n{history}",
    long_description_content_type='text/x-rst',
    
    # Requirements
    python_requires='>=3.6',
    install_requires=["numpy",
                      "xarray",
                      "gsw",
                      "scipy",
                      "xmltodict",
                      "pandas",
                      "munch",
                      "matplotlib"],
    extras_require={
        'test': [  # install these with: pip install ctdproc[test]
            "pytest>=3.8",
            "coverage>=4.5",
            "pytest-cov>=2.6",
            "tox>=3.3",
            "codecov>=2.0",
        ],
    },

    # Packaging
    packages=find_packages(include=["ctdproc", "ctdproc.*"],
                           exclude=["*.tests"]),
    include_package_data=True,
    zip_safe=False,

    platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"

    # Metadata
    project_urls={
        "Documentation": "https://ctdproc.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],

)
