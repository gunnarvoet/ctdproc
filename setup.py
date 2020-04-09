from setuptools import find_packages, setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name="pyctdproc",
      version="0.1.0",
      description="Library for CTD data processing",
      author="Gunnar Voet",
      author_email='gvoet@ucsd.edu',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="GNU GPL v3",
      url="https://github.com/gunnarvoet/pyctdproc",
      packages=find_packages(),
      )
