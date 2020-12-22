
from setuptools import setup, find_packages
import sys, os

setup(
    name="Bilateral GPs",
    description="",
    version="1.0",
    author="",
    author_email="maf820@nyu.edu",
    license="MIT",
    python_requires=">=3.6",
    install_requires=[],#
    packages=find_packages(),
    long_description=open("README.md").read(),
)