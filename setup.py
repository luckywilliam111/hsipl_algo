# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 23:34:06 2020

@author: user
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hsipl_algo",
    version="1.0.12",
    author="WEN",
    author_email="luckywilliam111@gmail.com",
    description="For HyperSpectral Image's Algorithm package",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luckywilliam111/hsipl_algo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)