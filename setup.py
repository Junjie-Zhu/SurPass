#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Generative conformational ensemble prediction for monomers, multidomain proteins, and protein complexes.",
    author="junjie zhu",
    author_email="shiroyuki@sjtu.edu.cn",
    url="https://github.com/junjie-zhu/SurPass",
    python_requires=">=3.11",
    install_requires=[],
    packages=find_packages(),
)
