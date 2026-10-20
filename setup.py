#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="surpass-ppi",
    version="0.1.0",
    description="Sequence-only inter-protein contact and interaction prediction.",
    author="junjie zhu",
    author_email="shiroyuki@sjtu.edu.cn",
    url="https://github.com/junjie-zhu/SurPass",
    python_requires=">=3.11",
    install_requires=[],
    packages=find_packages(),
)
