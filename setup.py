#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="anymat2vec",
    version="0.1",
    description="General Latent Representations of Materials From Text Mining",
    long_description=readme,
    author="Materials Scholar Development Team",
    author_email="amalietrewartha@lbl.gov, ardunn@lbl.gov",
    url="https://github.com/materialsintelligence/anymat2vec",
    license=license,
    packages=find_packages(),
    # test_suite="matscholar_web",
    # tests_require="tests",
)