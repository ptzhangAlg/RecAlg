# -*- coding:utf-8 -*-
from setuptools import setup

setup(
    name='rec_alg',
    version='0.1',
    description='end-to-end recommendation algorithm package with dataset processing',
    author='Pengtao Zhang',
    author_email='zpt1986@126.com',
    packages=['rec_alg'],
    install_requires=["tensorflow==1.14", "pydot", "graphviz", "numpy", "pandas", "scikit-learn==0.21.3"],
)
