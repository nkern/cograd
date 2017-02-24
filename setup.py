import sys
import os
try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name            = 'cograd',
    version         = '0.1',
    description     = 'Conjugate Gradient Algorithms in Python',
    author          = 'Nick Kern',
    url             = "http://github.com/nkern/cograd",
    packages        = ['cograd'],
    package_data    = {'cograd':['data/*.pkl']},
    setup_requires  = ['pytest-runner'],
    tests_require   = ['pytest']
    )

