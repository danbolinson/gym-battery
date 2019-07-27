#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='gym_battery',
      version='0.0.1',
      install_requires=['gym'],  # And any other dependencies foo needs
      #packages=find_packages(),
      package_data={'': ['*.xml']}
)
