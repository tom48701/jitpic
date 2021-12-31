#!/usr/bin/env python

from setuptools import setup, find_packages
import jitpic

setup(
   name='JitPIC',
   version=jitpic.__version__,
   author='Thomas Wilson',
   author_email='t.wilson@strath.ac.uk',
   packages=find_packages('.'),
   description='A 1D PIC code for teaching and learning',
)