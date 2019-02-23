#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


NAME = 'hellonlp'
DESCRIPTION = 'While I am learning NLP.'
AUTHOR = 'shuiruge'
AUTHOR_EMAIL = 'shuiruge@hotmail.com'
URL = 'https://github.com/shuiruge/hellonlp'
VERSION = '0.0.1'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license='MIT',
    url=URL,
    packages=find_packages(exclude=['dat.*', 'dat']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3+',
    ],
    zip_safe=False,
)
