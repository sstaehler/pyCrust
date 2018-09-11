#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 2
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev0'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

if __name__ == '__main__':
    setup(
        name='pyCrust',
        version=__version__,
        url='https://github.com/MarkWieczorek/pyCrust',
        license='',
        author='Mark Wieczorek',
        author_email='',
        description='Create a crustal thickness map of a planet from gravity and topography.',
        install_requires=['pyshtools', 'numpy', 'scipy'],
        packages = ['pyCrust'],
        package_data = {'pyCrust': [pjoin('data', '*')]},
    )
