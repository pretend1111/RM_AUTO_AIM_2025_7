from setuptools import find_packages
from setuptools import setup

setup(
    name='rm_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('rm_msgs', 'rm_msgs.*')),
)
