import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='protonet',
      version='0.0.1',
      author='Ulises Jeremias Cornejo Fandos, Gaston Gustavo Rios',
      author_email='ulisescf.24@gmail.com, okason1997@hotmail.com',
      license='MIT',
      description='Tensorflow v2 implementation of NIPS 2017 Paper Prototypical Networks for Few-shot Learning',
      long_description=long_description,
      url='https://github.com/ulises-jeremias/prototypical-networks-tf',
      packages=setuptools.find_packages(),
      install_requires=['Pillow'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"])

