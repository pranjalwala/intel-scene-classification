"""
setup.py
========
Package setup for intel-scene-classification.

Installing this package (pip install -e .) adds the `src` module
to the Python path so that `from src.data_loader import ...` works
from any working directory — useful when running scripts from outside
the project root.

Usage:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="intel-scene-classification",
    version="1.0.0",
    description="Scene Recognition via Residual Convolutional Architectures",
    author="Your Name",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.10.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "PyYAML>=6.0",
    ],
)
