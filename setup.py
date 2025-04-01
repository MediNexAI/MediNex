#!/usr/bin/env python3
"""
Setup script for the MediNex AI system.
"""

from setuptools import setup, find_packages
import os
import re

# Read version from ai/__init__.py
with open("ai/__init__.py", "r") as f:
    version_match = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE)
    if version_match:
        version = version_match.group(1)
    else:
        version = "0.0.1"

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

# Read long description from README.md
with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="medinex.life",
    version="1.0.2",
    author="MediNex AI Team",
    author_email="info@medinex.life",
    description="Advanced Medical Knowledge Assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MediNexAI/MediNex",
    packages=find_packages(),
    package_data={
        "": ["*.json", "*.md"]
    },
    entry_points={
        "console_scripts": [
            "medinex=app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
) 