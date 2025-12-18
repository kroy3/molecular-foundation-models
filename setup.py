#!/usr/bin/env python
"""Setup script for molecular-foundation-models package."""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="molecular-foundation-models",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@domain.com",
    description="Cross-domain foundation models for molecular electrostatics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/molecular-foundation-models",
    project_urls={
        "Bug Tracker": "https://github.com/YOUR_USERNAME/molecular-foundation-models/issues",
        "Documentation": "https://github.com/YOUR_USERNAME/molecular-foundation-models/blob/main/docs/",
        "Source Code": "https://github.com/YOUR_USERNAME/molecular-foundation-models",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=5.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.18",
        ],
        "viz": [
            "matplotlib>=3.5",
            "seaborn>=0.12",
            "plotly>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mfm-train=train:main",
            "mfm-evaluate=src.evaluate:main",
            "mfm-download=scripts.download_datasets:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "molecular modeling",
        "neural operators",
        "equivariant networks",
        "transfer learning",
        "computational physics",
        "drug discovery",
        "materials science",
        "deep learning",
        "pytorch",
    ],
)
