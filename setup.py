"""Package setup for MedExplain."""

from setuptools import find_packages, setup


setup(
    name="medexplain",
    version="0.1.0",
    description="Fine-tuned medical jargon to plain-language rewrite prototype.",
    package_dir={"": "src"},
    packages=find_packages("src"),
)
