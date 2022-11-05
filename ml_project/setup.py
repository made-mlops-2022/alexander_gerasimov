from setuptools import find_packages, setup


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="project",
    packages=find_packages(),
    version="0.1.0",
    description="ML project for ML_Ops course",
    author="Alexander Gerasimov",
    install_requires=required,
    license="MIT",
)
