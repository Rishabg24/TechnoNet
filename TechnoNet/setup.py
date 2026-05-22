from setuptools import setup, find_packages

setup(
    name="TechnoNet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "astropy",
        "astroquery",
        
        # add other dependencies as needed
    ],
    python_requires=">=3.8",
)