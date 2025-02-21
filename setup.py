from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="SAFAARI",  
    version="0.1.0",  
    
    authors=("Jun Wu", "Fatemeh Aminzadeh]"),
    description="SAFAARI: A domain adaptation method for single-cell data annotation & integration",
    long_description=open("README.md" ,encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VafaeeLab/SAFAARI",  
   
    packages=find_packages(include=["models", "scripts", "data"]),

    install_requires=requirements,  
    entry_points={
        "console_scripts": [
            "safaari-unsupervised=scripts.run_SAFAARI:main",  # Unsupervised annotation
            "safaari-supervised_integration=scripts.main_integration:main",  # Supervised integration
        ],
    },


    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
