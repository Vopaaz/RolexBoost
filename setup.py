from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rolexboost",
    version="0.0.1",
    author="Vopaaz, MichaelYangg",
    author_email="liyifan945@163.com",
    url="https://github.com/Vopaaz/RolexBoost",
    description="Unofficial implementation of RolexBoost: A Rotation-Based Boosting Algorithm With Adaptive Loss Functions",
    long_description = long_description,
    long_description_content_type="text/markdown",
    packages=["rolexboost"],
    install_requires=["scikit-learn>=0.22.0", "numpy>=1.18.0", "scipy>=1.4.0"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License"
    ]
)