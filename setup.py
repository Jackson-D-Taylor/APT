from setuptools import setup, find_packages

setup(
    name="APT",
    version="0.1.0",
    description="Algorithmic Pulsar Timing",
    author="Jackson Taylor, Camryn Phillips, Scott Ransom",
    url="",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "APTB=APT.scripts.APTB:main",
        ]
    },
    python_requires=">=3.7",
    install_requires=[
        "astropy",
        "numpy",
        "scipy",
        "loguru",
        "treelib",
        "colorama",
        "dataclasses",
        "pint-pulsar",
    ],
    zip_safe=False,
)
