import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple-pe",
    version="0.0.1",
    author="Stephen Fairhurst",
    author_email="stephen.fairhurst@ligo.org",
    description="A package to do simple GW parameter estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ligo.org/stephen-fairhurst/simple-pe",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'astropy>=3.2.3',
        'h5py',
        'lalsuite',
        'numpy>=1.15.4',
        'mpmath',
        'scipy',
        'sympy',
        'pycbc'
    ]
)
