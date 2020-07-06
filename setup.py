import setuptools

setuptools.setup(
    name="Online Group Modeling",
    version="1.0.3",
    author="Richard Sear",
    author_email="searri@gwu.edu",
    description="Package for using machine learning to analyze the behavior of online groups",
    url="https://github.com/gwclusterlab/ogm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
