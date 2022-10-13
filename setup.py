import setuptools
import os

req_file = os.path.dirname(os.path.realpath(__file__)) + "/requirements.txt"
if os.path.isfile(req_file):
    with open(req_file, "r", encoding="utf8") as infile:
        install_reqs = infile.read().splitlines()

setuptools.setup(
    name="Online Group Modeling",
    version="2.0.0",
    author="Richard Sear",
    author_email="searri@gwu.edu",
    description="Package for using machine learning to analyze the behavior of online groups",
    url="https://github.com/gwdonlab/ogm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=install_reqs,
)
