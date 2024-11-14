from setuptools import setup, find_packages

with open("README.md", "r") as source:
    long_description = source.read()

setup(
    name="proteinfertorch",
    version="1.0.0",
    author="Samir Char",
    packages=find_packages(),
    include_package_data=True,
    description="Unofficial PyTorch version of ProteInfer, originally implemented in TensorFlow 1.X and converted for PyTorch compatibility.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samirchar/proteinfertorch",
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)