import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ec-feature-selection", # Replace with your own username
    version="0.0.1",
    author="Ohad Volk",
    author_email="OhadVok@gmail.com",
    description="Feature Selection via Eigenvector Centrality",
    long_description=long_description,
    long_description_content_type="markdown",
    url="https://github.com/OhadVolk/ECFS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)