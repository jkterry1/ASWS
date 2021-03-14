import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ASWSEarlyStopping",
    version="",
    author="",
    author_email="",
    description="ASWS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=["numpy>=1.20.1", "matplotlib>=3.3.4", "scipy>=1.6.1", "tqdm>=4.57.0"],
    

)
