from setuptools import find_packages, setup


# Note:
#   The Hitchiker's guide to python provides an excellent, standard, method for creating python packages:
#       http://docs.python-guide.org/en/latest/writing/structure/
#
#   To deploy on PYPI follow the instructions at the bottom of:
#       https://packaging.python.org/tutorials/distributing-packages/#uploading-your-project-to-pypi

with open("README.md") as f:
    readme_text = f.read()

with open("LICENSE") as f:
    license_text = f.read()

setup(
    name="featuregen",
    version="0.1",
    py_modules=[],
    install_requires=["numpy", "scipy", "pandas", "scikit-learn"],
    url="https://www.github.com/jkapila/featuregen",
    license="MIT",
    author="Jitin Kapila",
    description="Feature Generatoin for Froecasting Purposes",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "docsrc")),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: ",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    keywords=["packages", "ml", "pandas", "forecasting"],
)
