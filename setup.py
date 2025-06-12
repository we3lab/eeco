from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup_requirements = []

test_requirements = [
    "black>=22.3.0",
    "flake8>=4.0.0",
    "codecov>=2.1.4",
    "pytest>=8.1.1",
    "pytest-cov>=3.0.0",
    "pytest-html>=3.1.1",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "Sphinx==7.0.1",
    "sphinx-rtd-theme==2.0.0",
    "tox>=3.24.5",
    "matplotlib>=3.8.4",
    "ipykernel"
]

requirements = [
    "pandas>=2.2.1",
    "numpy>=1.26.4",
    "cvxpy>=1.3.0",
    "pyomo>=6.7",
    "gurobipy>=11.0",
    "pint>=0.19.2",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ],
}

setup(
    author="WE3 Lab",
    author_email="fchapin@stanford.edu",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    description="Calculate electricity-related emissions and costs.",
    entry_points={},
    long_description=readme,
    long_description_content_type="text/x-rst",
    package_data={"electric_emission_cost": ["electric_emission_cost/data/*"]},
    include_package_data=True,
    keywords="electric-emission-cost",
    name="electric-emission-cost",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.9",
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    extras_require=extra_requirements,
    test_suite="tests",
    url="https://github.com/we3lab/electric-emission-cost",
    version="0.0.3",
    zip_safe=False,
)