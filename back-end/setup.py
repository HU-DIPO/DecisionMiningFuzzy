import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = ["Flask==2.1.0",
                "Flask-Cors==3.0.10",
                "flask-restful==0.3.9",
                "more_itertools==8.12.0",
                "numpy==1.22.3",
                "pandas==1.4.1",
                "python-dotenv==0.20.0",
                "scikit-learn==1.0.2",
                "scipy==1.8.0",
                "tqdm==4.63.1"]
extras = {
    "dev": [
        "flake8==4.0.1",
        "flake8-annotations==2.7.0",
        "flake8-docstrings==1.6.0",
        "pytest==7.1.1",
        "pytest-cov==3.0.0",
        "Sphinx==4.5.0",
        "sphinx-rtd-theme==1.0.0",
        "sphinxcontrib-napoleon==0.7"
    ]
}

setuptools.setup(
    name="DMFuzzy",
    version="1.5",
    author="""Casper Smet, Gijsbert Nutma, Martijn Knegt, Sam Leewis, Matthijs Berkhout""",
    author_email="matthijs.berkhout@hu.nl",
    description="Decision mining back-end",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HU-DIPO/DecisionMiningFuzzy/tree/main/back-end",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Windows 10, Linux",
    ],
    install_requires=requirements,
    extras_require=extras,
    python_requires='>=3.8',
)
