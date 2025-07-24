from setuptools import find_packages, setup

setup(
    name="climate-prediction",
    version="0.1.0",
    description="Climate Temperature Prediction using NASA Earth Data",
    author="AI Infrastructure Samples",
    author_email="team@example.com",
    url="https://github.com/nishaero/ai-infra-samples",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "fastapi>=0.100.0",
        "mlflow>=2.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "climate-train=src.models.training:main",
            "climate-api=src.api.main:main",
            "climate-fetch=src.data.ingestion:main",
        ],
    },
)
