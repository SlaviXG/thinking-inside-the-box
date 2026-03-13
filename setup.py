from setuptools import setup, find_packages

setup(
    name="thinking-inside-the-box",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "accelerate>=0.29.0",
        "kuzu>=0.4.0",
        "networkx>=3.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
    ],
)
