from setuptools import setup, find_packages

setup(
    name="label_space_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        "Pillow",
        "huggingface-hub"
    ],
    entry_points={
        "console_scripts": [
            "label_space_generator=label_space_generator.label_generator:main",
        ],
    },
)
