import setuptools

# todo: https://github.com/qubvel/segmentation_models.pytorch/blob/master/setup.py

long_description = "hi"

setuptools.setup(
    name="helper",
    version="1.0.0",
    author="Christina Bornberg",
    author_email="christina.bornberg@outlook.com",
    description="PyTorch helper functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chrisini/medical_ai",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)