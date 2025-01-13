from setuptools import setup, find_packages
    
setup(
    name="umbrella",
    version="0.1.0",
    description="A brief description of the umbrella package.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Zhuoming Chen",
    author_email="chenzhuoming911@gmail.com",
    url="https://github.com/Infini-AI-Lab/UMbreLLa",
    license="Apache-2.0",
    packages=find_packages(exclude=["examples", "app"]),
    python_requires=">=3.10",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
