from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="drone_cop",
    version="0.0.1",
    description="A demo of drone-based license plate identification and parking permit verification.",
    # package_dir={"": "app"},
    # packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://ignore.work",
    author="Quillaja",
    author_email="name@something.com",
    license="Unlicense",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    extras_require={
        "dev": ["pytest==7.4.3", "pyinstaller==6.1.0"],
    },
    python_requires=">=3.11",
)
