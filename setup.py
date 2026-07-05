from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="resume-screening-agent",
    version="1.0.0",
    author="Bhuvan Kambad",
    author_email="bkambad041@gmail.com",
    description="An ML-powered agent for ranking resumes against job descriptions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bot041/ResumeScreening-Agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "flake8>=6.0.0"],
    },
)
