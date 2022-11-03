from setuptools import find_packages, setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line]

setup(
    name='aug2clean',
    packages=find_packages(),
    version='0.1.0',
    description='AugClean: Automated Data Preparation for ML pipelines',
    author='Mohamed Abdelaal',
    author_email='mohamed.abdelaal@softwareag.com',
    license='MIT',
    keywords='data augmentation, data cleaning, error detection, ML pipelines',
    long_description=read('README.md'),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 1 - Planning"
    ]
)
