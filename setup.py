from setuptools import setup

setup(
    name='modelxglue',
    version='0.1',
    packages=['modelxglue', 'modelxglue.utils', 'modelxglue.models', 'modelxglue.features', 'modelxglue.conf'],
    url='https://github.com/models-lab/modelxglue',
    license='MIT',
    author='jesus',
    author_email='jesus.sanchez.cuadrado@gmail.com',
    description='A benchmarking framework for ML',
    install_requires=[
        line.strip() for line in open('requirements.txt')
    ],
    entry_points={
        'console_scripts': [
            'modelxglue=modelxglue.main:main',
            'modelxglue-reports=modelxglue.reports:main',
        ]
    },
)
