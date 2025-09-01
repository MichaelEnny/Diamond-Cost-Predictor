from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='diamond-price-predictor',
    version='1.0.0',
    author='Diamond Analytics Team',
    author_email='analytics@diamondpriceml.com',
    description='End-to-end MLOps pipeline for diamond price prediction using XGBoost and MLflow',
    long_description='A comprehensive machine learning system that predicts diamond prices based on physical and quality characteristics. Features automated data processing, model training, experiment tracking, and deployment infrastructure.',
    url='https://github.com/your-username/diamond-price-predictor',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=get_requirements('requirements_dev.txt'),
    packages=find_packages()
)