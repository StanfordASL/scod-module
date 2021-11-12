from setuptools import setup, find_packages

install_requires = [
    'numpy>=1.18.0',
    'torch>=1.8',
    'tqdm>=4.15.0',
]

setup(name="scod",
      author="Apoorva Sharma",
      install_requires=install_requires,
      packages=find_packages(),
      description='Equip arbitrary Pytorch models with OOD detection',
      version='0.1.0')

