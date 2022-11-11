from distutils.core import setup
from pathlib import Path


def get_install_requires():
    """Returns requirements.txt parsed to a list"""
    fname = Path(__file__).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets


requirements = get_install_requires()

setup(name='MolScribe',
      version='1.0',
      description='MolScribe',
      author='Yujie Qian',
      author_email='yujieq@csail.mit.edu',
      url='https://github.com/thomas0809/MolScribe',
      packages=['molscribe'],
      setup_requires=['numpy'],
      install_requires=requirements,
      )
