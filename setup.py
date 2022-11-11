from distutils.core import setup

setup(name='MolScribe',
      version='1.0',
      description='MolScribe',
      author='Yujie Qian',
      author_email='yujieq@csail.mit.edu',
      url='https://github.com/thomas0809/MolScribe',
      packages=['MolScribe'],
      install_requires=[
          'torch',
          'numpy',
          'pandas',
          'matplotlib',
          'transformers>=4.5.1',
          'tensorboardX',
          'SmilesPE==0.0.3',
          'OpenNMT-py==2.2.0',
          'rdkit-pypi==2021.3.2.2',
      ],
      )
