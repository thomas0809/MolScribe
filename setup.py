from distutils.core import setup

setup(name='MolScribe',
      version='1.1.1',
      description='MolScribe',
      author='Yujie Qian',
      author_email='yujieq@csail.mit.edu',
      url='https://github.com/thomas0809/MolScribe',
      packages=['molscribe', 'molscribe.indigo', 'molscribe.inference', 'molscribe.transformer'],
      package_dir={'molscribe': 'molscribe'},
      package_data={'molscribe': ['vocab/*']},
      python_requires='>=3.7',
      setup_requires=['numpy'],
      install_requires=[
          "numpy",
          "torch>=1.11.0",
          "pandas",
          "matplotlib",
          "opencv-python>=4.5.5.64",
          "SmilesPE==0.0.3",
          "OpenNMT-py==2.2.0",
          "rdkit>=2024.3.5",
          "albumentations==1.1.0",
          "timm==0.4.12"
      ])
