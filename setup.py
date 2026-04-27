from distutils.core import setup

def read_requirements():
    """Read the requirements.txt file and return a list of dependencies."""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return fh.read().splitlines()

setup(name='MolScribe',
      version='1.2.1',
      description='MolScribe',
      author='Yujie Qian',
      author_email='yujieq@csail.mit.edu',
      url='https://github.com/thomas0809/MolScribe',
      packages=['molscribe', 'molscribe.indigo', 'molscribe.inference', 'molscribe.transformer'],
      package_dir={'molscribe': 'molscribe'},
      package_data={'molscribe': ['vocab/*']},
      python_requires='>=3.7',
      setup_requires=['numpy'],
      install_requires=read_requirements(),
      )
