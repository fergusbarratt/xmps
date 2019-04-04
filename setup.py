from setuptools import setup
from Cython.Build import cythonize

setup(name='tmps',
      version='0.1',
      description='mps tangent space methods',
      url = 'https://github.com/fergusbarratt/mps',
      author='Fergus Barratt',
      author_email='fergus.barratt@kcl.ac.uk',
      license='GPL',
      packages=['pymps'],
      zip_safe=False
)
