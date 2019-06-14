from setuptools import setup

setup(name='xmps',
      version='0.1',
      description='tangent space methods for Matrix Product State',
      url = 'https://github.com/fergusbarratt/xmps',
      author='Fergus Barratt',
      author_email='fergus.barratt@kcl.ac.uk',
      license='GPL',
      packages=['xmps'],
      install_requires=[
          'cython',
          'numpy', 
          'scipy',
          'matplotlib']
      ,
      zip_safe=False
)
