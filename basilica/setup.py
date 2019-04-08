from setuptools import setup

setup(name='basilica',
      version='0.2.6',
      description='Client bindings for basilica.ai embeddings.',
      long_description='Client bindings for basilica.ai embeddings.',
      url='http://basilica.ai',
      author='Michael Lucy',
      author_email='mlucy@basilica.ai',
      license='MIT',
      packages=['basilica'],
      install_requires=[
          'requests',
          'six',
          'Pillow',
      ],
      zip_safe=True)
