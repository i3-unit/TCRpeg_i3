from setuptools import setup, find_packages
data_files_to_include = [('', ['README.md', 'LICENSE'])]
setup(name='tcrpeg',
      version='1.0.6',
      description='Recovering the repertoire probability, generating and encoding CDR3 sequences',
      long_description='To be added',
      url='https://github.com/jiangdada1221/TCRpeg',
      author='Yuepeng Jiang',
      author_email='jiangdada12344321@gmail.com',
      license='GPLv3',
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Healthcare Industry',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Scientific/Engineering :: Medical Science Apps.',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Natural Language :: English',
            'Programming Language :: Python :: 3',
            ],
      # packages=find_packages(),
      packages=find_packages(include=['tcrpeg', 'tcrpeg.*', 'tcrpeg_toolkit', 'tcrpeg_toolkit.*']),
      install_requires=['numpy','torch','matplotlib','tqdm','pandas','scikit-learn','Scipy'],
      data_files = data_files_to_include,
      include_package_data=True,
      zip_safe=False)

# Need to run pip install -e .