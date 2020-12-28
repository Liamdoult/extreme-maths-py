from setuptools import setup, find_packages, Extension
from os import path

from Cython.Build import cythonize

setup_file_location = path.abspath(path.dirname(__file__))
with open(path.join(setup_file_location, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='extreme_maths', # Change here
    version='0.1.0',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/LiamDoult/extreme-maths-py',
    author='Liam Doult',
    author_email='liam.doult@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='Maths GPU FAST',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    python_requires='>=3.5, <4',
    install_requires=["Cython"],
    ext_modules = cythonize(
        [Extension(
            "vectors", ["src/extreme_maths/vectors.pyx", "c/src/vector.c"],
            extra_compile_args = ["-I./c/include", "-O3"],
            language="c",
        )]
    ),
    extras_require={
        'dev': ['check-manifest'],
        'tests': [
            "pytest",
            "pytest-cov",
            "pytest-ordering",
            "numpy",
            "torch",
            "tqdm",
            "terminaltables",
            "termcolor",
        ],
    },
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/LiamDoult/python-template/issues',
        'Source': 'https://github.com/LiamDoult/python-template/',
    },
)
