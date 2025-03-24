from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("sega_learn/linear_models/linear_models_cython.pyx"),
    include_dirs=[np.get_include()]
)

# To build the cython code, run the following command in the terminal:
# The inplace flag is used to build the extension in the same directory as the source code.
# python setup.py build_ext --inplace


# Future work: automatically find all cython files in the project and build them.
# from setuptools import setup, Extension
# from Cython.Build import cythonize
# import os

# def find_pyx(path='.'):
#     pyx_files = []
#     for root, _, files in os.walk(path):
#         for file in files:
#             if file.endswith('.pyx'):
#                 pyx_files.append(os.path.join(root, file))
#     return pyx_files

# extensions = [Extension(
#     name=file[:-4].replace(os.path.sep, '.'), # Replace path separators with dots for module name
#     sources=[file],
# ) for file in find_pyx()]

# setup(
#     name='your_package_name',
#     ext_modules=cythonize(extensions),
#     packages=['your_package_name'] # If you have other Python modules
# )