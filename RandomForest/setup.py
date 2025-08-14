from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "DecisionTree",
        ["DecisionTree.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['/O2'],
        libraries=[],
    )
]
setup(
    ext_modules=cythonize(extensions, annotate=True)
)
#python setup.py build_ext --inplace

