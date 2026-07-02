from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import pybind11

extensions = [
    # Configuración Cython + OpenMP para Linux (GCC)
    Extension(
        "mandelbrot_cython",
        ["mandelbrot_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-fopenmp'],  # Flag correcto para Linux
        extra_link_args=['-fopenmp'],     # Flag correcto para Linux
    ),
    # Configuración pybind11
    Extension(
        "mandelbrot_cpp",
        ["mandelbrot_cpp.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++'
    )
]

setup(
    name="mandelbrot_performance",
    ext_modules=cythonize(extensions),
)