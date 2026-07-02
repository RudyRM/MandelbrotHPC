# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from cython.parallel import prange

# 1. Versión Cython pura (Compilación estática - Clase 4)
def compute_cython(double[:, :] grid, int max_iter):
    cdef int i, j, k
    cdef int height = grid.shape[0]
    cdef int width = grid.shape[1]
    cdef double c_re, c_im, z_re, z_im, z_re_sq, z_im_sq
    cdef int[:, :] output = np.zeros((height, width), dtype=np.int32)
    
    for i in range(height):
        for j in range(width):
            c_re = -2.0 + (j * 3.0 / width)
            c_im = -1.5 + (i * 3.0 / height)
            z_re = 0.0
            z_im = 0.0
            k = 0
            while k < max_iter:
                z_re_sq = z_re * z_re
                z_im_sq = z_im * z_im
                if z_re_sq + z_im_sq > 4.0:
                    break
                z_im = 2.0 * z_re * z_im + c_im
                z_re = z_re_sq - z_im_sq + c_re
                k += 1
            output[i, j] = k
    return np.asarray(output)

# 2. Versión Cython + OpenMP (CPU Multi-thread sin GIL - Clase 6)
def compute_openmp(double[:, :] grid, int max_iter, int num_threads):
    # Todas las declaraciones cdef se quedan arriba cumpliendo la regla de Cython
    cdef int i, j, k
    cdef int height = grid.shape[0]
    cdef int width = grid.shape[1]
    cdef double c_re, c_im, z_re, z_im, z_re_sq, z_im_sq
    cdef int[:, :] output = np.zeros((height, width), dtype=np.int32)
    
    # prange paraleliza el ciclo externo distribuyendo las filas en hilos nativos
    for i in prange(height, nogil=True, num_threads=num_threads, schedule='dynamic'):
        for j in range(width):
            c_re = -2.0 + (j * 3.0 / width)
            c_im = -1.5 + (i * 3.0 / height)
            z_re = 0.0
            z_im = 0.0
            
            # Usamos un for estático en rango. Cython y OpenMP interpretan automáticamente 
            # la variable 'k' del ciclo como PRIVADA para cada hilo, eliminando el error de reducción.
            for k in range(max_iter):
                z_re_sq = z_re * z_re
                z_im_sq = z_im * z_im
                
                if z_re_sq + z_im_sq > 4.0:
                    break
                    
                z_im = 2.0 * z_re * z_im + c_im
                z_re = z_re_sq - z_im_sq + c_re
                
            # Al terminar el ciclo for, guardamos cuántas iteraciones se alcanzaron
            output[i, j] = k
            
    return np.asarray(output)