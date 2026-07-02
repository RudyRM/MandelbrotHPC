# Usamos una base de Python oficial sobre Debian/Ubuntu estable
FROM python:3.11-slim

# Evita que Python escriba archivos .pyc y fuerza salida en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema esenciales para compilar C++ y OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Definir directorio de trabajo
WORKDIR /app

# Copiar los requerimientos primero para aprovechar la cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python (forzando JAX a solo CPU)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente del proyecto
COPY setup.py .
COPY mandelbrot_cpp.cpp .
COPY mandelbrot_cython.pyx .
COPY testray.py .

# Compilar los módulos de Cython y C++ (pybind11) inplace
RUN python3 setup.py build_ext --inplace

# Comando por defecto para correr el benchmark al iniciar el contenedor
CMD ["python3", "testray.py"]