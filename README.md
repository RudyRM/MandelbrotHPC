# MandelbrotHPC

# Para ejecutar con restricción de memoria
1. docker build -t mandelbrot-benchmark}
2. docker run --rm --memory="4g" --cpus="4" --volume=/sys:/sys:ro mandelbrot-benchmark
