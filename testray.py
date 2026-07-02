import time
import os
import resource
import numpy as np
import multiprocessing as mp
# NOTA: NO forzar 'spawn' en Linux. Con 'spawn' cada proceso hijo re-ejecuta
# TODOS los imports de nivel superior de este archivo (jax, ray),
# multiplicando por mp.cpu_count() la memoria usada y pudiendo colgar/crashear
# el equipo. El 'fork' por defecto de Linux comparte memoria copy-on-write
# con el proceso padre y es seguro aquí porque JAX está forzado a CPU.

# Cargar módulos compilados locales (.so en Linux)
try:
    import mandelbrot_cython
    import mandelbrot_cpp
except ImportError:
    print("ERROR: Primero debes compilar ejecutando: python3 setup.py build_ext --inplace")
    exit()

import jax
import jax.numpy as jnp
import ray

# Forzar a JAX a ejecutarse en CPU
jax.config.update('jax_platform_name', 'cpu')

# =====================================================================
# MEDICIÓN: energía (RAPL), memoria (RSS) y tiempo de CPU
# =====================================================================
# NOTA sobre Ray chunked (removido): en la corrida anterior, Ray chunked con
# CPU pinning explícito (os.sched_setaffinity) dio resultados ~5-8x peores
# que Ray naive de forma consistente y reproducible, incluso tras: (1)
# verificar que el mapeo núcleo físico -> CPU lógica era correcto vía
# /sys/devices/system/cpu/cpu*/topology/thread_siblings_list, y (2) descartar
# throttling térmico subiendo las pausas de enfriamiento de 5s a 30s sin
# cambio en el resultado (25.4s vs 25.5s -- prácticamente idéntico, lo cual
# es evidencia de una causa determinística y no térmica). La hipótesis más
# plausible es contención con los propios procesos internos de Ray (raylet,
# GCS, plasma store) que no estaban pineados y pudieron caer en las mismas
# CPUs reservadas para cómputo. Se documenta esto como hallazgo de
# discusión en el informe, pero se remueve del benchmark principal porque
# no aporta una comparación válida de rendimiento -- Ray naive ya cumple el
# rol de representar paralelismo distribuido en la comparativa.

def read_rapl_energy_uj(rapl_path='/sys/class/powercap/intel-rapl:0/energy_uj'):
    """
    Lee el contador de energía RAPL (Running Average Power Limit, sólo
    CPUs Intel) del dominio 'package' (paquete completo del procesador),
    en microjulios acumulados desde que arrancó el contador.

    Devuelve None si no está disponible: CPU no-Intel (AMD no expone RAPL
    por esta misma vía; usa un mecanismo distinto vía msr), sin permisos de
    lectura, o kernel sin soporte 'powercap'. En ese caso la energía se
    reporta como N/A en vez de inventar un número.

    Para habilitar en la mayoría de las distros si el archivo existe pero
    da PermissionError:
        sudo chmod -R a+r /sys/class/powercap/intel-rapl
    (el permiso se resetea al reiniciar; no es una fijación permanente).
    """
    try:
        with open(rapl_path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, PermissionError, ValueError, NotADirectoryError):
        return None

RAPL_AVAILABLE = read_rapl_energy_uj() is not None
if not RAPL_AVAILABLE:
    print("[WARN] RAPL no disponible en este sistema (CPU no-Intel, sin "
          "permisos, o sin soporte powercap). La columna de energía se "
          "reportará como N/A. Para intentar habilitarla: "
          "sudo chmod -R a+r /sys/class/powercap/intel-rapl\n")

# NOTA sobre RAPL y overflow: el contador de energy_uj es un entero de 32
# bits en algunas CPUs Intel y puede dar la vuelta (wrap-around) tras
# consumir suficiente energía continua (en la práctica, tras decenas de
# segundos a potencia alta). Cada medición individual de este benchmark
# dura pocos segundos, así que el riesgo es bajo, pero igual se valida que
# el delta no sea negativo antes de reportarlo (ver medir()).

def medir(func, *args, **kwargs):
    """
    Ejecuta func(*args, **kwargs) y devuelve (resultado, metricas), donde
    metricas incluye tiempo de pared, energía (si RAPL disponible), memoria
    residente pico del PROCESO PRINCIPAL, y tiempo de CPU (user+sys) del
    proceso principal y de sus hijos.

    LIMITACIÓN IMPORTANTE A DOCUMENTAR EN EL INFORME: resource.getrusage y
    os.times() con RUSAGE_SELF/hijos miden memoria y CPU vistos DESDE ESTE
    PROCESO. Para Multiprocessing esto sí captura a los hijos (children_*),
    porque son hijos directos esperados con wait(). Para Ray, en cambio, los
    workers son procesos gestionados por el runtime de Ray (no son hijos
    directos de este proceso en el sentido de wait()), por lo que
    cpu_children_* y el RSS de los workers de Ray NO quedan reflejados acá
    -- solo se ve el tiempo de CPU del proceso principal esperando en
    ray.get(). Si se necesita medir memoria/CPU real de los workers de Ray,
    hay que usar herramientas externas (ray.get_actor / métricas del
    dashboard de Ray, o /usr/bin/time -v por proceso, o cgroups).
    """
    energy_before = read_rapl_energy_uj()
    mem_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cpu_before = os.times()

    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - t0

    cpu_after = os.times()
    mem_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    energy_after = read_rapl_energy_uj()

    # ru_maxrss es el RSS PICO acumulado del proceso desde que arrancó (no
    # se resetea entre llamadas), por eso reportamos el valor absoluto tras
    # la llamada (memoria pico acumulada hasta ahora) y también el delta
    # respecto de antes de esta llamada específica (cuánto pico subió, si
    # es que subió -- puede ser 0 si un paso anterior ya usó más memoria).
    metrics = {
        'tiempo_s': elapsed,
        'mem_pico_total_mb': mem_after_kb / 1024.0,
        'mem_pico_delta_mb': max(0, mem_after_kb - mem_before_kb) / 1024.0,
        'cpu_user_s': cpu_after.user - cpu_before.user,
        'cpu_sys_s': cpu_after.system - cpu_before.system,
        'cpu_children_user_s': cpu_after.children_user - cpu_before.children_user,
        'cpu_children_sys_s': cpu_after.children_system - cpu_before.children_system,
    }

    if RAPL_AVAILABLE and energy_before is not None and energy_after is not None:
        delta_uj = energy_after - energy_before
        if delta_uj >= 0:
            metrics['energia_j'] = delta_uj / 1_000_000.0
            metrics['potencia_prom_w'] = (metrics['energia_j'] / elapsed) if elapsed > 0 else None
        else:
            # Overflow del contador RAPL detectado -- no reportar un
            # negativo sin sentido.
            metrics['energia_j'] = None
            metrics['potencia_prom_w'] = None
    else:
        metrics['energia_j'] = None
        metrics['potencia_prom_w'] = None

    return result, metrics


def reportar(nombre, metrics, t_base=None):
    """Imprime una línea de resultado uniforme para todos los métodos,
    incluyendo tiempo, speedup, CPU total (proceso + hijos), memoria pico y
    energía (si disponible). GPU no aplica en este benchmark: todos los
    métodos, incluido JAX, están forzados a ejecutar en CPU
    (jax_platform_name='cpu'); no hay ningún paso que use GPU."""
    cpu_propio = metrics['cpu_user_s'] + metrics['cpu_sys_s']
    cpu_hijos = metrics['cpu_children_user_s'] + metrics['cpu_children_sys_s']
    linea = f"{nombre:<28}: {metrics['tiempo_s']:.4f}s"
    if t_base is not None:
        linea += f"  (Speedup: {t_base/metrics['tiempo_s']:.1f}x)"
    linea += f" | CPU proc: {cpu_propio:.2f}s"
    if cpu_hijos > 0.001:
        linea += f" + hijos: {cpu_hijos:.2f}s"
    linea += f" | RSS pico: {metrics['mem_pico_total_mb']:.1f}MB"
    if metrics['energia_j'] is not None:
        linea += f" | Energía: {metrics['energia_j']:.2f}J ({metrics['potencia_prom_w']:.1f}W prom)"
    else:
        linea += " | Energía: N/A"
    print(linea)


# Configuración del tamaño del problema
HEIGHT, WIDTH = 2000, 2000
MAX_ITER = 150

grid_base = np.zeros((HEIGHT, WIDTH), dtype=np.float64)

# =====================================================================
# 1. PROGRAMACIÓN NORMAL (Python Secuencial - Clase 2)
# =====================================================================
def compute_normal(height, width, max_iter):
    output = np.zeros((height, width), dtype=np.int32)
    for i in range(height):
        for j in range(width):
            c_re = -2.0 + (j * 3.0 / width)
            c_im = -1.5 + (i * 3.0 / height)
            z_re, z_im = 0.0, 0.0
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
    return output

# =====================================================================
# 2. VECTORIZACIÓN (NumPy SIMD - Clase 3)
# =====================================================================
def compute_numpy(height, width, max_iter):
    y, x = np.ogrid[-1.5:1.5:complex(0, height), -2.0:1.0:complex(0, width)]
    c = x + 1j * y
    z = np.zeros(c.shape, dtype=np.complex128)
    output = np.zeros(c.shape, dtype=np.int32)

    for k in range(max_iter):
        mask = np.abs(z) <= 2.0
        z[mask] = z[mask]**2 + c[mask]
        output[mask] = k
    return output

# =====================================================================
# 3. PARALELISMO CPU (Multiprocessing - Clase 6)
# =====================================================================
def _worker_row(args):
    i, width, max_iter = args
    row = np.zeros(width, dtype=np.int32)
    c_im = -1.5 + (i * 3.0 / HEIGHT)
    for j in range(width):
        c_re = -2.0 + (j * 3.0 / width)
        z_re, z_im = 0.0, 0.0
        k = 0
        while k < max_iter:
            z_re_sq = z_re * z_re
            z_im_sq = z_im * z_im
            if z_re_sq + z_im_sq > 4.0:
                break
            z_im = 2.0 * z_re * z_im + c_im
            z_re = z_re_sq - z_im_sq + c_re
            k += 1
        row[j] = k
    return row

def compute_multiprocessing(height, width, max_iter):
    num_workers = min(4, mp.cpu_count())
    pool = mp.Pool(num_workers)
    tasks = [(i, width, max_iter) for i in range(height)]
    rows = pool.map(_worker_row, tasks)
    pool.close()
    pool.join()
    return np.array(rows)

# =====================================================================
# 4. COMPILACIÓN JIT (JAX - Clase 8)
# =====================================================================
@jax.jit
def _jax_mandelbrot_kernel(c_re, c_im, max_iter):
    def cond_fn(state):
        k, z_re, z_im = state
        return (k < max_iter) & (z_re*z_re + z_im*z_im <= 4.0)

    def body_fn(state):
        k, z_re, z_im = state
        return k + 1, z_re*z_re - z_im*z_im + c_re, 2.0*z_re*z_im + c_im

    k, _, _ = jax.lax.while_loop(cond_fn, body_fn, (0, 0.0, 0.0))
    return k

compute_jax_vmap = jax.jit(jax.vmap(jax.vmap(_jax_mandelbrot_kernel, in_axes=(0, None, None)), in_axes=(None, 0, None)))

# =====================================================================
# 5. PARALELISMO DISTRIBUIDO (Ray - Multi-nodo / Cluster)
# =====================================================================
@ray.remote
def _ray_worker_row(i, width, max_iter, height):
    row = np.zeros(width, dtype=np.int32)
    c_im = -1.5 + (i * 3.0 / height)
    for j in range(width):
        c_re = -2.0 + (j * 3.0 / width)
        z_re, z_im = 0.0, 0.0
        k = 0
        while k < max_iter:
            z_re_sq = z_re * z_re
            z_im_sq = z_im * z_im
            if z_re_sq + z_im_sq > 4.0:
                break
            z_im = 2.0 * z_re * z_im + c_im
            z_re = z_re_sq - z_im_sq + c_re
            k += 1
        row[j] = k
    return row

def compute_ray_naive(height, width, max_iter):
    futures = [_ray_worker_row.remote(i, width, max_iter, height) for i in range(height)]
    rows = ray.get(futures)
    return np.array(rows)

# =====================================================================
# EJECUCIÓN DEL BENCHMARK
# =====================================================================
if __name__ == "__main__":
    try:
        ray.shutdown()
    except Exception:
        pass

    ray.init(
        num_cpus=mp.cpu_count(),
        ignore_reinit_error=False,
        logging_level="ERROR",
        include_dashboard=False,
        object_store_memory=500_000_000,
    )

    print(f"--- Iniciando Comparativa Mandelbrot ({HEIGHT}x{WIDTH}) en Ubuntu --- \n")
    print(f"[DEBUG] mp.cpu_count() reporta: {mp.cpu_count()} núcleos lógicos")
    print(f"[DEBUG] Ray ve estos recursos: {ray.cluster_resources()}")
    print("[DEBUG] GPU: no aplica -- todos los métodos corren forzados a CPU "
          "(jax_platform_name='cpu'); no hay ningún paso en este benchmark "
          "que use GPU.\n")

    try:
        # [1] Python Normal
        res_normal, m_normal = medir(compute_normal, HEIGHT, WIDTH, MAX_ITER)
        t_normal = m_normal['tiempo_s']
        reportar("[1] Python Normal Secuencial", m_normal)

        # [2] NumPy Vectorizado
        _, m_np = medir(compute_numpy, HEIGHT, WIDTH, MAX_ITER)
        reportar("[2] NumPy Vectorizado (SIMD)", m_np, t_normal)

        # [3] Cython Estático
        _, m_cy = medir(mandelbrot_cython.compute_cython, grid_base, MAX_ITER)
        reportar("[3] Cython Estático", m_cy, t_normal)

        # [4] pybind11 (C++)
        _, m_cpp = medir(mandelbrot_cpp.compute_cpp, HEIGHT, WIDTH, MAX_ITER)
        reportar("[4] C++ Bindings (pybind11)", m_cpp, t_normal)

        # [5] Multiprocessing CPU
        _, m_mp = medir(compute_multiprocessing, HEIGHT, WIDTH, MAX_ITER)
        reportar("[5] Multiprocessing (CPU)", m_mp, t_normal)

        print("[DEBUG] Pausa de enfriamiento de 5s...\n")
        time.sleep(5)

        # [6] Cython + OpenMP
        _, m_omp = medir(mandelbrot_cython.compute_openmp, grid_base, MAX_ITER, num_threads=4)
        reportar("[6] OpenMP (Cython -fopenmp)", m_omp, t_normal)

        # [7] JAX JIT (CPU) -- warm-up obligatorio fuera de la medición, para
        # que el tiempo de compilación XLA no contamine el tiempo de
        # ejecución "en caliente" que se reporta.
        x_range = jnp.linspace(-2.0, 1.0, WIDTH)
        y_range = jnp.linspace(-1.5, 1.5, HEIGHT)
        _ = compute_jax_vmap(x_range, y_range, MAX_ITER)  # warm-up

        def _jax_run():
            r = compute_jax_vmap(x_range, y_range, MAX_ITER)
            r.block_until_ready()
            return r
        _, m_jax = medir(_jax_run)
        reportar("[7] JAX Compilación JIT (CPU)", m_jax, t_normal)

        print("[DEBUG] Pausa de enfriamiento de 5s antes de Ray...\n")
        time.sleep(5)

        # [8] Ray naive (una tarea remota por fila)
        _, m_ray = medir(compute_ray_naive, HEIGHT, WIDTH, MAX_ITER)
        reportar("[8] Ray (naive, por fila)", m_ray, t_normal)

    finally:
        ray.shutdown()