#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<int> compute_cpp(int height, int width, int max_iter) {
    auto result = py::array_t<int>({height, width});
    py::buffer_info buf = result.request();
    int *ptr = static_cast<int *>(buf.ptr);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            double c_re = -2.0 + (j * 3.0 / width);
            double c_im = -1.5 + (i * 3.0 / height);
            double z_re = 0.0, z_im = 0.0;
            int k = 0;
            
            while (k < max_iter) {
                double z_re_sq = z_re * z_re;
                double z_im_sq = z_im * z_im;
                if (z_re_sq + z_im_sq > 4.0) break;
                z_im = 2.0 * z_re * z_im + c_im;
                z_re = z_re_sq - z_im_sq + c_re;
                k++;
            }
            ptr[i * width + j] = k;
        }
    }
    return result;
}

PYBIND11_MODULE(mandelbrot_cpp, m) {
    m.def("compute_cpp", &compute_cpp, "Calcula Mandelbrot usando C++ puro con pybind11");
}
