#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <time.h>
#include <cmath>
#include <complex>
#include <chrono>

#include "a2-helpers.hpp"

using namespace std;

vector<gradient> gradients = {
    gradient({0, 0, 0}, {76, 57, 125}, 0.0, 0.010, 2000),
    gradient({76, 57, 125}, {255, 255, 255}, 0.010, 0.020, 2000),
    gradient({255, 255, 255}, {0, 0, 0}, 0.020, 0.050, 2000),
    gradient({0, 0, 0}, {0, 0, 0}, 0.050, 1.0, 2000)};

bool mandelbrot_kernel(complex<double> c, vector<int> &pixel)
{
    int max_iterations = 2048, iteration = 0;
    complex<double> z(0, 0);

    // Replace abs(z) which is calculated like 
    // sqrt(z.real()*z.real() + z.imag()*z.imag()) 
    // with just square of complex number because sqrt operation is costly 
    while (z.real()*z.real() + z.imag()*z.imag() <= 16 && (iteration < max_iterations))
    {
        z = z * z + c;
        ++iteration;
    }

    double length = sqrt(z.real() * z.real() + z.imag() * z.imag());
    long double m = (iteration + 1 - log(length) / log(2.0));
    double q = m / (double)max_iterations;

    q = iteration + 1 - log(log(length)) / log(2.0);
    q /= max_iterations;

    colorize(pixel, q, iteration, gradients);

    return (iteration < max_iterations);
}

int mandelbrot(Image &image, double ratio = 0.15)
{
    // move i, j to private variable in omp parallel block
    // make h, w, channels const
    const int h = image.height;
    const int w = image.width;
    const int channels = image.channels;
    ratio /= 10.0;

    vector<int> pixel = {0, 0, 0}; 
    // move complex<double> c to private variable in omp parallel block

    // array to use for marking pexels_inside to prevent race condition
    int pixels_inside_arr[w][h] = {0};

    #pragma omp parallel num_threads(16) shared(ratio, image, pixels_inside_arr)
    #pragma omp single nowait
    for (int j = 0; j < h; ++j)
    {
        #pragma omp task shared(ratio, image, pixels_inside_arr) firstprivate(j, pixel) 
        for (int i = 0; i < w; ++i)
        {
            double dx = (double)i / (w)*ratio - 1.10;
            double dy = (double)j / (h)*0.1 - 0.35;

            complex<double> c = complex<double>(dx, dy);

            if (mandelbrot_kernel(c, pixel))
                // Mark the pixel in the array
                ++pixels_inside_arr[i][j];

            for (int ch = 0; ch < channels; ++ch)
                image(ch, j, i) = pixel[ch];
        }
    }
    // Compute pixels_inside from array of flags pixels_inside_arr
    int pixels_inside = 0;
    for (int i = 0; i < w; ++i)
        for (int j = 0; j < h; ++j)
            pixels_inside += pixels_inside_arr[i][j];

    return pixels_inside;
}

void convolution_2d(Image &src, Image &dst, int kernel_width, double sigma, int nsteps=1)
{
    // Make h, w, channels, kernel, displ const
    const int h = src.height;
    const int w = src.width;
    const int channels = src.channels;
    const std::vector<std::vector<double>> kernel = get_2d_kernel(kernel_width, kernel_width, sigma);
    const int displ = (kernel.size() / 2);

    #pragma omp parallel num_threads(16) shared(nsteps, src, dst)
    #pragma omp single nowait 
    for (int step = 0; step < nsteps; ++step)
    {
        for (int ch = 0; ch < channels; ++ch)
        {
            for (int i = 0; i < h; ++i)
            {
                #pragma omp task shared(nsteps, src, dst) firstprivate(i, ch, step)
                for (int j = 0; j < w; ++j)
                {
                    double val = 0.0;
                    // Reduce the range from -displ to +displ to remove unnecessary if clauses
                    int k_start = i < displ ? -i : -displ;
                    int k_end = i >= h - displ ? h - i - 1 : displ;
                    for (int k = k_start; k <= k_end; ++k)
                    {
                        int l_start = j < displ ? -j : -displ;
                        int l_end = j >= w - displ ? w - j - 1 : displ;
                        for (int l = l_start; l <= l_end; ++l)
                            val += kernel[k + displ][l + displ] * src(ch, i + k, j + l);
                    }
                    dst(ch, i, j) = (int)(val > 255 ? 255 : (val < 0 ? 0 : val));
                }
            }
        }

        if ( step < nsteps - 1 ) {
            // Wait for all the tasks to finish the step
            #pragma omp taskwait
            Image tmp = src; src = dst; dst = tmp;
        }
    }
}

int main(int argc, char **argv)
{
    int width = 1536, height = 1024;
    double ratio = width / (double)height;

    double time;
    int i, j, pixels_inside = 0;

    int channels = 3;

    Image image(channels, height, width);

    Image filtered_image(channels, height, width);

    auto t1 = chrono::high_resolution_clock::now();

    pixels_inside = mandelbrot(image, ratio);

    auto t2 = chrono::high_resolution_clock::now();

    cout << "Mandelbrot time: " << chrono::duration<double>(t2 - t1).count() << endl;
    cout << "Total Mandelbrot pixels: " << pixels_inside << endl;

    auto t3 = chrono::high_resolution_clock::now(); 

    convolution_2d(image, filtered_image, 5, 0.37, 20);

    auto t4 = chrono::high_resolution_clock::now();

    cout << "Convolution time: " << chrono::duration<double>(t4 - t3).count() << endl;

    // Put Mandelbrot and Convolution time to files for computing the average later
    std::ofstream out;
    out.open("out_m.txt", std::ios_base::app);
    out << chrono::duration<double>(t2 - t1).count() << endl;
    out.close(); 

    out.open("out_c.txt", std::ios_base::app);
    out << chrono::duration<double>(t4 - t3).count() << endl;
    out.close(); 

    cout << "Total time: " << chrono::duration<double>((t4 - t3) + (t2-t1)).count() << endl;

    cout << endl; 

    std::ofstream ofs("mandelbrot-task.ppm", std::ofstream::out);
    ofs << "P3" << std::endl;
    ofs << width << " " << height << std::endl;
    ofs << 255 << std::endl;

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            ofs << " " << filtered_image(0, j, i) << " " << filtered_image(1, j, i) << " " << filtered_image(2, j, i) << std::endl;
        }
    }
    ofs.close();

    return 0;
}