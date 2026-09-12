#include "SplineTrajectory.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#ifndef BENCH_DEGREE
#define BENCH_DEGREE 7
#endif
#ifndef BENCH_DIMENSION
#define BENCH_DIMENSION 3
#endif
using Spline = SplineTrajectory::MinDerivativeSplineND<BENCH_DIMENSION, (BENCH_DEGREE + 1) / 2>;
#ifdef BENCH_OLD
using Matrix = Spline::MatrixType;
#else
using Matrix = Spline::CoefficientMatrix;
#endif
using Clock = std::chrono::steady_clock;
volatile double kernel_checksum = 0.0;

template<class Function> double measure(int repeats, Function function)
{
    std::vector<double> batches;
    for (int batch = 0; batch < 15; ++batch)
    {
        const auto start = Clock::now();
        for (int i = 0; i < repeats; ++i) function();
        batches.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count() / repeats);
    }
    std::sort(batches.begin(), batches.end());
    return batches[7];
}
int main(int argc, char **argv)
{
    const int repeats = argc > 1 ? std::max(1, std::atoi(argv[1])) : 10000;
    std::cout << "degree,dimension,pieces,first_build_us,build_p50_us,backward_p50_us,energy,gradient_checksum,object_bytes\n" << std::setprecision(17);
    for (int n : {2, 8, 32, 128})
    {
        std::vector<double> durations(n, 1.2);
        Matrix points(n + 1, BENCH_DIMENSION);
        for (int i = 0; i <= n; ++i)
            for (int j = 0; j < BENCH_DIMENSION; ++j) points(i, j) = 0.3*i + std::sin(i + 0.4*j);
        Spline spline;
        const auto first = Clock::now();
        spline.update(durations, points, 0.0);
        const double first_us = std::chrono::duration<double, std::micro>(Clock::now() - first).count();
        Matrix gc = Matrix::Constant(n * (BENCH_DEGREE + 1), BENCH_DIMENSION, 0.01);
        Eigen::VectorXd gt = Eigen::VectorXd::Constant(n, 0.02);
        Spline::Gradients gradient;
        gradient.resetTopology(n);
        auto backward = [&] {
#ifdef BENCH_OLD
            spline.propagateGrad(gc, gt, gradient);
            kernel_checksum += gradient.times.sum();
#else
            spline.backward(gc, gt, gradient);
            kernel_checksum += gradient.durations.sum();
#endif
        };
        backward();
        const double build = measure(repeats, [&] { spline.update(durations, points, 0.0); });
        const double reverse = measure(repeats, backward);
#ifdef BENCH_OLD
        const double energy = spline.getEnergy(), checksum = gradient.times.sum() + gradient.inner_points.sum();
#else
        const double energy = spline.energy(), checksum = gradient.durations.sum() + gradient.inner_points.sum();
#endif
        std::cout << BENCH_DEGREE << ',' << BENCH_DIMENSION << ',' << n << ',' << first_us << ','
                  << build << ',' << reverse << ',' << energy << ',' << checksum << ',' << sizeof(Spline) << '\n';
    }
}
