#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "ConvexHullBasis.hpp"

namespace
{
using Clock = std::chrono::steady_clock;
using PPoly = SplineTrajectory::PPolyND<3>;
using Matrix = PPoly::MatrixType;
using Hull = SplineTrajectory::ConvexHullRepresentation<3>;

volatile double benchmark_sink = 0.0;

struct Measurement
{
    std::string category;
    int segments = 0;
    int degree = 0;
    int derivative = 0;
    std::string basis;
    int subdivision_depth = 0;
    std::string direction;
    int output_controls = 0;
    double nanoseconds = 0.0;
};

template <typename Function>
double medianNanoseconds(Function &&function)
{
    for (int i = 0; i < 12; ++i)
        function();

    int iterations = 1;
    constexpr double target_seconds = 0.018;
    while (iterations < (1 << 24))
    {
        const auto begin = Clock::now();
        for (int i = 0; i < iterations; ++i)
            function();
        const double elapsed =
            std::chrono::duration<double>(Clock::now() - begin).count();
        if (elapsed >= target_seconds)
            break;
        const double multiplier =
            std::clamp(target_seconds / std::max(elapsed, 1e-9), 2.0, 16.0);
        iterations = std::min(
            1 << 24,
            std::max(iterations + 1,
                     static_cast<int>(std::ceil(iterations * multiplier))));
    }

    std::vector<double> samples;
    samples.reserve(9);
    for (int sample = 0; sample < 9; ++sample)
    {
        const auto begin = Clock::now();
        for (int i = 0; i < iterations; ++i)
            function();
        const double elapsed =
            std::chrono::duration<double, std::nano>(
                Clock::now() - begin).count();
        samples.push_back(elapsed / static_cast<double>(iterations));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

PPoly makePolynomial(int segments, int degree, std::mt19937 &generator)
{
    const int coefficients_per_segment = degree + 1;
    std::vector<double> breakpoints(segments + 1, 0.0);
    for (int segment = 0; segment < segments; ++segment)
    {
        const double duration =
            0.75 + 0.5 * static_cast<double>((segment * 37) % 101) / 100.0;
        breakpoints[segment + 1] = breakpoints[segment] + duration;
    }

    std::normal_distribution<double> normal(0.0, 0.25);
    Matrix coefficients(segments * coefficients_per_segment, 3);
    for (Eigen::Index row = 0; row < coefficients.rows(); ++row)
        for (Eigen::Index column = 0; column < coefficients.cols(); ++column)
            coefficients(row, column) = normal(generator);
    return PPoly(breakpoints, coefficients, coefficients_per_segment);
}

Matrix makeGradients(Eigen::Index rows, std::mt19937 &generator)
{
    std::normal_distribution<double> normal(0.0, 0.2);
    Matrix gradients(rows, 3);
    for (Eigen::Index row = 0; row < gradients.rows(); ++row)
        for (Eigen::Index column = 0; column < gradients.cols(); ++column)
            gradients(row, column) = normal(generator);
    return gradients;
}

void addForwardMeasurement(std::vector<Measurement> &measurements,
                           const std::string &category,
                           const PPoly &polynomial,
                           int segments,
                           int degree,
                           SplineTrajectory::ConvexHullBasis basis,
                           int derivative,
                           int subdivision_depth)
{
    const char *basis_name =
        basis == SplineTrajectory::ConvexHullBasis::Bezier
            ? "Bezier" : "MINVO";
    const int controls =
        segments * (degree - derivative + 1) *
        (basis == SplineTrajectory::ConvexHullBasis::Bezier
             ? (1 << subdivision_depth) : 1);
    const double elapsed = medianNanoseconds([&]()
    {
        const auto representation = Hull::fromPPoly(
            polynomial, basis, derivative, subdivision_depth);
        benchmark_sink +=
            representation.controls()(representation.controls().rows() - 1, 0);
    });
    measurements.push_back({
        category, segments, degree, derivative, basis_name,
        subdivision_depth, "forward", controls, elapsed});
}

void addBackwardMeasurement(std::vector<Measurement> &measurements,
                            const std::string &category,
                            const PPoly &polynomial,
                            int segments,
                            int degree,
                            SplineTrajectory::ConvexHullBasis basis,
                            int derivative,
                            int subdivision_depth,
                            std::mt19937 &generator)
{
    const char *basis_name =
        basis == SplineTrajectory::ConvexHullBasis::Bezier
            ? "Bezier" : "MINVO";
    const auto representation = Hull::fromPPoly(
        polynomial, basis, derivative, subdivision_depth);
    const Matrix gradients =
        makeGradients(representation.controls().rows(), generator);
    const double elapsed = medianNanoseconds([&]()
    {
        const auto result = representation.backward(gradients);
        benchmark_sink += result.coefficients(0, 0) +
                          result.durations(result.durations.size() - 1);
    });
    measurements.push_back({
        category, segments, degree, derivative, basis_name,
        subdivision_depth, "backward",
        static_cast<int>(representation.controls().rows()), elapsed});
}

void writeCsv(const std::string &path,
              const std::vector<Measurement> &measurements)
{
    std::ofstream stream(path);
    stream << std::setprecision(17);
    stream << "category,segments,degree,derivative,basis,subdivision_depth,"
              "direction,output_controls,nanoseconds,nanoseconds_per_segment\n";
    for (const auto &measurement : measurements)
    {
        stream << measurement.category << ','
               << measurement.segments << ','
               << measurement.degree << ','
               << measurement.derivative << ','
               << measurement.basis << ','
               << measurement.subdivision_depth << ','
               << measurement.direction << ','
               << measurement.output_controls << ','
               << measurement.nanoseconds << ','
               << measurement.nanoseconds / measurement.segments << '\n';
    }
}

Measurement findMeasurement(
    const std::vector<Measurement> &measurements,
    const std::string &category,
    int segments,
    int degree,
    int derivative,
    const std::string &basis,
    int depth,
    const std::string &direction)
{
    const auto iterator = std::find_if(
        measurements.begin(), measurements.end(),
        [&](const Measurement &measurement)
        {
            return measurement.category == category &&
                   measurement.segments == segments &&
                   measurement.degree == degree &&
                   measurement.derivative == derivative &&
                   measurement.basis == basis &&
                   measurement.subdivision_depth == depth &&
                   measurement.direction == direction;
        });
    if (iterator == measurements.end())
        throw std::runtime_error("Missing benchmark measurement.");
    return *iterator;
}

double microseconds(const Measurement &measurement)
{
    return measurement.nanoseconds * 1e-3;
}
}

int main(int argc, char **argv)
{
    std::mt19937 generator(20260719);
    std::vector<Measurement> measurements;
    const std::vector<int> segment_counts{1, 4, 16, 64, 256, 512, 1024};

    for (int segments : segment_counts)
    {
        const PPoly polynomial = makePolynomial(segments, 5, generator);

        for (int derivative : {1, 2})
        {
            const double elapsed = medianNanoseconds([&]()
            {
                const auto result = polynomial.derivative(derivative);
                benchmark_sink += result.getCoefficients()(
                    result.getCoefficients().rows() - 1, 0);
            });
            measurements.push_back({
                "segments", segments, 5, derivative, "Power", 0,
                "derivative", segments * (6 - derivative), elapsed});
        }

        for (int derivative : {0, 1, 2})
        {
            addForwardMeasurement(
                measurements, "segments", polynomial, segments, 5,
                SplineTrajectory::ConvexHullBasis::Bezier,
                derivative, 0);
            addForwardMeasurement(
                measurements, "segments", polynomial, segments, 5,
                SplineTrajectory::ConvexHullBasis::MINVO,
                derivative, 0);
        }
        addBackwardMeasurement(
            measurements, "segments", polynomial, segments, 5,
            SplineTrajectory::ConvexHullBasis::Bezier,
            0, 0, generator);
        addBackwardMeasurement(
            measurements, "segments", polynomial, segments, 5,
            SplineTrajectory::ConvexHullBasis::MINVO,
            0, 0, generator);
    }

    constexpr int subdivision_segments = 64;
    const PPoly subdivision_polynomial =
        makePolynomial(subdivision_segments, 5, generator);
    for (int depth = 0; depth <= 4; ++depth)
    {
        addForwardMeasurement(
            measurements, "subdivision", subdivision_polynomial,
            subdivision_segments, 5,
            SplineTrajectory::ConvexHullBasis::Bezier,
            0, depth);
        addBackwardMeasurement(
            measurements, "subdivision", subdivision_polynomial,
            subdivision_segments, 5,
            SplineTrajectory::ConvexHullBasis::Bezier,
            0, depth, generator);
    }

    constexpr int degree_segments = 64;
    for (int degree : {3, 5, 7})
    {
        const PPoly polynomial =
            makePolynomial(degree_segments, degree, generator);
        for (auto basis : {
                 SplineTrajectory::ConvexHullBasis::Bezier,
                 SplineTrajectory::ConvexHullBasis::MINVO})
        {
            addForwardMeasurement(
                measurements, "degree", polynomial, degree_segments,
                degree, basis, 0, 0);
            addBackwardMeasurement(
                measurements, "degree", polynomial, degree_segments,
                degree, basis, 0, 0, generator);
        }
    }

    const std::string csv_path =
        argc > 1 ? argv[1] : "convex_hull_benchmark.csv";
    writeCsv(csv_path, measurements);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nQuintic PPolyND<3>, median wall time [us]\n";
    std::cout << "| segments | PPoly d1 | PPoly d2 | Bz p | Bz v | Bz a |"
                 " MV p | MV v | MV a | Bz back | MV back |\n";
    std::cout << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (int segments : segment_counts)
    {
        auto value = [&](int derivative,
                         const std::string &basis,
                         const std::string &direction)
        {
            return microseconds(findMeasurement(
                measurements, "segments", segments, 5,
                derivative, basis, 0, direction));
        };
        std::cout << "| " << segments
                  << " | " << value(1, "Power", "derivative")
                  << " | " << value(2, "Power", "derivative")
                  << " | " << value(0, "Bezier", "forward")
                  << " | " << value(1, "Bezier", "forward")
                  << " | " << value(2, "Bezier", "forward")
                  << " | " << value(0, "MINVO", "forward")
                  << " | " << value(1, "MINVO", "forward")
                  << " | " << value(2, "MINVO", "forward")
                  << " | " << value(0, "Bezier", "backward")
                  << " | " << value(0, "MINVO", "backward")
                  << " |\n";
    }

    std::cout << "\nBezier subdivision, 64 quintic segments\n";
    std::cout << "| depth | pieces | controls | forward [us] |"
                 " backward [us] |\n";
    std::cout << "|---:|---:|---:|---:|---:|\n";
    for (int depth = 0; depth <= 4; ++depth)
    {
        const auto forward = findMeasurement(
            measurements, "subdivision", subdivision_segments,
            5, 0, "Bezier", depth, "forward");
        const auto backward = findMeasurement(
            measurements, "subdivision", subdivision_segments,
            5, 0, "Bezier", depth, "backward");
        std::cout << "| " << depth
                  << " | " << subdivision_segments * (1 << depth)
                  << " | " << forward.output_controls
                  << " | " << microseconds(forward)
                  << " | " << microseconds(backward) << " |\n";
    }

    std::cout << "\nDegree scaling, 64 segments\n";
    std::cout << "| degree | basis | forward [us] | backward [us] |\n";
    std::cout << "|---:|:---|---:|---:|\n";
    for (int degree : {3, 5, 7})
    {
        for (const std::string basis : {"Bezier", "MINVO"})
        {
            const auto forward = findMeasurement(
                measurements, "degree", degree_segments,
                degree, 0, basis, 0, "forward");
            const auto backward = findMeasurement(
                measurements, "degree", degree_segments,
                degree, 0, basis, 0, "backward");
            std::cout << "| " << degree << " | " << basis
                      << " | " << microseconds(forward)
                      << " | " << microseconds(backward) << " |\n";
        }
    }
    std::cout << "\nCSV: " << csv_path << '\n';
    std::cout << "sink: " << benchmark_sink << '\n';
    return 0;
}
