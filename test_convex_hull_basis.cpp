#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "ConvexHullBasis.hpp"

namespace
{
using PPoly = SplineTrajectory::PPolyND<3>;
using Hull = SplineTrajectory::ConvexHullRepresentation<3>;
using Matrix = PPoly::MatrixType;

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double relativeError(double actual, double expected)
{
    return std::abs(actual - expected) /
           std::max({1.0, std::abs(actual), std::abs(expected)});
}

PPoly makePolynomial(const std::vector<double> &durations,
                     const Matrix &coefficients,
                     int num_coefficients)
{
    std::vector<double> breaks(durations.size() + 1, 0.0);
    breaks.front() = 0.37;
    for (std::size_t i = 0; i < durations.size(); ++i)
        breaks[i + 1] = breaks[i] + durations[i];
    return PPoly(breaks, coefficients, num_coefficients);
}

double objective(const PPoly &polynomial,
                 SplineTrajectory::ConvexHullBasis basis,
                 int derivative,
                 int subdivision_depth,
                 const Matrix &weights)
{
    const auto representation = Hull::fromPPoly(
        polynomial, basis, derivative, subdivision_depth);
    return (representation.controls().array() * weights.array()).sum();
}

void checkRepresentationValues(const PPoly &polynomial,
                               SplineTrajectory::ConvexHullBasis basis,
                               int derivative,
                               int subdivision_depth)
{
    const auto representation = Hull::fromPPoly(
        polynomial, basis, derivative, subdivision_depth);
    const Eigen::MatrixXd transform =
        Hull::powerToControlMatrix(basis, representation.degree());
    const Eigen::MatrixXd inverse = transform.inverse();
    const int degree = representation.degree();

    for (int piece = 0; piece < representation.numPieces(); ++piece)
    {
        const Matrix controls = representation.pieceControls(piece);
        const Matrix normalized_power = inverse * controls;
        const auto &info = representation.pieceInfo(piece);

        for (int sample = 0; sample <= 20; ++sample)
        {
            const double u = static_cast<double>(sample) / 20.0;
            Eigen::RowVectorXd powers(degree + 1);
            powers(0) = 1.0;
            for (int k = 1; k <= degree; ++k)
                powers(k) = powers(k - 1) * u;

            const Eigen::Vector3d from_controls =
                (powers * normalized_power).transpose();
            const double local_time =
                info.start_time -
                polynomial.getBreakpoints()[info.source_segment] +
                u * info.duration;
            const Eigen::Vector3d expected =
                polynomial[info.source_segment].evaluate(local_time, derivative);
            require((from_controls - expected).norm() < 2e-9,
                    "Converted controls do not reproduce the polynomial derivative.");

            const Eigen::RowVectorXd hull_weights = powers * inverse;
            require(std::abs(hull_weights.sum() - 1.0) < 2e-9,
                    "Basis does not form a partition of unity.");
            require(hull_weights.minCoeff() > -2e-8,
                    "Basis is not non-negative on [0, 1].");
        }
    }
}

void checkBackward(SplineTrajectory::ConvexHullBasis basis,
                   int derivative,
                   int subdivision_depth,
                   std::mt19937 &generator)
{
    constexpr int num_segments = 3;
    constexpr int num_coefficients = 8;
    std::vector<double> durations{0.73, 1.21, 0.91};
    std::normal_distribution<double> normal(0.0, 0.5);

    Matrix coefficients(num_segments * num_coefficients, 3);
    for (Eigen::Index row = 0; row < coefficients.rows(); ++row)
        for (Eigen::Index col = 0; col < coefficients.cols(); ++col)
            coefficients(row, col) = normal(generator);

    PPoly polynomial = makePolynomial(durations, coefficients, num_coefficients);
    const auto representation = Hull::fromPPoly(
        polynomial, basis, derivative, subdivision_depth);

    Matrix weights(representation.controls().rows(), 3);
    for (Eigen::Index row = 0; row < weights.rows(); ++row)
        for (Eigen::Index col = 0; col < weights.cols(); ++col)
            weights(row, col) = normal(generator);

    const auto gradients = representation.backward(weights);
    constexpr double epsilon = 2e-7;
    double max_coefficient_error = 0.0;

    for (Eigen::Index row = 0; row < coefficients.rows(); ++row)
    {
        for (Eigen::Index col = 0; col < coefficients.cols(); ++col)
        {
            Matrix plus = coefficients;
            Matrix minus = coefficients;
            plus(row, col) += epsilon;
            minus(row, col) -= epsilon;
            const double numerical =
                (objective(makePolynomial(durations, plus, num_coefficients),
                           basis, derivative, subdivision_depth, weights) -
                 objective(makePolynomial(durations, minus, num_coefficients),
                           basis, derivative, subdivision_depth, weights)) /
                (2.0 * epsilon);
            max_coefficient_error = std::max(
                max_coefficient_error,
                relativeError(gradients.coefficients(row, col), numerical));
        }
    }

    double max_duration_error = 0.0;
    for (int segment = 0; segment < num_segments; ++segment)
    {
        std::vector<double> plus = durations;
        std::vector<double> minus = durations;
        plus[segment] += epsilon;
        minus[segment] -= epsilon;
        const double numerical =
            (objective(makePolynomial(plus, coefficients, num_coefficients),
                       basis, derivative, subdivision_depth, weights) -
             objective(makePolynomial(minus, coefficients, num_coefficients),
                       basis, derivative, subdivision_depth, weights)) /
            (2.0 * epsilon);
        max_duration_error = std::max(
            max_duration_error,
            relativeError(gradients.durations(segment), numerical));
    }

    require(max_coefficient_error < 3e-7,
            "Coefficient adjoint failed its finite-difference check.");
    require(max_duration_error < 8e-7,
            "Duration adjoint failed its finite-difference check.");
}
}

int main()
{
    try
    {
        std::mt19937 generator(42);
        constexpr int num_coefficients = 8;
        Matrix coefficients(2 * num_coefficients, 3);
        std::normal_distribution<double> normal(0.0, 0.3);
        for (Eigen::Index row = 0; row < coefficients.rows(); ++row)
            for (Eigen::Index col = 0; col < coefficients.cols(); ++col)
                coefficients(row, col) = normal(generator);
        const PPoly polynomial =
            makePolynomial({0.83, 1.17}, coefficients, num_coefficients);

        for (int derivative = 0; derivative < num_coefficients; ++derivative)
        {
            checkRepresentationValues(
                polynomial, SplineTrajectory::ConvexHullBasis::Bezier,
                derivative, 3);
            checkRepresentationValues(
                polynomial, SplineTrajectory::ConvexHullBasis::MINVO,
                derivative, 0);
        }

        checkBackward(
            SplineTrajectory::ConvexHullBasis::Bezier, 0, 3, generator);
        checkBackward(
            SplineTrajectory::ConvexHullBasis::Bezier, 2, 2, generator);
        checkBackward(
            SplineTrajectory::ConvexHullBasis::Bezier, 6, 1, generator);
        checkBackward(
            SplineTrajectory::ConvexHullBasis::MINVO, 0, 0, generator);
        checkBackward(
            SplineTrajectory::ConvexHullBasis::MINVO, 3, 0, generator);
        checkBackward(
            SplineTrajectory::ConvexHullBasis::MINVO, 7, 0, generator);

        std::cout << "test_convex_hull_basis passed" << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "test_convex_hull_basis failed: "
                  << error.what() << std::endl;
        return 1;
    }
}
