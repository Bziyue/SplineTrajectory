#define EIGEN_RUNTIME_NO_MALLOC
#include "TestSupport.hpp"
#include "SplineTrajectory.hpp"
#include <future>
#include <iostream>
#include <stdexcept>

using spline_test::require;

/** @brief Compare sampler output with an independent ascending-power derivative sum. */
template <int DIM, int ORDER>
void checkSampler()
{
    using Poly = SplineTrajectory::PPolyND<DIM, ORDER>;
    constexpr int count = ORDER == Eigen::Dynamic ? 11 : ORDER;
    typename Poly::CoefficientMatrix coefficients(4 * count, DIM);
    for (int i = 0; i < coefficients.rows(); ++i)
        for (int axis = 0; axis < DIM; ++axis)
            coefficients(i, axis) = std::sin(0.17 * (i + 1) * (axis + 1));
    const std::vector<double> knots{2.0, 2.3, 3.2, 3.25, 4.0};
    const Poly poly(knots, coefficients, count);
    auto sample = poly.template makeSampler<5>();
    auto verify = [&](double t) {
        int piece = 0;
        while (piece < 3 && t >= knots[piece + 1]) ++piece;
        const double dt = t - knots[piece];
        const auto values = sample.evaluateDerivatives(t);
        const auto combined = poly.template evaluateDerivatives<5>(t);
        for (int d = 0; d <= 5; ++d)
        {
            typename Poly::Vector expected = Poly::Vector::Zero();
            for (int k = d; k < count; ++k)
            {
                double factor = 1.0;
                for (int j = 0; j < d; ++j) factor *= k - j;
                expected += factor * std::pow(dt, k - d) * coefficients.row(piece * count + k).transpose();
            }
            const double tolerance = 2e-10 * std::max(1.0, expected.norm());
            require((values[d] - expected).norm() <= tolerance, "sequential derivative mismatch");
            require((combined[d] - expected).norm() <= tolerance, "combined derivative mismatch");
            require((poly.evaluate(t, d) - expected).norm() <= tolerance, "random derivative mismatch");
        }
    };
    Eigen::internal::set_is_malloc_allowed(false);
    for (double t : {1.9, 2.0, 2.0, 2.2, 2.3, 3.2, 3.25, 4.0, 4.1, 3.21, 2.0, 4.0}) verify(t);
    for (int k = 0; k <= 200; ++k) verify(2.0 + 0.01 * k);
    Eigen::internal::set_is_malloc_allowed(true);
    const auto invalid = sample.evaluateDerivatives(std::numeric_limits<double>::infinity());
    for (const auto &value : invalid) require(value.array().isNaN().all(), "nonfinite time must return NaN");

    // Each worker performs its first query on a never-sampled immutable source.
    const Poly cold(knots, coefficients, count);
    auto worker = [&cold]() {
        auto cursor = cold.template makeSampler<3>();
        double sum = 0;
        for (int k = 0; k < 1000; ++k) sum += cursor.evaluateDerivatives(2.0 + 0.001 * k)[1].sum();
        return sum;
    };
    auto first = std::async(std::launch::async, worker);
    auto second = std::async(std::launch::async, worker);
    require(first.get() == second.get(), "first concurrent reads differ");

    const Poly invalid_poly;
    auto invalid_sampler = invalid_poly.template makeSampler<3>();
    require(invalid_sampler.evaluateDerivatives(0)[0].array().isNaN().all(), "invalid source must return NaN");
}

int main()
{
    return spline_test::run("test_sampler", []
    {
        checkSampler<1, 4>(); checkSampler<3, 6>(); checkSampler<3, 8>();
        checkSampler<2, Eigen::Dynamic>();
    });
}
