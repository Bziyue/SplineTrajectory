#define EIGEN_RUNTIME_NO_MALLOC
#include "TestSupport.hpp"
#include "SplineTrajectory.hpp"
#include "SplineConvexHull.hpp"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace SplineTrajectory;
namespace {
using spline_test::require;
using spline_test::throws;
template<class Spline> void checkSpline()
{
    using Matrix = typename Spline::CoefficientMatrix;
    Spline curve;
    require(!curve.isValid() && curve.polynomial().numSegments() == 0 && curve.duration() == 0.0,
            "Default spline must be safely inspectable");
    const Matrix points = Matrix::Random(3, Spline::kDimension);
    const std::vector<double> durations{1.2, 1.7};
    if (const auto status = curve.update(durations, points, -2.0); !status) { throw std::runtime_error(status.message); }
    require(curve.isValid(), "Valid topology must recover after failure");
    auto copy = curve;
    auto moved = std::move(copy);
    require(!copy.isValid() && moved.isValid(), "Move transfers polynomial validity");
    auto snapshot = curve.copyPolynomial();
    const auto *storage = curve.polynomial().coefficients().data();
    typename Spline::Gradients gradient;
    curve.energyGradient(gradient);
    Matrix gc = Matrix::Ones(2 * Spline::kCoefficientCount, Spline::kDimension);
    Eigen::VectorXd gt = Eigen::VectorXd::Zero(2);
    Eigen::internal::set_is_malloc_allowed(false);
    if (const auto status = curve.update(durations, points, -2.0); !status) { throw std::runtime_error(status.message); }
    curve.backward(gc, gt, gradient);
    Eigen::internal::set_is_malloc_allowed(true);
    require(storage == curve.polynomial().coefficients().data(), "Fixed topology must reuse coefficient storage");
    require(!curve.update(durations, points, 1e308), "Unrepresentable cumulative knots must fail");
    require(!curve.isValid() && curve.polynomial().coefficients().rows() == 0,
            "Failed updates must not expose previous coefficients");
    require(snapshot.isValid() && snapshot.coefficients().isApprox(moved.polynomial().coefficients()),
            "Owning snapshots survive source updates");
    if (const auto status = curve.update(durations, points, 0.0); !status) { throw std::runtime_error(status.message); }
    require(curve.isValid(), "Failure must not poison future updates");
}
void checkPolynomial()
{
    using Poly = PPolyND<2, 6>;
    auto poly = Poly::constant({2.0, 3.0, 5.0}, Eigen::Vector2d(3.0, 4.0));
    require(poly.degree() == 0 && Poly::kMaxCoefficients == 6, "Capacity differs from degree");
    require(poly.evaluate(2.0, 8).isZero() && poly.derivative(8).value().evaluate(2.0).isZero(),
            "Derivatives above degree must be zero");
    require(!(poly.generateTimeSequence(1e20, 2e20, 1e-10)), "Expected failure status");
    require(poly.generateTimeSequence(3.0, 3.0, 0.1).value() == std::vector<double>{3.0}, "Zero-length interval");
    auto times = poly.generateTimeSequence(0.6).value();
    require(times.front() == 2.0 && times.back() == 5.0 && std::is_sorted(times.begin(), times.end()),
            "Generated samples include endpoints");
    Poly::CoefficientMatrix output(times.size(), 2);
    Eigen::internal::set_is_malloc_allowed(false);
    poly.evaluateInto(times, output);
    auto sampler = poly.makeSampler<3>();
    require(sampler.evaluate(3.0).isApprox(Eigen::Vector2d(3.0, 4.0)), "No warm-up sampling");
    Eigen::internal::set_is_malloc_allowed(true);
    auto begin = poly.begin(), end = poly.end();
    require(std::distance(begin, end) == 2 && (1 + begin)->index() == 1 && begin[1].index() == 1,
            "Random-access iterator arithmetic");
    require(begin < end && begin <= end && end > begin && end >= begin && end - 1 == begin + 1,
            "Random-access iterator ordering");
    std::advance(begin, 2); begin -= 1; begin += -1;
    require(begin == poly.begin(), "Iterator compound arithmetic");
    auto moved = std::move(poly);
    require(!poly.isValid() && moved.isValid(), "Polynomial move state");
}
template<int DIM> void checkHullDimension()
{
    using Poly = PPolyND<DIM, 8>;
    typename Poly::CoefficientMatrix coefficients = Poly::CoefficientMatrix::Random(16, DIM);
    auto polynomial = Poly::create({0.2, 1.4, 2.7}, coefficients, 8);
    auto hull = toMINVO(polynomial, 2, 2);
    typename Poly::CoefficientMatrix weights = Poly::CoefficientMatrix::Random(hull.controls().rows(), DIM);
    auto partials = hull.backward(weights);
    const double primal = (hull.controls().array() * weights.array()).sum();
    const double dual = (coefficients.array() * partials.coefficients.array()).sum();
    require(std::abs(primal - dual) < 1e-11 * std::max(1.0, std::abs(primal)),
            "Hull forward and adjoint must agree across dimensions");
}
void checkHull()
{
    using Poly1 = PPolyND<1, 4>;
    using Poly3 = PPolyND<3, 4>;
    auto p1 = Poly1::zero({0.0, 1.0}, 4);
    auto p3 = Poly3::zero({0.0, 1.0}, 4);
    ConvexHullWorkspace<1> h1;
    ConvexHullWorkspace<3> h3;
    require(!h1.isConfigured() && !h1.isValid(), "Default hull state");
    h1.configure(p1); h3.configure(p3);
    require(h1.isConfigured() && !h1.isValid() && h1.matches(p1, {}), "Configured hull state");
    require(h1.kernel() == h3.kernel(), "Identical kernels must be shared across spatial dimensions");
    h1.update(p1);
    require(h1.isValid(), "Updated hull state");
    auto moved = std::move(h1);
    require(!h1.isValid() && moved.isValid(), "Moved-from hull validity");
}
}
int main()
{
    return spline_test::run("test_trajectory_contracts", []
    {
        checkSpline<CubicSplineND<1>>();
        checkSpline<QuinticSplineND<3>>();
        checkSpline<SepticSplineND<4>>();
        checkPolynomial(); checkHull();
        checkHullDimension<1>(); checkHullDimension<2>(); checkHullDimension<6>(); checkHullDimension<10>();
    });
}
