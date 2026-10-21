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
    require(std::isnan(curve.energy()), "Invalid energy query must not report zero cost");
    throws<std::logic_error>([&] { curve.energyGradient(); });
    const Matrix points = Matrix::Random(3, Spline::kDimension);
    const std::vector<double> durations{1.2, 1.7};
    curve.update(std::vector<double>{}, points);
    require(!curve.isValid(), "Empty knots must fail without indexing front");
    curve.update(durations, Matrix::Zero(2, Spline::kDimension), 0.0);
    require(!curve.isValid(), "Wrong point topology must fail before Eigen operations");
    curve.update(durations, points, -2.0);
    require(curve.isValid(), "Valid topology must recover after failure");
    auto copy = curve;
    auto moved = std::move(copy);
    require(!copy.isValid() && moved.isValid(), "Move transfers polynomial validity");
    auto snapshot = curve.copyPolynomial();
    const auto *storage = curve.polynomial().coefficients().data();
    typename Spline::Gradients gradient;
    gradient.resetTopology(2);
    Matrix gc = Matrix::Ones(2 * Spline::kCoefficientCount, Spline::kDimension);
    Eigen::VectorXd gt = Eigen::VectorXd::Zero(2);
    Eigen::internal::set_is_malloc_allowed(false);
    curve.update(durations, points, -2.0);
    curve.backward(gc, gt, gradient);
    Eigen::internal::set_is_malloc_allowed(true);
    require(storage == curve.polynomial().coefficients().data(), "Fixed topology must reuse coefficient storage");
    curve.update({1.0, -1.0}, points, 0.0);
    require(!curve.isValid() && curve.polynomial().coefficients().rows() == 0,
            "Failed updates must not expose previous coefficients");
    require(snapshot.isValid() && snapshot.coefficients().isApprox(moved.polynomial().coefficients()),
            "Owning snapshots survive source updates");
    throws<std::invalid_argument>([&] { curve.backward(gc, gt, gradient); });
    curve.update(durations, points, 0.0);
    require(curve.isValid(), "Failure must not poison future updates");
}
void checkPolynomial()
{
    using Poly = PPolyND<2, 6>;
    auto poly = Poly::constant({2.0, 3.0, 5.0}, Eigen::Vector2d(3.0, 4.0));
    require(poly.degree() == 0 && Poly::kMaxCoefficients == 6, "Capacity differs from degree");
    throws<std::invalid_argument>([&] { poly.evaluate(2.0, -1); });
    throws<std::invalid_argument>([&] { poly.at(0).evaluateLocal(0.0, -1); });
    throws<std::invalid_argument>([&] { poly.derivative(-1); });
    throws<std::invalid_argument>([&] { poly.evaluate(std::vector<double>{}, -1); });
    require(poly.evaluate(2.0, 8).isZero() && poly.derivative(8).evaluate(2.0).isZero(),
            "Derivatives above degree must be zero");
    for (double step : {0.0, -0.1, std::numeric_limits<double>::infinity()})
        throws<std::invalid_argument>([&] { poly.generateTimeSequence(step); });
    throws<std::invalid_argument>([&] { poly.generateTimeSequence(2.0, 1.0, 0.1); });
    throws<std::length_error>([&] { poly.generateTimeSequence(1e20, 2e20, 1e-10); });
    require(poly.generateTimeSequence(3.0, 3.0, 0.1) == std::vector<double>{3.0}, "Zero-length interval");
    auto times = poly.generateTimeSequence(0.6);
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
    throws<std::out_of_range>([&] { (void)poly.at(-1); });
    auto moved = std::move(poly);
    require(!poly.isValid() && !poly.evaluate(0.0).allFinite() && moved.isValid(), "Polynomial move state");
    require(!Poly::zero({0.0, 1.0}, -1).isValid(), "Negative coefficient count is rejected before allocation");
}
template<int DIM> void checkHullDimension()
{
    using Poly = PPolyND<DIM, 8>;
    typename Poly::CoefficientMatrix coefficients = Poly::CoefficientMatrix::Random(16, DIM);
    Poly polynomial({0.2, 1.4, 2.7}, coefficients, 8);
    auto hull = toMINVO(polynomial, 2, 2);
    typename Poly::CoefficientMatrix weights = Poly::CoefficientMatrix::Random(hull.controls().rows(), DIM);
    auto partials = hull.backward(weights);
    const double primal = (hull.controls().array() * weights.array()).sum();
    const double dual = (coefficients.array() * partials.coefficients.array()).sum();
    require(std::abs(primal - dual) < 1e-11 * std::max(1.0, std::abs(primal)),
            "Hull forward and adjoint must agree across dimensions");
    const auto huge = Poly::zero({0.0, 1e200}, 8);
    hull.configure(huge);
    throws<std::overflow_error>([&] { hull.update(huge); });
    require(hull.isConfigured() && !hull.isValid(), "Conversion overflow must invalidate numeric state");
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
    throws<std::logic_error>([&] { h1.backward(Poly1::CoefficientMatrix{}); });
    h1.configure(p1); h3.configure(p3);
    require(h1.isConfigured() && !h1.isValid() && h1.matches(p1, {}), "Configured hull state");
    require(h1.kernel() == h3.kernel(), "Identical kernels must be shared across spatial dimensions");
    h1.update(p1);
    require(h1.isValid(), "Updated hull state");
    throws<std::invalid_argument>([&] { h1.update(Poly1{}); });
    require(h1.isConfigured() && !h1.isValid(), "Failed hull update invalidates numeric state");
    h1.update(p1);
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
