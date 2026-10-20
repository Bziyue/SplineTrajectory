/*
    MIT License

    Copyright (c) 2025 Deping Zhang (beiyuena@foxmail.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

/** @file MinimumDerivativeSpline.hpp
 * @brief Shared O(N) interpolation, energy and adjoint pipeline for clamped splines. */
#ifndef SPLINE_MINIMUM_DERIVATIVE_HPP
#define SPLINE_MINIMUM_DERIVATIVE_HPP
#include "PiecewisePolynomial.hpp"
#include "detail/HermiteKernel.hpp"
#include "detail/BlockLDLT.hpp"

// Keep compile-time index visitors inside their numeric kernel on GCC and Clang.
#if defined(__GNUC__) || defined(__clang__)
#define SPLINE_INLINE_VISITOR __attribute__((always_inline))
#else
#define SPLINE_INLINE_VISITOR
#endif
namespace SplineTrajectory {
/** @brief Minimum squared S-th derivative spline, specialized at compile time by order.
 * @tparam DIM Positive spatial dimension. @tparam S Minimum derivative order: 2, 3 or 4.
 * @note Waypoints use caller-defined coordinates and durations use seconds. One object owns
 * its polynomial, factors and workspaces. Updates and backward require exclusive access;
 * const queries on an unchanged object are reentrant. Copies own independent storage.
 * Invalid input or a failed factorization clears the polynomial. No regularization is applied. */
template<int DIM, int S>
class MinDerivativeSplineND
{
    static_assert(DIM > 0 && S >= 2 && S <= 4, "Supported spline dimensions are positive; orders are 2, 3 and 4");
    // M free derivatives per knot, K endpoint values / power coefficients, E polynomial degree.
    static constexpr int M = S - 1, K = 2 * S, E = 2 * S - 1;
    using Kernel = detail::HermiteKernel<S>;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Vector = Eigen::Matrix<double, DIM, 1>;
    using RowVector = Eigen::Matrix<double, 1, DIM>;
    static constexpr int kMatrixOptions = DIM == 1 ? Eigen::ColMajor : Eigen::RowMajor;
    using CoefficientMatrix = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;
    using Polynomial = PPolyND<DIM, K>;
    using BoundaryGradient = SplineTrajectory::BoundaryGradient<DIM>;
    using BoundaryGradients = BoundaryGradientPair<DIM>;
    using Gradients = SplineGradients<DIM>;
    static constexpr int kDimension = DIM, kDegree = E, kCoefficientCount = K, kMinDerivativeOrder = S;
private:
    using Block = Eigen::Matrix<double, M, M, (M == 1 ? Eigen::ColMajor : Eigen::RowMajor)>;
    std::vector<double> times_;
    CoefficientMatrix points_;
    BoundaryConditions<DIM> boundary_;
    Polynomial polynomial_;
    double start_ = 0.0;
    int segments_ = 0;
    // Knot-major derivative rows keep each v/a/j block and all spatial coordinates together.
    CoefficientMatrix derivatives_, adjoint_;
    std::vector<std::array<double, K>> inverse_powers_;
    detail::BlockLDLT<M> factor_;

public:
    /** @brief Construct an empty, invalid spline. */
    MinDerivativeSplineND() = default;
    /** @brief Copy interpolation inputs expressed as absolute knots.
     * @param knots N+1 finite, strictly increasing times in seconds.
     * @param points N+1 waypoint rows. @param boundary Physical endpoint derivatives. */
    MinDerivativeSplineND(const std::vector<double> &knots, const CoefficientMatrix &points,
                          const BoundaryConditions<DIM> &boundary = {})
    { update(knots, points, boundary); }
    /** @brief Copy interpolation inputs expressed as durations.
     * @param times N positive finite durations in seconds. @param points N+1 waypoint rows.
     * @param start Common time origin in seconds. @param boundary Physical endpoint derivatives. */
    MinDerivativeSplineND(const std::vector<double> &times, const CoefficientMatrix &points,
                          double start, const BoundaryConditions<DIM> &boundary = {})
    { update(times, points, start, boundary); }
    /** @brief Replace data using absolute knots; invalid input clears the polynomial.
     * @param knots Strictly increasing times. @param points Waypoints in the caller's frame.
     * @param boundary Physical endpoint derivatives. @note Invalidates all borrowed views. */
    void update(const std::vector<double> &knots, const CoefficientMatrix &points,
                const BoundaryConditions<DIM> &boundary = {})
    {
        detail::durationsFromKnots(knots, times_, start_);
        points_ = points; boundary_ = boundary;
        rebuild();
    }
    /** @brief Replace data using durations, reusing storage for an unchanged topology.
     * @param times Positive durations in seconds. @param points N+1 waypoint rows.
     * @param start Common time origin. @param boundary Physical endpoint derivatives.
     * @note Invalidates borrowed views; invalid input or numeric breakdown clears the polynomial. */
    void update(const std::vector<double> &times, const CoefficientMatrix &points, double start,
                const BoundaryConditions<DIM> &boundary = {})
    {
        times_ = times; points_ = points; start_ = start; boundary_ = boundary;
        rebuild();
    }
    /** @brief Inspect construction validity. @return True only for a finite published polynomial. */
    bool isValid() const { return polynomial_.isValid(); }
    /** @brief Spatial dimension. @return DIM. */
    int dimension() const { return DIM; }
    /** @brief Absolute start time. @return Seconds, or zero for an invalid polynomial. */
    double startTime() const { return polynomial_.startTime(); }
    /** @brief Absolute end time. @return Seconds, or zero for an invalid polynomial. */
    double endTime() const { return polynomial_.endTime(); }
    /** @brief Total duration. @return Seconds, or zero for an invalid polynomial. */
    double duration() const { return polynomial_.duration(); }
    /** @brief Input waypoint count. @return Number of stored rows. */
    size_t numPoints() const { return static_cast<size_t>(points_.rows()); }
    /** @brief Published segment count. @return Zero for an invalid polynomial. */
    int numSegments() const { return polynomial_.numSegments(); }
    /** @brief Borrow physical waypoints. @return Storage valid until update, assignment or move. */
    const CoefficientMatrix &waypoints() const { return points_; }
    /** @brief Borrow input durations. @return Seconds, valid until update, assignment or move. */
    const std::vector<double> &durations() const { return times_; }
    /** @brief Borrow absolute knots. @return Seconds, valid until update, assignment or move. */
    const std::vector<double> &breakpoints() const { return polynomial_.breakpoints(); }
    /** @brief Borrow endpoint derivatives. @return Physical values valid until the next mutation. */
    const BoundaryConditions<DIM> &boundary() const { return boundary_; }
    /** @brief Borrow the polynomial. @return View valid until update, assignment or move. */
    const Polynomial &polynomial() const & { return polynomial_; }
    /** @brief Prevent a borrowed polynomial from outliving a temporary spline. */
    const Polynomial &polynomial() const && = delete;
    /** @brief Copy an independent owning polynomial. @return Snapshot with independent lifetime. */
    Polynomial copyPolynomial() const { return polynomial_; }

    /** @brief Build physical-time basis rows through the fifth derivative.
     * @param t Local time in seconds. @param[out] p Position basis. @param[out] v Velocity basis.
     * @param[out] a Acceleration basis. @param[out] j Jerk basis. @param[out] s Snap basis.
     * @param[out] c Crackle basis. Derivatives beyond the polynomial degree are zero. */
    static void computeBasisFunctions(double t, Eigen::Matrix<double,1,K> &p,
        Eigen::Matrix<double,1,K> &v, Eigen::Matrix<double,1,K> &a,
        Eigen::Matrix<double,1,K> &j, Eigen::Matrix<double,1,K> &s, Eigen::Matrix<double,1,K> &c)
    {
        std::array<Eigen::Matrix<double,1,K> *,6> rows{{&p,&v,&a,&j,&s,&c}};
        detail::staticFor<0,6>([&](auto derivative) SPLINE_INLINE_VISITOR {
            constexpr int r = decltype(derivative)::value;
            rows[r]->setZero();
            double power = 1.0;
            detail::staticFor<r,K>([&](auto degree) SPLINE_INLINE_VISITOR {
                constexpr int k = decltype(degree)::value;
                (*rows[r])(k) = detail::factorial(k) / detail::factorial(k-r) * power;
                power *= t;
            });
        });
    }

    /** @brief Integrate squared minimum derivative using a nonnegative orthogonal expansion.
     * @return Energy in physical units; NaN when the spline is invalid. */
    double energy() const
    {
        if (!isValid()) return std::numeric_limits<double>::quiet_NaN();
        double value = 0.0;
        for (int i = 0; i < segments_; ++i)
        {
            double segment = 0.0;
            detail::staticFor<0,S>([&](auto mode) SPLINE_INLINE_VISITOR {
                constexpr int r = decltype(mode)::value;
                segment += (1.0/(2*r+1))*energyMode<r>(i).squaredNorm();
            });
            value += times_[i]*segment;
        }
        return value;
    }
    /** @brief Differentiate energy holding physical coefficients fixed.
     * @param[out] gradient N direct duration partials; existing storage is reused. */
    void energyDurationPartials(Eigen::VectorXd &gradient) const
    {
        requireValid(); gradient.resize(segments_);
        for (int i = 0; i < segments_; ++i)
            gradient(i) = derivative<S>(i, times_[i]).squaredNorm();
    }
    /** @brief Allocate direct duration partials. @return N physical partial derivatives. */
    Eigen::VectorXd energyDurationPartials() const
    { Eigen::VectorXd result; energyDurationPartials(result); return result; }
    /** @brief Differentiate energy holding durations fixed.
     * @param[out] gradient Physical power-coefficient partials in piece-major order. */
    void energyCoefficientPartials(CoefficientMatrix &gradient) const
    {
        requireValid(); gradient.setZero(segments_*K, DIM);
        for (int i = 0; i < segments_; ++i)
        {
            const auto modes = energyModes(i);
            double power = times_[i];
            detail::staticFor<0,S>([&](auto column) SPLINE_INLINE_VISITOR {
                constexpr int j = decltype(column)::value;
                RowVector row = RowVector::Zero();
                detail::staticFor<0,j+1>([&](auto mode) SPLINE_INLINE_VISITOR {
                    constexpr int r = decltype(mode)::value;
                    row += (2.0 * Kernel::E[r][j] / (2*r+1)) * modes.row(r);
                });
                gradient.row(i*K+S+j) = power * row;
                power *= times_[i];
            });
        }
    }
    /** @brief Allocate coefficient partials. @return Physical coefficient gradient matrix. */
    CoefficientMatrix energyCoefficientPartials() const
    { CoefficientMatrix result; energyCoefficientPartials(result); return result; }
    /** @brief Evaluate the Hamiltonian duration gradient of the optimized spline.
     * @param[out] gradient Total physical duration gradients, without an adjoint solve. */
    void energyDurationGradient(Eigen::VectorXd &gradient) const
    {
        requireValid(); gradient.resize(segments_);
        for (int i = 0; i < segments_; ++i)
        {
            const auto coefficients = polynomial_.coefficients_.template middleRows<K>(i*K);
            double value = -std::pow(detail::factorial(S),2) * coefficients.row(S).squaredNorm();
            detail::staticFor<1,S>([&](auto order) SPLINE_INLINE_VISITOR {
                constexpr int r = decltype(order)::value;
                constexpr double scale = ((S-r+1)%2 ? -2.0 : 2.0) * detail::factorial(r) * detail::factorial(K-r);
                value += scale * coefficients.row(r).dot(coefficients.row(K-r));
            });
            gradient(i) = value;
        }
    }
    /** @brief Allocate total duration gradients. @return N derivatives in physical units. */
    Eigen::VectorXd energyDurationGradient() const
    { Eigen::VectorXd result; energyDurationGradient(result); return result; }
    /** @brief Differentiate energy at interior waypoints using highest-derivative jumps.
     * @param[out] gradient N-1 waypoint gradients; endpoint positions are excluded. */
    void energyWaypointGradient(CoefficientMatrix &gradient) const
    {
        requireValid(); gradient.resize(segments_-1, DIM);
        constexpr double scale = (S%2 ? 2.0 : -2.0) * detail::factorial(E);
        for (int i = 1; i < segments_; ++i)
            gradient.row(i-1) = scale * (polynomial_.coefficients_.row(i*K-1) - polynomial_.coefficients_.row((i+1)*K-1));
    }
    /** @brief Allocate interior waypoint gradients. @return N-1 physical gradient rows. */
    CoefficientMatrix energyWaypointGradient() const
    { CoefficientMatrix result; energyWaypointGradient(result); return result; }
    /** @brief Evaluate energy boundary gradients through integration by parts.
     * @return Start/end position and derivative gradients; unused orders remain zero. */
    BoundaryGradients energyBoundaryGradient() const
    {
        requireValid(); BoundaryGradients result;
        detail::staticFor<0,S>([&](auto order) SPLINE_INLINE_VISITOR {
            constexpr int r = decltype(order)::value;
            constexpr double scale = (S-r)%2 ? -2.0 : 2.0;
            boundaryComponent<r>(result.start) = scale * detail::factorial(E-r) * polynomial_.coefficients_.row(E-r).transpose();
            boundaryComponent<r>(result.end) = -scale * derivative<E-r>(segments_-1, times_.back()).transpose();
        });
        return result;
    }
    /** @brief Compute all analytic energy gradients without a general adjoint.
     * @param[out] gradient Overwritten physical gradients; fixed-topology storage is reused. */
    void energyGradient(Gradients &gradient) const
    {
        energyWaypointGradient(gradient.inner_points);
        energyDurationGradient(gradient.durations);
        const auto boundary = energyBoundaryGradient();
        gradient.start = boundary.start; gradient.end = boundary.end;
    }
    /** @brief Allocate all energy gradients. @return Independently owned gradient buffers. */
    Gradients energyGradient() const
    { Gradients result; energyGradient(result); return result; }

    /** @brief Pull back coefficient and independent duration partials through interpolation.
     * @param coefficients Physical local-power partials, K*N rows in the polynomial's order.
     * @param times Independent duration partials holding those coefficients fixed.
     * @param[out] gradient Overwritten physical gradients; prepared storage is reused.
     * @note Requires a valid spline and matching shapes. The factorization is shared with construction. */
    void backward(const CoefficientMatrix &coefficients, const Eigen::VectorXd &times, Gradients &gradient)
    {
        if (!isValid() || coefficients.rows() != segments_*K || times.size() != segments_)
            throw std::invalid_argument("Adjoint requires a valid spline and matching partial shapes");
        gradient.resetTopology(segments_); gradient.setZero(); gradient.durations = times;
        adjoint_.setZero();

        // Transpose the local Hermite map; time derivatives hold endpoint physical derivatives fixed.
        for (int i = 0; i < segments_; ++i)
        {
            const auto &inv = inverse_powers_[i];
            double time_partial = 0.0;
            detail::staticFor<S,K>([&](auto degree) SPLINE_INLINE_VISITOR {
                constexpr int k = decltype(degree)::value;
                time_partial -= k * coefficients.row(i*K+k).dot(polynomial_.coefficients_.row(i*K+k));
            });
            detail::staticFor<1,K>([&](auto column) SPLINE_INLINE_VISITOR {
                constexpr int j = decltype(column)::value, r = j%S;
                RowVector partial = coefficientPullback<j>(i, coefficients, std::make_index_sequence<S>{});
                if constexpr (r == 0)
                {
                    addPoint(gradient,i,coefficients.row(i*K)-partial);
                    addPoint(gradient,i+1,partial);
                }
                else
                {
                    constexpr int side = j/S;
                    time_partial += r*partial.dot(derivatives_.row((i+side)*M+r-1));
                    if constexpr (side == 0)
                        partial += (1.0/detail::factorial(r))*coefficients.row(i*K+r);
                    adjoint_.row((i+side)*M+r-1) += partial;
                }
            });
            gradient.durations(i) += inv[1]*time_partial;
        }
        detail::staticFor<1,S>([&](auto order) SPLINE_INLINE_VISITOR {
            constexpr int r = decltype(order)::value;
            boundaryComponent<r>(gradient.start) = adjoint_.row(r-1).transpose();
            boundaryComponent<r>(gradient.end) = adjoint_.row(segments_*M+r-1).transpose();
        });
        adjoint_.template topRows<M>().setZero();
        adjoint_.template bottomRows<M>().setZero();
        factor_.solve(adjoint_, M);

        // The adjoint is itself a Hermite polynomial with zero waypoint values.
        // Integration by parts gives its endpoint forces; the polarized Hamiltonian
        // contracts the time derivative without forming dA/dT or a dense local Hessian.
        for (int i = 0; i < segments_; ++i)
        {
            Eigen::Matrix<double,S,DIM,kMatrixOptions> high;
            detail::staticFor<S,K>([&](auto degree) SPLINE_INLINE_VISITOR {
                constexpr int k = decltype(degree)::value;
                high.row(k-S) = derivativeCoefficient<k>(i, adjoint_, std::make_index_sequence<2*M>{});
            });
            constexpr double position_scale = (S%2 ? 1.0 : -1.0)*detail::factorial(E);
            const RowVector point = position_scale*high.row(S-1);
            addPoint(gradient,i,point); addPoint(gradient,i+1,-point);
            double time_partial = detail::factorial(S)*detail::factorial(S) *
                polynomial_.coefficients_.row(i*K+S).dot(high.row(0));
            detail::staticFor<1,S>([&](auto order) SPLINE_INLINE_VISITOR {
                constexpr int r = decltype(order)::value;
                constexpr double sign = (S-r+1)%2 ? -1.0 : 1.0;
                time_partial -= sign*detail::factorial(K-r) * (
                    adjoint_.row(i*M+r-1).dot(polynomial_.coefficients_.row(i*K+K-r)) +
                    detail::factorial(r)*polynomial_.coefficients_.row(i*K+r).dot(high.row(S-r)));
                constexpr double boundary_sign = (S-r)%2 ? -1.0 : 1.0;
                if (i == 0)
                    boundaryComponent<r>(gradient.start) -= boundary_sign*detail::factorial(E-r)*high.row(S-1-r).transpose();
                if (i+1 == segments_)
                {
                    RowVector end = RowVector::Zero();
                    detail::staticFor<0,r+1>([&](auto index) SPLINE_INLINE_VISITOR {
                        constexpr int k = E-decltype(index)::value;
                        end = end*times_[i] + (detail::factorial(k)/detail::factorial(k-E+r))*high.row(k-S);
                    });
                    boundaryComponent<r>(gradient.end) += boundary_sign*end.transpose();
                }
            });
            gradient.durations(i) += time_partial;
        }
    }
    /** @brief Allocate and return a coefficient-map pullback.
     * @param coefficients Physical coefficient partials. @param times Independent duration partials.
     * @return Independently owned physical gradients. */
    Gradients backward(const CoefficientMatrix &coefficients, const Eigen::VectorXd &times)
    { Gradients result; backward(coefficients,times,result); return result; }

private:
    /** @brief Require a published polynomial before numeric gradient queries. */
    void requireValid() const
    { if (!isValid()) throw std::logic_error("Energy gradients require a valid spline"); }
    /** @brief Select a physical boundary gradient at compile time.
     * @param[in,out] value Boundary gradient. @return Mutable component of order R. */
    template<int R> static Vector &boundaryComponent(BoundaryGradient &value)
    {
        if constexpr (R == 0) return value.p;
        else if constexpr (R == 1) return value.v;
        else if constexpr (R == 2) return value.a;
        else return value.j;
    }
    /** @brief Add a waypoint partial to its public output location.
     * @param[in,out] gradient Physical output gradients. @param index Knot index.
     * @param value Row to accumulate in the caller's coordinates. */
    void addPoint(Gradients &gradient, int index, const RowVector &value) const
    {
        if (index == 0) gradient.start.p += value.transpose();
        else if (index == segments_) gradient.end.p += value.transpose();
        else gradient.inner_points.row(index-1) += value;
    }
    /** @brief Evaluate a fixed Hermite derivative row using scalar arithmetic.
     * @param segment Valid piece. @param values Knot-major physical derivative rows.
     * @param indices Compile-time endpoint derivative indices. @return High power coefficient.
     * @note Position differences are added separately; adjoint waypoint values are identically zero. */
    template<int Degree, std::size_t... Index>
    EIGEN_ALWAYS_INLINE RowVector derivativeCoefficient(int segment, const CoefficientMatrix &values,
                                                       std::index_sequence<Index...> indices) const
    {
        (void)indices;
        const auto &inv = inverse_powers_[segment];
        RowVector result;
        for (int d=0; d<DIM; ++d)
            result(d) = (((Kernel::H[Degree][(Index/M)*S+Index%M+1]*inv[Degree-Index%M-1]) *
                          values((segment+Index/M)*M+Index%M,d)) + ...);
        return result;
    }
    /** @brief Evaluate one row of the transposed Hermite map without expression temporaries.
     * @param segment Valid piece. @param partials Physical coefficient partials.
     * @param indices High polynomial degrees. @return Endpoint partial from high coefficients. */
    template<int Column, std::size_t... Index>
    EIGEN_ALWAYS_INLINE RowVector coefficientPullback(int segment, const CoefficientMatrix &partials,
                                                     std::index_sequence<Index...> indices) const
    {
        (void)indices;
        const auto &inv = inverse_powers_[segment];
        RowVector result;
        for (int d=0; d<DIM; ++d)
            result(d) = (((Kernel::H[S+Index][Column]*inv[S+Index-Column%S]) *
                          partials(segment*K+S+Index,d)) + ...);
        return result;
    }
    /** @brief Evaluate one derivative with Horner's rule in local physical time.
     * @param segment Valid segment index. @param t Local seconds. @return Physical derivative row. */
    template<int R> RowVector derivative(int segment, double t) const
    {
        RowVector value = RowVector::Zero();
        detail::staticFor<0,K-R>([&](auto index) SPLINE_INLINE_VISITOR {
            constexpr int k = E-decltype(index)::value;
            value = value*t + (detail::factorial(k)/detail::factorial(k-R)) * polynomial_.coefficients_.row(segment*K+k);
        });
        return value;
    }
    /** @brief Evaluate one shifted-Legendre mode with Horner's rule.
     * @param segment Valid segment index. @return Mode R in physical derivative units. */
    template<int R> EIGEN_ALWAYS_INLINE RowVector energyMode(int segment) const
    {
        RowVector mode = Kernel::E[R][S-1]*polynomial_.coefficients_.row(segment*K+E);
        detail::staticFor<0,S-1-R>([&](auto index) SPLINE_INLINE_VISITOR {
            constexpr int j = S-2-decltype(index)::value;
            mode = mode*times_[segment] + Kernel::E[R][j]*polynomial_.coefficients_.row(segment*K+S+j);
        });
        double power = 1.0;
        for (int j = 0; j < R; ++j) power *= times_[segment];
        return power*mode;
    }
    /** @brief Gather orthogonal energy modes for the coefficient partials.
     * @param segment Valid segment index. @return Modes in physical derivative units. */
    Eigen::Matrix<double,S,DIM,kMatrixOptions> energyModes(int segment) const
    {
        Eigen::Matrix<double,S,DIM,kMatrixOptions> modes;
        detail::staticFor<0,S>([&](auto index) SPLINE_INLINE_VISITOR {
            constexpr int r = decltype(index)::value;
            modes.row(r) = energyMode<r>(segment);
        });
        return modes;
    }
    /** @brief Validate inputs, allocate once per topology, then assemble, solve and publish.
     * @note Failed numeric construction clears the polynomial and never exposes stale coefficients. */
    void rebuild()
    {
        if (!polynomial_.template beginSpline<S>(times_,points_,boundary_,start_,K))
        { segments_ = 0; return; }
        segments_ = static_cast<int>(times_.size());
        inverse_powers_.resize(segments_);
        derivatives_.resize((segments_+1)*M,DIM);
        adjoint_.resize((segments_+1)*M,DIM);
        factor_.resize(segments_-1);
        for (int i = 0; i < segments_; ++i)
        {
            auto &inv = inverse_powers_[i]; inv[0] = 1.0; inv[1] = 1.0/times_[i];
            for (int k = 2; k < K; ++k) inv[k] = inv[k-1]*inv[1];
        }
        derivatives_.row(0) = boundary_.start_velocity.transpose();
        derivatives_.row(segments_*M) = boundary_.end_velocity.transpose();
        if constexpr (S >= 3)
        {
            derivatives_.row(1) = boundary_.start_acceleration.transpose();
            derivatives_.row(segments_*M+1) = boundary_.end_acceleration.transpose();
        }
        if constexpr (S >= 4)
        {
            derivatives_.row(2) = boundary_.start_jerk.transpose();
            derivatives_.row(segments_*M+2) = boundary_.end_jerk.transpose();
        }
        if (!assemble()) { polynomial_.clear(); return; }
        factor_.solve(derivatives_,M);
        writeCoefficients();
        polynomial_.finishSpline();
    }
    /** @brief Assemble stationarity of the exact segment energy, in physical derivative coordinates.
     * @return True if every symmetric Schur block factors; false on numeric breakdown. */
    bool assemble()
    {
        for (int knot = 1; knot < segments_; ++knot)
        {
            const auto &left = inverse_powers_[knot-1], &right = inverse_powers_[knot];
            const RowVector dl = points_.row(knot)-points_.row(knot-1);
            const RowVector dr = points_.row(knot+1)-points_.row(knot);
            Block diagonal, previous;
            detail::staticFor<1,S>([&](auto row) SPLINE_INLINE_VISITOR {
                constexpr int r = decltype(row)::value;
                auto rhs = derivatives_.row(knot*M+r-1);
                rhs = -(Kernel::Q[S+r][S]*left[E-r])*dl - (Kernel::Q[r][S]*right[E-r])*dr;
                detail::staticFor<1,S>([&](auto column) SPLINE_INLINE_VISITOR {
                    constexpr int c = decltype(column)::value;
                    diagonal(r-1,c-1) = Kernel::Q[S+r][S+c]*left[E-r-c] + Kernel::Q[r][c]*right[E-r-c];
                    previous(r-1,c-1) = Kernel::Q[S+r][c]*left[E-r-c];
                    if (knot == 1) rhs -= previous(r-1,c-1)*derivatives_.row(c-1);
                    if (knot+1 == segments_)
                        rhs -= (Kernel::Q[r][S+c]*right[E-r-c])*derivatives_.row(segments_*M+c-1);
                });
            });
            if (!factor_.factor(knot-1,diagonal,previous)) return false;
        }
        return true;
    }
    /** @brief Apply one unit-interval Hermite row to scalar endpoint data.
     * @param data Endpoints relative to the start position, with derivatives scaled by T^r.
     * @param indices Nonconstant endpoint columns. @return Normalized power coefficient. */
    template<int Degree, std::size_t... Index>
    static EIGEN_ALWAYS_INLINE double hermiteValue(const std::array<double,K> &data,
                                                  std::index_sequence<Index...> indices)
    {
        (void)indices;
        return ((Kernel::H[Degree][Index+1]*data[Index+1]) + ...);
    }
    /** @brief Form coefficients in normalized time before converting to physical powers.
     * @note Each coordinate's endpoint values stay in registers across all polynomial degrees. */
    void writeCoefficients()
    {
        for (int i=0; i<segments_; ++i)
        {
            const auto &inv = inverse_powers_[i];
            for (int d=0; d<DIM; ++d)
            {
                std::array<double,K> data;
                data[0]=0.0; data[S]=points_(i+1,d)-points_(i,d);
                polynomial_.coefficients_(i*K,d)=points_(i,d);
                double power=1.0;
                detail::staticFor<1,S>([&](auto order) SPLINE_INLINE_VISITOR {
                    constexpr int r=decltype(order)::value;
                    power*=times_[i];
                    data[r]=power*derivatives_(i*M+r-1,d);
                    data[S+r]=power*derivatives_((i+1)*M+r-1,d);
                    polynomial_.coefficients_(i*K+r,d)=(1.0/detail::factorial(r))*derivatives_(i*M+r-1,d);
                });
                detail::staticFor<S,K>([&](auto degree) SPLINE_INLINE_VISITOR {
                    constexpr int k=decltype(degree)::value;
                    polynomial_.coefficients_(i*K+k,d)=inv[k]*hermiteValue<k>(data,std::make_index_sequence<K-1>{});
                });
            }
        }
    }

};
} // namespace SplineTrajectory
#undef SPLINE_INLINE_VISITOR
#endif
