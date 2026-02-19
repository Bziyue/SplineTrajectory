#include <Eigen/Dense>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "SplineTrajectory.hpp"
#include "gcopter/minco.hpp"

using std::cout;
using std::endl;

static inline double rel_err(double a, double b)
{
    const double denom = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) / denom;
}

static inline void print_triplet(const std::string &name, double spline_grad, double minco_grad, double num_grad)
{
    cout << std::left << std::setw(14) << name
         << " spline=" << std::setw(14) << spline_grad
         << " minco=" << std::setw(14) << minco_grad
         << " num=" << std::setw(14) << num_grad
         << " rel(s,num)=" << rel_err(spline_grad, num_grad)
         << " rel(m,num)=" << rel_err(minco_grad, num_grad)
         << " rel(s,m)=" << rel_err(spline_grad, minco_grad)
         << "\n";
}

template <typename SplineType>
double loss_from_spline(const SplineType &sp, double alpha_time)
{
    const auto &C = sp.getPPoly().getCoefficients();
    double Lt = 0.0;
    for (double h : sp.getTimeSegments())
    {
        Lt += h * h;
    }
    return C.squaredNorm() + alpha_time * Lt;
}

template <typename SplineType>
void make_partials(const SplineType &sp,
                   double alpha_time,
                   typename SplineType::MatrixType &dLdC,
                   Eigen::VectorXd &dLdT)
{
    dLdC = 2.0 * sp.getPPoly().getCoefficients();
    const auto &T = sp.getTimeSegments();
    dLdT.resize(static_cast<int>(T.size()));
    for (int i = 0; i < static_cast<int>(T.size()); ++i)
    {
        dLdT(i) = 2.0 * alpha_time * T[i];
    }
}

template <typename MatType>
Eigen::Matrix3Xd extract_inner_points_3d(const MatType &P)
{
    const int n_seg = static_cast<int>(P.rows()) - 1;
    Eigen::Matrix3Xd inner(3, n_seg - 1);
    for (int i = 0; i < n_seg - 1; ++i)
    {
        inner.col(i) = P.row(i + 1).transpose();
    }
    return inner;
}

template <typename MatType>
void build_minco_s2(const std::vector<double> &T,
                    const MatType &P,
                    const SplineTrajectory::BoundaryConditions<3> &bc,
                    minco::MINCO_S2NU &solver)
{
    const int n_seg = static_cast<int>(T.size());
    Eigen::Matrix<double, 3, 2> headPV, tailPV;
    headPV.col(0) = P.row(0).transpose();
    headPV.col(1) = bc.start_velocity;
    tailPV.col(0) = P.row(n_seg).transpose();
    tailPV.col(1) = bc.end_velocity;

    solver.setConditions(headPV, tailPV, n_seg);
    Eigen::Matrix3Xd inner = extract_inner_points_3d(P);
    const Eigen::Map<const Eigen::VectorXd> T_map(T.data(), n_seg);
    solver.setParameters(inner, T_map);
}

template <typename MatType>
void build_minco_s3(const std::vector<double> &T,
                    const MatType &P,
                    const SplineTrajectory::BoundaryConditions<3> &bc,
                    minco::MINCO_S3NU &solver)
{
    const int n_seg = static_cast<int>(T.size());
    Eigen::Matrix3d headPVA, tailPVA;
    headPVA.col(0) = P.row(0).transpose();
    headPVA.col(1) = bc.start_velocity;
    headPVA.col(2) = bc.start_acceleration;
    tailPVA.col(0) = P.row(n_seg).transpose();
    tailPVA.col(1) = bc.end_velocity;
    tailPVA.col(2) = bc.end_acceleration;

    solver.setConditions(headPVA, tailPVA, n_seg);
    Eigen::Matrix3Xd inner = extract_inner_points_3d(P);
    const Eigen::Map<const Eigen::VectorXd> T_map(T.data(), n_seg);
    solver.setParameters(inner, T_map);
}

template <typename MatType>
void build_minco_s4(const std::vector<double> &T,
                    const MatType &P,
                    const SplineTrajectory::BoundaryConditions<3> &bc,
                    minco::MINCO_S4NU &solver)
{
    const int n_seg = static_cast<int>(T.size());
    Eigen::Matrix<double, 3, 4> headPVAJ, tailPVAJ;
    headPVAJ.col(0) = P.row(0).transpose();
    headPVAJ.col(1) = bc.start_velocity;
    headPVAJ.col(2) = bc.start_acceleration;
    headPVAJ.col(3) = bc.start_jerk;
    tailPVAJ.col(0) = P.row(n_seg).transpose();
    tailPVAJ.col(1) = bc.end_velocity;
    tailPVAJ.col(2) = bc.end_acceleration;
    tailPVAJ.col(3) = bc.end_jerk;

    solver.setConditions(headPVAJ, tailPVAJ, n_seg);
    Eigen::Matrix3Xd inner = extract_inner_points_3d(P);
    const Eigen::Map<const Eigen::VectorXd> T_map(T.data(), n_seg);
    solver.setParameters(inner, T_map);
}

// ===================== Cubic =====================
template <int DIM>
void test_cubic()
{
    static_assert(DIM == 3, "MINCO boundary-gradient test currently assumes DIM=3.");

    using Spline = SplineTrajectory::CubicSplineND<DIM>;
    using Vec = typename Spline::VectorType;
    using Mat = typename Spline::MatrixType;

    cout << "\n=== CubicSplineND<" << DIM << "> boundary gradient check ===\n";

    const double alpha_time = 0.3;
    const double eps = 1e-6;
    const int n_seg = 4;
    const int n_pts = n_seg + 1;

    std::vector<double> T = {0.8, 1.1, 0.9, 1.3};

    Mat P(n_pts, DIM);
    for (int i = 0; i < n_pts; ++i)
    {
        Vec v;
        v.setZero();
        for (int d = 0; d < DIM; ++d)
        {
            v(d) = 0.2 * (i + 1) + 0.1 * (d + 1) * (i % 2 ? -1.0 : 1.0);
        }
        P.row(i) = v.transpose();
    }

    SplineTrajectory::BoundaryConditions<DIM> bc;
    bc.start_velocity.setConstant(0.15);
    bc.end_velocity.setConstant(-0.05);

    Spline sp(T, P, 0.0, bc);

    Mat dLdC;
    Eigen::VectorXd dLdT;
    make_partials(sp, alpha_time, dLdC, dLdT);
    auto g = sp.propagateGrad(dLdC, dLdT);

    minco::MINCO_S2NU minco_s2;
    build_minco_s2(T, P, bc, minco_s2);
    Eigen::MatrixX3d dLdC_minco = 2.0 * minco_s2.getCoeffs();
    const Eigen::Map<const Eigen::VectorXd> T_map(T.data(), n_seg);
    Eigen::VectorXd dLdT_minco = 2.0 * alpha_time * T_map;
    Eigen::Matrix3Xd gP_minco;
    Eigen::VectorXd gT_minco;
    Eigen::Matrix<double, 3, 2> gHead_minco, gTail_minco;
    minco_s2.propogateGrad(dLdC_minco, dLdT_minco, gP_minco, gT_minco, gHead_minco, gTail_minco);

    auto eval_loss = [&](const std::vector<double> &TT,
                         const Mat &PP,
                         const SplineTrajectory::BoundaryConditions<DIM> &bcc) {
        Spline tmp(TT, PP, 0.0, bcc);
        return loss_from_spline(tmp, alpha_time);
    };

    auto eval_loss_minco = [&](const std::vector<double> &TT,
                               const Mat &PP,
                               const SplineTrajectory::BoundaryConditions<DIM> &bcc) {
        minco::MINCO_S2NU solver;
        build_minco_s2(TT, PP, bcc, solver);
        const Eigen::Map<const Eigen::VectorXd> TT_map(TT.data(), static_cast<int>(TT.size()));
        return solver.getCoeffs().squaredNorm() + alpha_time * TT_map.squaredNorm();
    };

    cout << std::scientific << std::setprecision(6);
    cout << "[Compare] ||dL/dT_spline - dL/dT_minco|| = " << (g.times - gT_minco).norm() << "\n";
    cout << "[Compare] ||dL/dP_inner_spline - dL/dP_inner_minco|| = "
         << (g.inner_points - gP_minco.transpose()).norm() << "\n";

    cout << "\n[Check] dL/dP (inner points)\n";
    for (int k = 1; k <= n_pts - 2; ++k)
    {
        for (int d = 0; d < DIM; ++d)
        {
            auto Pp = P, Pm = P;
            Pp(k, d) += eps;
            Pm(k, d) -= eps;
            const double fp = eval_loss(T, Pp, bc);
            const double fm = eval_loss(T, Pm, bc);
            const double num = (fp - fm) / (2.0 * eps);
            const double ana = g.inner_points.row(k - 1)(d);
            cout << "P[" << k << "]." << d
                 << "  ana=" << ana << "  num=" << num
                 << "  rel=" << rel_err(ana, num) << "\n";
        }
    }

    cout << "\n[Check] dL/dT\n";
    for (int i = 0; i < n_seg; ++i)
    {
        auto Tp = T, Tm = T;
        Tp[i] += eps;
        Tm[i] -= eps;
        const double fp = eval_loss(Tp, P, bc);
        const double fm = eval_loss(Tm, P, bc);
        const double num = (fp - fm) / (2.0 * eps);
        const double ana = g.times(i);
        cout << "T[" << i << "]  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }

    cout << "\n[Check] boundary gradients (Spline vs MINCO vs Numeric)\n";
    for (int d = 0; d < DIM; ++d)
    {
        auto Pp = P, Pm = P;
        Pp(0, d) += eps;
        Pm(0, d) -= eps;
        const double num = (eval_loss_minco(T, Pp, bc) - eval_loss_minco(T, Pm, bc)) / (2.0 * eps);
        print_triplet("start_p." + std::to_string(d), g.start.p(d), gHead_minco(d, 0), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.start_velocity(d) += eps;
        bcm.start_velocity(d) -= eps;
        const double num = (eval_loss_minco(T, P, bcp) - eval_loss_minco(T, P, bcm)) / (2.0 * eps);
        print_triplet("start_v." + std::to_string(d), g.start.v(d), gHead_minco(d, 1), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto Pp = P, Pm = P;
        Pp(n_seg, d) += eps;
        Pm(n_seg, d) -= eps;
        const double num = (eval_loss_minco(T, Pp, bc) - eval_loss_minco(T, Pm, bc)) / (2.0 * eps);
        print_triplet("end_p." + std::to_string(d), g.end.p(d), gTail_minco(d, 0), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.end_velocity(d) += eps;
        bcm.end_velocity(d) -= eps;
        const double num = (eval_loss_minco(T, P, bcp) - eval_loss_minco(T, P, bcm)) / (2.0 * eps);
        print_triplet("end_v." + std::to_string(d), g.end.v(d), gTail_minco(d, 1), num);
    }
}

// ===================== Quintic =====================
template <int DIM>
void test_quintic()
{
    static_assert(DIM == 3, "MINCO boundary-gradient test currently assumes DIM=3.");

    using Spline = SplineTrajectory::QuinticSplineND<DIM>;
    using Vec = typename Spline::VectorType;
    using Mat = typename Spline::MatrixType;

    cout << "\n=== QuinticSplineND<" << DIM << "> boundary gradient check ===\n";

    const double alpha_time = 0.3;
    const double eps = 1e-6;
    const int n_seg = 4;
    const int n_pts = n_seg + 1;

    std::vector<double> T = {0.8, 1.1, 0.9, 1.3};

    Mat Pvec(n_pts, DIM);
    for (int i = 0; i < n_pts; ++i)
    {
        Vec v;
        v.setZero();
        for (int d = 0; d < DIM; ++d)
        {
            v(d) = 0.25 * (i + 1) + 0.07 * (d + 1) * (i % 2 ? -1.0 : 1.0);
        }
        Pvec.row(i) = v.transpose();
    }

    SplineTrajectory::BoundaryConditions<DIM> bc;
    bc.start_velocity.setConstant(0.12);
    bc.end_velocity.setConstant(-0.08);
    bc.start_acceleration.setConstant(0.05);
    bc.end_acceleration.setConstant(-0.03);

    Spline sp(T, Pvec, 0.0, bc);

    Mat dLdC;
    Eigen::VectorXd dLdT;
    make_partials(sp, alpha_time, dLdC, dLdT);
    auto g = sp.propagateGrad(dLdC, dLdT);

    minco::MINCO_S3NU minco_s3;
    build_minco_s3(T, Pvec, bc, minco_s3);
    Eigen::MatrixX3d dLdC_minco = 2.0 * minco_s3.getCoeffs();
    const Eigen::Map<const Eigen::VectorXd> T_map(T.data(), n_seg);
    Eigen::VectorXd dLdT_minco = 2.0 * alpha_time * T_map;
    Eigen::Matrix3Xd gP_minco;
    Eigen::VectorXd gT_minco;
    Eigen::Matrix3d gHead_minco, gTail_minco;
    minco_s3.propogateGrad(dLdC_minco, dLdT_minco, gP_minco, gT_minco, gHead_minco, gTail_minco);

    auto eval_loss = [&](const std::vector<double> &TT,
                         const Mat &PP,
                         const SplineTrajectory::BoundaryConditions<DIM> &bcc) {
        Spline tmp(TT, PP, 0.0, bcc);
        return loss_from_spline(tmp, alpha_time);
    };

    auto eval_loss_minco = [&](const std::vector<double> &TT,
                               const Mat &PP,
                               const SplineTrajectory::BoundaryConditions<DIM> &bcc) {
        minco::MINCO_S3NU solver;
        build_minco_s3(TT, PP, bcc, solver);
        const Eigen::Map<const Eigen::VectorXd> TT_map(TT.data(), static_cast<int>(TT.size()));
        return solver.getCoeffs().squaredNorm() + alpha_time * TT_map.squaredNorm();
    };

    cout << std::scientific << std::setprecision(6);
    cout << "[Compare] ||dL/dT_spline - dL/dT_minco|| = " << (g.times - gT_minco).norm() << "\n";
    cout << "[Compare] ||dL/dP_inner_spline - dL/dP_inner_minco|| = "
         << (g.inner_points - gP_minco.transpose()).norm() << "\n";

    cout << "\n[Check] dL/dP (inner points)\n";
    for (int k = 1; k <= n_pts - 2; ++k)
    {
        for (int d = 0; d < DIM; ++d)
        {
            auto Pp = Pvec, Pm = Pvec;
            Pp(k, d) += eps;
            Pm(k, d) -= eps;
            const double fp = eval_loss(T, Pp, bc);
            const double fm = eval_loss(T, Pm, bc);
            const double num = (fp - fm) / (2.0 * eps);
            const double ana = g.inner_points.row(k - 1)(d);
            cout << "P[" << k << "]." << d
                 << "  ana=" << ana << "  num=" << num
                 << "  rel=" << rel_err(ana, num) << "\n";
        }
    }

    cout << "\n[Check] dL/dT\n";
    for (int i = 0; i < n_seg; ++i)
    {
        auto Tp = T, Tm = T;
        Tp[i] += eps;
        Tm[i] -= eps;
        const double fp = eval_loss(Tp, Pvec, bc);
        const double fm = eval_loss(Tm, Pvec, bc);
        const double num = (fp - fm) / (2.0 * eps);
        const double ana = g.times(i);
        cout << "T[" << i << "]  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }

    cout << "\n[Check] boundary gradients (Spline vs MINCO vs Numeric)\n";
    for (int d = 0; d < DIM; ++d)
    {
        auto Pp = Pvec, Pm = Pvec;
        Pp(0, d) += eps;
        Pm(0, d) -= eps;
        const double num = (eval_loss_minco(T, Pp, bc) - eval_loss_minco(T, Pm, bc)) / (2.0 * eps);
        print_triplet("start_p." + std::to_string(d), g.start.p(d), gHead_minco(d, 0), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.start_velocity(d) += eps;
        bcm.start_velocity(d) -= eps;
        const double num = (eval_loss_minco(T, Pvec, bcp) - eval_loss_minco(T, Pvec, bcm)) / (2.0 * eps);
        print_triplet("start_v." + std::to_string(d), g.start.v(d), gHead_minco(d, 1), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.start_acceleration(d) += eps;
        bcm.start_acceleration(d) -= eps;
        const double num = (eval_loss_minco(T, Pvec, bcp) - eval_loss_minco(T, Pvec, bcm)) / (2.0 * eps);
        print_triplet("start_a." + std::to_string(d), g.start.a(d), gHead_minco(d, 2), num);
    }

    for (int d = 0; d < DIM; ++d)
    {
        auto Pp = Pvec, Pm = Pvec;
        Pp(n_seg, d) += eps;
        Pm(n_seg, d) -= eps;
        const double num = (eval_loss_minco(T, Pp, bc) - eval_loss_minco(T, Pm, bc)) / (2.0 * eps);
        print_triplet("end_p." + std::to_string(d), g.end.p(d), gTail_minco(d, 0), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.end_velocity(d) += eps;
        bcm.end_velocity(d) -= eps;
        const double num = (eval_loss_minco(T, Pvec, bcp) - eval_loss_minco(T, Pvec, bcm)) / (2.0 * eps);
        print_triplet("end_v." + std::to_string(d), g.end.v(d), gTail_minco(d, 1), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.end_acceleration(d) += eps;
        bcm.end_acceleration(d) -= eps;
        const double num = (eval_loss_minco(T, Pvec, bcp) - eval_loss_minco(T, Pvec, bcm)) / (2.0 * eps);
        print_triplet("end_a." + std::to_string(d), g.end.a(d), gTail_minco(d, 2), num);
    }
}

// ===================== Septic =====================
template <int DIM>
void test_septic()
{
    static_assert(DIM == 3, "MINCO boundary-gradient test currently assumes DIM=3.");

    using Spline = SplineTrajectory::SepticSplineND<DIM>;
    using Vec = typename Spline::VectorType;
    using Mat = typename Spline::MatrixType;

    cout << "\n=== SepticSplineND<" << DIM << "> boundary gradient check ===\n";

    const double alpha_time = 0.3;
    const double eps = 1e-6;
    const int n_seg = 4;
    const int n_pts = n_seg + 1;

    std::vector<double> T = {0.8, 1.1, 0.9, 1.3};

    Mat Pvec(n_pts, DIM);
    for (int i = 0; i < n_pts; ++i)
    {
        Vec v;
        v.setZero();
        for (int d = 0; d < DIM; ++d)
        {
            v(d) = 0.18 * (i + 1) + 0.09 * (d + 1) * (i % 2 ? -1.0 : 1.0);
        }
        Pvec.row(i) = v.transpose();
    }

    SplineTrajectory::BoundaryConditions<DIM> bc;
    bc.start_velocity.setConstant(0.11);
    bc.end_velocity.setConstant(-0.07);
    bc.start_acceleration.setConstant(0.04);
    bc.end_acceleration.setConstant(-0.02);
    bc.start_jerk.setConstant(0.015);
    bc.end_jerk.setConstant(-0.01);

    Spline sp(T, Pvec, 0.0, bc);

    typename Spline::MatrixType dLdC;
    Eigen::VectorXd dLdT;
    make_partials(sp, alpha_time, dLdC, dLdT);
    auto g = sp.propagateGrad(dLdC, dLdT);

    minco::MINCO_S4NU minco_s4;
    build_minco_s4(T, Pvec, bc, minco_s4);
    Eigen::MatrixX3d dLdC_minco = 2.0 * minco_s4.getCoeffs();
    const Eigen::Map<const Eigen::VectorXd> T_map(T.data(), n_seg);
    Eigen::VectorXd dLdT_minco = 2.0 * alpha_time * T_map;
    Eigen::Matrix3Xd gP_minco;
    Eigen::VectorXd gT_minco;
    Eigen::Matrix<double, 3, 4> gHead_minco, gTail_minco;
    minco_s4.propogateGrad(dLdC_minco, dLdT_minco, gP_minco, gT_minco, gHead_minco, gTail_minco);

    auto eval_loss = [&](const std::vector<double> &TT,
                         const Mat &PP,
                         const SplineTrajectory::BoundaryConditions<DIM> &bcc) {
        Spline tmp(TT, PP, 0.0, bcc);
        return loss_from_spline(tmp, alpha_time);
    };

    auto eval_loss_minco = [&](const std::vector<double> &TT,
                               const Mat &PP,
                               const SplineTrajectory::BoundaryConditions<DIM> &bcc) {
        minco::MINCO_S4NU solver;
        build_minco_s4(TT, PP, bcc, solver);
        const Eigen::Map<const Eigen::VectorXd> TT_map(TT.data(), static_cast<int>(TT.size()));
        return solver.getCoeffs().squaredNorm() + alpha_time * TT_map.squaredNorm();
    };

    cout << std::scientific << std::setprecision(6);
    cout << "[Compare] ||dL/dT_spline - dL/dT_minco|| = " << (g.times - gT_minco).norm() << "\n";
    cout << "[Compare] ||dL/dP_inner_spline - dL/dP_inner_minco|| = "
         << (g.inner_points - gP_minco.transpose()).norm() << "\n";

    cout << "\n[Check] dL/dP (inner points)\n";
    for (int k = 1; k <= n_pts - 2; ++k)
    {
        for (int d = 0; d < DIM; ++d)
        {
            auto Pp = Pvec, Pm = Pvec;
            Pp(k, d) += eps;
            Pm(k, d) -= eps;
            const double num = (eval_loss(T, Pp, bc) - eval_loss(T, Pm, bc)) / (2.0 * eps);
            const double ana = g.inner_points.row(k - 1)(d);
            cout << "P[" << k << "]." << d
                 << "  ana=" << ana << "  num=" << num
                 << "  rel=" << rel_err(ana, num) << "\n";
        }
    }

    cout << "\n[Check] dL/dT\n";
    for (int i = 0; i < n_seg; ++i)
    {
        auto Tp = T, Tm = T;
        Tp[i] += eps;
        Tm[i] -= eps;
        const double num = (eval_loss(Tp, Pvec, bc) - eval_loss(Tm, Pvec, bc)) / (2.0 * eps);
        const double ana = g.times(i);
        cout << "T[" << i << "]  ana=" << ana << "  num=" << num
             << "  rel=" << rel_err(ana, num) << "\n";
    }

    cout << "\n[Check] boundary gradients (Spline vs MINCO vs Numeric)\n";
    for (int d = 0; d < DIM; ++d)
    {
        auto Pp = Pvec, Pm = Pvec;
        Pp(0, d) += eps;
        Pm(0, d) -= eps;
        const double num = (eval_loss_minco(T, Pp, bc) - eval_loss_minco(T, Pm, bc)) / (2.0 * eps);
        print_triplet("start_p." + std::to_string(d), g.start.p(d), gHead_minco(d, 0), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.start_velocity(d) += eps;
        bcm.start_velocity(d) -= eps;
        const double num = (eval_loss_minco(T, Pvec, bcp) - eval_loss_minco(T, Pvec, bcm)) / (2.0 * eps);
        print_triplet("start_v." + std::to_string(d), g.start.v(d), gHead_minco(d, 1), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.start_acceleration(d) += eps;
        bcm.start_acceleration(d) -= eps;
        const double num = (eval_loss_minco(T, Pvec, bcp) - eval_loss_minco(T, Pvec, bcm)) / (2.0 * eps);
        print_triplet("start_a." + std::to_string(d), g.start.a(d), gHead_minco(d, 2), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.start_jerk(d) += eps;
        bcm.start_jerk(d) -= eps;
        const double num = (eval_loss_minco(T, Pvec, bcp) - eval_loss_minco(T, Pvec, bcm)) / (2.0 * eps);
        print_triplet("start_j." + std::to_string(d), g.start.j(d), gHead_minco(d, 3), num);
    }

    for (int d = 0; d < DIM; ++d)
    {
        auto Pp = Pvec, Pm = Pvec;
        Pp(n_seg, d) += eps;
        Pm(n_seg, d) -= eps;
        const double num = (eval_loss_minco(T, Pp, bc) - eval_loss_minco(T, Pm, bc)) / (2.0 * eps);
        print_triplet("end_p." + std::to_string(d), g.end.p(d), gTail_minco(d, 0), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.end_velocity(d) += eps;
        bcm.end_velocity(d) -= eps;
        const double num = (eval_loss_minco(T, Pvec, bcp) - eval_loss_minco(T, Pvec, bcm)) / (2.0 * eps);
        print_triplet("end_v." + std::to_string(d), g.end.v(d), gTail_minco(d, 1), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.end_acceleration(d) += eps;
        bcm.end_acceleration(d) -= eps;
        const double num = (eval_loss_minco(T, Pvec, bcp) - eval_loss_minco(T, Pvec, bcm)) / (2.0 * eps);
        print_triplet("end_a." + std::to_string(d), g.end.a(d), gTail_minco(d, 2), num);
    }
    for (int d = 0; d < DIM; ++d)
    {
        auto bcp = bc, bcm = bc;
        bcp.end_jerk(d) += eps;
        bcm.end_jerk(d) -= eps;
        const double num = (eval_loss_minco(T, Pvec, bcp) - eval_loss_minco(T, Pvec, bcm)) / (2.0 * eps);
        print_triplet("end_j." + std::to_string(d), g.end.j(d), gTail_minco(d, 3), num);
    }
}

int main()
{
    test_cubic<3>();
    test_quintic<3>();
    test_septic<3>();
    return 0;
}
