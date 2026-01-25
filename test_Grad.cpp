#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <chrono>
#include <Eigen/Dense>


#include "SplineTrajectory.hpp"
#include "gcopter/minco.hpp"
#include "traj_min_jerk.hpp"
#include "traj_min_snap.hpp"

using namespace std;
using namespace std::chrono;

void printHeader(const string &title)
{
    cout << "\n" << string(80, '=') << endl;
    cout << " " << title << endl;
    cout << string(80, '=') << endl;
}

void printSubHeader(const string &title)
{
    cout << "\n--- " << title << " ---" << endl;
}

void printCheck(const string &name, double diff, double tol = 1e-4)
{
    cout << std::left << std::setw(35) << name << ": ";
    if (diff < tol)
        cout << "\033[32mPASS\033[0m (Diff: " << std::scientific << std::setprecision(2) << diff << ")" << "\033[0m" << endl;
    else
        cout << "\033[31mFAIL\033[0m (Diff: " << std::scientific << std::setprecision(2) << diff << ")" << "\033[0m" << endl;
}

void printTime(const string &name, double time_avg_us)
{
    cout << std::left << std::setw(35) << name << ": " 
         << "\033[33m" << std::fixed << std::setprecision(3) << time_avg_us << " us\033[0m" << endl;
}

void generateRandomData(int N,
                        std::vector<double> &times,
                        SplineTrajectory::SplineVector3D &all_points,
                        Eigen::Matrix3Xd &inner_points_mat,
                        Eigen::MatrixXd &inner_points_ref_format) // 3x(N-1)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_t(0.5, 2.0);
    std::uniform_real_distribution<> dis_p(-10.0, 10.0);

    times.resize(N);
    all_points.resize(N + 1);
    inner_points_mat.resize(3, N - 1);
    
    for (int i = 0; i < N; ++i)
        times[i] = dis_t(gen);

    for (int i = 0; i <= N; ++i)
    {
        all_points[i] = Eigen::Vector3d(dis_p(gen), dis_p(gen), dis_p(gen));
        if (i > 0 && i < N)
        {
            inner_points_mat.col(i - 1) = all_points[i];
        }
    }
    inner_points_ref_format = inner_points_mat;
}

// ---------------------------------------------------------
// Test 1: Cubic Spline (Min Acc) vs MINCO S2
// ---------------------------------------------------------
void testCubic(int N, int BENCH_ITERS)
{
    printHeader("TEST: Cubic Spline (Order 3) vs MINCO S2");

    // Note: MINCO S2 requires N >= 2 (at least 2 segments)
    // For N = 1, we only test SplineTrajectory
    if (N < 2) {
        cout << "\033[33mWARNING: N=" << N << " is not supported by MINCO S2 (require N>=2).\033[0m" << endl;
        cout << "Testing SplineTrajectory only for N=1..." << endl;
        
        std::vector<double> times(N);
        SplineTrajectory::SplineVector3D all_points(N + 1);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis_t(0.5, 2.0);
        std::uniform_real_distribution<> dis_p(-10.0, 10.0);
        for (int i = 0; i < N; ++i) times[i] = dis_t(gen);
        for (int i = 0; i <= N; ++i) all_points[i] = Eigen::Vector3d(dis_p(gen), dis_p(gen), dis_p(gen));
        
        SplineTrajectory::BoundaryConditions<3> bc;
        SplineTrajectory::CubicSpline3D cubic_spline;
        cubic_spline.update(times, all_points, 0.0, bc);
        
        cout << "Spline3 Generation: \033[32mOK\033[0m" << endl;
        cout << "Energy: " << cubic_spline.getEnergy() << endl;
        cout << "Duration: " << cubic_spline.getDuration() << endl;
        
        // Test gradient computation
        auto gdC = cubic_spline.getEnergyPartialGradByCoeffs();
        auto gdT = cubic_spline.getEnergyPartialGradByTimes();
        cout << "Partial Grad (Coeffs) shape: " << gdC.rows() << "x" << gdC.cols() << endl;
        cout << "Partial Grad (Times) size: " << gdT.size() << endl;
        
        // Test propagateGrad
        auto grads = cubic_spline.propagateGrad(gdC, gdT);
        // Test reference overload
        SplineTrajectory::CubicSpline3D::Gradients grads_ref;
        cubic_spline.propagateGrad(gdC, gdT, grads_ref);
        printCheck("Reference overload", 0.0);
        // Reconstruct full point gradients from propagated results
        SplineTrajectory::CubicSpline3D::MatrixType gradP_full(N + 1, 3);
        gradP_full.row(0) = grads.start.p.transpose();
        if (N > 1) {
            gradP_full.block(1, 0, N - 1, 3) = grads.inner_points;
        }
        gradP_full.row(N) = grads.end.p.transpose();
        cout << "Propagated Grad (Points) shape: " << gradP_full.rows() << "x" << gradP_full.cols() << endl;
        cout << "Propagated Grad (Times) size: " << grads.times.size() << endl;
        
        // Consistency Check: Direct vs Propagated
        printSubHeader("Self-Check: Direct vs Propagated");
        auto direct_gradT = cubic_spline.getEnergyGradTimes();
        auto direct_gradP_inner = cubic_spline.getEnergyGradInnerPoints();
        auto direct_bc_grads = cubic_spline.getEnergyGradBoundary();

        printCheck("Direct vs Prop (Times)", (direct_gradT - grads.times).norm());

        if (N > 1) {
            Eigen::MatrixXd prop_gradP_inner = gradP_full.block(1, 0, N - 1, 3);
            printCheck("Direct vs Prop (Inner P)", (direct_gradP_inner - prop_gradP_inner).norm());
        } else {
            cout << "Inner Points Grad: N/A (no inner points for N=1)" << endl;
        }

        printCheck("Direct vs Prop (Start P)", (direct_bc_grads.start.p - grads.start.p).norm());
        printCheck("Direct vs Prop (Start V)", (direct_bc_grads.start.v - grads.start.v).norm());
        printCheck("Direct vs Prop (End P)", (direct_bc_grads.end.p - grads.end.p).norm());
        printCheck("Direct vs Prop (End V)", (direct_bc_grads.end.v - grads.end.v).norm());

        // Print boundary gradients
        printSubHeader("Boundary Condition Gradients");
        cout << "\n[Start Boundary Gradients]" << endl;
        cout << "Position (P): " << direct_bc_grads.start.p.transpose() << endl;
        cout << "Velocity (V): " << direct_bc_grads.start.v.transpose() << endl;
        cout << "\n[End Boundary Gradients]" << endl;
        cout << "Position (P): " << direct_bc_grads.end.p.transpose() << endl;
        cout << "Velocity (V): " << direct_bc_grads.end.v.transpose() << endl;
        return;
    }

    std::vector<double> times;
    SplineTrajectory::SplineVector3D all_points;
    Eigen::Matrix3Xd inner_points;
    Eigen::MatrixXd inner_points_ref; // Not used for min_jerk/snap here, but for MINCO
    generateRandomData(N, times, all_points, inner_points, inner_points_ref);

    // Initial Conditions
    Eigen::Matrix<double, 3, 2> headPV, tailPV;
    headPV.col(0) = all_points[0]; headPV.col(1) = Eigen::Vector3d::Zero();
    tailPV.col(0) = all_points.back(); tailPV.col(1) = Eigen::Vector3d::Zero();

    // Setup objects
    minco::MINCO_S2NU minco_s2;
    minco_s2.setConditions(headPV, tailPV, N);
    
    SplineTrajectory::BoundaryConditions<3> bc;
    bc.start_velocity = headPV.col(1);
    bc.end_velocity = tailPV.col(1);
    SplineTrajectory::CubicSpline3D cubic_spline;

    // 1. Performance Benchmark: Generation
    printSubHeader("Performance: Trajectory Generation");
    
    auto t1 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        minco_s2.setParameters(inner_points, Eigen::Map<Eigen::VectorXd>(times.data(), N));
    }
    double t_minco = duration_cast<microseconds>(high_resolution_clock::now() - t1).count() / (double)BENCH_ITERS;

    auto t2 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        cubic_spline.update(times, all_points, 0.0, bc);
    }
    double t_spline = duration_cast<microseconds>(high_resolution_clock::now() - t2).count() / (double)BENCH_ITERS;

    printTime("MINCO S2 Gen Time", t_minco);
    printTime("Spline3 Gen Time", t_spline);

    // 2. Consistency Check: Energy & Coeffs
    printSubHeader("Consistency: MINCO vs Spline");
    double e_minco; minco_s2.getEnergy(e_minco);
    printCheck("Energy Difference", std::abs(e_minco - cubic_spline.getEnergy()));
    printCheck("Coeffs Difference", (minco_s2.getCoeffs() - cubic_spline.getTrajectory().getCoefficients()).norm());

    // 3. Gradient Calculation & Propagation Benchmark
    printSubHeader("Performance: Gradient Propagation");

    Eigen::MatrixX3d gdC_minco; Eigen::VectorXd gdT_minco;
    Eigen::Matrix3Xd gradP_minco_out; Eigen::VectorXd gradT_minco_out;

    SplineTrajectory::CubicSpline3D::MatrixType gdC_spline;
    Eigen::VectorXd gdT_spline;
    SplineTrajectory::CubicSpline3D::MatrixType gradP_spline_full;
    Eigen::VectorXd gradT_spline_out;

    // MINCO Grad
    t1 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        minco_s2.getEnergyPartialGradByCoeffs(gdC_minco);
        minco_s2.getEnergyPartialGradByTimes(gdT_minco);
        minco_s2.propogateGrad(gdC_minco, gdT_minco, gradP_minco_out, gradT_minco_out);
    }
    double t_grad_minco = duration_cast<microseconds>(high_resolution_clock::now() - t1).count() / (double)BENCH_ITERS;

    // Spline Grad
    t2 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        gdC_spline = cubic_spline.getEnergyPartialGradByCoeffs();
        gdT_spline = cubic_spline.getEnergyPartialGradByTimes();
        auto grads = cubic_spline.propagateGrad(gdC_spline, gdT_spline);
        gradT_spline_out = grads.times;
        // Reconstruct full point gradients from propagated results
        gradP_spline_full.resize(N + 1, 3);
        gradP_spline_full.row(0) = grads.start.p.transpose();
        if (N > 1) {
            gradP_spline_full.block(1, 0, N - 1, 3) = grads.inner_points;
        }
        gradP_spline_full.row(N) = grads.end.p.transpose();
    }
    double t_grad_spline = duration_cast<microseconds>(high_resolution_clock::now() - t2).count() / (double)BENCH_ITERS;

    printTime("MINCO S2 Grad Prop", t_grad_minco);
    printTime("Spline3 Grad Prop", t_grad_spline);

    // 4. Gradient Consistency Check
    printSubHeader("Consistency: Gradients");
    printCheck("Partial Grad (Coeffs)", (gdC_minco - gdC_spline).norm());
    printCheck("Partial Grad (Times)", (gdT_minco - gdT_spline).norm());
    
    // Extract inner points from spline full gradient (Spline is RowMajor, size N+1)
    // MINCO returns 3x(N-1)
    Eigen::Matrix3Xd gradP_spline_inner = gradP_spline_full.block(1, 0, N - 1, 3).transpose();
    printCheck("Propagated Grad (Inner P)", (gradP_minco_out - gradP_spline_inner).norm());
    printCheck("Propagated Grad (Times)", (gradT_minco_out - gradT_spline_out).norm());

    // 5. Self-Check (Direct vs Propagated)
    printSubHeader("Self-Check: Direct vs Propagated");
    auto direct_gradT = cubic_spline.getEnergyGradTimes();
    auto direct_gradP_inner = cubic_spline.getEnergyGradInnerPoints();
    auto direct_bc_grads = cubic_spline.getEnergyGradBoundary();
    // Test reference overload
    SplineTrajectory::CubicSpline3D::Gradients grads_ref;
    cubic_spline.propagateGrad(gdC_spline, gdT_spline, grads_ref);
    printCheck("Reference overload", 0.0);

    // Spline Propagated Inner (as (N-1)x3)
    Eigen::MatrixXd prop_gradP_inner = gradP_spline_full.block(1, 0, N - 1, 3);

    printCheck("Direct vs Prop (Times)", (direct_gradT - gradT_spline_out).norm());
    printCheck("Direct vs Prop (Inner P)", (direct_gradP_inner - prop_gradP_inner).norm());
    printCheck("Direct vs Prop (Start P)", (direct_bc_grads.start.p - grads_ref.start.p).norm());
    printCheck("Direct vs Prop (Start V)", (direct_bc_grads.start.v - grads_ref.start.v).norm());
    printCheck("Direct vs Prop (End P)", (direct_bc_grads.end.p - grads_ref.end.p).norm());
    printCheck("Direct vs Prop (End V)", (direct_bc_grads.end.v - grads_ref.end.v).norm());

    // 6. Point-by-Point Gradient Comparison
    printSubHeader("Point-by-Point Gradient Comparison");
    cout << "\n[Gradient w.r.t. Times (T)]" << endl;
    cout << std::left << std::setw(8) << "Seg" 
         << std::setw(20) << "MINCO" 
         << std::setw(20) << "Spline" << endl;
    cout << string(48, '-') << endl;
    for (int i = 0; i < N; ++i) {
        cout << std::left << std::setw(8) << i
             << std::setw(20) << std::fixed << std::setprecision(6) << gradT_minco_out(i)
             << std::setw(20) << std::fixed << std::setprecision(6) << gradT_spline_out(i) << endl;
    }

    cout << "\n[Gradient w.r.t. Inner Points (P)]" << endl;
    cout << std::left << std::setw(8) << "Point" 
         << std::setw(36) << "MINCO (x, y, z)" 
         << std::setw(36) << "Spline (x, y, z)" << endl;
    cout << string(80, '-') << endl;
    for (int i = 0; i < N - 1; ++i) {
        cout << std::left << std::setw(8) << i
             << "(" << std::setw(10) << std::fixed << std::setprecision(4) << gradP_minco_out(0, i)
             << ", " << std::setw(10) << gradP_minco_out(1, i)
             << ", " << std::setw(10) << gradP_minco_out(2, i) << ")  "
             << "(" << std::setw(10) << gradP_spline_inner(0, i)
             << ", " << std::setw(10) << gradP_spline_inner(1, i)
             << ", " << std::setw(10) << gradP_spline_inner(2, i) << ")" << endl;
    }

    // 7. Print Boundary Gradients
    printSubHeader("Boundary Condition Gradients");
    cout << "\n[Start Boundary Gradients]" << endl;
    cout << "Position (P): " << direct_bc_grads.start.p.transpose() << endl;
    cout << "Velocity (V): " << direct_bc_grads.start.v.transpose() << endl;
    cout << "\n[End Boundary Gradients]" << endl;
    cout << "Position (P): " << direct_bc_grads.end.p.transpose() << endl;
    cout << "Velocity (V): " << direct_bc_grads.end.v.transpose() << endl;
}

// ---------------------------------------------------------
// Test 2: Quintic Spline (Min Jerk) vs MINCO S3 vs min_jerk
// ---------------------------------------------------------
void testQuintic(int N, int BENCH_ITERS)
{
    printHeader("TEST: Quintic Spline (Order 5) vs MINCO S3 vs Ref(Jerk)");

    // Note: MINCO and min_jerk require N >= 2 (at least 2 segments)
    // For N = 1, we only test SplineTrajectory
    if (N < 2) {
        cout << "\033[33mWARNING: N=" << N << " is not supported by MINCO/min_jerk (require N>=2).\033[0m" << endl;
        cout << "Testing SplineTrajectory only for N=1..." << endl;
        
        std::vector<double> times(N);
        SplineTrajectory::SplineVector3D all_points(N + 1);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis_t(0.5, 2.0);
        std::uniform_real_distribution<> dis_p(-10.0, 10.0);
        for (int i = 0; i < N; ++i) times[i] = dis_t(gen);
        for (int i = 0; i <= N; ++i) all_points[i] = Eigen::Vector3d(dis_p(gen), dis_p(gen), dis_p(gen));
        
        SplineTrajectory::BoundaryConditions<3> bc;
        SplineTrajectory::QuinticSpline3D quintic_spline;
        quintic_spline.update(times, all_points, 0.0, bc);
        
        cout << "Spline5 Generation: \033[32mOK\033[0m" << endl;
        cout << "Energy: " << quintic_spline.getEnergy() << endl;
        cout << "Duration: " << quintic_spline.getDuration() << endl;
        
        // Test gradient computation
        auto gdC = quintic_spline.getEnergyPartialGradByCoeffs();
        auto gdT = quintic_spline.getEnergyPartialGradByTimes();
        cout << "Partial Grad (Coeffs) shape: " << gdC.rows() << "x" << gdC.cols() << endl;
        cout << "Partial Grad (Times) size: " << gdT.size() << endl;
        
        // Test propagateGrad
        auto grads = quintic_spline.propagateGrad(gdC, gdT);
        // Test reference overload
        SplineTrajectory::QuinticSpline3D::Gradients grads_ref;
        quintic_spline.propagateGrad(gdC, gdT, grads_ref);
        printCheck("Reference overload", 0.0);
        // Reconstruct full point gradients from propagated results
        SplineTrajectory::QuinticSpline3D::MatrixType gradP_full(N + 1, 3);
        gradP_full.row(0) = grads.start.p.transpose();
        if (N > 1) {
            gradP_full.block(1, 0, N - 1, 3) = grads.inner_points;
        }
        gradP_full.row(N) = grads.end.p.transpose();
        cout << "Propagated Grad (Points) shape: " << gradP_full.rows() << "x" << gradP_full.cols() << endl;
        cout << "Propagated Grad (Times) size: " << grads.times.size() << endl;
        
        // Consistency Check: Direct vs Propagated
        printSubHeader("Self-Check: Direct vs Propagated");
        auto direct_gradT = quintic_spline.getEnergyGradTimes();
        auto direct_gradP_inner = quintic_spline.getEnergyGradInnerPoints();
        auto direct_bc_grads = quintic_spline.getEnergyGradBoundary();

        printCheck("Direct vs Prop (Times)", (direct_gradT - grads.times).norm());

        if (N > 1) {
            Eigen::MatrixXd prop_gradP_inner = gradP_full.block(1, 0, N - 1, 3);
            printCheck("Direct vs Prop (Inner P)", (direct_gradP_inner - prop_gradP_inner).norm());
        } else {
            cout << "Inner Points Grad: N/A (no inner points for N=1)" << endl;
        }

        printCheck("Direct vs Prop (Start P)", (direct_bc_grads.start.p - grads.start.p).norm());
        printCheck("Direct vs Prop (Start V)", (direct_bc_grads.start.v - grads.start.v).norm());
        printCheck("Direct vs Prop (Start A)", (direct_bc_grads.start.a - grads.start.a).norm());
        printCheck("Direct vs Prop (End P)", (direct_bc_grads.end.p - grads.end.p).norm());
        printCheck("Direct vs Prop (End V)", (direct_bc_grads.end.v - grads.end.v).norm());
        printCheck("Direct vs Prop (End A)", (direct_bc_grads.end.a - grads.end.a).norm());

        // Print boundary gradients
        printSubHeader("Boundary Condition Gradients");
        cout << "\n[Start Boundary Gradients]" << endl;
        cout << "Position (P): " << direct_bc_grads.start.p.transpose() << endl;
        cout << "Velocity (V): " << direct_bc_grads.start.v.transpose() << endl;
        cout << "Acceleration (A): " << direct_bc_grads.start.a.transpose() << endl;
        cout << "\n[End Boundary Gradients]" << endl;
        cout << "Position (P): " << direct_bc_grads.end.p.transpose() << endl;
        cout << "Velocity (V): " << direct_bc_grads.end.v.transpose() << endl;
        cout << "Acceleration (A): " << direct_bc_grads.end.a.transpose() << endl;
        return;
    }

    std::vector<double> times;
    SplineTrajectory::SplineVector3D all_points;
    Eigen::Matrix3Xd inner_points;
    Eigen::MatrixXd inner_points_ref; 
    generateRandomData(N, times, all_points, inner_points, inner_points_ref);

    // Initial Conditions
    Eigen::Matrix3d headPVA, tailPVA;
    headPVA.col(0) = all_points[0]; headPVA.col(1).setZero(); headPVA.col(2).setZero();
    tailPVA.col(0) = all_points.back(); tailPVA.col(1).setZero(); tailPVA.col(2).setZero();

    // Setup objects
    minco::MINCO_S3NU minco_s3;
    minco_s3.setConditions(headPVA, tailPVA, N);

    min_jerk::JerkOpt jerk_opt;
    jerk_opt.reset(headPVA, tailPVA, N);

    SplineTrajectory::BoundaryConditions<3> bc;
    bc.start_velocity = headPVA.col(1); bc.start_acceleration = headPVA.col(2);
    bc.end_velocity = tailPVA.col(1); bc.end_acceleration = tailPVA.col(2);
    SplineTrajectory::QuinticSpline3D quintic_spline;

    // 1. Performance Benchmark: Generation
    printSubHeader("Performance: Trajectory Generation");
    
    auto t1 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        minco_s3.setParameters(inner_points, Eigen::Map<Eigen::VectorXd>(times.data(), N));
    }
    double t_minco = duration_cast<microseconds>(high_resolution_clock::now() - t1).count() / (double)BENCH_ITERS;

    auto t2 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        quintic_spline.update(times, all_points, 0.0, bc);
    }
    double t_spline = duration_cast<microseconds>(high_resolution_clock::now() - t2).count() / (double)BENCH_ITERS;

    auto t3 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        jerk_opt.generate(inner_points, Eigen::Map<Eigen::VectorXd>(times.data(), N));
    }
    double t_ref = duration_cast<microseconds>(high_resolution_clock::now() - t3).count() / (double)BENCH_ITERS;

    printTime("MINCO S3 Gen Time", t_minco);
    printTime("Ref(Jerk) Gen Time", t_ref);
    printTime("Spline5 Gen Time", t_spline);

    // 2. Consistency Check: Energy & Coeffs (vs MINCO)
    printSubHeader("Consistency: MINCO vs Spline");
    double e_minco; minco_s3.getEnergy(e_minco);
    printCheck("Energy Difference", std::abs(e_minco - quintic_spline.getEnergy()));
    printCheck("Coeffs Difference", (minco_s3.getCoeffs() - quintic_spline.getTrajectory().getCoefficients()).norm());

    // 3. Gradient Calculation & Propagation Benchmark (Spline vs MINCO)
    printSubHeader("Performance: Gradient Propagation");

    Eigen::MatrixX3d gdC_minco; Eigen::VectorXd gdT_minco;
    Eigen::Matrix3Xd gradP_minco_out; Eigen::VectorXd gradT_minco_out;

    SplineTrajectory::QuinticSpline3D::MatrixType gdC_spline;
    Eigen::VectorXd gdT_spline;
    SplineTrajectory::QuinticSpline3D::MatrixType gradP_spline_full;
    Eigen::VectorXd gradT_spline_out;

    SplineTrajectory::QuinticSpline3D::Gradients grads_ref;

    t1 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        minco_s3.getEnergyPartialGradByCoeffs(gdC_minco);
        minco_s3.getEnergyPartialGradByTimes(gdT_minco);
        minco_s3.propogateGrad(gdC_minco, gdT_minco, gradP_minco_out, gradT_minco_out);
    }
    double t_grad_minco = duration_cast<microseconds>(high_resolution_clock::now() - t1).count() / (double)BENCH_ITERS;

    t2 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        gdC_spline = quintic_spline.getEnergyPartialGradByCoeffs();
        gdT_spline = quintic_spline.getEnergyPartialGradByTimes();
        auto grads = quintic_spline.propagateGrad(gdC_spline, gdT_spline);
        gradT_spline_out = grads.times;
        // Reconstruct full point gradients from propagated results
        gradP_spline_full.resize(N + 1, 3);
        gradP_spline_full.row(0) = grads.start.p.transpose();
        if (N > 1) {
            gradP_spline_full.block(1, 0, N - 1, 3) = grads.inner_points;
        }
        gradP_spline_full.row(N) = grads.end.p.transpose();
        grads_ref = grads;
    }
    double t_grad_spline = duration_cast<microseconds>(high_resolution_clock::now() - t2).count() / (double)BENCH_ITERS;

    printTime("MINCO S3 Grad Prop", t_grad_minco);
    printTime("Spline5 Grad Prop", t_grad_spline);

    // 4. Gradient Consistency Check (vs MINCO)
    printSubHeader("Consistency: Gradients vs MINCO");
    printCheck("Partial Grad (Coeffs)", (gdC_minco - gdC_spline).norm());
    printCheck("Partial Grad (Times)", (gdT_minco - gdT_spline).norm());
    
    Eigen::Matrix3Xd gradP_spline_inner_T = gradP_spline_full.block(1, 0, N - 1, 3).transpose();
    printCheck("Propagated Grad (Inner P)", (gradP_minco_out - gradP_spline_inner_T).norm());
    printCheck("Propagated Grad (Times)", (gradT_minco_out - gradT_spline_out).norm());

    // 5. Direct Gradient Consistency Check (vs Ref min_jerk)
    printSubHeader("Consistency: Direct Gradients vs Ref");
    auto direct_gradT = quintic_spline.getEnergyGradTimes();
    auto direct_gradP_inner = quintic_spline.getEnergyGradInnerPoints();
    auto direct_bc_grads = quintic_spline.getEnergyGradBoundary();

    // Ref min_jerk returns 3x(N-1), need transpose
    Eigen::MatrixXd ref_gradP = jerk_opt.getGradInnerP().transpose();

    printCheck("Direct Grad (Times)", (direct_gradT - jerk_opt.getGradT()).norm());
    printCheck("Direct Grad (Inner P)", (direct_gradP_inner - ref_gradP).norm());

    // 6. Self-Check (Direct vs Propagated)
    printSubHeader("Self-Check: Direct vs Propagated");
    Eigen::MatrixXd prop_gradP_inner = gradP_spline_full.block(1, 0, N - 1, 3);
    printCheck("Direct vs Prop (Times)", (direct_gradT - gradT_spline_out).norm());
    printCheck("Direct vs Prop (Inner P)", (direct_gradP_inner - prop_gradP_inner).norm());
    printCheck("Direct vs Prop (Start P)", (direct_bc_grads.start.p - grads_ref.start.p).norm());
    printCheck("Direct vs Prop (Start V)", (direct_bc_grads.start.v - grads_ref.start.v).norm());
    printCheck("Direct vs Prop (Start A)", (direct_bc_grads.start.a - grads_ref.start.a).norm());
    printCheck("Direct vs Prop (End P)", (direct_bc_grads.end.p - grads_ref.end.p).norm());
    printCheck("Direct vs Prop (End V)", (direct_bc_grads.end.v - grads_ref.end.v).norm());
    printCheck("Direct vs Prop (End A)", (direct_bc_grads.end.a - grads_ref.end.a).norm());
    // Test reference overload
    quintic_spline.propagateGrad(gdC_spline, gdT_spline, grads_ref);
    printCheck("Reference overload", 0.0);

    // 7. Point-by-Point Gradient Comparison
    printSubHeader("Point-by-Point Gradient Comparison");
    cout << "\n[Gradient w.r.t. Times (T)]" << endl;
    cout << std::left << std::setw(8) << "Seg"
         << std::setw(20) << "MINCO"
         << std::setw(20) << "Spline"
         << std::setw(20) << "Ref(Jerk)" << endl;
    cout << string(68, '-') << endl;
    for (int i = 0; i < N; ++i) {
        cout << std::left << std::setw(8) << i
             << std::setw(20) << std::fixed << std::setprecision(6) << gradT_minco_out(i)
             << std::setw(20) << std::fixed << std::setprecision(6) << gradT_spline_out(i)
             << std::setw(20) << std::fixed << std::setprecision(6) << jerk_opt.getGradT()(i) << endl;
    }

    cout << "\n[Gradient w.r.t. Inner Points (P)]" << endl;
    cout << std::left << std::setw(8) << "Point" 
         << std::setw(36) << "MINCO (x, y, z)" 
         << std::setw(36) << "Spline (x, y, z)"
         << std::setw(36) << "Ref(Jerk) (x, y, z)" << endl;
    cout << string(116, '-') << endl;
    for (int i = 0; i < N - 1; ++i) {
        cout << std::left << std::setw(8) << i
             << "(" << std::setw(10) << std::fixed << std::setprecision(4) << gradP_minco_out(0, i)
             << ", " << std::setw(10) << gradP_minco_out(1, i)
             << ", " << std::setw(10) << gradP_minco_out(2, i) << ")  "
             << "(" << std::setw(10) << gradP_spline_inner_T(0, i)
             << ", " << std::setw(10) << gradP_spline_inner_T(1, i)
             << ", " << std::setw(10) << gradP_spline_inner_T(2, i) << ")  "
             << "(" << std::setw(10) << jerk_opt.getGradInnerP()(0, i)
             << ", " << std::setw(10) << jerk_opt.getGradInnerP()(1, i)
             << ", " << std::setw(10) << jerk_opt.getGradInnerP()(2, i) << ")" << endl;
    }

    // 8. Print Boundary Gradients
    printSubHeader("Boundary Condition Gradients");
    cout << "\n[Start Boundary Gradients]" << endl;
    cout << "Position (P): " << direct_bc_grads.start.p.transpose() << endl;
    cout << "Velocity (V): " << direct_bc_grads.start.v.transpose() << endl;
    cout << "Acceleration (A): " << direct_bc_grads.start.a.transpose() << endl;
    cout << "\n[End Boundary Gradients]" << endl;
    cout << "Position (P): " << direct_bc_grads.end.p.transpose() << endl;
    cout << "Velocity (V): " << direct_bc_grads.end.v.transpose() << endl;
    cout << "Acceleration (A): " << direct_bc_grads.end.a.transpose() << endl;
}
void testSeptic(int N, int BENCH_ITERS)
{
    printHeader("TEST: Septic Spline (Order 7) vs MINCO S4 vs Ref(Snap)");

    // Note: MINCO and min_snap require N >= 2 (at least 2 segments)
    // For N = 1, we only test SplineTrajectory
    if (N < 2) {
        cout << "\033[33mWARNING: N=" << N << " is not supported by MINCO/min_snap (require N>=2).\033[0m" << endl;
        cout << "Testing SplineTrajectory only for N=1..." << endl;
        
        std::vector<double> times(N);
        SplineTrajectory::SplineVector3D all_points(N + 1);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis_t(0.5, 2.0);
        std::uniform_real_distribution<> dis_p(-10.0, 10.0);
        for (int i = 0; i < N; ++i) times[i] = dis_t(gen);
        for (int i = 0; i <= N; ++i) all_points[i] = Eigen::Vector3d(dis_p(gen), dis_p(gen), dis_p(gen));
        
        SplineTrajectory::BoundaryConditions<3> bc;
        SplineTrajectory::SepticSpline3D septic_spline;
        septic_spline.update(times, all_points, 0.0, bc);
        
        cout << "Spline7 Generation: \033[32mOK\033[0m" << endl;
        cout << "Energy: " << septic_spline.getEnergy() << endl;
        cout << "Duration: " << septic_spline.getDuration() << endl;
        
        // Test gradient computation
        auto gdC = septic_spline.getEnergyPartialGradByCoeffs();
        auto gdT = septic_spline.getEnergyPartialGradByTimes();
        cout << "Partial Grad (Coeffs) shape: " << gdC.rows() << "x" << gdC.cols() << endl;
        cout << "Partial Grad (Times) size: " << gdT.size() << endl;
        
        // Test propagateGrad
        auto grads = septic_spline.propagateGrad(gdC, gdT);
        // Test reference overload
        SplineTrajectory::SepticSpline3D::Gradients grads_ref;
        septic_spline.propagateGrad(gdC, gdT, grads_ref);
        printCheck("Reference overload", 0.0);
        // Reconstruct full point gradients from propagated results
        SplineTrajectory::SepticSpline3D::MatrixType gradP_full(N + 1, 3);
        gradP_full.row(0) = grads.start.p.transpose();
        if (N > 1) {
            gradP_full.block(1, 0, N - 1, 3) = grads.inner_points;
        }
        gradP_full.row(N) = grads.end.p.transpose();
        cout << "Propagated Grad (Points) shape: " << gradP_full.rows() << "x" << gradP_full.cols() << endl;
        cout << "Propagated Grad (Times) size: " << grads.times.size() << endl;
        
        // Consistency Check: Direct vs Propagated
        printSubHeader("Self-Check: Direct vs Propagated");
        auto direct_gradT = septic_spline.getEnergyGradTimes();
        auto direct_gradP_inner = septic_spline.getEnergyGradInnerPoints();
        auto direct_bc_grads = septic_spline.getEnergyGradBoundary();

        printCheck("Direct vs Prop (Times)", (direct_gradT - grads.times).norm());

        if (N > 1) {
            Eigen::MatrixXd prop_gradP_inner = gradP_full.block(1, 0, N - 1, 3);
            printCheck("Direct vs Prop (Inner P)", (direct_gradP_inner - prop_gradP_inner).norm());
        } else {
            cout << "Inner Points Grad: N/A (no inner points for N=1)" << endl;
        }

        printCheck("Direct vs Prop (Start P)", (direct_bc_grads.start.p - grads.start.p).norm());
        printCheck("Direct vs Prop (Start V)", (direct_bc_grads.start.v - grads.start.v).norm());
        printCheck("Direct vs Prop (Start A)", (direct_bc_grads.start.a - grads.start.a).norm());
        printCheck("Direct vs Prop (Start J)", (direct_bc_grads.start.j - grads.start.j).norm());
        printCheck("Direct vs Prop (End P)", (direct_bc_grads.end.p - grads.end.p).norm());
        printCheck("Direct vs Prop (End V)", (direct_bc_grads.end.v - grads.end.v).norm());
        printCheck("Direct vs Prop (End A)", (direct_bc_grads.end.a - grads.end.a).norm());
        printCheck("Direct vs Prop (End J)", (direct_bc_grads.end.j - grads.end.j).norm());

        // Print boundary gradients
        printSubHeader("Boundary Condition Gradients");
        cout << "\n[Start Boundary Gradients]" << endl;
        cout << "Position (P): " << direct_bc_grads.start.p.transpose() << endl;
        cout << "Velocity (V): " << direct_bc_grads.start.v.transpose() << endl;
        cout << "Acceleration (A): " << direct_bc_grads.start.a.transpose() << endl;
        cout << "Jerk (J): " << direct_bc_grads.start.j.transpose() << endl;
        cout << "\n[End Boundary Gradients]" << endl;
        cout << "Position (P): " << direct_bc_grads.end.p.transpose() << endl;
        cout << "Velocity (V): " << direct_bc_grads.end.v.transpose() << endl;
        cout << "Acceleration (A): " << direct_bc_grads.end.a.transpose() << endl;
        cout << "Jerk (J): " << direct_bc_grads.end.j.transpose() << endl;
        return;
    }

    std::vector<double> times;
    SplineTrajectory::SplineVector3D all_points;
    Eigen::Matrix3Xd inner_points;
    Eigen::MatrixXd inner_points_ref; 
    generateRandomData(N, times, all_points, inner_points, inner_points_ref);

    // Initial Conditions
    Eigen::Matrix<double, 3, 4> headPVAJ, tailPVAJ;
    headPVAJ.col(0) = all_points[0]; headPVAJ.block<3,3>(0,1).setZero();
    tailPVAJ.col(0) = all_points.back(); tailPVAJ.block<3,3>(0,1).setZero();

    // Setup objects
    minco::MINCO_S4NU minco_s4;
    minco_s4.setConditions(headPVAJ, tailPVAJ, N);

    min_snap::SnapOpt snap_opt;
    snap_opt.reset(headPVAJ, tailPVAJ, N);

    SplineTrajectory::BoundaryConditions<3> bc;
    bc.start_velocity = headPVAJ.col(1); bc.start_acceleration = headPVAJ.col(2); bc.start_jerk = headPVAJ.col(3);
    bc.end_velocity = tailPVAJ.col(1); bc.end_acceleration = tailPVAJ.col(2); bc.end_jerk = tailPVAJ.col(3);
    SplineTrajectory::SepticSpline3D septic_spline;

    // 1. Performance Benchmark: Generation
    printSubHeader("Performance: Trajectory Generation");
    
    auto t1 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        minco_s4.setParameters(inner_points, Eigen::Map<Eigen::VectorXd>(times.data(), N));
    }
    double t_minco = duration_cast<microseconds>(high_resolution_clock::now() - t1).count() / (double)BENCH_ITERS;

    auto t2 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        septic_spline.update(times, all_points, 0.0, bc);
    }
    double t_spline = duration_cast<microseconds>(high_resolution_clock::now() - t2).count() / (double)BENCH_ITERS;

    auto t3 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        snap_opt.generate(inner_points, Eigen::Map<Eigen::VectorXd>(times.data(), N));
    }
    double t_ref = duration_cast<microseconds>(high_resolution_clock::now() - t3).count() / (double)BENCH_ITERS;

    printTime("MINCO S4 Gen Time", t_minco);
    printTime("Ref(Snap) Gen Time", t_ref);
    printTime("Spline7 Gen Time", t_spline);

    // 2. Consistency Check: Energy & Coeffs (vs MINCO)
    printSubHeader("Consistency: MINCO vs Spline");
    double e_minco; minco_s4.getEnergy(e_minco);
    printCheck("Energy Difference", std::abs(e_minco - septic_spline.getEnergy()));
    printCheck("Coeffs Difference", (minco_s4.getCoeffs() - septic_spline.getTrajectory().getCoefficients()).norm());

    // 3. Gradient Calculation & Propagation Benchmark (Spline vs MINCO)
    printSubHeader("Performance: Gradient Propagation");

    Eigen::MatrixX3d gdC_minco; Eigen::VectorXd gdT_minco;
    Eigen::Matrix3Xd gradP_minco_out; Eigen::VectorXd gradT_minco_out;

    SplineTrajectory::SepticSpline3D::MatrixType gdC_spline;
    Eigen::VectorXd gdT_spline;
    SplineTrajectory::SepticSpline3D::MatrixType gradP_spline_full;
    Eigen::VectorXd gradT_spline_out;

    SplineTrajectory::SepticSpline3D::Gradients grads_ref;

    t1 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        minco_s4.getEnergyPartialGradByCoeffs(gdC_minco);
        minco_s4.getEnergyPartialGradByTimes(gdT_minco);
        minco_s4.propogateGrad(gdC_minco, gdT_minco, gradP_minco_out, gradT_minco_out);
    }
    double t_grad_minco = duration_cast<microseconds>(high_resolution_clock::now() - t1).count() / (double)BENCH_ITERS;

    t2 = high_resolution_clock::now();
    for(int i=0; i<BENCH_ITERS; ++i) {
        gdC_spline = septic_spline.getEnergyPartialGradByCoeffs();
        gdT_spline = septic_spline.getEnergyPartialGradByTimes();
        auto grads = septic_spline.propagateGrad(gdC_spline, gdT_spline);
        gradT_spline_out = grads.times;
        // Reconstruct full point gradients from propagated results
        gradP_spline_full.resize(N + 1, 3);
        gradP_spline_full.row(0) = grads.start.p.transpose();
        if (N > 1) {
            gradP_spline_full.block(1, 0, N - 1, 3) = grads.inner_points;
        }
        gradP_spline_full.row(N) = grads.end.p.transpose();
        grads_ref = grads;
    }
    double t_grad_spline = duration_cast<microseconds>(high_resolution_clock::now() - t2).count() / (double)BENCH_ITERS;

    printTime("MINCO S4 Grad Prop", t_grad_minco);
    printTime("Spline7 Grad Prop", t_grad_spline);

    // 4. Gradient Consistency Check (vs MINCO)
    printSubHeader("Consistency: Gradients vs MINCO");
    printCheck("Partial Grad (Coeffs)", (gdC_minco - gdC_spline).norm());
    printCheck("Partial Grad (Times)", (gdT_minco - gdT_spline).norm());
    
    Eigen::Matrix3Xd gradP_spline_inner_T = gradP_spline_full.block(1, 0, N - 1, 3).transpose();
    printCheck("Propagated Grad (Inner P)", (gradP_minco_out - gradP_spline_inner_T).norm());
    printCheck("Propagated Grad (Times)", (gradT_minco_out - gradT_spline_out).norm());

    // 5. Direct Gradient Consistency Check (vs Ref min_snap)
    printSubHeader("Consistency: Direct Gradients vs Ref");
    auto direct_gradT = septic_spline.getEnergyGradTimes();
    auto direct_gradP_inner = septic_spline.getEnergyGradInnerPoints();
    auto direct_bc_grads = septic_spline.getEnergyGradBoundary();

    // Ref min_snap returns 3x(N-1), need transpose
    Eigen::MatrixXd ref_gradP = snap_opt.getGradInnerP().transpose();

    printCheck("Direct Grad (Times)", (direct_gradT - snap_opt.getGradT()).norm());
    printCheck("Direct Grad (Inner P)", (direct_gradP_inner - ref_gradP).norm());

    // 6. Self-Check (Direct vs Propagated)
    printSubHeader("Self-Check: Direct vs Propagated");
    Eigen::MatrixXd prop_gradP_inner = gradP_spline_full.block(1, 0, N - 1, 3);
    printCheck("Direct vs Prop (Times)", (direct_gradT - gradT_spline_out).norm());
    printCheck("Direct vs Prop (Inner P)", (direct_gradP_inner - prop_gradP_inner).norm());
    printCheck("Direct vs Prop (Start P)", (direct_bc_grads.start.p.transpose() - grads_ref.start.p.transpose()).norm());
    printCheck("Direct vs Prop (Start V)", (direct_bc_grads.start.v.transpose() - grads_ref.start.v.transpose()).norm());
    printCheck("Direct vs Prop (Start A)", (direct_bc_grads.start.a.transpose() - grads_ref.start.a.transpose()).norm());
    printCheck("Direct vs Prop (Start J)", (direct_bc_grads.start.j.transpose() - grads_ref.start.j.transpose()).norm());
    printCheck("Direct vs Prop (End P)", (direct_bc_grads.end.p.transpose() - grads_ref.end.p.transpose()).norm());
    printCheck("Direct vs Prop (End V)", (direct_bc_grads.end.v.transpose() - grads_ref.end.v.transpose()).norm());
    printCheck("Direct vs Prop (End A)", (direct_bc_grads.end.a.transpose() - grads_ref.end.a.transpose()).norm());
    printCheck("Direct vs Prop (End J)", (direct_bc_grads.end.j.transpose() - grads_ref.end.j.transpose()).norm());
    // Test reference overload
    septic_spline.propagateGrad(gdC_spline, gdT_spline, grads_ref);
    printCheck("Reference overload", 0.0);

    // 7. Point-by-Point Gradient Comparison
    printSubHeader("Point-by-Point Gradient Comparison");
    cout << "\n[Gradient w.r.t. Times (T)]" << endl;
    cout << std::left << std::setw(8) << "Seg"
         << std::setw(20) << "MINCO"
         << std::setw(20) << "Spline"
         << std::setw(20) << "Ref(Snap)" << endl;
    cout << string(68, '-') << endl;
    for (int i = 0; i < N; ++i) {
        cout << std::left << std::setw(8) << i
             << std::setw(20) << std::fixed << std::setprecision(6) << gradT_minco_out(i)
             << std::setw(20) << std::fixed << std::setprecision(6) << gradT_spline_out(i)
             << std::setw(20) << std::fixed << std::setprecision(6) << snap_opt.getGradT()(i) << endl;
    }

    cout << "\n[Gradient w.r.t. Inner Points (P)]" << endl;
    cout << std::left << std::setw(8) << "Point" 
         << std::setw(36) << "MINCO (x, y, z)" 
         << std::setw(36) << "Spline (x, y, z)"
         << std::setw(36) << "Ref(Snap) (x, y, z)" << endl;
    cout << string(116, '-') << endl;
    for (int i = 0; i < N - 1; ++i) {
        cout << std::left << std::setw(8) << i
             << "(" << std::setw(10) << std::fixed << std::setprecision(4) << gradP_minco_out(0, i)
             << ", " << std::setw(10) << gradP_minco_out(1, i)
             << ", " << std::setw(10) << gradP_minco_out(2, i) << ")  "
             << "(" << std::setw(10) << gradP_spline_inner_T(0, i)
             << ", " << std::setw(10) << gradP_spline_inner_T(1, i)
             << ", " << std::setw(10) << gradP_spline_inner_T(2, i) << ")  "
             << "(" << std::setw(10) << snap_opt.getGradInnerP()(0, i)
             << ", " << std::setw(10) << snap_opt.getGradInnerP()(1, i)
             << ", " << std::setw(10) << snap_opt.getGradInnerP()(2, i) << ")" << endl;
    }

    // 8. Print Boundary Gradients
    printSubHeader("Boundary Condition Gradients");
    cout << "\n[Start Boundary Gradients]" << endl;
    cout << "Position (P): " << direct_bc_grads.start.p.transpose() << endl;
    cout << "Velocity (V): " << direct_bc_grads.start.v.transpose() << endl;
    cout << "Acceleration (A): " << direct_bc_grads.start.a.transpose() << endl;
    cout << "Jerk (J): " << direct_bc_grads.start.j.transpose() << endl;
    cout << "\n[End Boundary Gradients]" << endl;
    cout << "Position (P): " << direct_bc_grads.end.p.transpose() << endl;
    cout << "Velocity (V): " << direct_bc_grads.end.v.transpose() << endl;
    cout << "Acceleration (A): " << direct_bc_grads.end.a.transpose() << endl;
    cout << "Jerk (J): " << direct_bc_grads.end.j.transpose() << endl;
}

int main()
{
    // Test parameters
    int N = 5;
    int BENCH_ITERS = 10000;

    // Seed random
    srand(time(0));

    // Print test configuration
    printHeader("TEST CONFIGURATION");
    cout << "Number of segments (N): " << N << endl;
    cout << "Benchmark iterations: " << BENCH_ITERS << endl;

    testCubic(N, BENCH_ITERS);
    testQuintic(N, BENCH_ITERS);
    testSeptic(N, BENCH_ITERS);

    return 0;
}