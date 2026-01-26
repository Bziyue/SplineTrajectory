#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <random> 

#include "SplineTrajectory.hpp"

using namespace SplineTrajectory;
using namespace Eigen;

void printPass(const std::string& testName) {
    std::cout << "[PASS] " << testName << std::endl;
}

void printFail(const std::string& testName, const std::string& reason) {
    std::cerr << "[FAIL] " << testName << ": " << reason << std::endl;
    exit(1);
}

void assertVectorEq(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const std::string& msg, double tol = 1e-9) {
    if ((a - b).norm() > tol) {
        std::cerr << "Expected: " << b.transpose() << "\nActual:   " << a.transpose() << std::endl;
        printFail(msg, "Vector mismatch");
    }
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Quintic Spline (36 Segments) Unit Test" << std::endl;
    std::cout << "===========================================" << std::endl;

    const int DIM = 3;
    const int DEGREE = 5; 
    const int NUM_COEFFS = DEGREE + 1; 
    const int NUM_SEGS = 36;

    std::mt19937 gen(42); 
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    std::uniform_real_distribution<double> time_dist(0.5, 2.0);

    std::vector<double> breakpoints;
    breakpoints.reserve(NUM_SEGS + 1);
    double current_time = 0.0;
    breakpoints.push_back(current_time);
    for (int i = 0; i < NUM_SEGS; ++i) {
        current_time += time_dist(gen);
        breakpoints.push_back(current_time);
    }

    using PPoly = PPolyND<DIM>;
    PPoly::MatrixType coeffs(NUM_SEGS * NUM_COEFFS, DIM);
    
    int target_seg_idx = 10;
    Eigen::Matrix<double, NUM_COEFFS, DIM> target_seg_coeffs;

    for (int i = 0; i < NUM_SEGS; ++i) {
        for (int k = 0; k < NUM_COEFFS; ++k) {
            int row_idx = i * NUM_COEFFS + k;
            
            if (i == target_seg_idx) {
                if (k == 1) coeffs.row(row_idx) = Eigen::Vector3d(1, 1, 1);
                else if (k == 5) coeffs.row(row_idx) = Eigen::Vector3d(1, 1, 1);
                else coeffs.row(row_idx) = Eigen::Vector3d::Zero();
            } else {
                coeffs.row(row_idx) = Eigen::Vector3d(dist(gen), dist(gen), dist(gen));
            }
        }
    }
    
    target_seg_coeffs = coeffs.block<NUM_COEFFS, DIM>(target_seg_idx * NUM_COEFFS, 0);

    PPoly traj(breakpoints, coeffs, NUM_COEFFS);

    if (traj.isInitialized() && traj.getNumSegments() == NUM_SEGS && traj.getNumCoeffs() == NUM_COEFFS) {
        std::cout << "Generated Spline: " << NUM_SEGS << " segments, Total Duration: " << traj.getDuration() << "s" << std::endl;
        printPass("Initialization & Random Generation");
    } else {
        printFail("Initialization", "Failed to init PPolyND with random data");
    }

    {
        std::cout << "\nTesting STL Iterator on 36 segments..." << std::endl;
        int iter_count = 0;
        bool continuity_check = true;

        for (const auto& seg : traj) {
            if (seg.index() != iter_count) {
                printFail("Iterator", "Index mismatch");
            }

            double expected_duration = breakpoints[iter_count + 1] - breakpoints[iter_count];
            if (std::abs(seg.duration() - expected_duration) > 1e-9) {
                printFail("Iterator", "Duration mismatch");
            }

            if (iter_count == 5) {
                auto seg_coeffs = seg.getCoeffs(); 
                if ((seg_coeffs.row(0) - coeffs.row(5 * NUM_COEFFS)).norm() > 1e-9) {
                    printFail("Iterator", "Coefficient access mismatch");
                }
            }

            iter_count++;
        }

        if (iter_count == NUM_SEGS) {
            printPass("STL Iterator Full Traversal");
        } else {
            printFail("STL Iterator", "Did not traverse all segments");
        }
    }

    {
        std::cout << "\nTesting Quintic Evaluation (Target Segment 10)..." << std::endl;

        auto seg = traj[target_seg_idx];
        double dt = 0.5;

        Eigen::Vector3d ones(1, 1, 1);

        Eigen::Vector3d expected_pos = (0.5 + std::pow(0.5, 5)) * ones;
        assertVectorEq(seg.evaluate(dt, Deriv::Pos), expected_pos, "Quintic Pos");

        Eigen::Vector3d expected_vel = (1.0 + 5.0 * std::pow(0.5, 4)) * ones;
        assertVectorEq(seg.evaluate(dt, Deriv::Vel), expected_vel, "Quintic Vel");

        Eigen::Vector3d expected_crackle = 120.0 * ones;
        assertVectorEq(seg.evaluate(dt, Deriv::Crackle), expected_crackle, "Quintic Crackle (5th deriv)");

        assertVectorEq(seg.evaluate(dt, Deriv::Pop), Eigen::Vector3d::Zero(), "Quintic Pop (6th deriv - zero)");

        printPass("Quintic Derivative Evaluation (up to 6th order)");
    }

    {
        std::cout << "\nTesting Random Access..." << std::endl;
        
        auto first_seg = traj[0];
        auto last_seg = traj[NUM_SEGS - 1];

        if (first_seg.index() == 0 && last_seg.index() == 35) {
            printPass("Random Access indices");
        } else {
            printFail("Random Access", "Indices wrong");
        }

        try {
            auto out_bounds = traj.at(NUM_SEGS); 
            printFail("Exception", "traj.at(36) did not throw");
        } catch (const std::out_of_range& e) {
            printPass("Exception Handling (out_of_range)");
        }
    }

    {
        std::cout << "\nTesting Global Evaluation..." << std::endl;
        double seg_start_t = breakpoints[target_seg_idx];
        double global_t = seg_start_t + 0.5;

        Eigen::Vector3d global_val = traj.evaluate(global_t, Deriv::Crackle);
        
        assertVectorEq(global_val, Eigen::Vector3d(120, 120, 120), "Global Evaluate Crackle");
        
        printPass("Global Evaluation");
    }

    std::cout << "\n-------------------------------------------" << std::endl;
    std::cout << "  All Tests Passed Successfully!" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    return 0;
}