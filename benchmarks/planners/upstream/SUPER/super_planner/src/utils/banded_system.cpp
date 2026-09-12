#include <utils/optimization/banded_system.h>


using namespace geometry_utils;

// BandedSystem==================================================
void BandedSystem::create(const int &n, const int &p, const int &q) {
    // In case of re-creating before destroying
    destroy();
    N = n;
    lowerBw = p;
    upperBw = q;
    int actualSize = N * (lowerBw + upperBw + 1);
    ptrData = new double[actualSize];
    std::fill_n(ptrData, actualSize, 0.0);
    return;
}

void BandedSystem::destroy() {
    if (ptrData != nullptr) {
        delete[] ptrData;
        ptrData = nullptr;
    }
    return;
}

void BandedSystem::reset() {
    std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0);
    return;
}

void BandedSystem::factorizeLU() {
    int iM, jM;
    double cVl;
    for (int k = 0; k <= N - 2; k++) {
        iM = std::min(k + lowerBw, N - 1);
        cVl = operator()(k, k);
        for (int i = k + 1; i <= iM; i++) {
            if (operator()(i, k) != 0.0) {
                operator()(i, k) /= cVl;
            }
        }
        jM = std::min(k + upperBw, N - 1);
        for (int j = k + 1; j <= jM; j++) {
            cVl = operator()(k, j);
            if (cVl != 0.0) {
                for (int i = k + 1; i <= iM; i++) {
                    if (operator()(i, k) != 0.0) {
                        operator()(i, j) -= operator()(i, k) * cVl;
                    }
                }
            }
        }
    }
    return;
}

template<typename EIGENMAT>
void BandedSystem::solve(EIGENMAT &b) const {
    int iM;
    for (int j = 0; j <= N - 1; j++) {
        iM = std::min(j + lowerBw, N - 1);
        for (int i = j + 1; i <= iM; i++) {
            if (operator()(i, j) != 0.0) {
                b.row(i) -= operator()(i, j) * b.row(j);
            }
        }
    }
    for (int j = N - 1; j >= 0; j--) {
        b.row(j) /= operator()(j, j);
        iM = std::max(0, j - upperBw);
        for (int i = iM; i <= j - 1; i++) {
            if (operator()(i, j) != 0.0) {
                b.row(i) -= operator()(i, j) * b.row(j);
            }
        }
    }
    return;
}

template<typename EIGENMAT>
void BandedSystem::solveAdj(EIGENMAT &b) const {
    int iM;
    for (int j = 0; j <= N - 1; j++) {
        b.row(j) /= operator()(j, j);
        iM = std::min(j + upperBw, N - 1);
        for (int i = j + 1; i <= iM; i++) {
            if (operator()(j, i) != 0.0) {
                b.row(i) -= operator()(j, i) * b.row(j);
            }
        }
    }
    for (int j = N - 1; j >= 0; j--) {
        iM = std::max(0, j - lowerBw);
        for (int i = iM; i <= j - 1; i++) {
            if (operator()(j, i) != 0.0) {
                b.row(i) -= operator()(j, i) * b.row(j);
            }
        }
    }
    return;
}


template void BandedSystem::solve<Eigen::MatrixX3d>(Eigen::MatrixX3d &b) const;

template void BandedSystem::solve<Eigen::MatrixXd>(Eigen::MatrixXd &b) const;

template void BandedSystem::solveAdj<Eigen::MatrixXd>(Eigen::MatrixXd &b) const;

template void BandedSystem::solveAdj<Eigen::MatrixX3d>(Eigen::MatrixX3d &b) const;
