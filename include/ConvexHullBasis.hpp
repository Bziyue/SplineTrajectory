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

#ifndef SPLINE_TRAJECTORY_CONVEX_HULL_BASIS_HPP
#define SPLINE_TRAJECTORY_CONVEX_HULL_BASIS_HPP

#include "SplineTrajectory.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace SplineTrajectory
{
    enum class ConvexHullBasis
    {
        Bezier,
        MINVO
    };

    /**
     * @brief Convex-hull control-point representation of a PPolyND or one of its
     *        physical-time derivatives.
     *
     * Rows in controls() are piece-major. Every piece owns degree()+1 consecutive
     * rows. A Bezier subdivision depth s creates 2^s pieces for every source
     * polynomial segment. MINVO is available for degrees 0 through 7 and does not
     * subdivide.
     *
     * The object retains the source coefficients, durations, and one small basis
     * matrix needed by backward(), but it does not retain or sample the source
     * trajectory.
     */
    template <int DIM>
    class ConvexHullRepresentation
    {
    public:
        static constexpr int kMatrixOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using VectorType = Eigen::Matrix<double, DIM, 1>;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;

        struct PieceInfo
        {
            int source_segment = 0;
            int subdivision_index = 0;
            double source_fraction_begin = 0.0;
            double source_fraction_end = 1.0;
            double start_time = 0.0;
            double duration = 0.0;
        };

        struct BackwardResult
        {
            MatrixType coefficients;
            Eigen::VectorXd durations;
        };

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ConvexHullRepresentation() = default;

        template <int ORDER>
        static ConvexHullRepresentation fromPPoly(
            const PPolyND<DIM, ORDER> &polynomial,
            ConvexHullBasis basis,
            int derivative_order = 0,
            int subdivision_depth = 0)
        {
            if (!polynomial.isInitialized())
                throw std::invalid_argument("Convex-hull conversion requires an initialized PPolyND.");
            if (derivative_order < 0 || derivative_order > polynomial.getDegree())
                throw std::invalid_argument("Derivative order must be between zero and the polynomial degree.");
            if (subdivision_depth < 0 || subdivision_depth > 20)
                throw std::invalid_argument("Bezier subdivision depth must be in [0, 20].");
            if (basis == ConvexHullBasis::MINVO && subdivision_depth != 0)
                throw std::invalid_argument("MINVO conversion does not use Bezier subdivision.");

            ConvexHullRepresentation result;
            result.initialize(polynomial.getBreakpoints(),
                              polynomial.getCoefficients(),
                              polynomial.getNumCoeffs(),
                              basis,
                              derivative_order,
                              subdivision_depth);
            return result;
        }

        bool isInitialized() const { return initialized_; }
        ConvexHullBasis basis() const { return basis_; }
        int derivativeOrder() const { return derivative_order_; }
        int degree() const { return degree_; }
        int sourceDegree() const { return source_num_coeffs_ - 1; }
        int subdivisionDepth() const { return subdivision_depth_; }
        int piecesPerSegment() const { return leaves_per_segment_; }
        int numSourceSegments() const { return num_source_segments_; }
        int numPieces() const { return static_cast<int>(pieces_.size()); }
        int controlsPerPiece() const { return degree_ + 1; }

        const MatrixType &controls() const { return controls_; }
        const std::vector<PieceInfo> &pieces() const { return pieces_; }
        const PieceInfo &pieceInfo(int piece_index) const { return pieces_.at(piece_index); }

        auto pieceControls(int piece_index) const
        {
            if (piece_index < 0 || piece_index >= numPieces())
                throw std::out_of_range("Convex-hull piece index out of range.");
            return controls_.middleRows(piece_index * controlsPerPiece(), controlsPerPiece());
        }

        /**
         * @brief Reverse a scalar objective gradient from hull controls to the
         *        source PPolyND coefficients and independent segment durations.
         *
         * durations is the partial derivative with the local power coefficients
         * held fixed. It can be passed together with coefficients to a MINCO
         * spline's propagateGrad() method.
         */
        BackwardResult backward(const MatrixType &control_gradients) const
        {
            BackwardResult result;
            result.coefficients = MatrixType::Zero(
                num_source_segments_ * source_num_coeffs_, DIM);
            result.durations = Eigen::VectorXd::Zero(num_source_segments_);
            backward(control_gradients, result.coefficients, result.durations);
            return result;
        }

        void backward(const MatrixType &control_gradients,
                      MatrixType &coefficient_gradients,
                      Eigen::VectorXd &duration_gradients) const
        {
            if (!initialized_)
                throw std::logic_error("Cannot backpropagate an uninitialized representation.");
            if (control_gradients.rows() != controls_.rows() ||
                control_gradients.cols() != DIM)
                throw std::invalid_argument("Control-gradient dimensions do not match controls().");

            if (coefficient_gradients.rows() !=
                    num_source_segments_ * source_num_coeffs_ ||
                coefficient_gradients.cols() != DIM)
            {
                coefficient_gradients.resize(
                    num_source_segments_ * source_num_coeffs_, DIM);
            }
            coefficient_gradients.setZero();
            if (duration_gradients.size() != num_source_segments_)
                duration_gradients.resize(num_source_segments_);
            duration_gradients.setZero();

            const int cp = controlsPerPiece();
            MatrixType normalized_gradient(cp, DIM);
            for (int segment = 0; segment < num_source_segments_; ++segment)
            {
                MatrixType base_gradient;
                if (basis_ == ConvexHullBasis::Bezier && subdivision_depth_ > 0)
                {
                    std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>> level;
                    level.reserve(leaves_per_segment_);
                    const int first_piece = segment * leaves_per_segment_;
                    for (int leaf = 0; leaf < leaves_per_segment_; ++leaf)
                    {
                        level.emplace_back(control_gradients.middleRows(
                            (first_piece + leaf) * cp, cp));
                    }

                    while (level.size() > 1)
                    {
                        std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>> parent;
                        parent.reserve(level.size() / 2);
                        for (std::size_t i = 0; i < level.size(); i += 2)
                        {
                            parent.emplace_back(left_split_.transpose() * level[i] +
                                                right_split_.transpose() * level[i + 1]);
                        }
                        level.swap(parent);
                    }
                    base_gradient = std::move(level.front());
                }
                else
                {
                    base_gradient = control_gradients.middleRows(segment * cp, cp);
                }

                normalized_gradient.noalias() =
                    power_to_control_.transpose() * base_gradient;
                const double duration = source_durations_(segment);
                const double inv_duration = 1.0 / source_durations_(segment);
                for (int normalized_power = 0;
                     normalized_power <= degree_;
                     ++normalized_power)
                {
                    const int k = normalized_power + derivative_order_;
                    const double scale =
                        fallingFactorial(k, derivative_order_) *
                        std::pow(duration, normalized_power);
                    coefficient_gradients.row(
                        segment * source_num_coeffs_ + k) +=
                        scale * normalized_gradient.row(normalized_power);

                    if (normalized_power > 0)
                    {
                        const auto source_coefficient =
                            source_coefficients_.row(
                                segment * source_num_coeffs_ + k);
                        duration_gradients(segment) +=
                            (static_cast<double>(normalized_power) *
                             inv_duration * scale) *
                            normalized_gradient.row(normalized_power)
                                .dot(source_coefficient);
                    }
                }
            }
        }

        /**
         * @brief Matrix B such that control = B * normalized_ascending_power.
         */
        static Eigen::MatrixXd powerToControlMatrix(ConvexHullBasis basis, int degree)
        {
            if (degree < 0)
                throw std::invalid_argument("Polynomial degree must be non-negative.");
            return basis == ConvexHullBasis::Bezier
                       ? bezierPowerToControlMatrix(degree)
                       : minvoPowerToControlMatrix(degree);
        }

    private:
        ConvexHullBasis basis_{ConvexHullBasis::Bezier};
        int derivative_order_{0};
        int degree_{0};
        int subdivision_depth_{0};
        int leaves_per_segment_{1};
        int num_source_segments_{0};
        int source_num_coeffs_{0};
        bool initialized_{false};

        MatrixType controls_;
        MatrixType source_coefficients_;
        Eigen::VectorXd source_durations_;
        std::vector<PieceInfo> pieces_;
        Eigen::MatrixXd power_to_control_;
        Eigen::MatrixXd left_split_;
        Eigen::MatrixXd right_split_;

        static double binomial(int n, int k)
        {
            if (k < 0 || k > n)
                return 0.0;
            k = std::min(k, n - k);
            double result = 1.0;
            for (int i = 1; i <= k; ++i)
                result *= static_cast<double>(n - k + i) / static_cast<double>(i);
            return result;
        }

        static double fallingFactorial(int n, int count)
        {
            double result = 1.0;
            for (int i = 0; i < count; ++i)
                result *= static_cast<double>(n - i);
            return result;
        }

        static Eigen::MatrixXd bezierPowerToControlMatrix(int degree)
        {
            Eigen::MatrixXd matrix =
                Eigen::MatrixXd::Zero(degree + 1, degree + 1);
            for (int i = 0; i <= degree; ++i)
            {
                for (int k = 0; k <= i; ++k)
                    matrix(i, k) = binomial(i, k) / binomial(degree, k);
            }
            return matrix;
        }

        static Eigen::MatrixXd minvoPowerToControlMatrix(int degree)
        {
            // Generated once from the official MINVO degree 0--7 solution files:
            // getA_MV(degree, [0, 1]), followed by B=A^{-T} and ascending powers.
            // MINVO solution data: Copyright (c) 2020 Jesus Tordesillas Torres,
            // MIT Aerospace Controls Laboratory, BSD-3-Clause. See
            // THIRD_PARTY_NOTICES.md.
            Eigen::MatrixXd b(degree + 1, degree + 1);
            switch (degree)
            {
            case 0:
                b << 1.0;
                break;
            case 1:
                b << 1.0, 0.0,
                     1.0, 1.0;
                break;
            case 2:
                b << 0.9999999999999999, -0.07735026889434268, -0.07735026889434268,
                     1.0000000000000002,  0.5000000000000001,   0.16666666649618495,
                     1.0000000000000009,  1.0773502688943430,   1.0773502688943430;
                break;
            case 3:
                b << 1.0000000000000002, -0.07454782086322333, -0.05111494427287730, -0.03203276811028855,
                     1.0000000000000002,  0.20395194315405324, -0.04627261955020995, -0.09273094095157490,
                     1.0000000000000009,  0.79604805684594740,  0.54582349414168410,  0.34205725283878580,
                     1.0000000000000013,  1.07454782086322420,  1.09798069745357000,  1.10233139788132740;
                break;
            case 4:
                b << 0.9999999999999999, -0.07116896486408000, -0.04442474424479857, -0.03105263393515785, -0.02526085139144257,
                     0.9999999999999996,  0.09437199289145079, -0.06104394324429762, -0.06166517120990710, -0.04642786622261585,
                     1.0000000000000010,  0.50000000000000080,  0.18036721916374074,  0.02055082874561072, -0.04866949316244124,
                     1.0000000000000056,  0.90562800710855010,  0.75021207097280120,  0.59541736280266210,  0.45648118758542390,
                     1.0000000000000089,  1.07116896486408050,  1.09791318548336130,  1.11128529579300170,  1.11707707833671680;
                break;
            case 5:
                b << 0.9999999999999992, -0.06471861202015976, -0.03728008486153074, -0.02577637793955953, -0.02027573243110871, -0.01678273037232765,
                     0.9999999999999981,  0.03314986095853732, -0.06548114210849208, -0.05530463802397501, -0.04362718953107343, -0.03671639115121680,
                     0.9999999999999979,  0.33755289971473396,  0.05836232551993632, -0.02920033164741221, -0.04690387913496401, -0.04376447946695088,
                     1.0000000000000033,  0.66244710028526650,  0.38325652609046760,  0.19162860906301790,  0.06985980171536514, -0.00289284310805573,
                     1.0000000000000273,  0.96685013904146560,  0.86821913597443030,  0.75941162882288220,  0.65210506607971930,  0.55106609785798630,
                     1.0000000000000457,  1.06471861202016440,  1.09215713917878450,  1.10809195941544010,  1.11802371823857680,  1.12396005909786360;
                break;
            case 6:
                b << 0.99999999999999822, -0.060228927502977987, -0.033755692487772944, -0.023855836203916802, -0.019121785378066126, -0.016235170513163748, -0.014282189352672938,
                     0.99999999999999911,  0.004635647049236109, -0.057491939345686513, -0.043549624552338302, -0.033205509446734088, -0.026978635943569296, -0.022753790797390143,
                     1.00000000000000090,  0.22809674652208098,  0.000222735192258446, -0.045338417799847111, -0.046433799578532789, -0.040367247492681377, -0.035004509179091745,
                     1.00000000000000330,  0.50000000000000078,  0.20208114478650321,  0.053121717179754259, -0.013079987577292137, -0.037902830909487220, -0.043795656142563442,
                     0.99999999999999978,  0.77190325347790933,  0.54402924214809056,  0.36171638381037630,  0.22386929668608069,  0.12332604691066712,  0.052220886847337070,
                     0.99999999999998934,  0.99536435295072534,  0.93323676655581100,  0.85716686536754871,  0.77749876449153965,  0.69834970553022380,  0.62183490172905553,
                     0.99999999999998490,  1.06022892750292260,  1.08670216251813700,  1.10327554124949740,  1.11468311452284910,  1.12277231829914070,  1.12845695483490950;
                break;
            case 7:
                b << 0.9999999999999978, -0.05550234532091862, -0.03015649917329037, -0.02126221894516797, -0.01693417728228520, -0.01426182870806678, -0.01240847743086744, -0.01102231756545113,
                     0.9999999999999968, -0.01444679376549229, -0.05377370181851387, -0.03962596388999581, -0.03098345173577396, -0.02595197185380287, -0.02265469217115398, -0.02032295522350399,
                     0.9999999999999989,  0.16240204644755055, -0.01852864608397268, -0.03991647484819192, -0.03471412550958872, -0.02817909967291144, -0.02334086394352624, -0.01983518262002364,
                     1.0000000000000002,  0.37541364567871430,  0.09567905676542685, -0.00828476061039229, -0.03998426794247088, -0.04550931887029734, -0.04303904824639800, -0.03898747361466550,
                     0.9999999999999966,  0.62458635432125740,  0.34485176540797230,  0.16908099387050880,  0.06557453237678527,  0.00815792452254936, -0.02134796454472514, -0.03470825213367175,
                     1.0000000000000047,  0.83759795355248600,  0.65666726102096370,  0.49712439725365380,  0.36417171158916700,  0.25647652752942990,  0.17100937846907616,  0.10437656210132987,
                     1.0000000000000855,  1.01444679376569670,  0.97511988571265720,  0.92164523973107760,  0.86266536797520940,  0.80179130271730500,  0.74089987603029060,  0.68109926252276930,
                     1.0000000000001545,  1.05550234532123470,  1.08084819146883060,  1.09729975738827350,  1.10918508474248930,  1.11815986662014440,  1.12506079881288020,  1.13037277122710170;
                break;
            default:
                throw std::invalid_argument(
                    "Exact MINVO matrices are available for polynomial degrees 0 through 7.");
            }
            return b;
        }

        static void makeHalfSplitMatrices(int degree,
                                          Eigen::MatrixXd &left,
                                          Eigen::MatrixXd &right)
        {
            const int size = degree + 1;
            left = Eigen::MatrixXd::Zero(size, size);
            right = Eigen::MatrixXd::Zero(size, size);
            for (int i = 0; i <= degree; ++i)
            {
                const double left_scale = std::ldexp(1.0, -i);
                for (int j = 0; j <= i; ++j)
                    left(i, j) = binomial(i, j) * left_scale;

                const int remaining = degree - i;
                const double right_scale = std::ldexp(1.0, -remaining);
                for (int j = i; j <= degree; ++j)
                    right(i, j) = binomial(remaining, j - i) * right_scale;
            }
        }

        void initialize(const std::vector<double> &breakpoints,
                        const MatrixType &coefficients,
                        int source_num_coeffs,
                        ConvexHullBasis basis,
                        int derivative_order,
                        int subdivision_depth)
        {
            basis_ = basis;
            derivative_order_ = derivative_order;
            source_num_coeffs_ = source_num_coeffs;
            degree_ = source_num_coeffs_ - derivative_order_ - 1;
            subdivision_depth_ = subdivision_depth;
            leaves_per_segment_ =
                basis_ == ConvexHullBasis::Bezier ? (1 << subdivision_depth_) : 1;
            num_source_segments_ = static_cast<int>(breakpoints.size()) - 1;
            source_coefficients_ = coefficients;
            source_durations_.resize(num_source_segments_);

            if (basis_ == ConvexHullBasis::MINVO && degree_ > 7)
                throw std::invalid_argument(
                    "Exact MINVO matrices are available for polynomial degrees 0 through 7.");

            power_to_control_ = powerToControlMatrix(basis_, degree_);
            const int cp = controlsPerPiece();
            controls_.resize(num_source_segments_ * leaves_per_segment_ * cp, DIM);
            pieces_.clear();
            pieces_.reserve(num_source_segments_ * leaves_per_segment_);
            if (basis_ == ConvexHullBasis::Bezier && subdivision_depth_ > 0)
                makeHalfSplitMatrices(degree_, left_split_, right_split_);

            MatrixType normalized_derivative(controlsPerPiece(), DIM);
            MatrixType base_controls(controlsPerPiece(), DIM);
            for (int segment = 0; segment < num_source_segments_; ++segment)
            {
                const double duration = breakpoints[segment + 1] - breakpoints[segment];
                if (!(duration > 0.0) || !std::isfinite(duration))
                    throw std::invalid_argument(
                        "PPolyND breakpoints must have finite, strictly positive durations.");
                source_durations_(segment) = duration;

                for (int k = derivative_order_; k < source_num_coeffs_; ++k)
                {
                    const int normalized_power = k - derivative_order_;
                    const double scale =
                        fallingFactorial(k, derivative_order_) *
                        std::pow(duration, normalized_power);
                    normalized_derivative.row(normalized_power) =
                        scale * source_coefficients_.row(
                            segment * source_num_coeffs_ + k);
                }

                base_controls.noalias() =
                    power_to_control_ * normalized_derivative;

                std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>> level;
                level.emplace_back(base_controls);
                for (int depth = 0; depth < subdivision_depth_; ++depth)
                {
                    std::vector<MatrixType, Eigen::aligned_allocator<MatrixType>> children;
                    children.reserve(level.size() * 2);
                    for (const MatrixType &parent : level)
                    {
                        children.emplace_back(left_split_ * parent);
                        children.emplace_back(right_split_ * parent);
                    }
                    level.swap(children);
                }

                for (int leaf = 0; leaf < leaves_per_segment_; ++leaf)
                {
                    const int piece_index = segment * leaves_per_segment_ + leaf;
                    controls_.middleRows(piece_index * cp, cp) = level[leaf];

                    const double begin_fraction =
                        static_cast<double>(leaf) / static_cast<double>(leaves_per_segment_);
                    const double end_fraction =
                        static_cast<double>(leaf + 1) / static_cast<double>(leaves_per_segment_);
                    pieces_.push_back(PieceInfo{
                        segment,
                        leaf,
                        begin_fraction,
                        end_fraction,
                        breakpoints[segment] + begin_fraction * duration,
                        duration / static_cast<double>(leaves_per_segment_)});
                }
            }
            initialized_ = true;
        }
    };

    template <int DIM, int ORDER>
    inline ConvexHullRepresentation<DIM>
    toBezier(const PPolyND<DIM, ORDER> &polynomial,
             int derivative_order = 0,
             int subdivision_depth = 0)
    {
        return ConvexHullRepresentation<DIM>::fromPPoly(
            polynomial, ConvexHullBasis::Bezier,
            derivative_order, subdivision_depth);
    }

    template <int DIM, int ORDER>
    inline ConvexHullRepresentation<DIM>
    toMINVO(const PPolyND<DIM, ORDER> &polynomial,
            int derivative_order = 0)
    {
        return ConvexHullRepresentation<DIM>::fromPPoly(
            polynomial, ConvexHullBasis::MINVO,
            derivative_order, 0);
    }
}

#endif
