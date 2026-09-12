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
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
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
     * rows. A subdivision depth s creates 2^s pieces for every source polynomial
     * segment. Both Bezier and MINVO can be subdivided; MINVO is available for
     * degrees 0 through 7.
     *
     * Topology-only conversion matrices live in a shared immutable Kernel.
     * Each object retains only its source values, control points, temporal
     * metadata, and fixed-size scratch required by update()/backwardAdd(); it
     * does not retain or sample the source trajectory.
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

        struct Kernel
        {
            ConvexHullBasis basis = ConvexHullBasis::Bezier;
            int source_num_coeffs = 0;
            int derivative_order = 0;
            int degree = 0;
            int subdivision_depth = 0;
            int leaves_per_segment = 1;
            Eigen::MatrixXd stacked_power_to_control;
            Eigen::MatrixXd stacked_control_to_power_adjoint;
            Eigen::VectorXd derivative_factors;
            Eigen::VectorXd leaf_begin_fractions;

            std::size_t memoryBytes() const
            {
                return sizeof(Kernel) +
                       static_cast<std::size_t>(
                           stacked_power_to_control.size() +
                           stacked_control_to_power_adjoint.size() +
                           derivative_factors.size() +
                           leaf_begin_fractions.size()) *
                           sizeof(double);
            }
        };

        static constexpr std::size_t kDefaultMemoryBudgetBytes =
            std::size_t{64} * 1024 * 1024;

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
            ConvexHullRepresentation result;
            result.resetTopology(
                polynomial.getNumSegments(), polynomial.getNumCoeffs(),
                basis, derivative_order, subdivision_depth);
            result.update(polynomial);
            return result;
        }

        void resetTopology(int num_source_segments,
                           int source_num_coeffs,
                           ConvexHullBasis basis,
                           int derivative_order = 0,
                           int subdivision_depth = 0,
                           std::size_t memory_budget_bytes =
                               kDefaultMemoryBudgetBytes)
        {
            if (num_source_segments <= 0 || source_num_coeffs <= 0)
                throw std::invalid_argument(
                    "Convex-hull topology requires positive segment and coefficient counts.");
            if (derivative_order < 0 ||
                derivative_order >= source_num_coeffs)
                throw std::invalid_argument(
                    "Derivative order must be smaller than the source coefficient count.");

            const auto new_kernel = acquireKernel(
                basis, source_num_coeffs, derivative_order,
                subdivision_depth, memory_budget_bytes);
            const std::size_t rows =
                checkedProduct(
                    checkedProduct(
                        static_cast<std::size_t>(num_source_segments),
                        static_cast<std::size_t>(new_kernel->leaves_per_segment)),
                    static_cast<std::size_t>(new_kernel->degree + 1));
            std::size_t workspace_bytes =
                checkedProduct(
                    checkedProduct(
                        rows, static_cast<std::size_t>(DIM)),
                    sizeof(double));
            workspace_bytes = checkedAdd(
                workspace_bytes,
                checkedProduct(
                    checkedProduct(
                        checkedProduct(
                            static_cast<std::size_t>(
                                num_source_segments),
                            static_cast<std::size_t>(
                                source_num_coeffs)),
                        static_cast<std::size_t>(DIM)),
                    sizeof(double)));
            workspace_bytes = checkedAdd(
                workspace_bytes,
                checkedProduct(
                    checkedProduct(
                        static_cast<std::size_t>(
                            num_source_segments),
                        static_cast<std::size_t>(
                            new_kernel->degree + 1)),
                    sizeof(double)));
            workspace_bytes = checkedAdd(
                workspace_bytes,
                checkedProduct(
                    checkedProduct(
                        static_cast<std::size_t>(
                            new_kernel->degree + 1),
                        static_cast<std::size_t>(2 * DIM)),
                    sizeof(double)));
            workspace_bytes = checkedAdd(
                workspace_bytes,
                checkedProduct(
                    static_cast<std::size_t>(
                        num_source_segments),
                    sizeof(double)));
            workspace_bytes = checkedAdd(
                workspace_bytes,
                checkedProduct(
                    checkedProduct(
                        static_cast<std::size_t>(
                            num_source_segments),
                        static_cast<std::size_t>(
                            new_kernel->leaves_per_segment)),
                    sizeof(PieceInfo)));
            if (new_kernel->memoryBytes() + workspace_bytes >
                memory_budget_bytes)
                throw std::length_error(
                    "Convex-hull topology exceeds its memory budget.");
            if (rows > static_cast<std::size_t>(
                           std::numeric_limits<Eigen::Index>::max()))
                throw std::length_error(
                    "Convex-hull topology is too large for Eigen indices.");

            basis_ = basis;
            derivative_order_ = derivative_order;
            degree_ = new_kernel->degree;
            subdivision_depth_ = subdivision_depth;
            leaves_per_segment_ = new_kernel->leaves_per_segment;
            num_source_segments_ = num_source_segments;
            source_num_coeffs_ = source_num_coeffs;
            kernel_ = new_kernel;

            controls_.resize(static_cast<Eigen::Index>(rows), DIM);
            source_coefficients_.resize(
                num_source_segments_ * source_num_coeffs_, DIM);
            source_durations_.resize(num_source_segments_);
            duration_powers_.resize(num_source_segments_, degree_ + 1);
            normalized_derivative_.resize(degree_ + 1, DIM);
            normalized_gradient_.resize(degree_ + 1, DIM);
            pieces_.resize(num_source_segments_ * leaves_per_segment_);
            initialized_ = false;
        }

        template <int ORDER>
        void resetTopology(const PPolyND<DIM, ORDER> &polynomial,
                           ConvexHullBasis basis,
                           int derivative_order = 0,
                           int subdivision_depth = 0,
                           std::size_t memory_budget_bytes =
                               kDefaultMemoryBudgetBytes)
        {
            if (!polynomial.isInitialized())
                throw std::invalid_argument(
                    "Convex-hull conversion requires an initialized PPolyND.");
            resetTopology(polynomial.getNumSegments(),
                          polynomial.getNumCoeffs(), basis,
                          derivative_order, subdivision_depth,
                          memory_budget_bytes);
        }

        template <int ORDER>
        void update(const PPolyND<DIM, ORDER> &polynomial)
        {
            if (!polynomial.isInitialized())
                throw std::invalid_argument(
                    "Convex-hull conversion requires an initialized PPolyND.");
            if (polynomial.getNumSegments() != num_source_segments_ ||
                polynomial.getNumCoeffs() != source_num_coeffs_ ||
                !kernel_)
                throw std::invalid_argument(
                    "PPolyND topology does not match resetTopology().");
            update(polynomial.getBreakpoints(), polynomial.getCoefficients());
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
        const std::shared_ptr<const Kernel> &kernel() const { return kernel_; }

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
            backwardAdd(control_gradients, coefficient_gradients,
                        duration_gradients);
        }

        /**
         * @brief Allocation-free additive adjoint after resetTopology().
         *
         * The destination gradients must already have the exact topology. They
         * are accumulated rather than cleared, allowing several hull costs and
         * derivative orders to share one later spline propagateGrad() call.
         */
        void backwardAdd(const MatrixType &control_gradients,
                         MatrixType &coefficient_gradients,
                         Eigen::VectorXd &duration_gradients) const
        {
            if (!initialized_)
                throw std::logic_error(
                    "Cannot backpropagate an uninitialized representation.");
            if (control_gradients.rows() != controls_.rows() ||
                control_gradients.cols() != DIM)
                throw std::invalid_argument(
                    "Control-gradient dimensions do not match controls().");
            if (coefficient_gradients.rows() !=
                    num_source_segments_ * source_num_coeffs_ ||
                coefficient_gradients.cols() != DIM ||
                duration_gradients.size() != num_source_segments_)
                throw std::invalid_argument(
                    "backwardAdd() destinations must be pre-sized.");

            const int cp = controlsPerPiece();
            for (int segment = 0; segment < num_source_segments_; ++segment)
            {
                const int first_row =
                    segment * leaves_per_segment_ * cp;
                normalized_gradient_.noalias() =
                    kernel_->stacked_control_to_power_adjoint *
                    control_gradients.middleRows(
                        first_row, leaves_per_segment_ * cp);
                const double duration = source_durations_(segment);
                for (int normalized_power = 0;
                     normalized_power <= degree_;
                     ++normalized_power)
                {
                    const int k = normalized_power + derivative_order_;
                    const double scale =
                        kernel_->derivative_factors(normalized_power) *
                        duration_powers_(segment, normalized_power);
                    coefficient_gradients.row(
                        segment * source_num_coeffs_ + k) +=
                        scale *
                        normalized_gradient_.row(normalized_power);

                    if (normalized_power > 0)
                    {
                        const auto source_coefficient =
                            source_coefficients_.row(
                                segment * source_num_coeffs_ + k);
                        duration_gradients(segment) +=
                            (static_cast<double>(normalized_power) *
                             scale / duration) *
                            normalized_gradient_.row(normalized_power)
                                .dot(source_coefficient);
                    }
                }
            }
        }

        /**
         * @brief Add gradients of piece start times and durations to independent
         * source durations and the common trajectory start time.
         */
        void backwardPieceTimesAdd(
            const Eigen::Ref<const Eigen::VectorXd> &piece_start_gradients,
            const Eigen::Ref<const Eigen::VectorXd> &piece_duration_gradients,
            Eigen::Ref<Eigen::VectorXd> source_duration_gradients,
            double &source_start_time_gradient) const
        {
            if (!initialized_ ||
                piece_start_gradients.size() != numPieces() ||
                piece_duration_gradients.size() != numPieces() ||
                source_duration_gradients.size() != num_source_segments_)
                throw std::invalid_argument(
                    "Piece-time gradient dimensions do not match the workspace.");

            double later_start_sum = 0.0;
            for (int segment = num_source_segments_ - 1;
                 segment >= 0; --segment)
            {
                double this_start_sum = 0.0;
                double local_duration_gradient = 0.0;
                for (int leaf = 0; leaf < leaves_per_segment_; ++leaf)
                {
                    const int piece =
                        segment * leaves_per_segment_ + leaf;
                    const double start_gradient =
                        piece_start_gradients(piece);
                    this_start_sum += start_gradient;
                    local_duration_gradient +=
                        kernel_->leaf_begin_fractions(leaf) *
                            start_gradient +
                        piece_duration_gradients(piece) /
                            static_cast<double>(leaves_per_segment_);
                }
                source_duration_gradients(segment) +=
                    later_start_sum + local_duration_gradient;
                later_start_sum += this_start_sum;
            }
            source_start_time_gradient += later_start_sum;
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
        Eigen::MatrixXd duration_powers_;
        mutable MatrixType normalized_gradient_;
        MatrixType normalized_derivative_;
        std::vector<PieceInfo> pieces_;
        std::shared_ptr<const Kernel> kernel_;

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

        static std::size_t checkedProduct(std::size_t a, std::size_t b)
        {
            if (a != 0 &&
                b > std::numeric_limits<std::size_t>::max() / a)
                throw std::length_error(
                    "Convex-hull topology size overflow.");
            return a * b;
        }

        static std::size_t checkedAdd(std::size_t a, std::size_t b)
        {
            if (b > std::numeric_limits<std::size_t>::max() - a)
                throw std::length_error(
                    "Convex-hull topology size overflow.");
            return a + b;
        }

        static std::uint64_t kernelKey(ConvexHullBasis basis,
                                       int source_num_coeffs,
                                       int derivative_order,
                                       int subdivision_depth)
        {
            return (static_cast<std::uint64_t>(basis) << 56) |
                   (static_cast<std::uint64_t>(source_num_coeffs) << 40) |
                   (static_cast<std::uint64_t>(derivative_order) << 24) |
                   static_cast<std::uint64_t>(subdivision_depth);
        }

        static std::shared_ptr<const Kernel> buildKernel(
            ConvexHullBasis basis,
            int source_num_coeffs,
            int derivative_order,
            int subdivision_depth,
            std::size_t memory_budget_bytes)
        {
            if (subdivision_depth < 0)
                throw std::invalid_argument(
                    "Subdivision depth must be non-negative.");
            if (subdivision_depth >=
                std::numeric_limits<int>::digits - 1)
                throw std::length_error(
                    "Subdivision depth exceeds integer index capacity.");

            const int degree =
                source_num_coeffs - derivative_order - 1;
            if (basis == ConvexHullBasis::MINVO && degree > 7)
                throw std::invalid_argument(
                    "Exact MINVO matrices are available for polynomial degrees 0 through 7.");

            const int cp = degree + 1;
            const int leaves = int{1} << subdivision_depth;
            const std::size_t stacked_rows =
                checkedProduct(static_cast<std::size_t>(leaves),
                               static_cast<std::size_t>(cp));
            const std::size_t matrix_elements =
                checkedProduct(stacked_rows,
                               static_cast<std::size_t>(cp));
            std::size_t kernel_bytes = sizeof(Kernel);
            kernel_bytes = checkedAdd(
                kernel_bytes,
                checkedProduct(
                    checkedProduct(
                        matrix_elements, std::size_t{2}),
                    sizeof(double)));
            kernel_bytes = checkedAdd(
                kernel_bytes,
                checkedProduct(
                    checkedAdd(
                        static_cast<std::size_t>(cp),
                        static_cast<std::size_t>(leaves)),
                    sizeof(double)));
            if (kernel_bytes > memory_budget_bytes)
                throw std::length_error(
                    "Convex-hull kernel exceeds its memory budget.");

            auto kernel = std::make_shared<Kernel>();
            kernel->basis = basis;
            kernel->source_num_coeffs = source_num_coeffs;
            kernel->derivative_order = derivative_order;
            kernel->degree = degree;
            kernel->subdivision_depth = subdivision_depth;
            kernel->leaves_per_segment = leaves;
            kernel->stacked_power_to_control.resize(
                static_cast<Eigen::Index>(stacked_rows), cp);
            kernel->derivative_factors.resize(cp);
            kernel->leaf_begin_fractions.resize(leaves);

            const Eigen::MatrixXd basis_matrix =
                powerToControlMatrix(basis, degree);
            Eigen::MatrixXd restriction(cp, cp);
            std::vector<long double> a_powers(cp, 1.0L);
            std::vector<long double> h_powers(cp, 1.0L);
            for (int leaf = 0; leaf < leaves; ++leaf)
            {
                const long double a =
                    static_cast<long double>(leaf) /
                    static_cast<long double>(leaves);
                const long double h =
                    1.0L / static_cast<long double>(leaves);
                kernel->leaf_begin_fractions(leaf) =
                    static_cast<double>(a);
                for (int power = 1; power <= degree; ++power)
                {
                    a_powers[power] =
                        a_powers[power - 1] * a;
                    h_powers[power] =
                        h_powers[power - 1] * h;
                }
                restriction.setZero();
                for (int k = 0; k <= degree; ++k)
                {
                    for (int j = 0; j <= k; ++j)
                    {
                        restriction(j, k) =
                            static_cast<double>(
                                static_cast<long double>(binomial(k, j)) *
                                a_powers[k - j] *
                                h_powers[j]);
                    }
                }
                kernel->stacked_power_to_control.middleRows(
                    leaf * cp, cp).noalias() =
                    basis_matrix * restriction;
            }
            kernel->stacked_control_to_power_adjoint =
                kernel->stacked_power_to_control.transpose();
            for (int power = 0; power <= degree; ++power)
                kernel->derivative_factors(power) =
                    fallingFactorial(
                        power + derivative_order,
                        derivative_order);
            return kernel;
        }

        static std::shared_ptr<const Kernel> acquireKernel(
            ConvexHullBasis basis,
            int source_num_coeffs,
            int derivative_order,
            int subdivision_depth,
            std::size_t memory_budget_bytes)
        {
            using Cache =
                std::unordered_map<std::uint64_t,
                                   std::weak_ptr<const Kernel>>;
            static Cache cache;
            static std::mutex cache_mutex;
            const std::uint64_t key =
                kernelKey(basis, source_num_coeffs,
                          derivative_order, subdivision_depth);

            std::lock_guard<std::mutex> lock(cache_mutex);
            const auto found = cache.find(key);
            if (found != cache.end())
            {
                if (auto kernel = found->second.lock())
                {
                    if (kernel->memoryBytes() > memory_budget_bytes)
                        throw std::length_error(
                            "Cached convex-hull kernel exceeds the requested memory budget.");
                    return kernel;
                }
            }
            auto kernel = buildKernel(
                basis, source_num_coeffs, derivative_order,
                subdivision_depth, memory_budget_bytes);
            cache[key] = kernel;
            return kernel;
        }

        void update(const std::vector<double> &breakpoints,
                    const MatrixType &coefficients)
        {
            if (!kernel_ ||
                static_cast<int>(breakpoints.size()) !=
                    num_source_segments_ + 1 ||
                coefficients.rows() !=
                    num_source_segments_ * source_num_coeffs_ ||
                coefficients.cols() != DIM)
                throw std::invalid_argument(
                    "Convex-hull update dimensions do not match resetTopology().");

            source_coefficients_ = coefficients;
            const int cp = controlsPerPiece();
            for (int segment = 0; segment < num_source_segments_;
                 ++segment)
            {
                const double duration =
                    breakpoints[segment + 1] - breakpoints[segment];
                if (!(duration > 0.0) || !std::isfinite(duration))
                    throw std::invalid_argument(
                        "PPolyND breakpoints must have finite, strictly positive durations.");
                source_durations_(segment) = duration;
                duration_powers_(segment, 0) = 1.0;
                for (int power = 1; power <= degree_; ++power)
                    duration_powers_(segment, power) =
                        duration_powers_(segment, power - 1) *
                        duration;

                for (int power = 0; power <= degree_; ++power)
                {
                    const int source_power =
                        power + derivative_order_;
                    normalized_derivative_.row(power) =
                        kernel_->derivative_factors(power) *
                        duration_powers_(segment, power) *
                        source_coefficients_.row(
                            segment * source_num_coeffs_ +
                            source_power);
                }

                const int first_row =
                    segment * leaves_per_segment_ * cp;
                controls_.middleRows(
                    first_row,
                    leaves_per_segment_ * cp).noalias() =
                    kernel_->stacked_power_to_control *
                    normalized_derivative_;

                for (int leaf = 0; leaf < leaves_per_segment_;
                     ++leaf)
                {
                    const int piece_index =
                        segment * leaves_per_segment_ + leaf;
                    const double begin_fraction =
                        kernel_->leaf_begin_fractions(leaf);
                    pieces_[piece_index] = PieceInfo{
                        segment,
                        leaf,
                        begin_fraction,
                        static_cast<double>(leaf + 1) /
                            static_cast<double>(leaves_per_segment_),
                        breakpoints[segment] +
                            begin_fraction * duration,
                        duration /
                            static_cast<double>(leaves_per_segment_)};
                }
            }
            initialized_ = true;
        }
    };

    template <int DIM>
    using ConvexHullWorkspace = ConvexHullRepresentation<DIM>;

    template <int DIM>
    using ConvexHullKernel =
        typename ConvexHullRepresentation<DIM>::Kernel;

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
            int derivative_order = 0,
            int subdivision_depth = 0)
    {
        return ConvexHullRepresentation<DIM>::fromPPoly(
            polynomial, ConvexHullBasis::MINVO,
            derivative_order, subdivision_depth);
    }
}

#endif
