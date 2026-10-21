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

/** @file SplineConvexHull.hpp
 * @brief Shared immutable Bezier/MINVO kernels and exclusive differentiable hull workspaces. */

#ifndef SPLINE_CONVEX_HULL_HPP
#define SPLINE_CONVEX_HULL_HPP

#include "SplineTrajectory.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <map>
#include <array>
#include <vector>

namespace SplineTrajectory
{
    enum class ConvexHullBasis
    {
        Bezier,
        MINVO
    };

    /** @brief Topology-only conversion data, shared as const across dimensions and workspaces. */
    struct ConvexHullKernel
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


    };


    /** @brief Conversion basis, physical derivative order and binary subdivision depth. */
    struct HullOptions
    {
        ConvexHullBasis basis = ConvexHullBasis::Bezier;
        int derivative_order = 0;
        int subdivision_depth = 0;
    };

    namespace detail
    {
        struct HullKernelFactory
        {
            /** @brief Build the configured basis transform from ascending powers.
             * @param basis Supported Bezier or MINVO basis.
             * @param degree Supported nonnegative degree with representable conversion arithmetic.
             * @return Owned conversion matrix; supported degree and allocation size are caller preconditions.
             * @note Reentrant; allocation exceptions propagate normally. */
            static Eigen::MatrixXd powerToControlMatrix(ConvexHullBasis basis, int degree)
            {
                Eigen::MatrixXd result = basis == ConvexHullBasis::Bezier ?
                    bezierPowerToControlMatrix(degree) : minvoPowerToControlMatrix(degree);
                return result;
            }

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
                assert(false && "MINVO degree must be in [0, 7]");
                break;
            }
            return b;
        }

            using KernelKey = std::array<int, 4>;
            static KernelKey kernelKey(ConvexHullBasis basis, int count, int derivative, int depth)
            { return {static_cast<int>(basis), count, derivative, depth}; }

            /** @brief Precompute an immutable normalized-power conversion for trusted topology.
             * @param basis Supported conversion basis. @param source_num_coeffs Source coefficient count.
             * @param derivative_order Physical derivative order below the source coefficient count.
             * @param subdivision_depth Binary depth in [0, 16], asserted before shifting.
             * @return Shared owning kernel; allocation sizes must fit memory and index capacity. */
            static std::shared_ptr<const ConvexHullKernel> buildKernel(
                ConvexHullBasis basis,
                int source_num_coeffs,
                int derivative_order,
                int subdivision_depth)
            {
                assert(subdivision_depth >= 0 && subdivision_depth <= 16);
                const int degree = source_num_coeffs - derivative_order - 1;
                const int cp = degree + 1;
                const int leaves = int{1} << subdivision_depth;
                const Eigen::Index stacked_rows = Eigen::Index(leaves) * cp;

                auto kernel = std::make_shared<ConvexHullKernel>();
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

                const Eigen::MatrixXd basis_matrix = powerToControlMatrix(basis, degree);
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
                return std::shared_ptr<const ConvexHullKernel>(std::move(kernel));
            }

            /** @brief Reuse or construct the immutable conversion for a trusted topology.
             * @param basis Supported conversion basis. @param source_num_coeffs Source coefficient count.
             * @param derivative_order Physical derivative order. @param subdivision_depth Binary depth in [0, 16].
             * @return Shared owning kernel; cache access is synchronized and does not retain expired workspaces. */
            static std::shared_ptr<const ConvexHullKernel> acquireKernel(
                ConvexHullBasis basis,
                int source_num_coeffs,
                int derivative_order,
                int subdivision_depth)
            {
                using Cache =
                    std::map<KernelKey,
                                       std::weak_ptr<const ConvexHullKernel>>;
                static Cache cache;
                static std::mutex cache_mutex;
                const KernelKey key =
                    kernelKey(basis, source_num_coeffs,
                              derivative_order, subdivision_depth);

                std::lock_guard<std::mutex> lock(cache_mutex);
                const auto found = cache.find(key);
                if (found != cache.end())
                {
                    if (auto kernel = found->second.lock())
                        return kernel;
                }
                auto kernel = buildKernel(
                    basis, source_num_coeffs, derivative_order,
                    subdivision_depth);
                cache[key] = kernel;
                return kernel;
            }

        };
    }

    /**
     * @brief Convex-hull control-point representation of a PPolyND or one of its
     *        physical-time derivatives.
     *
     * Rows in controls() are piece-major. Every piece owns degree()+1 consecutive
     * rows. A subdivision depth s creates 2^s pieces for every source polynomial
     * segment. Both Bezier and MINVO can be subdivided; MINVO is available for
     * degrees 0 through 7.
     *
     * Topology-only conversion matrices live in a shared immutable ConvexHullKernel.
     * Each object retains only its source values, control points, temporal
     * metadata, and preallocated scratch required by update()/backwardAdd(); it
     * does not retain or sample the source trajectory.
     * Callers guarantee valid input polynomials, supported options, finite partials and matching
     * pre-sized input/output shapes, within a representable numerical working range.
     * These preconditions are not checked; controls and gradients are not scanned for overflow.
     * Configuration and factories trust topology and allocate directly. A configured workspace is invalid until updated.
     * Update requires a configured workspace; adjoints require updated controls. These state
     * preconditions are asserted in debug builds; methods return values or void directly.
     * Fixed-topology update/adjoint does not allocate; allocation exceptions during preparation
     * remain standard allocation failures.
     */
    template <int DIM>
    class ConvexHullWorkspace
    {
    public:
        static constexpr int kMatrixOptions = (DIM == 1) ? Eigen::ColMajor : Eigen::RowMajor;
        using Vector = Eigen::Matrix<double, DIM, 1>;
        using RowVector = Eigen::Matrix<double, 1, DIM>;
        using CoefficientMatrix = Eigen::Matrix<double, Eigen::Dynamic, DIM, kMatrixOptions>;

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
            CoefficientMatrix coefficients;
            Eigen::VectorXd durations;
        };

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ConvexHullWorkspace() = default;

        /** @brief Configure and update an owning hull workspace from one polynomial.
         * @param polynomial Valid source in physical time; values are copied, not borrowed.
         * @param options Conversion basis, physical derivative order and subdivision depth.
         * @return Independent mutable workspace with a shared immutable conversion kernel.
         * @pre Supported options and topology within available memory and index capacity.
         * @note Allocates and synchronizes the kernel cache; allocation failures propagate normally. */
        template <int MaxCoefficients>
        static ConvexHullWorkspace fromPPoly(
            const PPolyND<DIM, MaxCoefficients> &polynomial,
            HullOptions options = {})
        {
            ConvexHullWorkspace result;
            result.configure(polynomial, options);
            result.update(polynomial.breakpoints(), polynomial.coefficients());
            return result;
        }

        /** @brief Allocate a fixed topology and acquire its immutable conversion kernel.
         * @param num_source_segments Positive source segment count.
         * @param source_num_coeffs Positive source coefficient count per segment.
         * @param options Basis, physical derivative order and binary subdivision depth.
         * @note Topology sizes must fit supported indices and available memory.
         * @pre Positive counts, a supported basis, 0 <= derivative_order < source_num_coeffs,
         *      subdivision depth in [0, 16], and MINVO degree <= 7; resulting row counts fit int indices.
         * @note Successful configuration leaves isConfigured() true and isValid() false. */
        void configure(int num_source_segments, int source_num_coeffs, HullOptions options = {})
        {
            // Topology is caller-validated; configuration owns all allocation and cache lookup.
            initialized_ = false;
            kernel_ = detail::HullKernelFactory::acquireKernel(options.basis, source_num_coeffs,
                options.derivative_order, options.subdivision_depth);
            num_source_segments_ = num_source_segments;
            const int cp = kernel_->degree + 1;
            const Eigen::Index rows = Eigen::Index(num_source_segments_) * kernel_->stacked_power_to_control.rows();
            controls_.resize(rows, DIM);
            source_durations_.resize(num_source_segments_);
            duration_powers_.resize(num_source_segments_, cp);
            normalized_derivative_.resize(Eigen::Index(num_source_segments_) * cp, DIM);
            normalized_gradient_.resize(cp, DIM);
            pieces_.resize(static_cast<std::size_t>(num_source_segments_) * kernel_->leaves_per_segment);

            // Segment ownership and normalized intervals depend only on the configured topology.
            for (int segment = 0; segment < num_source_segments_; ++segment)
                for (int leaf = 0; leaf < kernel_->leaves_per_segment; ++leaf)
                {
                    auto &piece = pieces_[segment * kernel_->leaves_per_segment + leaf];
                    piece = PieceInfo{segment, leaf, kernel_->leaf_begin_fractions(leaf),
                                      static_cast<double>(leaf + 1) / kernel_->leaves_per_segment,
                                      0.0, 0.0};
                }
        }

        /** @brief Update numeric data with fixed topology, without allocation or locking.
         * @param polynomial Valid source matching the configured topology; data are copied.
         * @pre The workspace is configured; the caller guarantees source validity and matching topology.
         * @note Requires exclusive workspace access, as does backwardAdd(). */
        template <int MaxCoefficients>
        void update(const PPolyND<DIM, MaxCoefficients> &polynomial)
        {
            initialized_ = false;
            assert(isConfigured());
            update(polynomial.breakpoints(), polynomial.coefficients());
        }

        /** @brief True after topology configuration, even before the first numeric update. */
        bool isConfigured() const { return bool(kernel_); }
        /** @brief True after numeric update; arithmetic range is guaranteed by the caller. */
        bool isValid() const { return kernel_ && initialized_; }
        /** @brief Compare topology and conversion options without allocating or locking. */
        template <int MaxCoefficients>
        bool matches(const PPolyND<DIM, MaxCoefficients> &polynomial, HullOptions options) const
        {
            return kernel_ && polynomial.isValid() && polynomial.numSegments() == num_source_segments_ &&
                polynomial.coefficientCount() == kernel_->source_num_coeffs &&
                options.basis == basis() && options.derivative_order == derivativeOrder() &&
                options.subdivision_depth == subdivisionDepth();
        }

        /** @brief Configure storage; may allocate and synchronize the shared kernel cache.
         * @param polynomial Valid source; only its topology is read during this call.
         * @param options Supported basis, derivative order and subdivision depth.
         * @note Topology sizes must fit supported indices and available memory.
         * @note Requires exclusive workspace access; the source is neither retained nor changed. */
        template <int MaxCoefficients>
        void configure(const PPolyND<DIM, MaxCoefficients> &polynomial, HullOptions options = {})
        {
            configure(polynomial.numSegments(), polynomial.coefficientCount(), options);
        }
        ConvexHullBasis basis() const { return kernel_ ? kernel_->basis : ConvexHullBasis::Bezier; }
        int derivativeOrder() const { return kernel_ ? kernel_->derivative_order : 0; }
        int degree() const { return kernel_ ? kernel_->degree : 0; }
        int sourceDegree() const { return kernel_ ? kernel_->source_num_coeffs - 1 : -1; }
        int subdivisionDepth() const { return kernel_ ? kernel_->subdivision_depth : 0; }
        int piecesPerSegment() const { return kernel_ ? kernel_->leaves_per_segment : 0; }
        int numSourceSegments() const { return kernel_ ? num_source_segments_ : 0; }
        int numPieces() const { return static_cast<int>(pieces_.size()); }
        int controlsPerPiece() const { return kernel_ ? kernel_->degree + 1 : 0; }

        const CoefficientMatrix &controls() const { return controls_; }
        const std::vector<PieceInfo> &pieces() const { return pieces_; }
        /** @brief Borrow metadata for a configured piece.
         * @param piece_index Index in [0, numPieces()). @return Borrow valid until reconfiguration.
         * @pre The workspace is configured and the index is in range. */
        const PieceInfo &pieceInfo(int piece_index) const
        { assert(isConfigured()); return pieces_[piece_index]; }
        const std::shared_ptr<const ConvexHullKernel> &kernel() const { return kernel_; }

        /** @brief Borrow controls of a valid piece without copying.
         * @param piece_index Index in [0, numPieces()). @return View valid until workspace mutation.
         * @pre The latest update succeeded; the piece index is in range. */
        auto pieceControls(int piece_index) const
        {
            assert(isValid());
            return controls_.middleRows(piece_index * controlsPerPiece(), controlsPerPiece());
        }

        /** @brief Allocate and return source coefficient and independent-duration partials.
         * @param control_gradients Finite partials matching the piece-major rows of controls().
         * @return Owned gradients; durations holds local physical-time coefficients fixed.
         * @pre The workspace contains updated controls.
         * @note Requires exclusive workspace access. Pass both outputs to the spline adjoint. */
        BackwardResult backward(const CoefficientMatrix &control_gradients)
        {
            assert(isValid());
            BackwardResult result;
            result.coefficients = CoefficientMatrix::Zero(
                num_source_segments_ * kernel_->source_num_coeffs, DIM);
            result.durations = Eigen::VectorXd::Zero(num_source_segments_);
            accumulateControlGradients(control_gradients, result.coefficients, result.durations);
            return result;
        }

        /** @brief Overwrite pre-sized source partials without allocating or resizing.
         * @param control_gradients Finite partials matching the piece-major rows of controls().
         * @param[out] coefficient_gradients Source coefficient-shaped output in caller-owned storage.
         * @param[out] duration_gradients One independent physical-duration partial per source segment.
         * @pre The workspace contains updated controls and the caller supplies matching shapes.
         * @note Inputs/outputs must not alias each other or
         *       workspace storage. Requires exclusive workspace and output access. */
        void backwardInto(const CoefficientMatrix &control_gradients,
                          CoefficientMatrix &coefficient_gradients,
                          Eigen::VectorXd &duration_gradients)
        {
            assert(isValid());

            coefficient_gradients.setZero();
            duration_gradients.setZero();
            accumulateControlGradients(control_gradients, coefficient_gradients, duration_gradients);
        }

        /**
         * @brief Allocation-free additive adjoint after updating a configured workspace.
         *
         * The destination gradients must already have the exact topology. They
         * are accumulated rather than cleared, allowing several hull costs and
         * derivative orders to share one later spline backward() call.
         * @param control_gradients Finite partials matching the piece-major rows of controls().
         * @param[in,out] coefficient_gradients Source local-power partials, accumulated in place.
         * @param[in,out] duration_gradients Independent time partials at fixed source coefficients.
         * @pre The workspace contains updated controls and the caller supplies matching shapes.
         * @note Inputs/outputs must not alias each other or
         *       workspace storage. Mutates scratch storage; requires exclusive workspace/output access.
         */
        void backwardAdd(const CoefficientMatrix &control_gradients,
                         Eigen::Ref<CoefficientMatrix> coefficient_gradients,
                         Eigen::Ref<Eigen::VectorXd> duration_gradients)
        {
            assert(isValid());
            accumulateControlGradients(control_gradients, coefficient_gradients, duration_gradients);
        }

        /**
         * @brief Add gradients of piece start times and durations to independent
         * source durations and the common trajectory start time.
         * @param piece_start_gradients Finite partials, one per hull piece, per second.
         * @param piece_duration_gradients Finite duration partials, one per hull piece, per second.
         * @param[in,out] source_duration_gradients Finite accumulated partials, one per source segment.
         * @param[in,out] source_start_time_gradient Finite accumulated common-start partial, per second.
         * @pre Caller guarantees matching shapes and exclusive output access.
         * @pre The workspace contains updated controls and piece times. */
        void backwardPieceTimesAdd(
            const Eigen::Ref<const Eigen::VectorXd> &piece_start_gradients,
            const Eigen::Ref<const Eigen::VectorXd> &piece_duration_gradients,
            Eigen::Ref<Eigen::VectorXd> source_duration_gradients,
            double &source_start_time_gradient) const
        {
            assert(isValid());

            const double inv_leaves = 1.0 / kernel_->leaves_per_segment;
            double later_start_sum = 0.0;
            for (int segment = num_source_segments_ - 1;
                 segment >= 0; --segment)
            {
                double this_start_sum = 0.0;
                double local_duration_gradient = 0.0;
                for (int leaf = 0; leaf < kernel_->leaves_per_segment; ++leaf)
                {
                    const int piece =
                        segment * kernel_->leaves_per_segment + leaf;
                    const double start_gradient =
                        piece_start_gradients(piece);
                    this_start_sum += start_gradient;
                    local_duration_gradient +=
                        kernel_->leaf_begin_fractions(leaf) *
                            start_gradient +
                        piece_duration_gradients(piece) * inv_leaves;
                }
                source_duration_gradients(segment) +=
                    later_start_sum + local_duration_gradient;
                later_start_sum += this_start_sum;
            }
            source_start_time_gradient += later_start_sum;
        }

        /**
         * @brief Matrix B such that control = B * normalized_ascending_power.
         * @param basis Supported Bezier or MINVO basis.
         * @param degree Nonnegative polynomial degree; MINVO requires degree <= 7.
         * @return Owned conversion matrix; degree and allocation size are caller preconditions. */
        static Eigen::MatrixXd powerToControlMatrix(ConvexHullBasis basis, int degree)
        { return detail::HullKernelFactory::powerToControlMatrix(basis, degree); }

    private:
        int num_source_segments_{0};
        bool initialized_{false};

        CoefficientMatrix controls_;
        Eigen::VectorXd source_durations_;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> duration_powers_;
        CoefficientMatrix normalized_gradient_;
        CoefficientMatrix normalized_derivative_;
        std::vector<PieceInfo> pieces_;
        std::shared_ptr<const ConvexHullKernel> kernel_;

        /** @brief Accumulate the hull adjoint using caller-validated shapes and finite partials.
         * @param control_gradients Borrowed piece-major input partials.
         * @param[in,out] coefficient_gradients Additive local physical-time coefficient partials.
         * @param[in,out] duration_gradients Additive duration partials at fixed source coefficients.
         * @pre The workspace is valid and all arithmetic stays within the caller's working range.
         * @note Uses exclusive scratch storage; no allocation occurs and buffers must not alias. */
        void accumulateControlGradients(const CoefficientMatrix &control_gradients,
                                        Eigen::Ref<CoefficientMatrix> coefficient_gradients,
                                        Eigen::Ref<Eigen::VectorXd> duration_gradients)
        {
            const int cp = controlsPerPiece();
            for (int segment = 0; segment < num_source_segments_; ++segment)
            {
                const int first_row =
                    segment * kernel_->leaves_per_segment * cp;
                normalized_gradient_.noalias() =
                    kernel_->stacked_control_to_power_adjoint *
                    control_gradients.middleRows(
                        first_row, kernel_->leaves_per_segment * cp);
                const double duration = source_durations_(segment);
                for (int normalized_power = 0;
                     normalized_power <= kernel_->degree;
                     ++normalized_power)
                {
                    const int k = normalized_power + kernel_->derivative_order;
                    const double scale =
                        kernel_->derivative_factors(normalized_power) *
                        duration_powers_(segment, normalized_power);
                    coefficient_gradients.row(
                        segment * kernel_->source_num_coeffs + k) +=
                        scale *
                        normalized_gradient_.row(normalized_power);

                    if (normalized_power > 0)
                    {
                        duration_gradients(segment) +=
                            (static_cast<double>(normalized_power) / duration) *
                            normalized_gradient_.row(normalized_power).dot(
                                normalized_derivative_.row(segment * cp + normalized_power));
                    }
                }
            }
        }

        /** @brief Refresh hull controls and adjoint caches at the configured topology.
         * @param breakpoints Strictly increasing physical times, one per segment boundary.
         * @param coefficients Borrowed local-power coefficients; normalized derivatives are retained for the adjoint.
         * @pre The workspace is configured and the source matches its topology and numerical range.
         * @note Requires exclusive workspace access; fixed-topology updates do not allocate. */
        void update(const std::vector<double> &breakpoints,
                    const CoefficientMatrix &coefficients)
        {
            const int cp = controlsPerPiece();
            for (int segment = 0; segment < num_source_segments_;
                 ++segment)
            {
                const double duration =
                    breakpoints[segment + 1] - breakpoints[segment];
                source_durations_(segment) = duration;
                duration_powers_(segment, 0) = 1.0;
                for (int power = 1; power <= kernel_->degree; ++power)
                    duration_powers_(segment, power) =
                        duration_powers_(segment, power - 1) *
                        duration;

                // Cache the normalized derivative once for both controls and the time adjoint.
                auto normalized = normalized_derivative_.middleRows(segment * cp, cp);
                for (int power = 0; power < cp; ++power)
                    normalized.row(power) = kernel_->derivative_factors(power) *
                        duration_powers_(segment, power) *
                        coefficients.row(segment * kernel_->source_num_coeffs + power + kernel_->derivative_order);

                // Accumulate two adjacent controls in registers. This small, dimension-independent
                // block avoids packed-matrix overhead and preserves each control's summation order.
                const auto &transform = kernel_->stacked_power_to_control;
                const int control_count = kernel_->leaves_per_segment * cp;
                const int first_row = segment * control_count;
                int row = 0;
                for (; row + 1 < control_count; row += 2)
                {
                    Eigen::Matrix<double, 2, DIM> pair = Eigen::Matrix<double, 2, DIM>::Zero();
                    for (int power = 0; power < cp; ++power)
                        pair.noalias() += transform.template block<2, 1>(row, power) * normalized.row(power);
                    controls_.template middleRows<2>(first_row + row) = pair;
                }
                if (row < control_count)
                {
                    RowVector control = RowVector::Zero();
                    for (int power = 0; power < cp; ++power)
                        control += transform(row, power) * normalized.row(power);
                    controls_.row(first_row + row) = control;
                }

                const double piece_duration = duration / kernel_->leaves_per_segment;
                for (int leaf = 0; leaf < kernel_->leaves_per_segment;
                     ++leaf)
                {
                    auto &piece = pieces_[segment * kernel_->leaves_per_segment + leaf];
                    piece.start_time = breakpoints[segment] + piece.source_fraction_begin * duration;
                    piece.duration = piece_duration;
                }
            }
            initialized_ = true;
        }
    };

    /** @brief Create an owning Bezier hull for one polynomial or its physical derivative.
     * @param polynomial Valid source; copied values do not borrow its lifetime.
     * @param derivative_order Physical derivative order, between zero and the source degree.
     * @param subdivision_depth Binary subdivision depth in [0, 16].
     * @return Configured, updated workspace; may allocate; topology and allocation size are caller preconditions. */
    template <int DIM, int MaxCoefficients>
    inline ConvexHullWorkspace<DIM>
    toBezier(const PPolyND<DIM, MaxCoefficients> &polynomial,
             int derivative_order = 0,
             int subdivision_depth = 0)
    {
        return ConvexHullWorkspace<DIM>::fromPPoly(
            polynomial, {ConvexHullBasis::Bezier, derivative_order, subdivision_depth});
    }

    /** @brief Create an owning MINVO hull for a polynomial or derivative of degree zero through seven.
     * @param polynomial Valid source; copied values do not borrow its lifetime.
     * @param derivative_order Physical derivative order, between zero and the source degree.
     * @param subdivision_depth Binary subdivision depth in [0, 16].
     * @return Configured, updated workspace; may allocate; topology and allocation size are caller preconditions. */
    template <int DIM, int MaxCoefficients>
    inline ConvexHullWorkspace<DIM>
    toMINVO(const PPolyND<DIM, MaxCoefficients> &polynomial,
            int derivative_order = 0,
            int subdivision_depth = 0)
    {
        return ConvexHullWorkspace<DIM>::fromPPoly(
            polynomial, {ConvexHullBasis::MINVO, derivative_order, subdivision_depth});
    }
}

#endif
