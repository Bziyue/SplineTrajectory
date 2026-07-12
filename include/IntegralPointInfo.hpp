#pragma once

#include <type_traits>

namespace SplineTrajectory
{
struct IntegralPointInfo final
{
    int segment_index = 0;
    int segment_count = 0;
    int step_index = 0;
    int step_count = 0;
    double alpha = 0.0;
    double segment_duration = 0.0;
    double step_size = 0.0;
    double local_time = 0.0;
    double global_time = 0.0;

    constexpr bool isSegmentStart() const noexcept { return step_index == 0; }
    constexpr bool isSegmentEnd() const noexcept { return step_index == step_count; }
    constexpr bool isTrajectoryStart() const noexcept
    {
        return segment_index == 0 && isSegmentStart();
    }
    constexpr bool isTrajectoryEnd() const noexcept
    {
        return segment_index + 1 == segment_count && isSegmentEnd();
    }
    constexpr int interiorBoundaryIndex() const noexcept
    {
        if (isSegmentEnd() && segment_index + 1 < segment_count)
        {
            return segment_index;
        }
        if (isSegmentStart() && segment_index > 0)
        {
            return segment_index - 1;
        }
        return -1;
    }
};

static_assert(std::is_standard_layout<IntegralPointInfo>::value,
              "IntegralPointInfo must remain a plain value type");
static_assert(std::is_trivially_copyable<IntegralPointInfo>::value,
              "IntegralPointInfo must remain cheap to pass and optimize");
} // namespace SplineTrajectory
