# Standalone frozen planner kernels. ROS and downstream checkouts are not required.
find_program(SPLINE_PLANNER_PYTHON NAMES python3)
if(NOT SPLINE_PLANNER_PYTHON)
    message(FATAL_ERROR "Python 3 is required to verify and bind the frozen planner sources")
endif()
set(planner_generated ${CMAKE_CURRENT_BINARY_DIR}/upstream)
execute_process(COMMAND ${SPLINE_PLANNER_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/prepare_upstream.py
    ${CMAKE_CURRENT_SOURCE_DIR}/upstream ${planner_generated}
    RESULT_VARIABLE planner_extract_result)
if(NOT planner_extract_result EQUAL 0)
    message(FATAL_ERROR "Frozen planner source verification/extraction failed")
endif()
file(GLOB_RECURSE planner_frozen_files ${CMAKE_CURRENT_SOURCE_DIR}/upstream/*)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${planner_frozen_files}
    ${CMAKE_CURRENT_SOURCE_DIR}/prepare_upstream.py
    ${CMAKE_CURRENT_SOURCE_DIR}/upstream/manifest.json)

add_library(planner_super_native STATIC
    ${planner_generated}/super/minco.cpp
    ${planner_generated}/super/banded_system.cpp
    ${planner_generated}/super/lbfgs.cpp)
target_link_libraries(planner_super_native PRIVATE Eigen3::Eigen)
set_target_properties(planner_super_native PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_include_directories(planner_super_native SYSTEM PRIVATE ${planner_generated}/super)
# Compile original translation units without treating upstream diagnostics as owned warnings.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(planner_super_native PRIVATE -fno-fast-math)
endif()

foreach(planner gcopter super)
    spline_executable(benchmark_${planner}_planner planner_benchmark.cpp)
    set_target_properties(benchmark_${planner}_planner PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    target_include_directories(benchmark_${planner}_planner SYSTEM BEFORE PRIVATE ${planner_generated}/${planner})
    if(planner STREQUAL "gcopter")
        target_compile_definitions(benchmark_${planner}_planner PRIVATE PLANNER_GCOPTER=1)
    else()
        target_compile_definitions(benchmark_${planner}_planner PRIVATE PLANNER_GCOPTER=0)
        target_link_libraries(benchmark_${planner}_planner PRIVATE planner_super_native)
    endif()
    if(BUILD_TESTING)
        add_test(NAME planner_${planner}_equivalence COMMAND benchmark_${planner}_planner
            --data ${CMAKE_CURRENT_SOURCE_DIR}/data
            --output ${CMAKE_CURRENT_BINARY_DIR}/checks/${planner}
            --validate-only --limit-per-file 1)
    endif()
endforeach()
