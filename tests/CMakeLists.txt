add_custom_target(tests)
function(spline_test name category)
    spline_executable(${name} ${category}/${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support)
    # Keep allocation assertions active; numerical failure also exercises normal Release semantics.
    if(NOT name STREQUAL "test_numerical_failure")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${name} PRIVATE -UNDEBUG)
        elseif(MSVC)
            target_compile_options(${name} PRIVATE /UNDEBUG)
        endif()
    endif()
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS "unit;${category}")
    add_dependencies(tests ${name})
endfunction()

spline_test(test_polynomial trajectory)
spline_test(test_sampler trajectory)
spline_test(test_trajectory_contracts trajectory)
spline_test(test_energy_gradients gradients)
spline_test(test_boundary_gradients gradients)
spline_test(test_cost_gradients gradients)
spline_test(test_optimizer optimizer)
spline_test(test_optimizer_contracts optimizer)
spline_test(test_numerical_failure optimizer)
spline_test(test_convex_hull convex_hull)
target_compile_definitions(test_convex_hull PRIVATE EIGEN_RUNTIME_NO_MALLOC)

find_program(SPLINE_PYTHON_EXECUTABLE NAMES python3)
if(NOT SPLINE_PYTHON_EXECUTABLE)
    message(FATAL_ERROR "Python 3 is required for compile-contract tests")
endif()
get_target_property(spline_eigen_include Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
add_test(NAME compile_contracts COMMAND ${SPLINE_PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/compile/check_contracts.py
    ${CMAKE_CXX_COMPILER} ${SplineTrajectory_SOURCE_DIR}/include ${spline_eigen_include})
set_tests_properties(compile_contracts PROPERTIES LABELS "compile")
add_test(NAME readme_quickstart COMMAND ${SPLINE_PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/compile/check_readme.py
    ${CMAKE_CXX_COMPILER} ${SplineTrajectory_SOURCE_DIR}/include ${spline_eigen_include}
    ${SplineTrajectory_SOURCE_DIR}/README.md)
set_tests_properties(readme_quickstart PROPERTIES LABELS "compile")
