#pragma once
#include <iostream>
#include <stdexcept>

namespace spline_test
{
inline void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class Exception, class Function>
void throws(Function function)
{
    try { function(); }
    catch (const Exception &) { return; }
    throw std::runtime_error("Expected exception was not thrown");
}

/** @brief Run a test suite with consistent exit codes and failure diagnostics. */
template<class Function>
int run(const char *name, Function function)
{
    try
    {
        function();
        std::cout << name << ": passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << name << ": " << error.what() << '\n';
        return 1;
    }
}
}
