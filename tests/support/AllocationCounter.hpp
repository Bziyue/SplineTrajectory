#ifndef SPLINE_TEST_ALLOCATION_COUNTER_HPP
#define SPLINE_TEST_ALLOCATION_COUNTER_HPP
#include <atomic>
#include <cstdlib>
#include <new>

// Test-only replacement allocation functions; include in exactly one translation unit.
namespace allocation_test {
std::atomic<std::size_t> calls{0};
std::atomic<bool> enabled{false};
void begin() { calls = 0; enabled = true; }
std::size_t end() { enabled = false; return calls.load(); }
void *allocate(std::size_t size, std::size_t alignment = 0)
{
    if (enabled.load(std::memory_order_relaxed)) calls.fetch_add(1, std::memory_order_relaxed);
    void *result = nullptr;
    if (alignment == 0) result = std::malloc(size ? size : 1);
    else if (posix_memalign(&result, alignment, size ? size : 1) != 0) result = nullptr;
    if (!result) throw std::bad_alloc();
    return result;
}
}
void *operator new(std::size_t size) { return allocation_test::allocate(size); }
void *operator new[](std::size_t size) { return allocation_test::allocate(size); }
void *operator new(std::size_t size, std::align_val_t alignment)
{ return allocation_test::allocate(size, static_cast<std::size_t>(alignment)); }
void *operator new[](std::size_t size, std::align_val_t alignment)
{ return allocation_test::allocate(size, static_cast<std::size_t>(alignment)); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void *p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t, std::align_val_t) noexcept { std::free(p); }
#endif
