#ifndef SELECT_HELPER_HPP
#define SELECT_HELPER_HPP

#include "catlass/layout/layout.hpp"
using namespace AscendC;
using namespace Catlass;

template <typename Layout, typename ElementType, typename = void>
struct LayoutBInitializer {
    CATLASS_DEVICE
    static Layout create(uint32_t k, uint32_t n) {
        return Layout{k, n};
    }
};

template <typename Layout, typename ElementType>
struct LayoutBInitializer<Layout, ElementType,
    std::enable_if_t<std::is_same_v<Layout, layout::zN>>
> {
    CATLASS_DEVICE
    static Layout create(uint32_t k, uint32_t n) {
        return Layout::template MakeLayout<ElementType>(k, n);
    }
};
#endif // SELECT_HELPER_HPP