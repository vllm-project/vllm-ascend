#pragma once

namespace ascrip {

// Typed floor remainder: valid domain excludes zero and signed-minimum / -1.
// Correct only a nonzero remainder whose sign differs from the divisor. The
// correction stays between zero and the divisor and cannot overflow T.
//
// It is written twice because the attributes do not compose. A `__simt_vf__` body may only call
// a `__simt_callee__` function, and a `__simt_callee__` function may only be called by one, so
// a single definition carrying both is callable from SIMT and from nowhere else.
// `cce/host.py` picks the namespace from the function it is printing into, and
// helper consistency checks checks that the two bodies stay identical.
template <typename T>
__aicore__ inline T FloorMod(T a, T b)
{
    T r = a % b;
    return r != 0 && ((r < 0) != (b < 0)) ? r + b : r;
}

namespace simt {

template <typename T>
__simt_callee__ inline T FloorMod(T a, T b)
{
    T r = a % b;
    return r != 0 && ((r < 0) != (b < 0)) ? r + b : r;
}

}  // namespace simt

}  // namespace ascrip
