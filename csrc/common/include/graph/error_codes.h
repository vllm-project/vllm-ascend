#pragma once

// OpsBase 9.2 includes graph/error_codes.h, while Metadef 9.1 provides the
// same definitions through graph/ge_error_codes.h. Prefer the legacy header
// when it is available so mixed CANN installations do not recurse through a
// duplicate include path; otherwise forward to the active Metadef header.
#if defined(__has_include)
#if __has_include("graph/ge_error_codes.h")
#include "graph/ge_error_codes.h"
#elif defined(__has_include_next)
#if __has_include_next("graph/error_codes.h")
#include_next "graph/error_codes.h"
#else
#error "No compatible CANN graph error codes header was found"
#endif
#else
#error "No compatible CANN graph error codes header was found"
#endif
#else
#include "graph/ge_error_codes.h"
#endif
