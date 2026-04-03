// Shim header for standalone build.
//
// These kernel sources include `kernel_vec_intf.h` as an umbrella header,
// but CANN provides the actual vector operator interfaces under `basic_api/`.
//
// For simplicity, include the generic kernel operator interface, which pulls in
// the needed vec primitives for these kernels.

#pragma once

#include "kernel_operator.h"

