// Shim header for standalone build.
//
// The chunk_gated_delta_rule kernel includes `kernel_cube_intf.h` as an umbrella header.
// CANN exposes equivalent cube/matrix operator interfaces via `kernel_operator.h` and friends,
// so we re-route to the generic operator interface.

#pragma once

#include "kernel_operator.h"

