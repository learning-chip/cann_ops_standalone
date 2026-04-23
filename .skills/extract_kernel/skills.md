## Mandatory requirements for NPU kernel extraction tasks

**Definition of done (all are required):**

1. **Compile** the kernel with `bisheng` similar to other working examples in this repo
2. **Execute** it on a real NPU via torch-npu (PyTorch with `device="npu"`).
3. **Verify** numerical correctness against a PyTorch or NumPy reference.

Until all three succeed, the task is **not finished**. Do not treat "code written" or "compiles only" as completion.

**You MUST:**

- Compile extracted NPU kernel source code via `bisheng`, and load and launch the generated custom lib via `ctypes`
- Run the compile and NPU execution yourself and fix compile errors, runtime errors, and test failures by iterating until the kernel and its test scripts pass.
- Record the exact reproducing commands in that subdirectory’s `README.md` when the work is done so the user can re-run and confirm.

**You MUST NOT:**

- Ask the user to manually compile, run, or verify your new, still-untested code as a substitute for doing it yourself.
- Only test pre-existing APIs in `torch_npu` package, to pretend that it is the custom kernel built from standalone source.
