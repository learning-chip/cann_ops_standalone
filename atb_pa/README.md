# ATB paged attention benchmark (`bench_atb_pa_gqa_paged.py`)

Benchmarks **`torch_npu._npu_paged_attention`** and **`torch_npu.atb._npu_paged_attention_v2`** (`mask_type=0`, no mask) on decode GQA shapes. KV uses **paged cache** (`block_table`, `context_lens`, `block_size=128`) with the same logical FLOP and byte model as [`bench_ifa_gpa.py`](../ifa/bench_ifa_gpa.py) / [`bench_ifa_gpa_paged.py`](../ifa/bench_ifa_gpa_paged.py) (Event-based mean timer).

**Case names** below match the rows in [`ifa/README.md`](../ifa/README.md) (dense IFA table) for easy comparison; numbers here are for **paged** ATB APIs, not dense `npu_incre_flash_attention`.

## Measured throughput (latest `python3 bench_atb_pa_gqa_paged.py --suite ifa-gpa` on this environment)

**Setup:** `torch.float16`, warmup **5**, benchmark iters **20**, **`npu:3`**, paged KV **`block_size=128`**.

### `torch_npu._npu_paged_attention`

| Case | Time (ms) | TFLOP/s | Bandwidth (GiB/s) |
|------|-----------|---------|-------------------|
| Qwen3-0.6B GQA b1 h16/kv8 kv2048 | 0.1593 | 0.1053 | 49.09 |
| Qwen3-1.7B GQA b1 h16/kv8 kv4096 | 0.1543 | 0.2174 | 101.28 |
| Qwen3-4B GQA b1 h32/kv8 kv2048 | 0.1308 | 0.2565 | 59.84 |
| Qwen3-8B GQA b1 h32/kv8 kv4096 | 0.1413 | 0.4748 | 110.67 |
| Qwen3-8B GQA b1 h32/kv8 kv8192 | 0.1461 | 0.9184 | 213.93 |
| Qwen3-14B GQA b1 h40/kv8 kv2048 | 0.1308 | 0.3207 | 59.88 |
| Qwen3-32B GQA b1 h64/kv8 kv2048 | 0.1361 | 0.4929 | 57.61 |
| MHA synthetic b1 h32/kv32 kv2048 (not Qwen3-8B) | 0.1404 | 0.2391 | 222.75 |
| Qwen3-8B GQA b4 h32/kv8 kv2048 | 0.1455 | 0.9225 | 215.21 |
| Qwen3-8B GQA b8 h32/kv8 kv2048 | 0.1544 | 1.7387 | 405.62 |
| Qwen3-8B GQA b16 h32/kv8 kv2048 | 0.1581 | 3.3947 | 791.93 |
| Qwen3-8B GQA b32 h32/kv8 kv2048 | 0.2382 | 4.5068 | 1051.38 |
| Qwen3-8B GQA b64 h32/kv8 kv2048 | 0.4616 | 4.6521 | 1085.28 |

### `torch_npu.atb._npu_paged_attention_v2`

| Case | Time (ms) | TFLOP/s | Bandwidth (GiB/s) |
|------|-----------|---------|-------------------|
| Qwen3-0.6B GQA b1 h16/kv8 kv2048 | 0.2145 | 0.0782 | 36.46 |
| Qwen3-1.7B GQA b1 h16/kv8 kv4096 | 0.2006 | 0.1672 | 77.91 |
| Qwen3-4B GQA b1 h32/kv8 kv2048 | 0.1848 | 0.1816 | 42.36 |
| Qwen3-8B GQA b1 h32/kv8 kv4096 | 0.1888 | 0.3554 | 82.82 |
| Qwen3-8B GQA b1 h32/kv8 kv8192 | 0.1966 | 0.6828 | 159.05 |
| Qwen3-14B GQA b1 h40/kv8 kv2048 | 0.1869 | 0.2244 | 41.90 |
| Qwen3-32B GQA b1 h64/kv8 kv2048 | 0.1920 | 0.3495 | 40.85 |
| MHA synthetic b1 h32/kv32 kv2048 (not Qwen3-8B) | 0.1862 | 0.1802 | 167.90 |
| Qwen3-8B GQA b4 h32/kv8 kv2048 | 0.2145 | 0.6258 | 145.99 |
| Qwen3-8B GQA b8 h32/kv8 kv2048 | 0.2432 | 1.1035 | 257.44 |
| Qwen3-8B GQA b16 h32/kv8 kv2048 | 0.2838 | 1.8915 | 441.25 |
| Qwen3-8B GQA b32 h32/kv8 kv2048 | 0.3543 | 3.0303 | 706.92 |
| Qwen3-8B GQA b64 h32/kv8 kv2048 | 0.4774 | 4.4987 | 1049.49 |

Run: `python3 bench_atb_pa_gqa_paged.py --suite ifa-gpa` (optional: `--device N`, `--bf16`, `--warmup N`, `--iters N`, `--no-ifa` to skip IFA paged baseline when comparing only ATB ops). Default suite `--suite paged` uses the cases in [`bench_ifa_gpa_paged.py`](../ifa/bench_ifa_gpa_paged.py).
