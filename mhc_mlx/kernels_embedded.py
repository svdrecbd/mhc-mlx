"""Embedded Metal kernels."""

RESIDUAL_ADD_AGG_METAL = r"""
// MLX Metal kernel body.
//
// Fuses Residual Add + mHC Aggregate.
//
// Computes:
// - out[b, n, c] = x[b, n, c] + res[b, n, c]
// - y_agg[b, c] = sum_i (out[b, i, c] * H_pre[i])
//
// Inputs:
// - x: [B, n, C]
// - res: [B, n, C]
// - H_pre: [n]
//
// Outputs:
// - out: [B, n, C]
// - y_agg: [B, C]

typedef {{OUT_T}} OUT_T;

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if (b >= (uint)B) return;

// Load H_pre into shared memory
threadgroup float H[MAX_N];
for (uint i = lane; i < uint(n); i += threads_per_threadgroup.x) {
    H[i] = H_pre[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base_token = b * uint(n) * uint(C);
uint base_y = b * uint(C);

if ((C % 4) == 0) {
    // Vectorized path
    uint c_vec = c * 4;
    if (c_vec >= (uint)C) return;

    float4 agg = 0.0f;
    for (int i = 0; i < n; ++i) {
        uint idx = base_token + uint(i) * uint(C) + c_vec;
        
        // Load x and res
        float4 xv = float4(*(const device float4*)(x + idx));
        float4 rv = float4(*(const device float4*)(res + idx));
        
        // Add
        float4 val = xv + rv;
        
        // Store
        *(device float4*)(out + idx) = val;
        
        // Aggregate
        agg += val * H[i];
    }
    
    // Store y_agg
    *(device float4*)(y_agg + base_y + c_vec) = agg;

} else {
    // Scalar fallback
    if (c >= (uint)C) return;

    float agg = 0.0f;
    for (int i = 0; i < n; ++i) {
        uint idx = base_token + uint(i) * uint(C) + c;
        float val = float(x[idx]) + float(res[idx]);
        out[idx] = (OUT_T)val;
        agg += val * H[i];
    }
    y_agg[base_y + c] = agg;
}

"""

MHC_BACKWARD_RMS_REDUCE_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Reduces partial sums to inv_rms and d_r:
// inv_rms[b] = rsqrt(mean(y_agg[b, :]^2) + eps)
// d_r[b] = sum_c d_y_norm[b, c] * y_agg[b, c] * rms_weight[c]
//
// Expected row-contiguous shapes:
// - y_agg:      [B, C] float32
// - partial_sq: [B, T] float32
// - partial_dr: [B, T] float32
// - inv_rms:    [B]    float32
// - d_r:        [B]    float32

constexpr int MAX_TPG = {{MAX_TPG}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;
uint b = thread_position_in_grid.y;
uint tpg = threads_per_threadgroup.x;

int B = y_agg_shape[0];
int C = y_agg_shape[1];
int tiles = partial_sq_shape[1];

if ((int)b >= B) {
    return;
}

threadgroup float reduce_sq[MAX_TPG];
threadgroup float reduce_dr[MAX_TPG];

float partial_sq_sum = 0.0f;
float partial_dr_sum = 0.0f;
for (uint t = lane; t < (uint)tiles; t += tpg) {
    uint idx = uint(b) * uint(tiles) + t;
    partial_sq_sum += partial_sq[idx];
    partial_dr_sum += partial_dr[idx];
}

reduce_sq[lane] = partial_sq_sum;
reduce_dr[lane] = partial_dr_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_sq[lane] += reduce_sq[lane + stride];
        reduce_dr[lane] += reduce_dr[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_sq[0] += reduce_sq[active - 1u];
        reduce_dr[0] += reduce_dr[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    float mean_sq = reduce_sq[0] / float(C);
    inv_rms[b] = metal::rsqrt(mean_sq + EPS);
    d_r[b] = reduce_dr[0];
}

"""

SINKHORN_KNOPP_BACKWARD_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Backward for Sinkhorn-Knopp:
// Given H_res (logits) and dM, compute dH_res for:
// M = sinkhorn_knopp(exp(H_res)).
//
// Expected row-contiguous shapes:
// - H_res: [n, n] float32
// - dM:    [n, n] float32
// - dH_res:[n, n] float32

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};
constexpr int ITERS = {{ITERS}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;

int n = H_res_shape[0];
bool use_lane_norm = (n <= 32) && (tpg >= uint(n));

threadgroup float H[MAX_N * MAX_N];
threadgroup float reduce_buf[MAX_TPG];
threadgroup float row_scale[ITERS * MAX_N];
threadgroup float col_scale[ITERS * MAX_N];
threadgroup uchar row_mask[ITERS * MAX_N];
threadgroup uchar col_mask[ITERS * MAX_N];

uint nn = (uint)(n * n);
for (uint idx = lane; idx < nn; idx += tpg) {
    H[idx] = metal::exp(float(H_res[idx]));
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int it = 0; it < ITERS; ++it) {
    if (use_lane_norm) {
        if (lane < uint(n)) {
            int r = int(lane);
            float s = 0.0f;
            for (int c = 0; c < n; ++c) {
                s += H[r * n + c];
            }
            uchar mask = (s > EPS) ? uchar(1) : uchar(0);
            float scale = (mask != 0u) ? (1.0f / s) : 1.0f;
            row_scale[it * MAX_N + r] = scale;
            row_mask[it * MAX_N + r] = mask;
            for (int c = 0; c < n; ++c) {
                H[r * n + c] *= scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(n)) {
            int c = int(lane);
            float s = 0.0f;
            for (int r = 0; r < n; ++r) {
                s += H[r * n + c];
            }
            uchar mask = (s > EPS) ? uchar(1) : uchar(0);
            float scale = (mask != 0u) ? (1.0f / s) : 0.0f;
            col_scale[it * MAX_N + c] = scale;
            col_mask[it * MAX_N + c] = mask;
            for (int r = 0; r < n; ++r) {
                H[r * n + c] *= scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        continue;
    }

    // Row norm
    for (int r = 0; r < n; ++r) {
        float partial = 0.0f;
        for (uint c = lane; c < uint(n); c += tpg) {
            partial += H[r * n + int(c)];
        }
        reduce_buf[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint active = tpg;
        while (active > 1) {
            uint stride = active / 2;
            if (lane < stride) {
                reduce_buf[lane] += reduce_buf[lane + stride];
            }
            if ((active & 1u) != 0u && lane == 0u) {
                reduce_buf[0] += reduce_buf[active - 1u];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            active = stride + (active & 1u);
        }

        if (lane == 0u) {
            float s = reduce_buf[0];
            uchar mask = (s > EPS) ? uchar(1) : uchar(0);
            float scale = (mask != 0u) ? (1.0f / s) : 1.0f;
            row_scale[it * MAX_N + r] = scale;
            row_mask[it * MAX_N + r] = mask;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = row_scale[it * MAX_N + r];
        for (uint c = lane; c < uint(n); c += tpg) {
            H[r * n + int(c)] *= scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Col norm
    for (int c = 0; c < n; ++c) {
        float partial = 0.0f;
        for (uint r = lane; r < uint(n); r += tpg) {
            partial += H[int(r) * n + c];
        }
        reduce_buf[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint active = tpg;
        while (active > 1) {
            uint stride = active / 2;
            if (lane < stride) {
                reduce_buf[lane] += reduce_buf[lane + stride];
            }
            if ((active & 1u) != 0u && lane == 0u) {
                reduce_buf[0] += reduce_buf[active - 1u];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            active = stride + (active & 1u);
        }

        if (lane == 0u) {
            float s = reduce_buf[0];
            uchar mask = (s > EPS) ? uchar(1) : uchar(0);
            float scale = (mask != 0u) ? (1.0f / s) : 0.0f;
            col_scale[it * MAX_N + c] = scale;
            col_mask[it * MAX_N + c] = mask;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = col_scale[it * MAX_N + c];
        for (uint r = lane; r < uint(n); r += tpg) {
            H[int(r) * n + c] *= scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Initialize dH_res with dM.
for (uint idx = lane; idx < nn; idx += tpg) {
    dH_res[idx] = dM[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (int it = ITERS - 1; it >= 0; --it) {
    if (use_lane_norm) {
        if (lane < uint(n)) {
            int c = int(lane);
            float scale = col_scale[it * MAX_N + c];
            uchar mask = col_mask[it * MAX_N + c];

            if (mask == 0u) {
                for (int r = 0; r < n; ++r) {
                    uint idx = uint(r) * uint(n) + uint(c);
                    dH_res[idx] = 0.0f;
                    H[idx] = 0.0f;
                }
            } else {
                float dot = 0.0f;
                for (int r = 0; r < n; ++r) {
                    uint idx = uint(r) * uint(n) + uint(c);
                    float H_out = H[idx];
                    float H_in = H_out / scale;
                    dot += dH_res[idx] * H_in;
                }
                float scale_sq = scale * scale;
                for (int r = 0; r < n; ++r) {
                    uint idx = uint(r) * uint(n) + uint(c);
                    float H_out = H[idx];
                    float H_in = H_out / scale;
                    float g = dH_res[idx];
                    float d_in = scale * g - scale_sq * dot;
                    dH_res[idx] = d_in;
                    H[idx] = H_in;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(n)) {
            int r = int(lane);
            float scale = row_scale[it * MAX_N + r];
            uchar mask = row_mask[it * MAX_N + r];

            if (mask != 0u) {
                float dot = 0.0f;
                for (int c = 0; c < n; ++c) {
                    uint idx = uint(r) * uint(n) + uint(c);
                    float H_out = H[idx];
                    float H_in = H_out / scale;
                    dot += dH_res[idx] * H_in;
                }
                float scale_sq = scale * scale;
                for (int c = 0; c < n; ++c) {
                    uint idx = uint(r) * uint(n) + uint(c);
                    float H_out = H[idx];
                    float H_in = H_out / scale;
                    float g = dH_res[idx];
                    float d_in = scale * g - scale_sq * dot;
                    dH_res[idx] = d_in;
                    H[idx] = H_in;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        continue;
    }

    // Backprop through col norm.
    for (int c = 0; c < n; ++c) {
        float scale = col_scale[it * MAX_N + c];
        uchar mask = col_mask[it * MAX_N + c];

        if (mask == 0u) {
            for (uint r = lane; r < uint(n); r += tpg) {
                uint idx = uint(r) * uint(n) + uint(c);
                dH_res[idx] = 0.0f;
                H[idx] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        float partial = 0.0f;
        for (uint r = lane; r < uint(n); r += tpg) {
            uint idx = uint(r) * uint(n) + uint(c);
            float H_out = H[idx];
            float H_in = H_out / scale;
            partial += dH_res[idx] * H_in;
        }
        reduce_buf[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint active = tpg;
        while (active > 1) {
            uint stride = active / 2;
            if (lane < stride) {
                reduce_buf[lane] += reduce_buf[lane + stride];
            }
            if ((active & 1u) != 0u && lane == 0u) {
                reduce_buf[0] += reduce_buf[active - 1u];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            active = stride + (active & 1u);
        }

        float dot = reduce_buf[0];
        float scale_sq = scale * scale;
        for (uint r = lane; r < uint(n); r += tpg) {
            uint idx = uint(r) * uint(n) + uint(c);
            float H_out = H[idx];
            float H_in = H_out / scale;
            float g = dH_res[idx];
            float d_in = scale * g - scale_sq * dot;
            dH_res[idx] = d_in;
            H[idx] = H_in;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Backprop through row norm.
    for (int r = 0; r < n; ++r) {
        float scale = row_scale[it * MAX_N + r];
        uchar mask = row_mask[it * MAX_N + r];

        if (mask == 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        float partial = 0.0f;
        for (uint c = lane; c < uint(n); c += tpg) {
            uint idx = uint(r) * uint(n) + uint(c);
            float H_out = H[idx];
            float H_in = H_out / scale;
            partial += dH_res[idx] * H_in;
        }
        reduce_buf[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint active = tpg;
        while (active > 1) {
            uint stride = active / 2;
            if (lane < stride) {
                reduce_buf[lane] += reduce_buf[lane + stride];
            }
            if ((active & 1u) != 0u && lane == 0u) {
                reduce_buf[0] += reduce_buf[active - 1u];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            active = stride + (active & 1u);
        }

        float dot = reduce_buf[0];
        float scale_sq = scale * scale;
        for (uint c = lane; c < uint(n); c += tpg) {
            uint idx = uint(r) * uint(n) + uint(c);
            float H_out = H[idx];
            float H_in = H_out / scale;
            float g = dH_res[idx];
            float d_in = scale * g - scale_sq * dot;
            dH_res[idx] = d_in;
            H[idx] = H_in;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Chain rule through exp: dH_res *= exp(H_res) (stored in H).
for (uint idx = lane; idx < nn; idx += tpg) {
    dH_res[idx] *= H[idx];
}

"""

STREAM_MIX_BACKWARD_DX_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Computes dx for stream mixing:
// dx[b, i, c] = sum_k M[k, i] * d_out[b, k, c]
//
// Expected row-contiguous shapes:
// - M:     [n, n]    float32
// - d_out: [B, n, C] float32
// - dx:    [B, n, C] float32

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint k = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = d_out_shape[0];
int n = d_out_shape[1];
int C = d_out_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || (int)c >= C) {
    return;
}

threadgroup float P[MAX_N * MAX_N];
uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    P[idx] = M[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);

if ((C % 4) == 0) {
    // Vectorized path
    uint c_vec = c * 4;
    if ((int)c_vec >= C) return;

    float4 acc = 0.0f;
    for (int k_idx = 0; k_idx < n; ++k_idx) {
        float p_val = P[k_idx * n + i];
        float4 dout_val = float4(*(const device float4*)(d_out + base + uint(k_idx) * uint(C) + c_vec));
        acc += p_val * dout_val;
    }
    *(device float4*)(dx + base + uint(i) * uint(C) + c_vec) = acc;

} else {
    // Scalar fallback
    if ((int)c >= C) return;

    float acc = 0.0f;
    for (int k_idx = 0; k_idx < n; ++k_idx) {
        acc += P[k_idx * n + i] * float(d_out[base + uint(k_idx) * uint(C) + c]);
    }
    dx[base + uint(i) * uint(C) + c] = acc;
}

"""

STREAM_MIX_ADD_RMS_FP16_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Half2-optimized stream mix + RMS add, writing float16 output.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float16

constexpr int MAX_N = {{MAX_N}};

uint c2 = thread_position_in_grid.x * 2u;
uint k = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || (int)c2 >= C) {
    return;
}

bool has_second = (int)(c2 + 1u) < C;

threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += threads_per_threadgroup.x) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

uint out_idx = base + uint(i) * uint(C) + c2;
float inv = float(inv_rms[b]);

if (has_second) {
    float2 acc = float2(0.0f);
    for (int j = 0; j < n; ++j) {
        uint idx = base + uint(j) * uint(C) + c2;
        float2 xf = float2(0.0f);
        if (((idx & 1u) == 0u) && has_second) {
            const device packed_half2* x_ptr = (const device packed_half2*)(x + idx);
            half2 xv = half2(*x_ptr);
            xf = float2(float(xv.x), float(xv.y));
        } else {
            float x0 = float(x[idx]);
            float x1 = has_second ? float(x[idx + 1u]) : 0.0f;
            xf = float2(x0, x1);
        }
        acc += Ms[i * n + j] * xf;
    }

    float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
    float y1 = float(y_agg[base_bc + c2 + 1u]) * inv * float(rms_weight[c2 + 1u]);
    float out0 = acc.x + Hpost[i] * y0;
    float out1 = acc.y + Hpost[i] * y1;

    device packed_half2* out_ptr = (device packed_half2*)(out + out_idx);
    *out_ptr = packed_half2(half2(half(out0), half(out1)));
} else {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        uint idx = base + uint(j) * uint(C) + c2;
        acc += Ms[i * n + j] * float(x[idx]);
    }
    float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
    out[out_idx] = half(acc + Hpost[i] * y0);
}

"""

MHC_BACKWARD_GRADS_FUSED_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Optimized Fused Backward Gradients (n <= 32).
//
// Computes:
// - dM[i, j] = sum_{b, c} d_out[b, i, c] * x[b, j, c]
// - dH_pre[i] = sum_{b, c} d_y_agg[b, c] * x[b, i, c]
// - dH_post[i] = sum_{b, c} d_out[b, i, c] * y_norm[b, c]
// - d_rms_weight[c] (atomic add)
//
// Strategy:
// - TPG = 32. Each thread 'i' owns row 'i' of dM.
// - Grid-stride loop over (b, c).
// - Inside loop:
//   - Thread 'i' loads x[b, i, c] and dout[b, i, c].
//   - Share x[b, :, c] via simd_shuffle (no shared mem needed!).
//   - Accumulate dM[i, :] into 32 registers.
//   - Accumulate dH_pre[i] and dH_post[i] into registers.
//   - Compute d_rms contribution and atomic-add to global.
// - After loop:
//   - Atomic-add 32 registers to global dM[i, :].
//   - Atomic-add dH registers to global dH.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - d_out:      [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32
// - rms_weight: [C]       float32
// - dM:         [n, n]    float32 (atomic accum)
// - dH_pre:     [n]       float32 (atomic accum)
// - dH_post:    [n]       float32 (atomic accum)
// - d_rms_weight: [C]     float32 (atomic accum)

constexpr int MAX_N = {{MAX_N}};

uint tid = thread_position_in_grid.x;
uint tpg = threads_per_threadgroup.x; // Should be 32
uint lane = thread_position_in_threadgroup.x;
uint gsize = grid_dim.x; // Total threads

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

// This kernel is specialized for n <= 32 and TPG=32
if (tpg != 32 || lane >= 32) {
    return;
}

// Helper for atomic float add
inline void atomic_add_float(device float* addr, float val) {
    device atomic_uint* addr_u = (device atomic_uint*)addr;
    uint old = atomic_load_explicit(addr_u, memory_order_relaxed);
    uint expected = old;
    uint desired;
    do {
        expected = old;
        desired = as_type<uint>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr_u, &old, desired, memory_order_relaxed, memory_order_relaxed));
}

// Registers for dM row 'lane' (32 elements)
float acc_dM[32];
for (int k = 0; k < 32; ++k) {
    acc_dM[k] = 0.0f;
}
float acc_dH_pre = 0.0f;
float acc_dH_post = 0.0f;

// Total work items = B * C
uint total_items = uint(B) * uint(C);

// Each threadgroup processes a set of (b, c) pairs.
// Since TPG=32 and we use simd functions, the whole threadgroup works on the SAME (b, c) pair?
// Yes, strategy: Threadgroup 'g' processes pairs p, p+G, p+2G...
// where G is number of threadgroups.
// Inside step: All 32 threads work on pair 'p'.
// Thread 'lane' handles stream 'lane'.

uint gid = tid / 32u; // Threadgroup ID
uint num_groups = gsize / 32u;

for (uint idx = gid; idx < total_items; idx += num_groups) {
    uint b = idx / uint(C);
    uint c = idx % uint(C);
    
    // Load scalars for this (b, c)
    // Only needed by threads active (lane < n)
    // But since n <= 32 and we assumed TPG=32, lane < n check needed for loads.
    
    // Actually, simple mask:
    bool active = (lane < uint(n));
    
    float xi = 0.0f;
    float douti = 0.0f;
    
    uint base = uint(b) * uint(n) * uint(C) + uint(c);
    if (active) {
        xi = float(x[base + lane * uint(C)]);
        douti = float(d_out[base + lane * uint(C)]);
    }
    
    // Precompute d_y_agg, y_norm
    // These are scalar per (b, c), so all threads in group compute same value.
    // Redundant but cheap.
    float inv = float(inv_rms[b]);
    float dr = float(d_r[b]);
    float inv3 = inv * inv * inv;
    float d_mean_sq = -0.5f * dr * inv3;
    
    uint idx_bc = uint(b) * uint(C) + uint(c);
    float y_agg_val = float(y_agg[idx_bc]);
    float d_y_norm_val = float(d_y_norm[idx_bc]);
    float rw = float(rms_weight[c]);
    
    float d_y_agg = d_y_norm_val * inv * rw + d_mean_sq * (2.0f / float(C)) * y_agg_val;
    float y_norm = y_agg_val * inv * rw;
    
    // Accumulate dH
    if (active) {
        acc_dH_pre += d_y_agg * xi;
        acc_dH_post += douti * y_norm;
    }
    
    // d_rms_weight atomic add
    // d_rms_weight[c] += d_y_norm * y_agg * inv_rms
    // This assumes we only want d_rms_weight from this backward.
    // Only one thread should add per (b, c)? 
    // Or we duplicate adds? 
    // Let thread 0 do it.
    if (lane == 0u) {
        float val = d_y_norm_val * y_agg_val * inv;
        atomic_add_float((device float*)&d_rms_weight[c], val);
    }
    
    // Accumulate dM row 'lane'
    // dM[lane, j] += dout[lane] * x[j]
    if (active) {
        float my_dout = douti;
        
        // Broadcast x[j] from thread j
        for (int j = 0; j < n; ++j) {
            float xj = simd_shuffle(xi, j); // Get xi from thread j
            acc_dM[j] += my_dout * xj;
        }
    }
}

// Final atomic update to global memory
if (lane < uint(n)) {
    // dH
    atomic_add_float((device float*)&dH_pre[lane], acc_dH_pre);
    atomic_add_float((device float*)&dH_post[lane], acc_dH_post);
    
    // dM row
    for (int j = 0; j < n; ++j) {
        atomic_add_float((device float*)&dM[lane * uint(n) + uint(j)], acc_dM[j]);
    }
}

"""

STREAM_MIX_ADD_RMS_TILE_BF16_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// BF16-tiled stream mix + RMS add for fixed n (TILE_N).
// Computes a [TILE_N x (TILE_C*2)] block of output per threadgroup.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] bfloat16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] bfloat16

constexpr int TILE_N = {{TILE_N}};
constexpr int TILE_C = {{TILE_C}};

#define BF16_TO_FLOAT(v) (as_type<float>(uint(v) << 16))
#define FLOAT_TO_BF16(v) (ushort((as_type<uint>(v) + 0x7FFFu + ((as_type<uint>(v) >> 16) & 1u)) >> 16))

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint gid_x = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)b >= B) {
    return;
}

uint tile = gid_x / tpg;
uint c2_offset = lane % TILE_C;
uint i = lane / TILE_C;

if ((int)i >= n) {
    return;
}

uint c2 = tile * (TILE_C * 2u) + c2_offset * 2u;

threadgroup float M_tile[TILE_N * TILE_N];
threadgroup float Hpost[TILE_N];
threadgroup float2 X_tile[TILE_N * TILE_C];

uint mn = uint(n * n);
for (uint idx = lane; idx < mn; idx += tpg) {
    M_tile[idx] = M[idx];
}
if (lane < uint(n)) {
    Hpost[lane] = H_post[lane];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

const device ushort* x_bits = reinterpret_cast<const device ushort*>(x);
device ushort* out_bits = reinterpret_cast<device ushort*>(out);

bool has0 = (int)c2 < C;
bool has1 = (int)(c2 + 1u) < C;

float2 xval = float2(0.0f);
if (has0) {
    uint idx = base + uint(i) * uint(C) + c2;
    if (has1) {
        if ((idx & 1u) == 0u) {
            const device packed_ushort2* x_ptr = (const device packed_ushort2*)(x_bits + idx);
            ushort2 xv = ushort2(*x_ptr);
            xval = float2(BF16_TO_FLOAT(xv.x), BF16_TO_FLOAT(xv.y));
        } else {
            xval = float2(BF16_TO_FLOAT(x_bits[idx]), BF16_TO_FLOAT(x_bits[idx + 1u]));
        }
    } else {
        xval = float2(BF16_TO_FLOAT(x_bits[idx]), 0.0f);
    }
}
X_tile[uint(i) * TILE_C + c2_offset] = xval;
threadgroup_barrier(mem_flags::mem_threadgroup);

float2 acc = float2(0.0f);
for (int j = 0; j < n; ++j) {
    acc += M_tile[i * n + j] * X_tile[uint(j) * TILE_C + c2_offset];
}

if (has0) {
    float inv = float(inv_rms[b]);
    float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
    float out0 = acc.x + Hpost[i] * y0;
    uint out_idx = base + uint(i) * uint(C) + c2;
    if (has1) {
        float y1 = float(y_agg[base_bc + c2 + 1u]) * inv * float(rms_weight[c2 + 1u]);
        float out1 = acc.y + Hpost[i] * y1;
        device packed_ushort2* out_ptr = (device packed_ushort2*)(out_bits + out_idx);
        *out_ptr = packed_ushort2(ushort2(FLOAT_TO_BF16(out0), FLOAT_TO_BF16(out1)));
    } else {
        out_bits[out_idx] = FLOAT_TO_BF16(out0);
    }
}

"""

MHC_FORWARD_RMS_REDUCE_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Reduces partial sums to inv_rms:
// inv_rms[b] = rsqrt(mean(y_agg[b, :]^2) + eps)
//
// Expected row-contiguous shapes:
// - y_agg:     [B, C] float32
// - partial_sq:[B, T] float32
// - inv_rms:   [B]    float32

constexpr int MAX_TPG = {{MAX_TPG}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;
uint b = thread_position_in_grid.y;
uint tpg = threads_per_threadgroup.x;

int B = y_agg_shape[0];
int C = y_agg_shape[1];
int tiles = partial_sq_shape[1];

if ((int)b >= B) {
    return;
}

threadgroup float reduce_buf[MAX_TPG];

float partial = 0.0f;
for (uint t = lane; t < (uint)tiles; t += tpg) {
    partial += partial_sq[uint(b) * uint(tiles) + t];
}

reduce_buf[lane] = partial;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    float mean_sq = reduce_buf[0] / float(C);
    inv_rms[b] = metal::rsqrt(mean_sq + EPS);
}

"""

STREAM_MIX_ADD_RMS_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// This fuses:
// - stream mixing: out[b,i,c] = sum_j M[i,j] * x[b,j,c]
// - final add: out += H_post[i] * (y_agg[b,c] * inv_rms[b] * rms_weight[c])
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32/float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32 (activated)
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float32

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint k = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || (int)c >= C) {
    return;
}

threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += threads_per_threadgroup.x) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

float acc = 0.0f;
for (int j = 0; j < n; ++j) {
    acc += Ms[i * n + j] * float(x[base + uint(j) * uint(C) + c]);
}

float y = float(y_agg[base_bc + c]) * float(inv_rms[b]) * float(rms_weight[c]);
uint out_idx = base + uint(i) * uint(C) + c;
out[out_idx] = acc + Hpost[i] * y;

"""

MHC_BACKWARD_DM_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Computes dM for stream mixing:
// dM[i, j] = sum_{b, c} d_out[b, i, c] * x[b, j, c]
//
// Expected row-contiguous shapes:
// - x:     [B, n, C] float32
// - d_out: [B, n, C] float32
// - dM:    [n, n]    float32

constexpr int MAX_TPG = {{MAX_TPG}};
constexpr int MAX_SIMDGROUPS = (MAX_TPG + 31) / 32;

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint idx = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int nn = n * n;
if ((int)idx >= nn) {
    return;
}

int i = (int)(idx / (uint)n);
int j = (int)(idx - (uint)(i * n));

threadgroup float simd_buf[MAX_SIMDGROUPS];

float partial = 0.0f;
uint C4 = (uint(C) / 4u) * 4u;

for (int b = 0; b < B; ++b) {
    uint base = uint(b) * uint(n) * uint(C);
    uint base_i = base + uint(i) * uint(C);
    uint base_j = base + uint(j) * uint(C);

    for (uint c = lane * 4u; c < C4; c += tpg * 4u) {
        const device packed_float4* dout_ptr =
            (const device packed_float4*)(d_out + base_i + c);
        const device packed_float4* x_ptr =
            (const device packed_float4*)(x + base_j + c);
        float4 doutv = float4(*dout_ptr);
        float4 xv = float4(*x_ptr);
        partial += dot(doutv, xv);
    }

    for (uint c = C4 + lane; c < uint(C); c += tpg) {
        float doutv = float(d_out[base_i + c]);
        float xv = float(x[base_j + c]);
        partial += doutv * xv;
    }
}

float simd_sum = metal::simd_sum(partial);
if (thread_index_in_simdgroup == 0u) {
    simd_buf[simdgroup_index_in_threadgroup] = simd_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (lane == 0u) {
    float total = 0.0f;
    uint simd_groups = (tpg + 31u) / 32u;
    for (uint g = 0; g < simd_groups; ++g) {
        total += simd_buf[g];
    }
    dM[idx] = total;
}

"""

STREAM_MIX_ADD_RMS_TILE2D_BF16_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// 2D threadgroup variant of stream mix + RMS add (bf16 output).
// Uses threadgroup.y == n so a group computes all i in [0,n) for a batch b,
// amortizing M and H_post loads across the channel tile.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] bfloat16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] bfloat16

constexpr int MAX_N = {{MAX_N}};

#define BF16_TO_FLOAT(v) (as_type<float>(uint(v) << 16))
#define FLOAT_TO_BF16(v) (ushort((as_type<uint>(v) + 0x7FFFu + ((as_type<uint>(v) >> 16) & 1u)) >> 16))

uint c2 = thread_position_in_grid.x * 2u;
uint k = thread_position_in_grid.y;
uint lane_x = thread_position_in_threadgroup.x;
uint lane_y = thread_position_in_threadgroup.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || i >= n || (int)c2 >= C) {
    return;
}

threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint tpg_x = threads_per_threadgroup.x;
uint tpg_y = threads_per_threadgroup.y;
uint tcount = tpg_x * tpg_y;
uint lane = lane_y * tpg_x + lane_x;

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += tcount) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < (uint)n; idx += tcount) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

const device ushort* x_bits = reinterpret_cast<const device ushort*>(x);
device ushort* out_bits = reinterpret_cast<device ushort*>(out);

bool has1 = (int)(c2 + 1u) < C;

float2 acc = float2(0.0f);
for (int j = 0; j < n; ++j) {
    uint idx = base + uint(j) * uint(C) + c2;
    float2 xval = float2(0.0f);
    if (has1) {
        if ((idx & 1u) == 0u) {
            const device packed_ushort2* x_ptr = (const device packed_ushort2*)(x_bits + idx);
            ushort2 xv = ushort2(*x_ptr);
            xval = float2(BF16_TO_FLOAT(xv.x), BF16_TO_FLOAT(xv.y));
        } else {
            xval = float2(BF16_TO_FLOAT(x_bits[idx]), BF16_TO_FLOAT(x_bits[idx + 1u]));
        }
    } else {
        xval = float2(BF16_TO_FLOAT(x_bits[idx]), 0.0f);
    }
    acc += Ms[i * n + j] * xval;
}

float inv = float(inv_rms[b]);
float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
float out0 = acc.x + Hpost[i] * y0;
uint out_idx = base + uint(i) * uint(C) + c2;
if (has1) {
    float y1 = float(y_agg[base_bc + c2 + 1u]) * inv * float(rms_weight[c2 + 1u]);
    float out1 = acc.y + Hpost[i] * y1;
    if ((out_idx & 1u) == 0u) {
        device packed_ushort2* out_ptr = (device packed_ushort2*)(out_bits + out_idx);
        *out_ptr = packed_ushort2(ushort2(FLOAT_TO_BF16(out0), FLOAT_TO_BF16(out1)));
    } else {
        out_bits[out_idx] = FLOAT_TO_BF16(out0);
        out_bits[out_idx + 1u] = FLOAT_TO_BF16(out1);
    }
} else {
    out_bits[out_idx] = FLOAT_TO_BF16(out0);
}

"""

MHC_BACKWARD_D_RMS_WEIGHT_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Computes d_rms_weight:
// d_rms_weight[c] = sum_b d_y_norm[b, c] * y_agg[b, c] * inv_rms[b]
//
// Expected row-contiguous shapes:
// - y_agg:        [B, C] float32
// - d_y_norm:     [B, C] float32
// - inv_rms:      [B]    float32
// - d_rms_weight: [C]    float32

uint c = thread_position_in_grid.x;

int B = y_agg_shape[0];
int C = y_agg_shape[1];

if ((int)c >= C) {
    return;
}

float acc = 0.0f;
for (int b = 0; b < B; ++b) {
    uint base = uint(b) * uint(C);
    float inv = inv_rms[b];
    float y_val = y_agg[base + c];
    float d_y_val = d_y_norm[base + c];
    acc += d_y_val * y_val * inv;
}

d_rms_weight[c] = acc;

"""

MHC_BACKWARD_DH_PRE_POST_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Computes dH_pre and dH_post in one pass:
// dH_pre[i]  = sum_{b, c} d_y_agg[b, c] * x[b, i, c]
// dH_post[i] = sum_{b, c} d_out[b, i, c] * y_norm[b, c]
// where y_norm = y_agg * inv_rms * rms_weight.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - d_out:      [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32
// - rms_weight: [C]       float32
// - dH_pre:     [n]       float32
// - dH_post:    [n]       float32

constexpr int MAX_TPG = {{MAX_TPG}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint i = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)i >= n) {
    return;
}

threadgroup float reduce_pre[MAX_TPG];
threadgroup float reduce_post[MAX_TPG];

float partial_pre = 0.0f;
float partial_post = 0.0f;

for (int b = 0; b < B; ++b) {
    float inv = inv_rms[b];
    float dr = d_r[b];
    float inv3 = inv * inv * inv;
    float d_mean_sq = -0.5f * dr * inv3;
    uint base = uint(b) * uint(n) * uint(C);
    uint base_bc = uint(b) * uint(C);
    for (uint c = lane; c < uint(C); c += tpg) {
        float y_agg_val = y_agg[base_bc + c];
        float d_y_norm_val = d_y_norm[base_bc + c];
        float d_y_agg = d_y_norm_val * inv * float(rms_weight[c])
            + d_mean_sq * (2.0f / float(C)) * y_agg_val;
        float xv = float(x[base + uint(i) * uint(C) + c]);
        partial_pre += d_y_agg * xv;

        float y_norm = y_agg_val * inv * float(rms_weight[c]);
        float doutv = float(d_out[base + uint(i) * uint(C) + c]);
        partial_post += doutv * y_norm;
    }
}

reduce_pre[lane] = partial_pre;
reduce_post[lane] = partial_post;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_pre[lane] += reduce_pre[lane + stride];
        reduce_post[lane] += reduce_post[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_pre[0] += reduce_pre[active - 1u];
        reduce_post[0] += reduce_post[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    dH_pre[i] = reduce_pre[0];
    dH_post[i] = reduce_post[0];
}

"""

MHC_BACKWARD_DH_PRE_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Computes dH_pre for the aggregate branch:
// dH_pre[i] = sum_{b, c} d_y_agg[b, c] * x[b, i, c]
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32
// - rms_weight: [C]       float32
// - dH_pre:     [n]       float32

constexpr int MAX_TPG = {{MAX_TPG}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint i = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)i >= n) {
    return;
}

threadgroup float reduce_buf[MAX_TPG];

float partial = 0.0f;
for (int b = 0; b < B; ++b) {
    float inv = inv_rms[b];
    float dr = d_r[b];
    float inv3 = inv * inv * inv;
    float d_mean_sq = -0.5f * dr * inv3;
    uint base = uint(b) * uint(n) * uint(C);
    uint base_bc = uint(b) * uint(C);
    for (uint c = lane; c < uint(C); c += tpg) {
        float y_agg_val = y_agg[base_bc + c];
        float d_y_norm_val = d_y_norm[base_bc + c];
        float d_y_agg = d_y_norm_val * inv * float(rms_weight[c])
            + d_mean_sq * (2.0f / float(C)) * y_agg_val;
        float xv = float(x[base + uint(i) * uint(C) + c]);
        partial += d_y_agg * xv;
    }
}

reduce_buf[lane] = partial;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    dH_pre[i] = reduce_buf[0];
}

"""

MHC_BACKWARD_DX_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Computes dx for the fused mHC forward:
// dx = M^T * d_out + d_y_agg * H_pre
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - M:          [n, n]    float32
// - H_pre:      [n]       float32 (activated)
// - rms_weight: [C]       float32
// - d_out:      [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32
// - dx:         [B, n, C] float32

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint k = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || (int)c >= C) {
    return;
}

threadgroup float P[MAX_N * MAX_N];
threadgroup float Hpre[MAX_N];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    P[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += threads_per_threadgroup.x) {
    Hpre[idx] = H_pre[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

float inv = inv_rms[b];
float dr = d_r[b];
float inv3 = inv * inv * inv;
float d_mean_sq = -0.5f * dr * inv3;

float y_agg_val = y_agg[base_bc + c];
float d_y_norm_val = d_y_norm[base_bc + c];
float d_y_agg = d_y_norm_val * inv * float(rms_weight[c])
    + d_mean_sq * (2.0f / float(C)) * y_agg_val;

float dx_agg = d_y_agg * Hpre[i];

float dx_mix = 0.0f;
for (int k_idx = 0; k_idx < n; ++k_idx) {
    dx_mix += P[k_idx * n + i] * float(d_out[base + uint(k_idx) * uint(C) + c]);
}

dx[base + uint(i) * uint(C) + c] = dx_mix + dx_agg;

"""

MHC_FORWARD_AGG_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Computes:
// - y_agg[b, c] = sum_i H_pre[i] * x[b, i, c]
// - partial_sq[b, tile] = sum_{c in tile} y_agg[b, c]^2
//
// Expected row-contiguous shapes:
// - x:         [B, n, C] float32/float16
// - H_pre:     [n]       float32 (activated)
// - y_agg:     [B, C]    float32
// - partial_sq:[B, T]    float32 (T = ceil_div(C, threads_per_group))

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};

uint lane = thread_position_in_threadgroup.x;
uint c = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
uint tpg = threads_per_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)b >= B) {
    return;
}

uint tiles = (uint(C) + tpg - 1u) / tpg;
uint tile = c / tpg;

threadgroup float Hpre[MAX_N];
threadgroup float reduce_buf[MAX_TPG];

for (uint idx = lane; idx < uint(n); idx += tpg) {
    Hpre[idx] = H_pre[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

int c_i = int(c);
uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

float y_val = 0.0f;
if (c_i < C) {
    for (int i = 0; i < n; ++i) {
        y_val += Hpre[i] * float(x[base + uint(i) * uint(C) + uint(c_i)]);
    }
    y_agg[base_bc + uint(c_i)] = y_val;
}

float local_sq = (c_i < C) ? (y_val * y_val) : 0.0f;
reduce_buf[lane] = local_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    partial_sq[uint(b) * tiles + tile] = reduce_buf[0];
}

"""

MHC_BACKWARD_PREP_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Precomputes intermediates for mHC backward:
// - y_agg[b, c] = sum_i H_pre[i] * x[b, i, c]
// - d_y_norm[b, c] = sum_i H_post[i] * d_out[b, i, c]
// - inv_rms[b] and d_r[b] for RMSNorm backward
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - H_pre:      [n]       float32 (activated)
// - H_post:     [n]       float32 (activated)
// - rms_weight: [C]       float32
// - d_out:      [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint token = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)token >= B) {
    return;
}

threadgroup float Hpre[MAX_N];
threadgroup float Hpost[MAX_N];
threadgroup float reduce_buf[MAX_TPG];

for (uint idx = lane; idx < uint(n); idx += tpg) {
    Hpre[idx] = H_pre[idx];
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = token * uint(n) * uint(C);
uint base_bc = token * uint(C);

float local_sum = 0.0f;
float local_dr = 0.0f;

for (uint c = lane; c < uint(C); c += tpg) {
    float y_agg_val = 0.0f;
    float d_y_norm_val = 0.0f;
    for (int i = 0; i < n; ++i) {
        uint idx = base + uint(i) * uint(C) + c;
        float xv = float(x[idx]);
        float doutv = float(d_out[idx]);
        y_agg_val += Hpre[i] * xv;
        d_y_norm_val += Hpost[i] * doutv;
    }
    y_agg[base_bc + c] = y_agg_val;
    d_y_norm[base_bc + c] = d_y_norm_val;
    local_sum += y_agg_val * y_agg_val;
    local_dr += d_y_norm_val * y_agg_val * float(rms_weight[c]);
}

reduce_buf[lane] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

float mean_sq = reduce_buf[0] / float(C);
float inv = metal::rsqrt(mean_sq + EPS);

reduce_buf[lane] = local_dr;
threadgroup_barrier(mem_flags::mem_threadgroup);

active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    inv_rms[token] = inv;
    d_r[token] = reduce_buf[0];
}

"""

MHC_BACKWARD_DX_COL_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Optimized "Column-Parallel" Backward DX.
//
// Computes dx for the fused mHC forward:
// dx = M^T * d_out + d_y_agg * H_pre
//
// Optimization:
// - Loads d_out[b, :, c] (entire column) into registers once.
// - Performs M^T @ d_out_col in registers.
// - Reduces d_out reads from O(n^2) to O(n) per column.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - M:          [n, n]    float32
// - H_pre:      [n]       float32 (activated)
// - rms_weight: [C]       float32
// - d_out:      [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32
// - dx:         [B, n, C] float32

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if (b >= (uint)B || c >= (uint)C) {
    return;
}

// Load M and H_pre into threadgroup memory
threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpre[MAX_N];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += threads_per_threadgroup.x) {
    Hpre[idx] = H_pre[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Compute d_y_agg scalar part
// d_y_agg = d_y_norm * inv * rms_weight + d_mean_sq * (2/C) * y_agg
float inv = float(inv_rms[b]);
float dr = float(d_r[b]);
float inv3 = inv * inv * inv;
float d_mean_sq = -0.5f * dr * inv3;

uint idx_bc = uint(b) * uint(C) + c;
float y_agg_val = float(y_agg[idx_bc]);
float d_y_norm_val = float(d_y_norm[idx_bc]);
float d_y_agg = d_y_norm_val * inv * float(rms_weight[c])
    + d_mean_sq * (2.0f / float(C)) * y_agg_val;

// Load d_out column d_out[b, :, c] into registers
float d_out_col[MAX_N];
uint base = uint(b) * uint(n) * uint(C) + c;
uint stride = uint(C);

for (int k = 0; k < n; ++k) {
    d_out_col[k] = float(d_out[base + uint(k) * stride]);
}

// Compute dx
// dx[i] = (sum_k M[k, i] * d_out[k]) + d_y_agg * H_pre[i]
// Note: We need M^T, so we sum over rows of M column i.
// M is [n, n], flat index [k*n + i]

for (int i = 0; i < n; ++i) {
    float dx_mix = 0.0f;
    for (int k = 0; k < n; ++k) {
        // M[k, i] is Ms[k * n + i]
        dx_mix += Ms[k * n + i] * d_out_col[k];
    }
    
    float val = dx_mix + d_y_agg * Hpre[i];
    dx[base + uint(i) * stride] = val;
}

"""

SINKHORN_KNOPP_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Computes: M = sinkhorn_knopp(exp(H_res_raw))
//
// Expected row-contiguous shapes:
// - H_res: [n, n] float32 (raw logits)
// - out:   [n, n] float32

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};
constexpr int ITERS = {{ITERS}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;

int n = H_res_shape[0];
bool use_lane_norm = (n <= 32) && (tpg >= uint(n));

threadgroup float H[MAX_N * MAX_N];
threadgroup float reduce_buf[MAX_TPG];

uint nn = (uint)(n * n);

// Step 1: Find global max for stability
float local_max = -1e38f; // Init with small number
for (uint idx = lane; idx < nn; idx += tpg) {
    local_max = metal::max(local_max, float(H_res[idx]));
}

float global_max;
if (false) { // was tpg <= 32
    global_max = metal::simd_max(local_max);
} else {
    reduce_buf[lane] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max
    uint active = tpg;
    while (active > 1) {
        uint stride = active / 2;
        if (lane < stride) {
            reduce_buf[lane] = metal::max(reduce_buf[lane], reduce_buf[lane + stride]);
        }
        if ((active & 1u) != 0u && lane == 0u) {
            reduce_buf[0] = metal::max(reduce_buf[0], reduce_buf[active - 1u]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active = stride + (active & 1u);
    }
    global_max = reduce_buf[0];
}

// Step 2: Exponentiate with max subtraction
for (uint idx = lane; idx < nn; idx += tpg) {
    H[idx] = metal::exp(float(H_res[idx]) - global_max);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Sinkhorn-Knopp: alternate row/col normalization
for (int it = 0; it < ITERS; ++it) {
    if (use_lane_norm) {
        if (lane < uint(n)) {
            int r = int(lane);
            float s = 0.0f;
            for (int c = 0; c < n; ++c) {
                s += H[r * n + c];
            }
            float scale = (s > EPS) ? (1.0f / s) : 1.0f;
            for (int c = 0; c < n; ++c) {
                H[r * n + c] *= scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(n)) {
            int c = int(lane);
            float s = 0.0f;
            for (int r = 0; r < n; ++r) {
                s += H[r * n + c];
            }
            float scale = (s > EPS) ? (1.0f / s) : 0.0f;
            for (int r = 0; r < n; ++r) {
                H[r * n + c] *= scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        continue;
    }

    // Row norm
    for (int r = 0; r < n; ++r) {
        float partial = 0.0f;
        for (uint c = lane; c < uint(n); c += tpg) {
            partial += H[r * n + int(c)];
        }
        reduce_buf[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint active = tpg;
        while (active > 1) {
            uint stride = active / 2;
            if (lane < stride) {
                reduce_buf[lane] += reduce_buf[lane + stride];
            }
            if ((active & 1u) != 0u && lane == 0u) {
                reduce_buf[0] += reduce_buf[active - 1u];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            active = stride + (active & 1u);
        }

        float s = reduce_buf[0];
        float scale = (s > EPS) ? (1.0f / s) : 1.0f;
        for (uint c = lane; c < uint(n); c += tpg) {
            H[r * n + int(c)] *= scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Col norm
    for (int c = 0; c < n; ++c) {
        float partial = 0.0f;
        for (uint r = lane; r < uint(n); r += tpg) {
            partial += H[int(r) * n + c];
        }
        reduce_buf[lane] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint active = tpg;
        while (active > 1) {
            uint stride = active / 2;
            if (lane < stride) {
                reduce_buf[lane] += reduce_buf[lane + stride];
            }
            if ((active & 1u) != 0u && lane == 0u) {
                reduce_buf[0] += reduce_buf[active - 1u];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            active = stride + (active & 1u);
        }

        float s = reduce_buf[0];
        float scale = (s > EPS) ? (1.0f / s) : 0.0f;
        for (uint r = lane; r < uint(n); r += tpg) {
            H[int(r) * n + c] *= scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

for (uint idx = lane; idx < nn; idx += tpg) {
    out[idx] = H[idx];
}

"""

MHC_BACKWARD_DH_POST_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Computes dH_post for the distribute branch:
// dH_post[i] = sum_{b, c} d_out[b, i, c] * y_norm[b, c]
// where y_norm = y_agg * inv_rms * rms_weight.
//
// Expected row-contiguous shapes:
// - d_out:      [B, n, C] float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - dH_post:    [n]       float32

constexpr int MAX_TPG = {{MAX_TPG}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint i = thread_position_in_grid.y;

int B = d_out_shape[0];
int n = d_out_shape[1];
int C = d_out_shape[2];

if ((int)i >= n) {
    return;
}

threadgroup float reduce_buf[MAX_TPG];

float partial = 0.0f;
for (int b = 0; b < B; ++b) {
    float inv = inv_rms[b];
    uint base = uint(b) * uint(n) * uint(C);
    uint base_bc = uint(b) * uint(C);
    for (uint c = lane; c < uint(C); c += tpg) {
        float y_norm = y_agg[base_bc + c] * inv * float(rms_weight[c]);
        float doutv = float(d_out[base + uint(i) * uint(C) + c]);
        partial += doutv * y_norm;
    }
}

reduce_buf[lane] = partial;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    dH_post[i] = reduce_buf[0];
}

"""

STREAM_MIX_ADD_RMS_COL_BF16_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Optimized "Column-Parallel" Mix + Add + RMS Apply for BF16.
//
// Fuses:
// - stream mixing: out[b,i,c] = sum_j M[i,j] * x[b,j,c]
// - final add: out += H_post[i] * (y_agg[b,c] * inv_rms[b] * rms_weight[c])
//
// Optimization:
// - Loads x[b, :, c] (entire column) using vectorized BF16 loads (packed_ushort2).
// - Performs M @ x_col in registers.
// - Writes output as BF16.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] bfloat16
// - M:          [n, n]    float32
// - H_post:     [n]       float32 (activated)
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] bfloat16

constexpr int MAX_N = {{MAX_N}};

#define BF16_TO_FLOAT(v) (as_type<float>(uint(v) << 16))
#define FLOAT_TO_BF16(v) (ushort((as_type<uint>(v) + 0x7FFFu + ((as_type<uint>(v) >> 16) & 1u)) >> 16))

// We process 2 channels at a time (vectorized load)
uint c2 = thread_position_in_grid.x * 2u;
uint b = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if (b >= (uint)B || (int)c2 >= C) {
    return;
}

bool has_second = (int)(c2 + 1u) < C;

// Load M and H_post into threadgroup memory
threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += threads_per_threadgroup.x) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Load input column x[b, :, c2] into registers
// We load pairs if possible
float2 x_col[MAX_N]; // .x is channel c2, .y is channel c2+1

const device ushort* x_bits = reinterpret_cast<const device ushort*>(x);
device ushort* out_bits = reinterpret_cast<device ushort*>(out);

uint base_x = uint(b) * uint(n) * uint(C) + c2;
uint stride_x = uint(C);

for (int j = 0; j < n; ++j) {
    uint idx = base_x + uint(j) * stride_x;
    
    // Check alignment for vector load?
    // MLX buffers are usually aligned?
    // Packed types handle unaligned?
    // If address is 2-byte aligned (ushort), packed_ushort2 works?
    // Ideally we assume alignment.
    // But stride C might make rows unaligned if C is odd.
    // So we use scalar loads if unsure, or verify alignment.
    // Or just use scalar loads and pack manually to avoid bus errors.
    // But scalar loads of bf16 (ushort) are 2 bytes. 
    // Two scalar loads = 4 bytes.
    // Vector load `packed_ushort2` is efficient.
    
    // Let's use scalar loads for safety unless we know C is even.
    // If C is odd, `idx` for j=1 is `base + C`. If `base` is aligned 4, `base+C` is not.
    // So we cannot assume vector alignment across rows.
    // However, we process (c2, c2+1). These are adjacent in memory.
    // `x[idx]` and `x[idx+1]`.
    // We can load them as a pair IF valid.
    
    float v0 = BF16_TO_FLOAT(x_bits[idx]);
    float v1 = 0.0f;
    if (has_second) {
        v1 = BF16_TO_FLOAT(x_bits[idx + 1u]);
    }
    x_col[j] = float2(v0, v1);
}

// Precompute y_dist scalar parts
float inv = float(inv_rms[b]);
uint idx_bc = uint(b) * uint(C) + c2;

float y_agg0 = float(y_agg[idx_bc]);
float rw0 = float(rms_weight[c2]);
float y_norm0 = y_agg0 * inv * rw0;

float y_norm1 = 0.0f;
if (has_second) {
    float y_agg1 = float(y_agg[idx_bc + 1u]);
    float rw1 = float(rms_weight[c2 + 1u]);
    y_norm1 = y_agg1 * inv * rw1;
}

// Compute Mix + Add
for (int i = 0; i < n; ++i) {
    float2 acc = float2(0.0f);
    int row_offset = i * n;
    
    for (int j = 0; j < n; ++j) {
        float m_val = Ms[row_offset + j];
        acc += m_val * x_col[j];
    }
    
    // Add distributed branch
    float hp = Hpost[i];
    float out0 = acc.x + hp * y_norm0;
    
    // Store
    uint out_idx = base_x + uint(i) * stride_x;
    out_bits[out_idx] = FLOAT_TO_BF16(out0);
    
    if (has_second) {
        float out1 = acc.y + hp * y_norm1;
        out_bits[out_idx + 1u] = FLOAT_TO_BF16(out1);
    }
}

"""

MHC_BACKWARD_FUSED_DX_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Fused backward for mHC:
// - computes y_agg, d_y_norm, inv_rms, d_r
// - computes dx
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - M:          [n, n]    float32
// - H_pre:      [n]       float32 (activated)
// - H_post:     [n]       float32 (activated)
// - rms_weight: [C]       float32
// - d_out:      [B, n, C] float32
// - dx:         [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint token = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)token >= B) {
    return;
}

threadgroup float P[MAX_N * MAX_N];
threadgroup float Hpre[MAX_N];
threadgroup float Hpost[MAX_N];
threadgroup float reduce_buf[MAX_TPG];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += tpg) {
    P[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += tpg) {
    Hpre[idx] = H_pre[idx];
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = token * uint(n) * uint(C);
uint base_bc = token * uint(C);

float local_sum = 0.0f;
float local_dr = 0.0f;

for (uint c = lane; c < uint(C); c += tpg) {
    float y_agg_val = 0.0f;
    float d_y_norm_val = 0.0f;
    for (int i = 0; i < n; ++i) {
        uint idx = base + uint(i) * uint(C) + c;
        float xv = float(x[idx]);
        float doutv = float(d_out[idx]);
        y_agg_val += Hpre[i] * xv;
        d_y_norm_val += Hpost[i] * doutv;
    }
    y_agg[base_bc + c] = y_agg_val;
    d_y_norm[base_bc + c] = d_y_norm_val;
    local_sum += y_agg_val * y_agg_val;
    local_dr += d_y_norm_val * y_agg_val * float(rms_weight[c]);
}

reduce_buf[lane] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

float mean_sq = reduce_buf[0] / float(C);
float inv = metal::rsqrt(mean_sq + EPS);

reduce_buf[lane] = local_dr;
threadgroup_barrier(mem_flags::mem_threadgroup);

active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

float dr = reduce_buf[0];
if (lane == 0u) {
    inv_rms[token] = inv;
    d_r[token] = dr;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float inv3 = inv * inv * inv;
float d_mean_sq = -0.5f * dr * inv3;
float scale = 2.0f / float(C);

for (uint c = lane; c < uint(C); c += tpg) {
    float y_agg_val = y_agg[base_bc + c];
    float d_y_norm_val = d_y_norm[base_bc + c];
    float d_y_agg = d_y_norm_val * inv * float(rms_weight[c])
        + d_mean_sq * scale * y_agg_val;

    for (int i = 0; i < n; ++i) {
        float dx_mix = 0.0f;
        for (int k_idx = 0; k_idx < n; ++k_idx) {
            dx_mix += P[k_idx * n + i] * float(d_out[base + uint(k_idx) * uint(C) + c]);
        }
        dx[base + uint(i) * uint(C) + c] = dx_mix + d_y_agg * Hpre[i];
    }
}

"""

MHC_BACKWARD_PREP_TILE_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Tile-parallel backward prep:
// - y_agg[b, c] = sum_i H_pre[i] * x[b, i, c]
// - d_y_norm[b, c] = sum_i H_post[i] * d_out[b, i, c]
// - partial_sq[b, tile] = sum_{c in tile} y_agg[b, c]^2
// - partial_dr[b, tile] = sum_{c in tile} d_y_norm[b, c] * y_agg[b, c] * rms_weight[c]
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32/float16
// - H_pre:      [n]       float32 (activated)
// - H_post:     [n]       float32 (activated)
// - rms_weight: [C]       float32
// - d_out:      [B, n, C] float32/float16
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - partial_sq: [B, T]    float32
// - partial_dr: [B, T]    float32

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};

uint lane = thread_position_in_threadgroup.x;
uint c = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
uint tpg = threads_per_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)b >= B) {
    return;
}

uint tiles = (uint(C) + tpg - 1u) / tpg;
uint tile = c / tpg;

threadgroup float Hpre[MAX_N];
threadgroup float Hpost[MAX_N];
threadgroup float reduce_sq[MAX_TPG];
threadgroup float reduce_dr[MAX_TPG];

for (uint idx = lane; idx < uint(n); idx += tpg) {
    Hpre[idx] = H_pre[idx];
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

int c_i = int(c);
uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

float y_val = 0.0f;
float d_y_norm_val = 0.0f;
if (c_i < C) {
    for (int i = 0; i < n; ++i) {
        uint idx = base + uint(i) * uint(C) + uint(c_i);
        y_val += Hpre[i] * float(x[idx]);
        d_y_norm_val += Hpost[i] * float(d_out[idx]);
    }
    y_agg[base_bc + uint(c_i)] = y_val;
    d_y_norm[base_bc + uint(c_i)] = d_y_norm_val;
}

float local_sq = (c_i < C) ? (y_val * y_val) : 0.0f;
float local_dr = 0.0f;
if (c_i < C) {
    local_dr = d_y_norm_val * y_val * float(rms_weight[c_i]);
}

reduce_sq[lane] = local_sq;
reduce_dr[lane] = local_dr;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_sq[lane] += reduce_sq[lane + stride];
        reduce_dr[lane] += reduce_dr[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_sq[0] += reduce_sq[active - 1u];
        reduce_dr[0] += reduce_dr[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    uint out_idx = uint(b) * tiles + tile;
    partial_sq[out_idx] = reduce_sq[0];
    partial_dr[out_idx] = reduce_dr[0];
}

"""

STREAM_MIX_ADD_RMS_BF16_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// BF16-optimized stream mix + RMS add, writing bfloat16 output.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] bfloat16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] bfloat16

constexpr int MAX_N = {{MAX_N}};

#define BF16_TO_FLOAT(v) (as_type<float>(uint(v) << 16))
#define FLOAT_TO_BF16(v) (ushort((as_type<uint>(v) + 0x7FFFu + ((as_type<uint>(v) >> 16) & 1u)) >> 16))

uint c2 = thread_position_in_grid.x * 2u;
uint k = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || (int)c2 >= C) {
    return;
}

bool has_second = (int)(c2 + 1u) < C;

threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += threads_per_threadgroup.x) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);
uint out_idx = base + uint(i) * uint(C) + c2;

const device ushort* x_bits = reinterpret_cast<const device ushort*>(x);
device ushort* out_bits = reinterpret_cast<device ushort*>(out);

float inv = float(inv_rms[b]);

if (has_second) {
    float2 acc = float2(0.0f);
    for (int j = 0; j < n; ++j) {
        uint idx = base + uint(j) * uint(C) + c2;
        float2 xf = float2(0.0f);
        if (((idx & 1u) == 0u) && has_second) {
            const device packed_ushort2* x_ptr = (const device packed_ushort2*)(x_bits + idx);
            ushort2 xv = ushort2(*x_ptr);
            xf = float2(BF16_TO_FLOAT(xv.x), BF16_TO_FLOAT(xv.y));
        } else {
            float x0 = BF16_TO_FLOAT(x_bits[idx]);
            float x1 = has_second ? BF16_TO_FLOAT(x_bits[idx + 1u]) : 0.0f;
            xf = float2(x0, x1);
        }
        acc += Ms[i * n + j] * xf;
    }

    float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
    float y1 = float(y_agg[base_bc + c2 + 1u]) * inv * float(rms_weight[c2 + 1u]);
    float out0 = acc.x + Hpost[i] * y0;
    float out1 = acc.y + Hpost[i] * y1;

    device packed_ushort2* out_ptr = (device packed_ushort2*)(out_bits + out_idx);
    *out_ptr = packed_ushort2(ushort2(FLOAT_TO_BF16(out0), FLOAT_TO_BF16(out1)));
} else {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        uint idx = base + uint(j) * uint(C) + c2;
        acc += Ms[i * n + j] * BF16_TO_FLOAT(x_bits[idx]);
    }
    float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
    out_bits[out_idx] = FLOAT_TO_BF16(acc + Hpost[i] * y0);
}

"""

MHC_FORWARD_AGG_BF16_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Computes:
// - y_agg[b, c] = sum_i H_pre[i] * x[b, i, c]
// - partial_sq[b, tile] = sum_{c in tile} y_agg[b, c]^2
//
// Expected row-contiguous shapes:
// - x:         [B, n, C] bfloat16
// - H_pre:     [n]       float32 (activated)
// - y_agg:     [B, C]    float32
// - partial_sq:[B, T]    float32 (T = ceil_div(C, threads_per_group))

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};

#define BF16_TO_FLOAT(v) (as_type<float>(uint(v) << 16))

uint lane = thread_position_in_threadgroup.x;
uint c = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
uint tpg = threads_per_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)b >= B) {
    return;
}

uint tiles = (uint(C) + tpg - 1u) / tpg;
uint tile = c / tpg;

threadgroup float Hpre[MAX_N];
threadgroup float reduce_buf[MAX_TPG];

for (uint idx = lane; idx < uint(n); idx += tpg) {
    Hpre[idx] = H_pre[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

int c_i = int(c);
uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

const device ushort* x_bits = reinterpret_cast<const device ushort*>(x);

float y_val = 0.0f;
if (c_i < C) {
    for (int i = 0; i < n; ++i) {
        uint idx = base + uint(i) * uint(C) + uint(c_i);
        y_val += Hpre[i] * BF16_TO_FLOAT(x_bits[idx]);
    }
    y_agg[base_bc + uint(c_i)] = y_val;
}

float local_sq = (c_i < C) ? (y_val * y_val) : 0.0f;
reduce_buf[lane] = local_sq;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

if (lane == 0u) {
    partial_sq[uint(b) * tiles + tile] = reduce_buf[0];
}

"""

STREAM_MIX_ADD_RMS_TILE_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Tile-parallel variant of stream_mix_add_rms using a 2D threadgroup.
// Threadgroup.y == n so a group computes all i in [0,n) for a batch b
// and a tile of channels, amortizing M and H_post loads.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32/float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float32

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint k = thread_position_in_grid.y;
uint lane_x = thread_position_in_threadgroup.x;
uint lane_y = thread_position_in_threadgroup.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || i >= n || (int)c >= C) {
    return;
}

threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint tpg_x = threads_per_threadgroup.x;
uint tpg_y = threads_per_threadgroup.y;
uint tcount = tpg_x * tpg_y;
uint lane = lane_y * tpg_x + lane_x;

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += tcount) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < (uint)n; idx += tcount) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

float acc = 0.0f;
for (int j = 0; j < n; ++j) {
    acc += Ms[i * n + j] * float(x[base + uint(j) * uint(C) + c]);
}

float y = float(y_agg[base_bc + c]) * float(inv_rms[b]) * float(rms_weight[c]);
uint out_idx = base + uint(i) * uint(C) + c;
out[out_idx] = acc + Hpost[i] * y;

"""

STREAM_MIX_ADD_RMS_TILE_FP16_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Half2-tiled stream mix + RMS add for fixed n (TILE_N).
// Computes a [TILE_N x (TILE_C*2)] block of output per threadgroup.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float16

constexpr int TILE_N = {{TILE_N}};
constexpr int TILE_C = {{TILE_C}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint gid_x = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)b >= B) {
    return;
}

uint tile = gid_x / tpg;
uint c2_offset = lane % TILE_C;
uint i = lane / TILE_C;

if ((int)i >= n) {
    return;
}

uint c2 = tile * (TILE_C * 2u) + c2_offset * 2u;

threadgroup float M_tile[TILE_N * TILE_N];
threadgroup float Hpost[TILE_N];
threadgroup float2 X_tile[TILE_N * TILE_C];

uint mn = uint(n * n);
for (uint idx = lane; idx < mn; idx += tpg) {
    M_tile[idx] = M[idx];
}
if (lane < uint(n)) {
    Hpost[lane] = H_post[lane];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

bool has0 = (int)c2 < C;
bool has1 = (int)(c2 + 1u) < C;

float2 xval = float2(0.0f);
if (has0) {
    uint idx = base + uint(i) * uint(C) + c2;
    if (has1) {
        if ((idx & 1u) == 0u) {
            const device packed_half2* x_ptr = (const device packed_half2*)(x + idx);
            half2 xv = half2(*x_ptr);
            xval = float2(float(xv.x), float(xv.y));
        } else {
            xval = float2(float(x[idx]), float(x[idx + 1u]));
        }
    } else {
        xval = float2(float(x[idx]), 0.0f);
    }
}
X_tile[uint(i) * TILE_C + c2_offset] = xval;
threadgroup_barrier(mem_flags::mem_threadgroup);

float2 acc = float2(0.0f);
for (int j = 0; j < n; ++j) {
    acc += M_tile[i * n + j] * X_tile[uint(j) * TILE_C + c2_offset];
}

if (has0) {
    float inv = float(inv_rms[b]);
    float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
    float out0 = acc.x + Hpost[i] * y0;
    uint out_idx = base + uint(i) * uint(C) + c2;
    if (has1) {
        float y1 = float(y_agg[base_bc + c2 + 1u]) * inv * float(rms_weight[c2 + 1u]);
        float out1 = acc.y + Hpost[i] * y1;
        device packed_half2* out_ptr = (device packed_half2*)(out + out_idx);
        *out_ptr = packed_half2(half2(half(out0), half(out1)));
    } else {
        out[out_idx] = half(out0);
    }
}

"""

STREAM_MIX_ADD_RMS_COL_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Optimized "Column-Parallel" Mix + Add + RMS Apply.
//
// Fuses:
// - stream mixing: out[b,i,c] = sum_j M[i,j] * x[b,j,c]
// - final add: out += H_post[i] * (y_agg[b,c] * inv_rms[b] * rms_weight[c])
//
// Optimization:
// - Loads x[b, :, c] (entire column) into registers once.
// - Performs M @ x_col in registers.
// - Reduces x reads from O(n^2) to O(n) per column (factor of n reduction).
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32/float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32 (activated)
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float32

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if (b >= (uint)B || c >= (uint)C) {
    return;
}

// Load M and H_post into threadgroup memory for shared access
threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += threads_per_threadgroup.x) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += threads_per_threadgroup.x) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base_token = b * uint(n) * uint(C);
uint base_token_y = b * uint(C);
float inv_rms_val = inv_rms[b];

// Load x column into registers
float x_col[MAX_N];
for (int i = 0; i < n; ++i) {
    x_col[i] = float(x[base_token + uint(i) * uint(C) + c]);
}

float y_val = y_agg[base_token_y + c];
float y_n = y_val * inv_rms_val * float(rms_weight[c]);

// Compute M * x_col + H_post * y_norm
for (int i = 0; i < n; ++i) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    int row_offset = i * n;
    
    // Unroll 2x (8 elements per step) for ILP
    for (int j = 0; j < n; j += 8) {
        float4 m_vec0 = float4(Ms[row_offset + j], Ms[row_offset + j + 1], Ms[row_offset + j + 2], Ms[row_offset + j + 3]);
        float4 x_vec0 = float4(x_col[j], x_col[j + 1], x_col[j + 2], x_col[j + 3]);
        acc0 += dot(m_vec0, x_vec0);

        float4 m_vec1 = float4(Ms[row_offset + j + 4], Ms[row_offset + j + 5], Ms[row_offset + j + 6], Ms[row_offset + j + 7]);
        float4 x_vec1 = float4(x_col[j + 4], x_col[j + 5], x_col[j + 6], x_col[j + 7]);
        acc1 += dot(m_vec1, x_vec1);
    }
    
    out[base_token + uint(i) * uint(C) + c] = (acc0 + acc1) + Hpost[i] * y_n;
}


"""

STREAM_MIX_ADD_RMS_TILE_F32_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// Tiled stream mix + RMS add for fixed n (TILE_N).
// Computes a [TILE_N x TILE_C] block of output per threadgroup.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32/float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float32

constexpr int TILE_N = {{TILE_N}};
constexpr int TILE_C = {{TILE_C}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint gid_x = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if ((int)b >= B) {
    return;
}

uint tile = gid_x / tpg;
uint c_offset = lane % TILE_C;
uint i = lane / TILE_C;

if ((int)i >= n) {
    return;
}

uint c = tile * TILE_C + c_offset;

threadgroup float M_tile[TILE_N * TILE_N];
threadgroup float Hpost[TILE_N];
threadgroup float X_tile[TILE_N * TILE_C];

uint mn = uint(n * n);
for (uint idx = lane; idx < mn; idx += tpg) {
    M_tile[idx] = M[idx];
}
if (lane < uint(n)) {
    Hpost[lane] = H_post[lane];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

float xval = 0.0f;
if ((int)c < C) {
    xval = float(x[base + uint(i) * uint(C) + c]);
}
X_tile[uint(i) * TILE_C + c_offset] = xval;
threadgroup_barrier(mem_flags::mem_threadgroup);

float acc = 0.0f;
for (int j = 0; j < n; ++j) {
    acc += M_tile[i * n + j] * X_tile[uint(j) * TILE_C + c_offset];
}

if ((int)c < C) {
    float y = float(y_agg[base_bc + c]) * float(inv_rms[b]) * float(rms_weight[c]);
    uint out_idx = base + uint(i) * uint(C) + c;
    out[out_idx] = acc + Hpost[i] * y;
}

"""

STREAM_MIX_ADD_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// This fuses:
// - stream mixing: out[b,i,c] = sum_j M[i,j] * x[b,j,c]
// - final add: out += y_dist[b,i,c]
//
// Expected row-contiguous shapes:
// - x:      [B, n, C] float32
// - M:      [n, n]    float32
// - y_dist: [B, n, C] float32
// - out:    [B, n, C] float32

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint k = thread_position_in_grid.y;
uint t = thread_position_in_threadgroup.x;

int n = x_shape[1];
int C = x_shape[2];

// Each y-slice corresponds to one (b, i).
int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

// Load M into threadgroup memory once per threadgroup.
threadgroup float Ms[MAX_N * MAX_N];

uint mn = (uint)(n * n);
for (uint idx = t; idx < mn; idx += threads_per_threadgroup.x) {
    Ms[idx] = M[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if ((int)c >= C) {
    return;
}

int base = (b * n * C);
float acc = 0.0f;

// M is stored row-major: M[i*n + j]
for (int j = 0; j < n; j++) {
    acc += Ms[i * n + j] * x[base + j * C + (int)c];
}

int out_idx = base + i * C + (int)c;
out[out_idx] = acc + y_dist[out_idx];

"""

MHC_FUSED_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// This fuses:
// - stream aggregate + RMSNorm
// - stream mix: out[b,i,c] = sum_j M[i,j] * x[b,j,c]
// - final add: out += H_post[i] * y_norm[b,c]
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - M:          [n, n]    float32
// - H_pre:      [n]       float32 (activated)
// - H_post:     [n]       float32 (activated)
// - rms_weight: [C]       float32
// - out:        [B, n, C] OUT_T

typedef {{OUT_T}} OUT_T;

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint token = thread_position_in_grid.y;

int n = x_shape[1];
int C = x_shape[2];

threadgroup float P[MAX_N * MAX_N];
threadgroup float Hpre[MAX_N];
threadgroup float Hpost[MAX_N];
threadgroup float reduce_buf[MAX_TPG];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += tpg) {
    P[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += tpg) {
    Hpre[idx] = H_pre[idx];
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base_token = token * uint(n) * uint(C);

float local_sum = 0.0f;
for (uint c = lane; c < uint(C); c += tpg) {
    float y_agg = 0.0f;
    for (int i = 0; i < n; ++i) {
        y_agg += Hpre[i] * float(x[base_token + uint(i) * uint(C) + c]);
    }
    local_sum += y_agg * y_agg;
}
reduce_buf[lane] = local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);

uint active = tpg;
while (active > 1) {
    uint stride = active / 2;
    if (lane < stride) {
        reduce_buf[lane] += reduce_buf[lane + stride];
    }
    if ((active & 1u) != 0u && lane == 0u) {
        reduce_buf[0] += reduce_buf[active - 1u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    active = stride + (active & 1u);
}

float mean_sq = reduce_buf[0] / float(C);
float inv_rms = metal::rsqrt(mean_sq + EPS);

bool fast_n4 = (n == 4);
if (fast_n4) {
    for (uint c = lane; c < uint(C); c += tpg) {
        float x0 = float(x[base_token + 0u * uint(C) + c]);
        float x1 = float(x[base_token + 1u * uint(C) + c]);
        float x2 = float(x[base_token + 2u * uint(C) + c]);
        float x3 = float(x[base_token + 3u * uint(C) + c]);

        float y_agg = Hpre[0] * x0 + Hpre[1] * x1 + Hpre[2] * x2 + Hpre[3] * x3;
        float y_norm = y_agg * inv_rms * float(rms_weight[c]);

        float acc0 = P[0] * x0 + P[1] * x1 + P[2] * x2 + P[3] * x3;
        float acc1 = P[4] * x0 + P[5] * x1 + P[6] * x2 + P[7] * x3;
        float acc2 = P[8] * x0 + P[9] * x1 + P[10] * x2 + P[11] * x3;
        float acc3 = P[12] * x0 + P[13] * x1 + P[14] * x2 + P[15] * x3;

        out[base_token + 0u * uint(C) + c] = (OUT_T)(acc0 + Hpost[0] * y_norm);
        out[base_token + 1u * uint(C) + c] = (OUT_T)(acc1 + Hpost[1] * y_norm);
        out[base_token + 2u * uint(C) + c] = (OUT_T)(acc2 + Hpost[2] * y_norm);
        out[base_token + 3u * uint(C) + c] = (OUT_T)(acc3 + Hpost[3] * y_norm);
    }
} else if ((C % 4) == 0) {
    // Vectorized path for generic n, aligned C
    for (uint c_vec = lane; c_vec * 4 < uint(C); c_vec += tpg) {
        uint c = c_vec * 4;
        
        // Load x columns into registers as float4
        float4 xvals[MAX_N];
        float4 y_agg_vec = 0.0f;
        
        for (int j = 0; j < n; ++j) {
            float4 xv = float4(*(const device float4*)(x + base_token + uint(j) * uint(C) + c));
            xvals[j] = xv;
            y_agg_vec += Hpre[j] * xv;
        }
        
        float4 w_vec = float4(*(const device float4*)(rms_weight + c));
        float4 y_norm_vec = y_agg_vec * inv_rms * w_vec;
        
        for (int i = 0; i < n; ++i) {
            float4 acc_vec = 0.0f;
            int row = i * n;
            for (int j = 0; j < n; ++j) {
                acc_vec += P[row + j] * xvals[j];
            }
            uint out_idx = base_token + uint(i) * uint(C) + c;
            float4 res = acc_vec + Hpost[i] * y_norm_vec;

            if (sizeof(OUT_T) == 4) {
                *(device float4*)(out + out_idx) = res;
            } else {
                out[out_idx + 0] = (OUT_T)res.x;
                out[out_idx + 1] = (OUT_T)res.y;
                out[out_idx + 2] = (OUT_T)res.z;
                out[out_idx + 3] = (OUT_T)res.w;
            }
        }
    }
} else {
    // Scalar fallback
    for (uint c = lane; c < uint(C); c += tpg) {
        float y_agg = 0.0f;
        float xvals[MAX_N];

        for (int j = 0; j < n; ++j) {
            float xv = float(x[base_token + uint(j) * uint(C) + c]);
            xvals[j] = xv;
            y_agg += Hpre[j] * xv;
        }

        float y_norm = y_agg * inv_rms * float(rms_weight[c]);

        for (int i = 0; i < n; ++i) {
            float acc = 0.0f;
            int row = i * n;
            for (int j = 0; j < n; ++j) {
                acc += P[row + j] * xvals[j];
            }
            uint out_idx = base_token + uint(i) * uint(C) + c;
            out[out_idx] = (OUT_T)(acc + Hpost[i] * y_norm);
        }
    }
}

"""

STREAM_MIX_ADD_RMS_TILE2D_FP16_METAL = r"""
// MLX Metal kernel body (not a full .metal file).
//
// 2D threadgroup variant of stream mix + RMS add (half output).
// Uses threadgroup.y == n so a group computes all i in [0,n) for a batch b,
// amortizing M and H_post loads across the channel tile.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] float16

constexpr int MAX_N = {{MAX_N}};

uint c2 = thread_position_in_grid.x * 2u;
uint k = thread_position_in_grid.y;
uint lane_x = thread_position_in_threadgroup.x;
uint lane_y = thread_position_in_threadgroup.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || i >= n || (int)c2 >= C) {
    return;
}

threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint tpg_x = threads_per_threadgroup.x;
uint tpg_y = threads_per_threadgroup.y;
uint tcount = tpg_x * tpg_y;
uint lane = lane_y * tpg_x + lane_x;

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += tcount) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < (uint)n; idx += tcount) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

bool has1 = (int)(c2 + 1u) < C;

float2 acc = float2(0.0f);
for (int j = 0; j < n; ++j) {
    uint idx = base + uint(j) * uint(C) + c2;
    float2 xval = float2(0.0f);
    if (has1) {
        if ((idx & 1u) == 0u) {
            const device packed_half2* x_ptr = (const device packed_half2*)(x + idx);
            half2 xv = half2(*x_ptr);
            xval = float2(float(xv.x), float(xv.y));
        } else {
            xval = float2(float(x[idx]), float(x[idx + 1u]));
        }
    } else {
        xval = float2(float(x[idx]), 0.0f);
    }
    acc += Ms[i * n + j] * xval;
}

float inv = float(inv_rms[b]);
float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
float out0 = acc.x + Hpost[i] * y0;
uint out_idx = base + uint(i) * uint(C) + c2;
if (has1) {
    float y1 = float(y_agg[base_bc + c2 + 1u]) * inv * float(rms_weight[c2 + 1u]);
    float out1 = acc.y + Hpost[i] * y1;
    if ((out_idx & 1u) == 0u) {
        device packed_half2* out_ptr = (device packed_half2*)(out + out_idx);
        *out_ptr = packed_half2(half2(half(out0), half(out1)));
    } else {
        out[out_idx] = half(out0);
        out[out_idx + 1u] = half(out1);
    }
} else {
    out[out_idx] = half(out0);
}

"""

