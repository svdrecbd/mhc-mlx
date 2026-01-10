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
