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
for (uint idx = lane; idx < nn; idx += tpg) {
    H[idx] = metal::exp(float(H_res[idx]));
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
