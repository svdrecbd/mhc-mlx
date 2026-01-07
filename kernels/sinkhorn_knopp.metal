// MLX Metal kernel body (not a full .metal file).
//
// Computes: M = sinkhorn_knopp(I + H_res)
//
// Expected row-contiguous shapes:
// - H_res: [n, n] float32
// - out:   [n, n] float32

constexpr int MAX_N = {{MAX_N}};
constexpr int ITERS = {{ITERS}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;

int n = H_res_shape[0];

if (lane == 0) {
    float H[MAX_N * MAX_N];

    // Load residual + identity
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            int idx = r * n + c;
            float v = float(H_res[idx]);
            if (r == c) {
                v += 1.0f;
            }
            H[idx] = v;
        }
    }

    // Sinkhorn-Knopp: alternate row/col normalization
    for (int it = 0; it < ITERS; ++it) {
        // Row norm
        for (int r = 0; r < n; ++r) {
            float s = 0.0f;
            for (int c = 0; c < n; ++c) {
                s += H[r * n + c];
            }
            float scale = (s > EPS) ? (1.0f / s) : 1.0f;
            for (int c = 0; c < n; ++c) {
                H[r * n + c] *= scale;
            }
        }

        // Col norm
        for (int c = 0; c < n; ++c) {
            float s = 0.0f;
            for (int r = 0; r < n; ++r) {
                s += H[r * n + c];
            }
            float scale = (s > EPS) ? (1.0f / s) : 0.0f;
            for (int r = 0; r < n; ++r) {
                H[r * n + c] *= scale;
            }
        }
    }

    for (int i = 0; i < n * n; ++i) {
        out[i] = H[i];
    }
}
