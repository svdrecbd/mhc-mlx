\
/*
mhc_mix.metal (MLX custom kernel source body)

MLX will generate the full Metal function signature automatically.
This file must contain only the kernel body (no [[kernel]] signature, no includes).
*/

constexpr int N = 4;
constexpr int ITER = 20;
constexpr float EPS = 1e-6f;

// Thread identifiers
uint lane = thread_position_in_threadgroup.x;
uint tpg  = threads_per_threadgroup.x;
uint token = thread_position_in_grid.y;

// x is [tokens, N, C] row-contiguous
int C = x_shape[2];

// Shared 4x4 Sinkhorn matrix for this token
threadgroup float P[16];

// 1) Build P once per token
if (lane == 0) {
    float H[16];
    float maxv = -1e30f;

    uint logit_base = token * (N * N);

    // load logits and find max
    for (int i = 0; i < N * N; ++i) {
        float v = float(logits[logit_base + uint(i)]);
        H[i] = v;
        maxv = (v > maxv) ? v : maxv;
    }

    // exp(x - max)
    for (int i = 0; i < N * N; ++i) {
        H[i] = metal::exp(H[i] - maxv);
    }

    // Sinkhorn-Knopp: alternate row/col normalization
    for (int it = 0; it < ITER; ++it) {
        // row norm
        for (int r = 0; r < N; ++r) {
            float s = 0.0f;
            s += H[r * N + 0];
            s += H[r * N + 1];
            s += H[r * N + 2];
            s += H[r * N + 3];
            float inv = 1.0f / (s + EPS);
            H[r * N + 0] *= inv;
            H[r * N + 1] *= inv;
            H[r * N + 2] *= inv;
            H[r * N + 3] *= inv;
        }

        // col norm
        for (int c = 0; c < N; ++c) {
            float s = 0.0f;
            s += H[0 * N + c];
            s += H[1 * N + c];
            s += H[2 * N + c];
            s += H[3 * N + c];
            float inv = 1.0f / (s + EPS);
            H[0 * N + c] *= inv;
            H[1 * N + c] *= inv;
            H[2 * N + c] *= inv;
            H[3 * N + c] *= inv;
        }
    }

    // write to threadgroup memory
    for (int i = 0; i < N * N; ++i) {
        P[i] = H[i];
    }
}

threadgroup_barrier(mem_flags::mem_threadgroup);

// 2) Mix streams: each thread handles channels lane, lane+tpg, lane+2*tpg, ...
// layout index: ((token * N + stream) * C + channel)
uint base_token = token * uint(N) * uint(C);

for (uint c = lane; c < uint(C); c += tpg) {
    // load x for 4 streams at channel c
    float x0 = float(x[base_token + 0u * uint(C) + c]);
    float x1 = float(x[base_token + 1u * uint(C) + c]);
    float x2 = float(x[base_token + 2u * uint(C) + c]);
    float x3 = float(x[base_token + 3u * uint(C) + c]);

    // y0..y3 = P @ xvec
    float y0 = P[0]  * x0 + P[1]  * x1 + P[2]  * x2 + P[3]  * x3;
    float y1 = P[4]  * x0 + P[5]  * x1 + P[6]  * x2 + P[7]  * x3;
    float y2 = P[8]  * x0 + P[9]  * x1 + P[10] * x2 + P[11] * x3;
    float y3 = P[12] * x0 + P[13] * x1 + P[14] * x2 + P[15] * x3;

    out[base_token + 0u * uint(C) + c] = y0;
    out[base_token + 1u * uint(C) + c] = y1;
    out[base_token + 2u * uint(C) + c] = y2;
    out[base_token + 3u * uint(C) + c] = y3;
}
