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

float acc = 0.0f;
for (int k_idx = 0; k_idx < n; ++k_idx) {
    acc += P[k_idx * n + i] * float(d_out[base + uint(k_idx) * uint(C) + c]);
}

dx[base + uint(i) * uint(C) + c] = acc;
