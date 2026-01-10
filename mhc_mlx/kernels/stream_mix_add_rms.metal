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
