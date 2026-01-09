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

threadgroup float reduce_buf[MAX_TPG];

float partial = 0.0f;
for (int b = 0; b < B; ++b) {
    uint base = uint(b) * uint(n) * uint(C);
    for (uint c = lane; c < uint(C); c += tpg) {
        float doutv = float(d_out[base + uint(i) * uint(C) + c]);
        float xv = float(x[base + uint(j) * uint(C) + c]);
        partial += doutv * xv;
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
    dM[idx] = reduce_buf[0];
}
