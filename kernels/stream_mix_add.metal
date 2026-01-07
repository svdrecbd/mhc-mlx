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
