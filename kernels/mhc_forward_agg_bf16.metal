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
