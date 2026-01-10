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
