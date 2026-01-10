// MLX Metal kernel body (not a full .metal file).
//
// Computes dH_post for the distribute branch:
// dH_post[i] = sum_{b, c} d_out[b, i, c] * y_norm[b, c]
// where y_norm = y_agg * inv_rms * rms_weight.
//
// Expected row-contiguous shapes:
// - d_out:      [B, n, C] float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - dH_post:    [n]       float32

constexpr int MAX_TPG = {{MAX_TPG}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint i = thread_position_in_grid.y;

int B = d_out_shape[0];
int n = d_out_shape[1];
int C = d_out_shape[2];

if ((int)i >= n) {
    return;
}

threadgroup float reduce_buf[MAX_TPG];

float partial = 0.0f;
for (int b = 0; b < B; ++b) {
    float inv = inv_rms[b];
    uint base = uint(b) * uint(n) * uint(C);
    uint base_bc = uint(b) * uint(C);
    for (uint c = lane; c < uint(C); c += tpg) {
        float y_norm = y_agg[base_bc + c] * inv * float(rms_weight[c]);
        float doutv = float(d_out[base + uint(i) * uint(C) + c]);
        partial += doutv * y_norm;
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
    dH_post[i] = reduce_buf[0];
}
