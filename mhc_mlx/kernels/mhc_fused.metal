// MLX Metal kernel body (not a full .metal file).
//
// This fuses:
// - stream aggregate + RMSNorm
// - stream mix: out[b,i,c] = sum_j M[i,j] * x[b,j,c]
// - final add: out += H_post[i] * y_norm[b,c]
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - M:          [n, n]    float32
// - H_pre:      [n]       float32 (activated)
// - H_post:     [n]       float32 (activated)
// - rms_weight: [C]       float32
// - out:        [B, n, C] float32

constexpr int MAX_N = {{MAX_N}};
constexpr int MAX_TPG = {{MAX_TPG}};
constexpr float EPS = {{EPS}};

uint lane = thread_position_in_threadgroup.x;
uint tpg = threads_per_threadgroup.x;
uint token = thread_position_in_grid.y;

int n = x_shape[1];
int C = x_shape[2];

threadgroup float P[MAX_N * MAX_N];
threadgroup float Hpre[MAX_N];
threadgroup float Hpost[MAX_N];
threadgroup float reduce_buf[MAX_TPG];

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += tpg) {
    P[idx] = M[idx];
}
for (uint idx = lane; idx < uint(n); idx += tpg) {
    Hpre[idx] = H_pre[idx];
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base_token = token * uint(n) * uint(C);

float local_sum = 0.0f;
for (uint c = lane; c < uint(C); c += tpg) {
    float y_agg = 0.0f;
    for (int i = 0; i < n; ++i) {
        y_agg += Hpre[i] * float(x[base_token + uint(i) * uint(C) + c]);
    }
    local_sum += y_agg * y_agg;
}
reduce_buf[lane] = local_sum;
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

float mean_sq = reduce_buf[0] / float(C);
float inv_rms = metal::rsqrt(mean_sq + EPS);

bool fast_n4 = (n == 4);
for (uint c = lane; c < uint(C); c += tpg) {
    if (fast_n4) {
        float x0 = float(x[base_token + 0u * uint(C) + c]);
        float x1 = float(x[base_token + 1u * uint(C) + c]);
        float x2 = float(x[base_token + 2u * uint(C) + c]);
        float x3 = float(x[base_token + 3u * uint(C) + c]);

        float y_agg = Hpre[0] * x0 + Hpre[1] * x1 + Hpre[2] * x2 + Hpre[3] * x3;
        float y_norm = y_agg * inv_rms * float(rms_weight[c]);

        float acc0 = P[0] * x0 + P[1] * x1 + P[2] * x2 + P[3] * x3;
        float acc1 = P[4] * x0 + P[5] * x1 + P[6] * x2 + P[7] * x3;
        float acc2 = P[8] * x0 + P[9] * x1 + P[10] * x2 + P[11] * x3;
        float acc3 = P[12] * x0 + P[13] * x1 + P[14] * x2 + P[15] * x3;

        out[base_token + 0u * uint(C) + c] = acc0 + Hpost[0] * y_norm;
        out[base_token + 1u * uint(C) + c] = acc1 + Hpost[1] * y_norm;
        out[base_token + 2u * uint(C) + c] = acc2 + Hpost[2] * y_norm;
        out[base_token + 3u * uint(C) + c] = acc3 + Hpost[3] * y_norm;
    } else {
        float y_agg = 0.0f;
        float xvals[MAX_N];

        for (int j = 0; j < n; ++j) {
            float xv = float(x[base_token + uint(j) * uint(C) + c]);
            xvals[j] = xv;
            y_agg += Hpre[j] * xv;
        }

        float y_norm = y_agg * inv_rms * float(rms_weight[c]);

        for (int i = 0; i < n; ++i) {
            float acc = 0.0f;
            int row = i * n;
            for (int j = 0; j < n; ++j) {
                acc += P[row + j] * xvals[j];
            }
            uint out_idx = base_token + uint(i) * uint(C) + c;
            out[out_idx] = acc + Hpost[i] * y_norm;
        }
    }
}
