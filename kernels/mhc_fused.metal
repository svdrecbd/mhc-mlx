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
// - H_pre:      [n]       float32
// - H_post:     [n]       float32
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
if (lane < MAX_TPG) {
    reduce_buf[lane] = local_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (lane == 0) {
    float sum_sq = 0.0f;
    uint limit = tpg;
    if (limit > MAX_TPG) {
        limit = MAX_TPG;
    }
    for (uint i = 0; i < limit; ++i) {
        sum_sq += reduce_buf[i];
    }
    reduce_buf[0] = sum_sq;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float mean_sq = reduce_buf[0] / float(C);
float inv_rms = 1.0f / metal::sqrt(mean_sq + EPS);

for (uint c = lane; c < uint(C); c += tpg) {
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
