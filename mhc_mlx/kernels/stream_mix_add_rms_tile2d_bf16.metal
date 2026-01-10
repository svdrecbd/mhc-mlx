// MLX Metal kernel body (not a full .metal file).
//
// 2D threadgroup variant of stream mix + RMS add (bf16 output).
// Uses threadgroup.y == n so a group computes all i in [0,n) for a batch b,
// amortizing M and H_post loads across the channel tile.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] bfloat16
// - M:          [n, n]    float32
// - H_post:     [n]       float32
// - y_agg:      [B, C]    float32
// - inv_rms:    [B]       float32
// - rms_weight: [C]       float32
// - out:        [B, n, C] bfloat16

constexpr int MAX_N = {{MAX_N}};

#define BF16_TO_FLOAT(v) (as_type<float>(uint(v) << 16))
#define FLOAT_TO_BF16(v) (ushort((as_type<uint>(v) + 0x7FFFu + ((as_type<uint>(v) >> 16) & 1u)) >> 16))

uint c2 = thread_position_in_grid.x * 2u;
uint k = thread_position_in_grid.y;
uint lane_x = thread_position_in_threadgroup.x;
uint lane_y = thread_position_in_threadgroup.y;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

int b = (int)(k / (uint)n);
int i = (int)(k - (uint)(b * n));

if (b >= B || i >= n || (int)c2 >= C) {
    return;
}

threadgroup float Ms[MAX_N * MAX_N];
threadgroup float Hpost[MAX_N];

uint tpg_x = threads_per_threadgroup.x;
uint tpg_y = threads_per_threadgroup.y;
uint tcount = tpg_x * tpg_y;
uint lane = lane_y * tpg_x + lane_x;

uint mn = (uint)(n * n);
for (uint idx = lane; idx < mn; idx += tcount) {
    Ms[idx] = M[idx];
}
for (uint idx = lane; idx < (uint)n; idx += tcount) {
    Hpost[idx] = H_post[idx];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base = uint(b) * uint(n) * uint(C);
uint base_bc = uint(b) * uint(C);

const device ushort* x_bits = reinterpret_cast<const device ushort*>(x);
device ushort* out_bits = reinterpret_cast<device ushort*>(out);

bool has1 = (int)(c2 + 1u) < C;

float2 acc = float2(0.0f);
for (int j = 0; j < n; ++j) {
    uint idx = base + uint(j) * uint(C) + c2;
    float2 xval = float2(0.0f);
    if (has1) {
        if ((idx & 1u) == 0u) {
            const device packed_ushort2* x_ptr = (const device packed_ushort2*)(x_bits + idx);
            ushort2 xv = ushort2(*x_ptr);
            xval = float2(BF16_TO_FLOAT(xv.x), BF16_TO_FLOAT(xv.y));
        } else {
            xval = float2(BF16_TO_FLOAT(x_bits[idx]), BF16_TO_FLOAT(x_bits[idx + 1u]));
        }
    } else {
        xval = float2(BF16_TO_FLOAT(x_bits[idx]), 0.0f);
    }
    acc += Ms[i * n + j] * xval;
}

float inv = float(inv_rms[b]);
float y0 = float(y_agg[base_bc + c2]) * inv * float(rms_weight[c2]);
float out0 = acc.x + Hpost[i] * y0;
uint out_idx = base + uint(i) * uint(C) + c2;
if (has1) {
    float y1 = float(y_agg[base_bc + c2 + 1u]) * inv * float(rms_weight[c2 + 1u]);
    float out1 = acc.y + Hpost[i] * y1;
    if ((out_idx & 1u) == 0u) {
        device packed_ushort2* out_ptr = (device packed_ushort2*)(out_bits + out_idx);
        *out_ptr = packed_ushort2(ushort2(FLOAT_TO_BF16(out0), FLOAT_TO_BF16(out1)));
    } else {
        out_bits[out_idx] = FLOAT_TO_BF16(out0);
        out_bits[out_idx + 1u] = FLOAT_TO_BF16(out1);
    }
} else {
    out_bits[out_idx] = FLOAT_TO_BF16(out0);
}
