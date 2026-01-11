// MLX Metal kernel body.
//
// Fuses Residual Add + mHC Aggregate.
//
// Computes:
// - out[b, n, c] = x[b, n, c] + res[b, n, c]
// - y_agg[b, c] = sum_i (out[b, i, c] * H_pre[i])
//
// Inputs:
// - x: [B, n, C]
// - res: [B, n, C]
// - H_pre: [n]
//
// Outputs:
// - out: [B, n, C]
// - y_agg: [B, C]

typedef {{OUT_T}} OUT_T;

constexpr int MAX_N = {{MAX_N}};

uint c = thread_position_in_grid.x;
uint b = thread_position_in_grid.y;
uint lane = thread_position_in_threadgroup.x;

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

if (b >= (uint)B) return;

// Load H_pre into shared memory
threadgroup float H[MAX_N];
for (uint i = lane; i < uint(n); i += threads_per_threadgroup.x) {
    H[i] = H_pre[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

uint base_token = b * uint(n) * uint(C);
uint base_y = b * uint(C);

if ((C % 4) == 0) {
    // Vectorized path
    uint c_vec = c * 4;
    if (c_vec >= (uint)C) return;

    float4 agg = 0.0f;
    for (int i = 0; i < n; ++i) {
        uint idx = base_token + uint(i) * uint(C) + c_vec;
        
        // Load x and res
        float4 xv = float4(*(const device float4*)(x + idx));
        float4 rv = float4(*(const device float4*)(res + idx));
        
        // Add
        float4 val = xv + rv;
        
        // Store
        *(device float4*)(out + idx) = val;
        
        // Aggregate
        agg += val * H[i];
    }
    
    // Store y_agg
    *(device float4*)(y_agg + base_y + c_vec) = agg;

} else {
    // Scalar fallback
    if (c >= (uint)C) return;

    float agg = 0.0f;
    for (int i = 0; i < n; ++i) {
        uint idx = base_token + uint(i) * uint(C) + c;
        float val = float(x[idx]) + float(res[idx]);
        out[idx] = (OUT_T)val;
        agg += val * H[i];
    }
    y_agg[base_y + c] = agg;
}
