// MLX Metal kernel body (not a full .metal file).
//
// Optimized Fused Backward Gradients (n <= 32).
//
// Computes:
// - dM[i, j] = sum_{b, c} d_out[b, i, c] * x[b, j, c]
// - dH_pre[i] = sum_{b, c} d_y_agg[b, c] * x[b, i, c]
// - dH_post[i] = sum_{b, c} d_out[b, i, c] * y_norm[b, c]
// - d_rms_weight[c] (atomic add)
//
// Strategy:
// - TPG = 32. Each thread 'i' owns row 'i' of dM.
// - Grid-stride loop over (b, c).
// - Inside loop:
//   - Thread 'i' loads x[b, i, c] and dout[b, i, c].
//   - Share x[b, :, c] via simd_shuffle (no shared mem needed!).
//   - Accumulate dM[i, :] into 32 registers.
//   - Accumulate dH_pre[i] and dH_post[i] into registers.
//   - Compute d_rms contribution and atomic-add to global.
// - After loop:
//   - Atomic-add 32 registers to global dM[i, :].
//   - Atomic-add dH registers to global dH.
//
// Expected row-contiguous shapes:
// - x:          [B, n, C] float32
// - d_out:      [B, n, C] float32
// - y_agg:      [B, C]    float32
// - d_y_norm:   [B, C]    float32
// - inv_rms:    [B]       float32
// - d_r:        [B]       float32
// - rms_weight: [C]       float32
// - dM:         [n, n]    float32 (atomic accum)
// - dH_pre:     [n]       float32 (atomic accum)
// - dH_post:    [n]       float32 (atomic accum)
// - d_rms_weight: [C]     float32 (atomic accum)

constexpr int MAX_N = {{MAX_N}};

uint tid = thread_position_in_grid.x;
uint tpg = threads_per_threadgroup.x; // Should be 32
uint lane = thread_position_in_threadgroup.x;
uint gsize = grid_dim.x; // Total threads

int B = x_shape[0];
int n = x_shape[1];
int C = x_shape[2];

// This kernel is specialized for n <= 32 and TPG=32
if (tpg != 32 || lane >= 32) {
    return;
}

// Helper for atomic float add
inline void atomic_add_float(device float* addr, float val) {
    device atomic_uint* addr_u = (device atomic_uint*)addr;
    uint old = atomic_load_explicit(addr_u, memory_order_relaxed);
    uint expected = old;
    uint desired;
    do {
        expected = old;
        desired = as_type<uint>(as_type<float>(expected) + val);
    } while (!atomic_compare_exchange_weak_explicit(
        addr_u, &old, desired, memory_order_relaxed, memory_order_relaxed));
}

// Registers for dM row 'lane' (32 elements)
float acc_dM[32];
for (int k = 0; k < 32; ++k) {
    acc_dM[k] = 0.0f;
}
float acc_dH_pre = 0.0f;
float acc_dH_post = 0.0f;

// Total work items = B * C
uint total_items = uint(B) * uint(C);

// Each threadgroup processes a set of (b, c) pairs.
// Since TPG=32 and we use simd functions, the whole threadgroup works on the SAME (b, c) pair?
// Yes, strategy: Threadgroup 'g' processes pairs p, p+G, p+2G...
// where G is number of threadgroups.
// Inside step: All 32 threads work on pair 'p'.
// Thread 'lane' handles stream 'lane'.

uint gid = tid / 32u; // Threadgroup ID
uint num_groups = gsize / 32u;

for (uint idx = gid; idx < total_items; idx += num_groups) {
    uint b = idx / uint(C);
    uint c = idx % uint(C);
    
    // Load scalars for this (b, c)
    // Only needed by threads active (lane < n)
    // But since n <= 32 and we assumed TPG=32, lane < n check needed for loads.
    
    // Actually, simple mask:
    bool active = (lane < uint(n));
    
    float xi = 0.0f;
    float douti = 0.0f;
    
    uint base = uint(b) * uint(n) * uint(C) + uint(c);
    if (active) {
        xi = float(x[base + lane * uint(C)]);
        douti = float(d_out[base + lane * uint(C)]);
    }
    
    // Precompute d_y_agg, y_norm
    // These are scalar per (b, c), so all threads in group compute same value.
    // Redundant but cheap.
    float inv = float(inv_rms[b]);
    float dr = float(d_r[b]);
    float inv3 = inv * inv * inv;
    float d_mean_sq = -0.5f * dr * inv3;
    
    uint idx_bc = uint(b) * uint(C) + uint(c);
    float y_agg_val = float(y_agg[idx_bc]);
    float d_y_norm_val = float(d_y_norm[idx_bc]);
    float rw = float(rms_weight[c]);
    
    float d_y_agg = d_y_norm_val * inv * rw + d_mean_sq * (2.0f / float(C)) * y_agg_val;
    float y_norm = y_agg_val * inv * rw;
    
    // Accumulate dH
    if (active) {
        acc_dH_pre += d_y_agg * xi;
        acc_dH_post += douti * y_norm;
    }
    
    // d_rms_weight atomic add
    // d_rms_weight[c] += d_y_norm * y_agg * inv_rms
    // This assumes we only want d_rms_weight from this backward.
    // Only one thread should add per (b, c)? 
    // Or we duplicate adds? 
    // Let thread 0 do it.
    if (lane == 0u) {
        float val = d_y_norm_val * y_agg_val * inv;
        atomic_add_float((device float*)&d_rms_weight[c], val);
    }
    
    // Accumulate dM row 'lane'
    // dM[lane, j] += dout[lane] * x[j]
    if (active) {
        float my_dout = douti;
        
        // Broadcast x[j] from thread j
        for (int j = 0; j < n; ++j) {
            float xj = simd_shuffle(xi, j); // Get xi from thread j
            acc_dM[j] += my_dout * xj;
        }
    }
}

// Final atomic update to global memory
if (lane < uint(n)) {
    // dH
    atomic_add_float((device float*)&dH_pre[lane], acc_dH_pre);
    atomic_add_float((device float*)&dH_post[lane], acc_dH_post);
    
    // dM row
    for (int j = 0; j < n; ++j) {
        atomic_add_float((device float*)&dM[lane * uint(n) + uint(j)], acc_dM[j]);
    }
}
