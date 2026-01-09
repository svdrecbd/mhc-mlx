// MLX Metal kernel body (not a full .metal file).
//
// Computes d_rms_weight:
// d_rms_weight[c] = sum_b d_y_norm[b, c] * y_agg[b, c] * inv_rms[b]
//
// Expected row-contiguous shapes:
// - y_agg:        [B, C] float32
// - d_y_norm:     [B, C] float32
// - inv_rms:      [B]    float32
// - d_rms_weight: [C]    float32

uint c = thread_position_in_grid.x;

int B = y_agg_shape[0];
int C = y_agg_shape[1];

if ((int)c >= C) {
    return;
}

float acc = 0.0f;
for (int b = 0; b < B; ++b) {
    uint base = uint(b) * uint(C);
    float inv = inv_rms[b];
    float y_val = y_agg[base + c];
    float d_y_val = d_y_norm[base + c];
    acc += d_y_val * y_val * inv;
}

d_rms_weight[c] = acc;
