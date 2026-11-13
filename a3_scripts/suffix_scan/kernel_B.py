import triton
import triton.language as tl

@triton.jit
def _suffix_local_scan_kernel_early_exit(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int32  [B]  batch_idx -> req_state_idx
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    scratch_ptr,  # *int64  [B, scratch_stride]  (output)
    scratch_stride,
    Lp1,  # int64 packing radix (> any position)
    MAX_DEPTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per (request, block): best (match_len, end_pos) packed into one int64.

    end_pos is the exclusive end of an earlier occurrence of the request's
    suffix in its own history; the draft continues from end_pos. Longer
    matches win, then later positions.
    """
    b = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + req).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)
    row = token_ids_ptr + req * token_ids_stride

    scratch_off = b * scratch_stride + blk
    # Blocks past the last candidate (an occurrence must leave a token).
    if blk * BLOCK >= seq_len:
        tl.store(scratch_ptr + scratch_off, tl.zeros((), tl.int64))
        return

    pos = blk * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    active = (pos >= 1) & (pos < seq_len) & (num_sampled > 0)
    # Keep indirect-load masks independent of loop-carried match state so
    # Triton-Ascend can lower the loop. Matching still stops at the first miss.
    candidate_mask = active
    match_len = tl.zeros([BLOCK], dtype=tl.int64)
    t = 0
    while (t < MAX_DEPTH) & (tl.sum(active.to(tl.int32), axis=0) > 0):
        # Request token t back from its end (newest first).
        tail_pos = seq_len - 1 - t
        tail_tok = tl.load(row + tl.maximum(tail_pos, 0), mask=tail_pos >= 0, other=-2)
        src_pos = pos - 1 - t
        load_mask = candidate_mask & (src_pos >= 0) & (tail_pos >= 0)
        tok = tl.load(row + tl.minimum(tl.maximum(src_pos, 0), token_ids_stride - 1), mask=load_mask, other=-1)
        active = active & load_mask & (tok == tail_tok)
        match_len += active.to(tl.int64)
        t += 1

    score = tl.where(match_len > 0, match_len * Lp1 + pos, 0)
    tl.store(scratch_ptr + scratch_off, tl.max(score, axis=0))
