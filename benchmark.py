import pytest
import torch

import triton
import triton.language as tl

from configs import *

from flash_atten_fp import attention
from flash_atten_int8 import attention_int8
from flash_atten_full_int8 import attention_full_int8

try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for causal in [False]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=[2**i for i in range(10, 15)],
            line_arg="provider",
            line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) + ["triton-int8"] +
            ["triton-full-int8"] + (["flash"] if HAS_FLASH else []),
            line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) + ["Triton [int8]"] +
            ["Triton [full int8]"] + (["Flash-2"] if HAS_FLASH else []),
            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-"), ("m", "-")],
            ylabel="ms",
            plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-causal={causal}",
            args={
                "H": N_HEADS,
                "BATCH": BATCH,
                "HEAD_DIM": HEAD_DIM,
                "causal": causal,
            },
        ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, provider, device="cuda"):
    warmup = 25
    rep = 100
    dtype = torch.float16
    if "triton" in provider:
        if "int8" in provider:
            if "full" in provider:
                q = torch.randint(-128, 127, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device, requires_grad=False)
                k = torch.randint(-128, 127, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device, requires_grad=False)
                v = torch.randint(-128, 127, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device, requires_grad=False)
                q_scale = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=False)
                k_scale = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=False)
                v_scale = torch.randn((BATCH, H),        dtype=dtype, device=device, requires_grad=False)
                sm_scale = 1.3
                fn = lambda: attention_full_int8(q, k, v, q_scale, k_scale, v_scale, causal, sm_scale)
            else:
                q = torch.randint(-128, 127, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device, requires_grad=False)
                k = torch.randint(-128, 127, (BATCH, H, N_CTX, HEAD_DIM), dtype=torch.int8, device=device, requires_grad=False)
                v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
                q_scale = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=False)
                k_scale = torch.randn((BATCH, H, N_CTX), dtype=dtype, device=device, requires_grad=False)
                sm_scale = 1.3
                fn = lambda: attention_int8(q, k, v, q_scale, k_scale, causal, sm_scale)
        else:
            q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
            k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
            v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
            if "fp8" in provider:
                q = q.to(torch.float8_e5m2)
                k = k.to(torch.float8_e5m2)
                v = v.permute(0, 1, 3, 2).contiguous()
                v = v.permute(0, 1, 3, 2)
                v = v.to(torch.float8_e5m2)
            sm_scale = 1.3
            fn = lambda: attention(q, k, v, causal, sm_scale)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
    

@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    # triton implementation
    tri_out = attention(q, k, v, causal, sm_scale).half()
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
