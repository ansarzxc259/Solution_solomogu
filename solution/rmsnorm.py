"""Fused RMSNorm (forward + backward) in Triton for Qwen2RMSNorm.

Numerics follow transformers' Qwen2RMSNorm:
    x32 = x.float(); xhat = x32 * rsqrt(mean(x32^2) + eps)
    out = weight * xhat.to(x.dtype)        # dtype = promote(weight, x)
The norm weights are fp32 after prepare_model_for_kbit_training, so the
output is fp32 there; we keep exactly that dtype behaviour.

Memory: only the input (fp16) and a per-row rstd are saved for backward,
instead of fp32 copies of the activations.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_fwd(X, W, Y, RSTD, stride_x, stride_y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    # round the normalised value to the input dtype, like the reference does
    xn = (x * rstd).to(X.dtype.element_ty).to(tl.float32)
    y = w * xn
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + row, rstd)


@triton.jit
def _rms_bwd(DY, X, W, RSTD, DX, stride_dy, stride_x, stride_dx, N,
             BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + row * stride_dy + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD + row)
    xhat = x * rstd
    dxh = dy * w
    c = tl.sum(dxh * xhat, axis=0) / N
    dx = rstd * (dxh - xhat * c)
    tl.store(DX + row * stride_dx + cols, dx.to(DX.dtype.element_ty), mask=mask)


def _warps(block):
    return min(16, max(4, block // 256))


class RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, N = x2.shape
        out_dtype = torch.promote_types(w.dtype, x.dtype)
        y = torch.empty((M, N), dtype=out_dtype, device=x.device)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(N)
        _rms_fwd[(M,)](x2, w, y, rstd, x2.stride(0), y.stride(0), N, eps,
                       BLOCK=block, num_warps=_warps(block))
        ctx.save_for_backward(x2, w, rstd)
        ctx.shape = shape
        ctx.block = block
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        M, N = x2.shape
        dy2 = dy.reshape(-1, N).contiguous()
        dx = torch.empty_like(x2)
        _rms_bwd[(M,)](dy2, x2, w, rstd, dx, dy2.stride(0), x2.stride(0),
                       dx.stride(0), N, BLOCK=ctx.block,
                       num_warps=_warps(ctx.block))
        dw = None
        if ctx.needs_input_grad[1]:  # frozen in LoRA training, kept for safety
            xhat = x2.float() * rstd[:, None]
            dw = (dy2.float() * xhat).sum(0).to(w.dtype)
        return dx.view(ctx.shape), dw, None


def patch_rmsnorm():
    """Replace Qwen2RMSNorm.forward. Safe to call several times."""
    try:
        from transformers.models.qwen2 import modeling_qwen2 as m
    except Exception:
        return False
    cls = m.Qwen2RMSNorm
    if getattr(cls, "_soup_triton_patched", False):
        return True
    orig = cls.forward

    def forward(self, hidden_states):
        if (not isinstance(hidden_states, torch.Tensor)
                or not hidden_states.is_cuda
                or hidden_states.dtype not in (torch.float16, torch.float32)):
            return orig(self, hidden_states)
        return RMSNormFn.apply(hidden_states, self.weight, self.variance_epsilon)

    cls.forward = forward
    cls._soup_triton_patched = True
    return True
