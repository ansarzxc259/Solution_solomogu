import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def ref(x, w, eps):
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return w * h.to(x.dtype)


@pytest.mark.parametrize("w_dtype", [torch.float32, torch.float16])
def test_rmsnorm_fwd_bwd(w_dtype):
    from solution.rmsnorm import RMSNormFn

    torch.manual_seed(0)
    x = torch.randn(4, 130, 1536, device="cuda", dtype=torch.float16)
    w = torch.randn(1536, device="cuda", dtype=w_dtype)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    y_ref = ref(x1, w, 1e-6)
    y = RMSNormFn.apply(x2, w, 1e-6)
    assert y.dtype == y_ref.dtype
    torch.testing.assert_close(y.float(), y_ref.float(), rtol=2e-3, atol=2e-3)

    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    y.backward(g)
    torch.testing.assert_close(x2.grad.float(), x1.grad.float(), rtol=5e-3, atol=5e-3)
