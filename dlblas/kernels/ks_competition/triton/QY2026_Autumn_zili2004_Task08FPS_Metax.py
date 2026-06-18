# -*- coding: utf-8 -*-
import torch
import triton
import triton.language as tl


@triton.jit
def _fps_3d_kernel(
    x_ptr,
    start_ptr,
    selected_ptr,
    N: tl.constexpr,
    NSAMPLE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x0 = tl.load(x_ptr + offs * 3 + 0, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + offs * 3 + 1, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + offs * 3 + 2, mask=mask, other=0.0).to(tl.float32)

    cur = tl.load(start_ptr).to(tl.int64)

    dist_min = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
    dist_min = tl.where(offs == cur, 0.0, dist_min)

    tl.store(selected_ptr + 0, cur)

    i = 1
    while i < NSAMPLE:
        cx0 = tl.load(x_ptr + cur * 3 + 0).to(tl.float32)
        cx1 = tl.load(x_ptr + cur * 3 + 1).to(tl.float32)
        cx2 = tl.load(x_ptr + cur * 3 + 2).to(tl.float32)

        dx0 = x0 - cx0
        dx1 = x1 - cx1
        dx2 = x2 - cx2

        d = dx0 * dx0 + dx1 * dx1 + dx2 * dx2
        d = tl.where(mask, d, float("inf"))

        dist_min = tl.minimum(dist_min, d)

        valid_dist = tl.where(mask, dist_min, -float("inf"))
        max_val = tl.max(valid_dist, axis=0)

        # torch.argmax 在相同最大值时返回第一个最大下标
        idx_candidate = tl.where((valid_dist == max_val) & mask, offs, N)
        cur = tl.min(idx_candidate, axis=0).to(tl.int64)

        tl.store(selected_ptr + i, cur)
        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        num_samples,
        random_start=True,
    ):
        N, D = x.shape
        device = x.device

        # 官方输入固定为 x=[1000, 3], num_samples=256
        if x.is_cuda and D == 3 and N <= 1024:
            x = x.contiguous()

            selected = torch.empty(
                num_samples,
                dtype=torch.long,
                device=device,
            )

            if random_start:
                start_idx = torch.randint(
                    0,
                    N,
                    (1,),
                    device=device,
                    dtype=torch.long,
                )
            else:
                start_idx = torch.zeros(
                    (1,),
                    device=device,
                    dtype=torch.long,
                )

            _fps_3d_kernel[(1,)](
                x,
                start_idx,
                selected,
                N,
                num_samples,
                BLOCK_N=1024,
                num_warps=8,
            )

            return selected

        # fallback，保持和参考实现一致
        distances = torch.full((N,), float("inf"), device=device)
        selected = torch.zeros(num_samples, dtype=torch.long, device=device)

        if random_start:
            start_idx = torch.randint(0, N, (1,), device=device)
        else:
            start_idx = torch.tensor([0], device=device)

        selected[0] = start_idx
        distances[start_idx] = 0

        for i in range(1, num_samples):
            current_point = x[selected[i - 1]].unsqueeze(0)
            dist_to_current = torch.sum((x - current_point) ** 2, dim=1)
            distances = torch.min(distances, dist_to_current)
            selected[i] = torch.argmax(distances)

        return selected

def get_inputs():
    x = torch.randn(1000, 3, device="cuda")
    num_samples = 256
    return [x, num_samples]


def get_init_inputs():
    return []