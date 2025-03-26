import warp as wp
import torch
import numpy as np
from warp_utils import MyTape, from_torch_safe, CondTape
import pdb

# Initialize Warp
wp.init()

# Define the simple kernel
@wp.kernel
def mat33_square_sum(
    x: wp.array(dtype=wp.mat33d),
    loss: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    v = x[tid]
    for i in range(3):
        for j in range(3):
            wp.atomic_add(loss, 0, v[i, j] * v[i, j])

# PyTorch autograd wrapper
class SimpleSquareLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_tensor):
        assert x_tensor.dtype == torch.float64
        wp_x = from_torch_safe(x_tensor.contiguous(), dtype=wp.mat33d, requires_grad=True)
        wp_loss = wp.zeros(1, dtype=wp.float64, requires_grad=True, device=wp_x.device)

        tape = wp.Tape()
        with tape:
            wp.launch(
                kernel=mat33_square_sum,
                dim=x_tensor.shape[0],
                inputs=[wp_x, wp_loss],
                device=wp_x.device,
            )

        # convert wp_x to torch tensor before saving
        # wp_x_tensor = wp.to_torch(wp_x)
        # ctx.save_for_backward(wp_x_tensor)
        ctx.wp_x = wp_x
        ctx.wp_loss = wp_loss
        ctx.tape = tape
        # print(f"Forward wp_loss: {wp_loss}")
        return wp.to_torch(wp_loss).detach().requires_grad_(False)

    @staticmethod
    def backward(ctx, grad_output):
        ctx.tape.backward(ctx.wp_loss)
        grad_x = wp.to_torch(ctx.wp_x.grad)
        output = grad_output * grad_x
        ctx.tape.zero() # without this, the returned grad will double
        return output



# Test finite difference vs autodiff
def check_fd_vs_autograd():
    N = 1
    device = "cuda:0"

    x = torch.randn(N, 3, 3, dtype=torch.float64, device=device, requires_grad=True)
    print("Input tensor:\n", x)

    # Autograd
    loss = SimpleSquareLoss.apply(x)
    loss.backward()
    grad_auto = x.grad.clone()
    print("Autograd gradient:\n", grad_auto)

    # Finite Difference
    for relative_eps in [1e-5, 1e-6, 1e-7]:
        print(f"\n--- Finite Difference with relative_eps = {relative_eps} ---")
        grad_fd = torch.zeros_like(x)
        x_fd = x.detach().clone()

        with torch.no_grad():
            for n in range(N):
                for i in range(3):
                    for j in range(3):
                        base_val = x_fd[n, i, j].item()
                        perturbation = relative_eps * max(abs(base_val), 1.0)  # absolute fallback
                        perturb = torch.zeros_like(x)
                        perturb[n, i, j] = perturbation

                        loss_plus = SimpleSquareLoss.apply(x_fd + perturb).item()
                        loss_minus = SimpleSquareLoss.apply(x_fd - perturb).item()
                        grad_fd[n, i, j] = (loss_plus - loss_minus) / (2 * perturbation)

        # Compare
        error = torch.norm(grad_fd - grad_auto) / torch.norm(grad_fd + grad_auto)
        print("Finite difference gradient:\n", grad_fd)
        # print("Max abs diff:", (grad_fd - grad_auto).abs().max().item())
        # print("Relative error:", error.item())


if __name__ == "__main__":
    check_fd_vs_autograd()
