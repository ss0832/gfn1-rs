# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional PyTorch interop for GFN1-RS parameter optimization.

PyTorch is imported lazily inside the factory and is **not** a dependency of
``gfn1_rs`` (nothing here is imported unless you call it). It wraps the native
``Gfn1NativeCalculator.parameter_energy_and_gradient`` so the GFN1 total energy
becomes differentiable with respect to a chosen set of model parameters
(addressed by ``glob:`` / ``elem:`` / ``pair:`` target strings).

Example::

    import torch
    from gfn1_rs import Gfn1NativeCalculator, default_param_path
    from gfn1_rs.torch_interop import parameter_energy_function

    calc = Gfn1NativeCalculator(param_path=default_param_path())
    energy_fn = parameter_energy_function(
        calc, numbers=[1, 1], positions=[[0, 0, 0], [0.74, 0, 0]],
        targets=["glob:ks", "elem:1:GAM"])
    p = torch.tensor(calc.parameter_values(["glob:ks", "elem:1:GAM"]),
                     dtype=torch.float64, requires_grad=True)
    e = energy_fn(p)          # GFN1 total free energy (Hartree), differentiable
    e.backward()
    print(p.grad)             # dE/dp
"""

from __future__ import annotations


def parameter_energy_function(calc, numbers, positions, targets, unit="angstrom", step=1.0e-4):
    """Return a callable mapping a 1-D parameter-value tensor (one entry per
    target) to the GFN1 total free energy (Hartree), differentiable through a
    ``torch.autograd.Function`` whose backward pass uses the finite-difference
    parameter gradient ``dE/dp``."""
    import torch  # lazy import; torch is not a gfn1_rs dependency

    numbers = list(numbers)
    positions = [list(p) for p in positions]
    targets = list(targets)

    class _Gfn1ParameterEnergy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, values):
            vals = [float(v) for v in values.detach().cpu().reshape(-1).tolist()]
            energy, grad = calc.parameter_energy_and_gradient(
                numbers=numbers,
                positions=positions,
                targets=targets,
                values=vals,
                unit=unit,
                step=step,
            )
            ctx.save_for_backward(
                torch.as_tensor(grad, dtype=values.dtype, device=values.device)
            )
            return torch.as_tensor(energy, dtype=values.dtype, device=values.device)

        @staticmethod
        def backward(ctx, grad_output):
            (grad,) = ctx.saved_tensors
            return grad_output * grad

    return _Gfn1ParameterEnergy.apply
