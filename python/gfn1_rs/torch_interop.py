# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional PyTorch interop for GFN1-RS parameter optimization.

**Units: this is an atomic-units API.** It sits on the native
``Gfn1NativeCalculator``, not on the ASE layer, so the energy it returns is in
**Hartree**, the gradient is Hartree per unit parameter, and the parameters
themselves are in the native units of the GFN1 parameter file. (``positions`` /
``unit`` follow the native convention: Angstrom by default, ``unit="bohr"`` for
atomic units.) Only :class:`gfn1_rs.ase.GFN1RSCalculator` converts to ASE's
Angstrom / eV units.

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
    target) to the GFN1 total free energy, differentiable through a
    ``torch.autograd.Function`` whose backward pass uses the finite-difference
    parameter gradient ``dE/dp``.

    **Atomic units** (this is a native, non-ASE API): the returned energy is in
    **Hartree** and ``p.grad`` is Hartree per unit parameter. ``positions`` are in
    ``unit`` (default ``"angstrom"``; ``"bohr"`` for atomic units) and ``step`` is
    the finite-difference step in the parameter's own unit.
    """
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
