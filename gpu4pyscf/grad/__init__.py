from gpu4pyscf.lib.backends import BACKEND_NAME

if BACKEND_NAME == 'cupy':
    from gpu4pyscf.grad import rhf
    from gpu4pyscf.grad.rhf import Gradients as RHF
    from gpu4pyscf.grad.rks import Gradients as RKS
    from gpu4pyscf.grad.uhf import Gradients as UHF
    from gpu4pyscf.grad.uks import Gradients as UKS
    from . import tdrhf, tdrks, tduhf, tduks
else:
    from gpu4pyscf.grad._cpu_fallback import (
        RHFGradients as RHF,
        RKSGradients as RKS,
        UHFGradients as UHF,
        UKSGradients as UKS,
    )
