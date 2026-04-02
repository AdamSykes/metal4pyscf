from gpu4pyscf.lib.backends import BACKEND_NAME

from . import rks
from .rks import RKS, KohnShamDFT

from .uks import UKS

if BACKEND_NAME == 'cupy':
    from .rks_lowmem import RKS as LRKS
    from .gks import GKS
    from .roks import ROKS
    from .gen_grid import Grids
else:
    from pyscf.dft import GKS, ROKS
    from pyscf.dft.gen_grid import Grids

def KS(mol, xc='LDA,VWN'):
    if mol.spin == 0:
        return RKS(mol, xc)
    else:
        return UKS(mol, xc)
