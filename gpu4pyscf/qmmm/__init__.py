from gpu4pyscf.lib.backends import BACKEND_NAME

if BACKEND_NAME == 'cupy':
    from gpu4pyscf.qmmm import chelpg
    from gpu4pyscf.qmmm.itrf import *
else:
    from pyscf.qmmm import mm_mole, add_mm_charges
