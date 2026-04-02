from gpu4pyscf.lib.backends import BACKEND_NAME

if BACKEND_NAME == 'cupy':
    from gpu4pyscf.pbc import scf, dft
else:
    from pyscf.pbc import scf, dft
