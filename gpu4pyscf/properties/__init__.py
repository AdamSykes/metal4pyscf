from gpu4pyscf.lib.backends import BACKEND_NAME

if BACKEND_NAME == 'cupy':
    from gpu4pyscf.properties import polarizability, ir, shielding, raman
