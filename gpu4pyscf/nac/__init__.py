from gpu4pyscf.lib.backends import BACKEND_NAME

if BACKEND_NAME == 'cupy':
    from . import tdrhf, tdrks, tdrks_ris, finite_diff
