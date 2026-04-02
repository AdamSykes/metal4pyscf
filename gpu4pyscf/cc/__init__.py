from gpu4pyscf.lib.backends import BACKEND_NAME

if BACKEND_NAME == 'cupy':
    from . import ccsd_incore
else:
    from pyscf.cc import CCSD, RCCSD, UCCSD
