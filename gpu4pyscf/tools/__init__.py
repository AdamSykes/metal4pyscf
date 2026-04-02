from gpu4pyscf.lib.backends import BACKEND_NAME

if BACKEND_NAME == 'cupy':
    from .method_config import get_default_config, method_from_config
