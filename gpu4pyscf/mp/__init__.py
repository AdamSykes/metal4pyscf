from gpu4pyscf.lib.backends import BACKEND_NAME

if BACKEND_NAME == 'cupy':
    from . import mp2, dfmp2, dfump2, dfmp2_old, dfmp2_addons, dfmp2_drivers

    def MP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
        if mf.istype('UHF'):
            raise NotImplementedError
        elif mf.istype('GHF'):
            raise NotImplementedError
        else:
            return RMP2(mf, frozen, mo_coeff, mo_occ)

    def RMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
        if mf.istype('UHF'):
            raise RuntimeError('RMP2 cannot be used with UHF method.')
        if not mf.istype('RHF'):
            mf = mf.to_rhf()
        if getattr(mf, 'with_df', None) and getattr(mf.with_df, 'auxbasis', None) is not None:
            return dfmp2.DFMP2(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ, auxbasis=mf.with_df.auxbasis)
        else:
            return mp2.RMP2(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
else:
    from pyscf.mp import MP2, RMP2
