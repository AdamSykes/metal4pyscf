"""
Metal GPU-accelerated evaluation of Gaussian basis functions on grid points.

Single kernel launch: each thread handles one (grid_point, shell) pair.
All shells processed in one dispatch.
"""

import numpy as np
import mlx.core as mx


def _ncart(l):
    return (l + 1) * (l + 2) // 2


def _cart2sph_matrix(l):
    from pyscf.gto.mole import cart2sph
    return np.asarray(cart2sph(l, normalized='sp'))


# Single kernel for all angular momenta. Each thread writes to a unique
# region of the output (no race conditions) since ao_off is unique per shell.
_EVAL_AO_SOURCE = '''
uint gid = thread_position_in_grid.x;  // grid point
uint sid = thread_position_in_grid.y;  // shell

if (gid >= ngrids || sid >= nshells) return;

float acx = shell_data[sid * 8 + 0];  // atom x
float acy = shell_data[sid * 8 + 1];  // atom y
float acz = shell_data[sid * 8 + 2];  // atom z
float fac = shell_data[sid * 8 + 3];  // normalization
int nprim_s = (int)shell_data[sid * 8 + 4];
int ao_off_s = (int)shell_data[sid * 8 + 5];
int exp_off = (int)shell_data[sid * 8 + 6];
int ang = (int)shell_data[sid * 8 + 7];

float rx = gridx[gid] - acx;
float ry = gridy[gid] - acy;
float rz = gridz[gid] - acz;
float rr = rx*rx + ry*ry + rz*rz;

float ce = 0.0f;
for (int p = 0; p < nprim_s; p++) {
    ce += coeffs[exp_off + p] * exp(-exps[exp_off + p] * rr);
}
ce *= fac;

uint s = ngrids;  // stride
if (ang == 0) {
    out[ao_off_s*s + gid] = ce;
} else if (ang == 1) {
    out[(ao_off_s  )*s+gid] = ce*rx;
    out[(ao_off_s+1)*s+gid] = ce*ry;
    out[(ao_off_s+2)*s+gid] = ce*rz;
} else if (ang == 2) {
    out[(ao_off_s  )*s+gid] = ce*rx*rx;
    out[(ao_off_s+1)*s+gid] = ce*rx*ry;
    out[(ao_off_s+2)*s+gid] = ce*rx*rz;
    out[(ao_off_s+3)*s+gid] = ce*ry*ry;
    out[(ao_off_s+4)*s+gid] = ce*ry*rz;
    out[(ao_off_s+5)*s+gid] = ce*rz*rz;
} else if (ang == 3) {
    out[(ao_off_s  )*s+gid] = ce*rx*rx*rx;
    out[(ao_off_s+1)*s+gid] = ce*rx*rx*ry;
    out[(ao_off_s+2)*s+gid] = ce*rx*rx*rz;
    out[(ao_off_s+3)*s+gid] = ce*rx*ry*ry;
    out[(ao_off_s+4)*s+gid] = ce*rx*ry*rz;
    out[(ao_off_s+5)*s+gid] = ce*rx*rz*rz;
    out[(ao_off_s+6)*s+gid] = ce*ry*ry*ry;
    out[(ao_off_s+7)*s+gid] = ce*ry*ry*rz;
    out[(ao_off_s+8)*s+gid] = ce*ry*rz*rz;
    out[(ao_off_s+9)*s+gid] = ce*rz*rz*rz;
} else if (ang == 4) {
    float rx2=rx*rx, ry2=ry*ry, rz2=rz*rz;
    out[(ao_off_s+ 0)*s+gid] = ce*rx2*rx2;
    out[(ao_off_s+ 1)*s+gid] = ce*rx2*rx*ry;
    out[(ao_off_s+ 2)*s+gid] = ce*rx2*rx*rz;
    out[(ao_off_s+ 3)*s+gid] = ce*rx2*ry2;
    out[(ao_off_s+ 4)*s+gid] = ce*rx2*ry*rz;
    out[(ao_off_s+ 5)*s+gid] = ce*rx2*rz2;
    out[(ao_off_s+ 6)*s+gid] = ce*rx*ry2*ry;
    out[(ao_off_s+ 7)*s+gid] = ce*rx*ry2*rz;
    out[(ao_off_s+ 8)*s+gid] = ce*rx*ry*rz2;
    out[(ao_off_s+ 9)*s+gid] = ce*rx*rz2*rz;
    out[(ao_off_s+10)*s+gid] = ce*ry2*ry2;
    out[(ao_off_s+11)*s+gid] = ce*ry2*ry*rz;
    out[(ao_off_s+12)*s+gid] = ce*ry2*rz2;
    out[(ao_off_s+13)*s+gid] = ce*ry*rz2*rz;
    out[(ao_off_s+14)*s+gid] = ce*rz2*rz2;
}
'''

_eval_ao_kernel = mx.fast.metal_kernel(
    name='eval_ao_single',
    input_names=['gridx', 'gridy', 'gridz', 'exps', 'coeffs', 'shell_data'],
    output_names=['out'],
    source=_EVAL_AO_SOURCE,
)

# ---------------------------------------------------------------------------
# Deriv=1 kernel: values + d/dx + d/dy + d/dz
# Output layout: [values(ncart*ngrids) | dx(ncart*ngrids) | dy(...) | dz(...)]
# ---------------------------------------------------------------------------
_EVAL_AO_DERIV1_SOURCE = '''
uint gid = thread_position_in_grid.x;
uint sid = thread_position_in_grid.y;
if (gid >= ngrids || sid >= nshells) return;

float acx = shell_data[sid*8+0];
float acy = shell_data[sid*8+1];
float acz = shell_data[sid*8+2];
float fac = shell_data[sid*8+3];
int nprim_s = (int)shell_data[sid*8+4];
int ao_off_s = (int)shell_data[sid*8+5];
int exp_off = (int)shell_data[sid*8+6];
int ang = (int)shell_data[sid*8+7];

float rx = gridx[gid] - acx;
float ry = gridy[gid] - acy;
float rz = gridz[gid] - acz;
float rr = rx*rx + ry*ry + rz*rz;

float ce = 0.0f, ce_2a = 0.0f;
for (int p = 0; p < nprim_s; p++) {
    float c = coeffs[exp_off+p];
    float a = exps[exp_off+p];
    float e = exp(-a * rr);
    ce += c * e;
    ce_2a += c * e * a;
}
ce *= fac;
ce_2a *= -2.0f * fac;

uint s = ngrids;           // stride between AOs
uint blk = ncart_total * s;  // stride between components (val, dx, dy, dz)

// Pointers into the 4 output blocks
#define GTO(i)  out[(ao_off_s+(i))*s + gid]
#define GTOX(i) out[blk + (ao_off_s+(i))*s + gid]
#define GTOY(i) out[2*blk + (ao_off_s+(i))*s + gid]
#define GTOZ(i) out[3*blk + (ao_off_s+(i))*s + gid]

if (ang == 0) {
    GTO(0) = ce;
    GTOX(0) = ce_2a*rx;  GTOY(0) = ce_2a*ry;  GTOZ(0) = ce_2a*rz;
}
else if (ang == 1) {
    GTO(0) = ce*rx;  GTO(1) = ce*ry;  GTO(2) = ce*rz;
    float ax = ce_2a*rx, ay = ce_2a*ry, az = ce_2a*rz;
    GTOX(0) = ax*rx+ce;  GTOX(1) = ax*ry;     GTOX(2) = ax*rz;
    GTOY(0) = ay*rx;     GTOY(1) = ay*ry+ce;  GTOY(2) = ay*rz;
    GTOZ(0) = az*rx;     GTOZ(1) = az*ry;     GTOZ(2) = az*rz+ce;
}
else if (ang == 2) {
    GTO(0)=ce*rx*rx; GTO(1)=ce*rx*ry; GTO(2)=ce*rx*rz;
    GTO(3)=ce*ry*ry; GTO(4)=ce*ry*rz; GTO(5)=ce*rz*rz;
    float ax=ce_2a*rx, ay=ce_2a*ry, az=ce_2a*rz;
    GTOX(0)=(ax*rx+2*ce)*rx; GTOX(1)=(ax*rx+ce)*ry;   GTOX(2)=(ax*rx+ce)*rz;
    GTOX(3)=ax*ry*ry;        GTOX(4)=ax*ry*rz;        GTOX(5)=ax*rz*rz;
    GTOY(0)=ay*rx*rx;        GTOY(1)=(ay*ry+ce)*rx;   GTOY(2)=ay*rx*rz;
    GTOY(3)=(ay*ry+2*ce)*ry; GTOY(4)=(ay*ry+ce)*rz;   GTOY(5)=ay*rz*rz;
    GTOZ(0)=az*rx*rx;        GTOZ(1)=az*rx*ry;         GTOZ(2)=(az*rz+ce)*rx;
    GTOZ(3)=az*ry*ry;        GTOZ(4)=(az*rz+ce)*ry;    GTOZ(5)=(az*rz+2*ce)*rz;
}
else if (ang == 3) {
    GTO(0)=ce*rx*rx*rx; GTO(1)=ce*rx*rx*ry; GTO(2)=ce*rx*rx*rz;
    GTO(3)=ce*rx*ry*ry; GTO(4)=ce*rx*ry*rz; GTO(5)=ce*rx*rz*rz;
    GTO(6)=ce*ry*ry*ry; GTO(7)=ce*ry*ry*rz; GTO(8)=ce*ry*rz*rz;
    GTO(9)=ce*rz*rz*rz;
    float ax=ce_2a*rx, ay=ce_2a*ry, az=ce_2a*rz;
    GTOX(0)=(ax*rx+3*ce)*rx*rx; GTOX(1)=(ax*rx+2*ce)*rx*ry; GTOX(2)=(ax*rx+2*ce)*rx*rz;
    GTOX(3)=(ax*rx+ce)*ry*ry;   GTOX(4)=(ax*rx+ce)*ry*rz;   GTOX(5)=(ax*rx+ce)*rz*rz;
    GTOX(6)=ax*ry*ry*ry;        GTOX(7)=ax*ry*ry*rz;        GTOX(8)=ax*ry*rz*rz;
    GTOX(9)=ax*rz*rz*rz;
    GTOY(0)=ay*rx*rx*rx;        GTOY(1)=(ay*ry+ce)*rx*rx;   GTOY(2)=ay*rx*rx*rz;
    GTOY(3)=(ay*ry+2*ce)*rx*ry; GTOY(4)=(ay*ry+ce)*rx*rz;   GTOY(5)=ay*rx*rz*rz;
    GTOY(6)=(ay*ry+3*ce)*ry*ry; GTOY(7)=(ay*ry+2*ce)*ry*rz; GTOY(8)=(ay*ry+ce)*rz*rz;
    GTOY(9)=ay*rz*rz*rz;
    GTOZ(0)=az*rx*rx*rx;        GTOZ(1)=az*rx*rx*ry;        GTOZ(2)=(az*rz+ce)*rx*rx;
    GTOZ(3)=az*rx*ry*ry;        GTOZ(4)=(az*rz+ce)*rx*ry;   GTOZ(5)=(az*rz+2*ce)*rx*rz;
    GTOZ(6)=az*ry*ry*ry;        GTOZ(7)=(az*rz+ce)*ry*ry;   GTOZ(8)=(az*rz+2*ce)*ry*rz;
    GTOZ(9)=(az*rz+3*ce)*rz*rz;
}

#undef GTO
#undef GTOX
#undef GTOY
#undef GTOZ
'''

_eval_ao_deriv1_kernel = mx.fast.metal_kernel(
    name='eval_ao_deriv1',
    input_names=['gridx', 'gridy', 'gridz', 'exps', 'coeffs', 'shell_data'],
    output_names=['out'],
    source=_EVAL_AO_DERIV1_SOURCE,
)


def eval_ao_metal(mol, coords, deriv=0):
    """Evaluate AO basis functions on grid points using Metal GPU.

    Single kernel launch for all shells. Each thread = one (grid, shell) pair.

    Args:
        deriv: 0 for values, 1 for values + gradients (d/dx, d/dy, d/dz)

    Returns:
        deriv=0: (ngrids, nao) array
        deriv=1: (4, ngrids, nao) array [values, dx, dy, dz]
    """
    if deriv > 1:
        raise NotImplementedError('Metal eval_ao supports deriv=0 and deriv=1')

    coords = np.asarray(coords, dtype=np.float64, order='C')
    ngrids = coords.shape[0]
    nao = mol.nao
    cart = mol.cart

    gridx = mx.array(coords[:, 0].astype(np.float32))
    gridy = mx.array(coords[:, 1].astype(np.float32))
    gridz = mx.array(coords[:, 2].astype(np.float32))

    ncart_total = sum(_ncart(mol.bas_angular(i)) for i in range(mol.nbas))
    nshells = mol.nbas

    # Pack all shell metadata into a single (nshells, 8) float array
    # [atom_x, atom_y, atom_z, fac, nprim, ao_off, exp_off, ang]
    shell_data = np.zeros((nshells, 8), dtype=np.float32)
    all_exps = []
    all_coeffs = []
    exp_offset = 0
    cart_ao_off = 0

    for ish in range(nshells):
        l = mol.bas_angular(ish)
        atom_id = mol.bas_atom(ish)
        ac = mol.atom_coord(atom_id)
        nprim = mol.bas_nprim(ish)

        if l == 0:
            fac = 0.282094791773878143
        elif l == 1:
            fac = 0.488602511902919921
        else:
            fac = 1.0

        shell_data[ish] = [ac[0], ac[1], ac[2], fac, nprim,
                           cart_ao_off, exp_offset, l]

        all_exps.append(mol.bas_exp(ish).astype(np.float32))
        all_coeffs.append(mol._libcint_ctr_coeff(ish).flatten().astype(np.float32))
        exp_offset += nprim
        cart_ao_off += _ncart(l)

    exps_gpu = mx.array(np.concatenate(all_exps))
    coeffs_gpu = mx.array(np.concatenate(all_coeffs))
    shell_data_gpu = mx.array(shell_data.ravel())

    # Single kernel launch: grid=(ngrids, nshells)
    THREADS_X = 256
    grid_x = ((ngrids + THREADS_X - 1) // THREADS_X) * THREADS_X

    ncomp = 4 if deriv == 1 else 1
    output_size = ncomp * ncart_total * ngrids

    if deriv == 0:
        kernel = _eval_ao_kernel
        template = [('ngrids', ngrids), ('nshells', nshells)]
    else:
        kernel = _eval_ao_deriv1_kernel
        template = [('ngrids', ngrids), ('nshells', nshells),
                    ('ncart_total', ncart_total)]

    result = kernel(
        inputs=[gridx, gridy, gridz, exps_gpu, coeffs_gpu, shell_data_gpu],
        grid=(grid_x, nshells, 1),
        threadgroup=(THREADS_X, 1, 1),
        output_shapes=[(output_size,)],
        output_dtypes=[mx.float32],
        template=template,
    )
    # Keep as MLX array for cart2sph on GPU
    if deriv == 0:
        ao_cart_gpu = result[0].reshape(ncart_total, ngrids)  # (ncart, ngrids)
    else:
        ao_cart_gpu = result[0].reshape(4, ncart_total, ngrids)  # (4, ncart, ngrids)

    if not cart and ncart_total != nao:
        # Cart-to-spherical on Metal GPU
        shell_mapping = []
        ao_off_cart = 0
        ao_off_sph = 0
        for ish in range(mol.nbas):
            l = mol.bas_angular(ish)
            ncart_l = _ncart(l)
            nsph_l = ncart_l if l <= 1 else 2 * l + 1
            shell_mapping.append((l, ao_off_cart, ao_off_cart + ncart_l,
                                  ao_off_sph, ao_off_sph + nsph_l))
            ao_off_cart += ncart_l
            ao_off_sph += nsph_l

        def _apply_c2s(ao_cart_2d):
            """Apply cart2sph to a (ncart_total, ngrids) array → (nao, ngrids)."""
            parts = []
            for l, c0, c1, s0, s1 in shell_mapping:
                block = ao_cart_2d[c0:c1]
                if l <= 1:
                    parts.append(block)
                else:
                    c2s_gpu = mx.array(_cart2sph_matrix(l).T.astype(np.float32))
                    parts.append(c2s_gpu @ block)
            return mx.concatenate(parts, axis=0)

        if deriv == 0:
            ao_sph_gpu = _apply_c2s(ao_cart_gpu)
            mx.eval(ao_sph_gpu)
            return np.array(ao_sph_gpu).T.astype(np.float64)
        else:
            # Apply c2s to each of the 4 components
            result_parts = []
            for comp in range(4):
                result_parts.append(_apply_c2s(ao_cart_gpu[comp]))
            ao_sph_gpu = mx.stack(result_parts)  # (4, nao, ngrids)
            mx.eval(ao_sph_gpu)
            return np.array(ao_sph_gpu).transpose(0, 2, 1).astype(np.float64)

    mx.eval(ao_cart_gpu)
    if deriv == 0:
        return np.array(ao_cart_gpu).T.astype(np.float64)
    else:
        return np.array(ao_cart_gpu).transpose(0, 2, 1).astype(np.float64)


# ---------------------------------------------------------------------------
# Reusable building blocks for numint_metal
# ---------------------------------------------------------------------------

def _prepare_shell_data(mol):
    """Precompute shell metadata arrays (reusable across grid batches)."""
    nshells = mol.nbas
    ncart_total = sum(_ncart(mol.bas_angular(i)) for i in range(nshells))

    shell_data = np.zeros((nshells, 8), dtype=np.float32)
    all_exps = []
    all_coeffs = []
    exp_offset = 0
    cart_ao_off = 0
    shell_mapping = []
    sph_off = 0

    for ish in range(nshells):
        l = mol.bas_angular(ish)
        atom_id = mol.bas_atom(ish)
        ac = mol.atom_coord(atom_id)
        nprim = mol.bas_nprim(ish)
        ncart_l = _ncart(l)
        nsph_l = ncart_l if l <= 1 else 2 * l + 1

        fac = {0: 0.282094791773878143, 1: 0.488602511902919921}.get(l, 1.0)
        shell_data[ish] = [ac[0], ac[1], ac[2], fac, nprim,
                           cart_ao_off, exp_offset, l]
        shell_mapping.append((l, cart_ao_off, ncart_l, sph_off, nsph_l))

        all_exps.append(mol.bas_exp(ish).astype(np.float32))
        all_coeffs.append(mol._libcint_ctr_coeff(ish).flatten().astype(np.float32))
        exp_offset += nprim
        cart_ao_off += ncart_l
        sph_off += nsph_l

    exps_gpu = mx.array(np.concatenate(all_exps))
    coeffs_gpu = mx.array(np.concatenate(all_coeffs))
    return shell_data, exps_gpu, coeffs_gpu, ncart_total, shell_mapping


def _eval_ao_batch_gpu(mol, coords, deriv, shell_data, exps_gpu, coeffs_gpu,
                       ncart_total, shell_mapping, ngrids):
    """Evaluate AOs for a grid batch on Metal GPU. Returns mx.array (stays on GPU).

    Returns: (nao, ngrids) for deriv=0, (4, nao, ngrids) for deriv=1.
    """
    coords = np.asarray(coords)
    gridx = mx.array(coords[:, 0].astype(np.float32))
    gridy = mx.array(coords[:, 1].astype(np.float32))
    gridz = mx.array(coords[:, 2].astype(np.float32))
    shell_data_gpu = mx.array(shell_data.ravel())
    nshells = mol.nbas
    nao = mol.nao
    cart = mol.cart

    THREADS_X = 256
    grid_x = ((ngrids + THREADS_X - 1) // THREADS_X) * THREADS_X
    ncomp = 4 if deriv == 1 else 1

    if deriv == 0:
        kernel = _eval_ao_kernel
        template = [('ngrids', ngrids), ('nshells', nshells)]
    else:
        kernel = _eval_ao_deriv1_kernel
        template = [('ngrids', ngrids), ('nshells', nshells),
                    ('ncart_total', ncart_total)]

    result = kernel(
        inputs=[gridx, gridy, gridz, exps_gpu, coeffs_gpu, shell_data_gpu],
        grid=(grid_x, nshells, 1),
        threadgroup=(THREADS_X, 1, 1),
        output_shapes=[(ncomp * ncart_total * ngrids,)],
        output_dtypes=[mx.float32],
        template=template,
    )

    if deriv == 0:
        ao_cart = result[0].reshape(ncart_total, ngrids)
    else:
        ao_cart = result[0].reshape(4, ncart_total, ngrids)

    # Cart-to-spherical on GPU
    if not cart and ncart_total != nao:
        def _apply_c2s(ao_2d):
            parts = []
            for l, c0, ncart_l, s0, nsph_l in shell_mapping:
                block = ao_2d[c0:c0 + ncart_l]
                if l <= 1:
                    parts.append(block)
                else:
                    c2s_gpu = mx.array(_cart2sph_matrix(l).T.astype(np.float32))
                    parts.append(c2s_gpu @ block)
            return mx.concatenate(parts, axis=0)

        if deriv == 0:
            return _apply_c2s(ao_cart)
        else:
            return mx.stack([_apply_c2s(ao_cart[c]) for c in range(4)])

    return ao_cart
