import os

import numpy as np


MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(MODULE_DIR, 'tables')


def format_redshift_tag(z):
    """Match on-disk table naming convention (e.g. 0.341 -> 0p341)."""
    zstr = f"{z:.3f}" if z != 0.0 else '0'
    return zstr.replace('.', 'p')


def _table_path(z, param_set, stem):
    ztag = format_redshift_tag(z)
    filename = f'{stem}_z{ztag}_{param_set}.dat'
    return os.path.join(TABLES_DIR, f'z{ztag}', param_set, filename)


def load_table(z, param_set, stem, unpack=False):
    """Load a lookup-table by stem (without the _zX_param suffix)."""
    path = _table_path(z, param_set, stem)
    return np.loadtxt(path, unpack=unpack)


def load_linear_corr_table(z, param_set):
    return load_table(z, param_set, 'corrfuncs')


def load_psi_corrs_table(z, param_set):
    return load_table(z, param_set, 'corrfuncs_psi_corrs')


def load_clpt_xi_u_tables(z, param_set):
    r, xi_clpt = load_table(z, param_set, 'corrfuncs_xi_zel_clpt', unpack=True)
    r_u, U_clpt = load_table(z, param_set, 'corrfuncs_U_clpt', unpack=True)
    if not np.allclose(r, r_u):
        raise ValueError('CLPT xi and U tables are not aligned in r.')
    return r, xi_clpt, U_clpt


def load_gravz_table(z, param_set):
    return load_table(z, param_set, 'corrfuncs_gravz')
