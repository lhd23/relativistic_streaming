from dataclasses import dataclass
from typing import Mapping, Union

import numpy as np

CSPEED = 2.998e5  # km/s


@dataclass(frozen=True)
class CosmologyParams:
    """Canonical cosmology parameter container used across the codebase."""

    H0: float
    ombh2: float
    omch2: float
    omk: float
    mnu: float
    tau: float
    As: float
    ns: float
    r: float
    pivot_scalar: float
    sigma80: float
    f0: float

    @classmethod
    def from_omegas(
        cls,
        *,
        H0: float,
        Om0: float,
        Ob0: float,
        omk: float,
        mnu: float,
        tau: float,
        As: float,
        ns: float,
        r: float,
        pivot_scalar: float,
        sigma80: float,
        f0: float,
    ):
        h = H0 / 100.0
        return cls(
            H0=H0,
            ombh2=Ob0 * h * h,
            omch2=(Om0 - Ob0) * h * h,
            omk=omk,
            mnu=mnu,
            tau=tau,
            As=As,
            ns=ns,
            r=r,
            pivot_scalar=pivot_scalar,
            sigma80=sigma80,
            f0=f0,
        )

    @property
    def h(self) -> float:
        return self.H0 / 100.0

    @property
    def H0_100(self) -> float:
        return 100.0

    @property
    def Om0(self) -> float:
        return (self.ombh2 + self.omch2) / self.h**2

    @property
    def Ob0(self) -> float:
        return self.ombh2 / self.h**2

    def to_dict(self) -> dict:
        """Return a backward-compatible dictionary view."""
        return {
            'H0': self.H0,
            'ombh2': self.ombh2,
            'omch2': self.omch2,
            'omk': self.omk,
            'mnu': self.mnu,
            'tau': self.tau,
            'As': self.As,
            'ns': self.ns,
            'r': self.r,
            'pivot_scalar': self.pivot_scalar,
            'sigma80': self.sigma80,
            'f0': self.f0,
            'h': self.h,
            'H0_100': self.H0_100,
            'Om0': self.Om0,
            'Ob0': self.Ob0,
        }


# Planck plik best fit (table 1, column 1 in 2018 paper)
PLANCK = CosmologyParams(
    H0=67.32,
    ombh2=0.022383,
    omch2=0.12011,
    omk=0.0,
    mnu=0.0,
    tau=0.0543,
    As=1e-10 * np.exp(3.0448),
    ns=0.96605,
    r=0.0,
    pivot_scalar=0.05,
    sigma80=0.8234,
    f0=0.5265,
)

# WMAP7 maximum likelihood (used by RayGalGroup sims)
WMAP = CosmologyParams.from_omegas(
    H0=72.0,
    Om0=0.25733,
    Ob0=0.04356,
    omk=0.0,
    mnu=0.0,
    tau=0.085,
    As=2.42e-9,
    ns=0.963,
    r=0.0,
    pivot_scalar=0.002,
    sigma80=0.7995,
    f0=0.4702,
)

# Euclid reference parameters (Planck 2015-like)
EUCLID = CosmologyParams.from_omegas(
    H0=67.0,
    Om0=0.319,
    Ob0=0.049,
    omk=0.0,
    mnu=0.06,
    tau=0.058,
    As=2.1e-9,
    ns=0.96,
    r=0.0,
    pivot_scalar=0.05,
    sigma80=0.83,
    f0=0.5330,
)

PARAM_PRESETS = {
    'planck': PLANCK,
    'wmap': WMAP,
    'euclid': EUCLID,
}

# Backward-compatible aliases for legacy imports.
pars = {name: cfg.to_dict() for name, cfg in PARAM_PRESETS.items()}
planck_pars = pars['planck']
wmap_pars = pars['wmap']
euclid_pars = pars['euclid']


HALO_COSMO_PAR_MAP = {
    'planck': 'planck18-only',
    'wmap': 'WMAP7-only',
    'euclid': 'planck15-only',
}


# Colossus helper defaults
pars_colossus = {
    'flat': True,
    'H0': PLANCK.H0,
    'Om0': PLANCK.Om0,
    'Ob0': PLANCK.Ob0,
    'sigma8': 0.81,
    'ns': PLANCK.ns,
}


def get_preset(name: str) -> CosmologyParams:
    if name not in PARAM_PRESETS:
        valid = ', '.join(sorted(PARAM_PRESETS))
        raise ValueError(f'Unknown param_set "{name}". Expected one of: {valid}')
    return PARAM_PRESETS[name]


def get_params(param_set_or_params: Union[str, CosmologyParams, Mapping]) -> dict:
    """Resolve input cosmology to a normalized parameter dictionary."""
    if isinstance(param_set_or_params, CosmologyParams):
        return param_set_or_params.to_dict()
    if isinstance(param_set_or_params, str):
        return get_preset(param_set_or_params).to_dict()
    if isinstance(param_set_or_params, Mapping):
        return dict(param_set_or_params)
    raise TypeError('Expected param_set name, CosmologyParams, or mapping.')


def get_halo_cosmo_name(param_set: str) -> str:
    """Map local preset names to Colossus cosmology identifiers."""
    if param_set not in HALO_COSMO_PAR_MAP:
        valid = ', '.join(sorted(HALO_COSMO_PAR_MAP))
        raise ValueError(f'param_set {param_set} not valid; choose from: {valid}')
    return HALO_COSMO_PAR_MAP[param_set]


if __name__ == '__main__':
    pass
