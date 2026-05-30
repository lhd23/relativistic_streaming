# Relativistic Streaming Model

Python code accompanying the paper:

> **Gravitational redshift from large-scale structure: nonlinearities, anti-symmetries, and the dipole**
> L. Dam & C. Bonvin (2025)
> [arXiv:2506.22431](https://arxiv.org/abs/2506.22431)

This code computes the galaxy two-point correlation function in redshift space using the
**Gaussian Streaming Model (GSM)**, with support for relativistic and wide-angle corrections
and a one-halo contribution for the gravitational redshift.

## Models

Three model classes are provided in `corrfuncs_models.py`, differing in their coordinate system and integration method:

| Class | Coordinates | Integration | Use case |
|---|---|---|---|
| `gsm` | (chi1, chi2, varmu) | 2D (`dblquad`) | Two-point correlations in the full wide-angle regime |
| `gsm_sdu` | (s, d, mu) | 3D (inherits `gsm`) | Extract multipoles from two-point correlations |
| `gsm_DOL` | (s, mu) | 1D (`quad`) | Two-point correlations in the distant-observer limit |

Here `chi1`, `chi2` are the redshift-space comoving distances of the two tracers, `varmu` is the cosine of their angular separation, `s` is the redshift-space pair separation, `d` is their mean distance, and `mu` is the cosine of the angle between the separation vector and the line of sight.

## Optional physics flags

| Flag | Effect |
|---|---|
| `bA` | linear bias of tracer A |
| `bB` | linear bias of tracer B (choose different `bA` and `bB` when computing odd multipoles) |
| `with_gravz` | Include gravitational redshift (only in the mean) |
| `with_rsd` | Include redshift-space distortions (includes full wide-angle corrections) |
| `with_clpt` | Use CLPT (Convolution Lagrangian PT) tables for the real-space correlation function and mean pairwise velocity |
| `with_lightcone` | Include lightcone corrections |
| `with_lookback` | Include lookback-time corrections |
| `with_all_cov` | Incldue gravitational redshift contributions to the covariance |

Halo-model one-halo contributions are supported via the `MA`, `MB`, `cA`, `cB` parameters
(NFW halo masses and concentrations for each tracer population).
Note: input real-space correlation functions (see [here](https://arxiv.org/pdf/2506.22431#section*.56)) are read
from tables hardcoded using the [RayGal simulation cosmology](https://arxiv.org/abs/1803.04294) (`z=0.341`, WMAP7 cosmology).
This is for reasons of speed. For different cosmologies and redshifts, new tables should be generated.

If one is only interested in RSD and their complete wide-angle corrections, set `with_gravz=False`.
This returns the model described in [arXiv:2307.01294](https://arxiv.org/abs/2307.01294).

## Quick start

```python
import numpy as np
from corrfuncs_models import gsm, gsm_sdu

# Full wide-angle GSM at z=0.341 (no lightcone, linear theory)
F = gsm(bA=2., bB=1., z=0.341, with_lightcone=False, with_clpt=False, param_set='wmap')

# Evaluate 1 + xi_s for a pair configuration
val = F(chi1=500, chi2=1000, varmu=np.cos(20.0 * np.pi / 180.0))
print(val)

# Compute multipoles using
G = gsm_sdu(bA=2., bB=1., z=0.341, with_gravz=True, with_clpt=True, param_set='wmap')

# Dipole multipole at separation s=60 Mpc/h
xi1 = G.multipole(ell=1, s=60.0) # note: slow due to triple integration!
print(xi1)
```

## Dependencies

- Python >= 3.8
- `numpy`, `scipy`

Light use is also made of
- `camb` (for basic cosmological quantities)
- `colossus` (for halo recipes)

## Citation

If you use this code, please cite:

```
@article{Dam:2025,
  author        = {Dam, Lawrence and Bonvin, Camille},
  title         = {Gravitational redshift from large-scale structure:
                   nonlinearities, anti-symmetries, and the dipole},
  year          = {2025},
  eprint        = {2506.22431},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO}
}
```

## License

MIT
