"""Driver module for GSM correlation-function models.
"""

import numpy as np

from corrfuncs_models import gsm, gsm_sdu, gsm_DOL

__all__ = ['gsm', 'gsm_sdu', 'gsm_DOL', 'main']


def main():
    F = gsm(z=0.341, with_lightcone=False, with_clpt=False, param_set='wmap')
    print(F(chi1=500, chi2=1000, varmu=np.cos(20.0 * np.pi / 180.0)))


if __name__ == '__main__':
    main()
