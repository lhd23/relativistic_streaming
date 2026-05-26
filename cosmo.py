import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1

import fftlog

CSPEED = 2.998e5 # km/s

# Note importing camb messes up the parallelisation

# This module computes pk using CAMB. This is slow and for tabulating only.
# To limit CAMB exposure, background quantities such as H(z) and Om(z) are computed by hand


def _build_camb_params(z, pardict):
    import camb
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=pardict['H0'],
        ombh2=pardict['ombh2'],
        omch2=pardict['omch2'],
        omk=pardict['omk'],
        tau=pardict['tau'],
        mnu=pardict['mnu'],
    )
    pars.InitPower.set_params(
        As=pardict['As'],
        ns=pardict['ns'],
        r=pardict['r'],
        pivot_scalar=pardict['pivot_scalar'],
    )
    pars.set_matter_power(redshifts=[z], kmax=1e3)
    return pars


def pk(z,pardict):
    """Return linear P(k) from CAMB at redshift z."""
    import camb
    pars = _build_camb_params(z, pardict)
    #Linear spectra
    results = camb.get_results(pars)
    k,z,pk = results.get_matter_power_spectrum(minkh=1.3e-5, maxkh=1e3, npoints=5000)
    return (k,pk[0,:]) # k is in units h/Mpc

def sigma8(z,pardict):
    import camb
    pars = _build_camb_params(z, pardict)
    results = camb.get_results(pars)
    sig8 = results.get_sigma8()
    return sig8[0]

def xi(z,pardict):
    k,P = pk(z,pardict)
    F = fftlog.CorrelationFFTLog(k,P)
    r = F.qv
    xi = F.xi00
    return (r,xi)

def xi_Psidelta(z,pardict): # <Psi/c^2 delta> (dimensionless)
    aH = aH(z,pardict) # km/s/(Mpc/h)
    Om = Om(z,pardict)
    k,P = pk(z,pardict)
    F = fftlog.CorrelationFFTLog(k,P)
    r = F.qv
    xi_Psidelta = -3./2. * Om * (aH/CSPEED)**2 * F.xi0m2
    return (r,xi_Psidelta)

def xi_div(z,pardict):
    k,P = pk(z,pardict)
    F = fftlog.CorrelationFFTLog(k,P)
    r = F.qv
    f = f_growthrate(z,pardict)
    d = conformal_distance(z,pardict)
    xi1_epsilon = -f * 2./3 * (F.xi00 + F.xi20)
    epsilon = r/d
    return (r,epsilon*xi1_epsilon)

def f_growthrate(z,pardict):
    import camb
    pars = _build_camb_params(z, pardict)
    results = camb.get_results(pars)
    fsig8 = results.get_fsigma8()
    sig8 = results.get_sigma8()
    return fsig8[0] / sig8[0]

def D_growthfactor(z,pardict):
    import camb
    # This is a very inefficient way to compute D(z)
    k,p0 = pk(0.,pardict)
    k,p = pk(z,pardict)
    D2 = p/p0 # D0=1 and D<=1 in generalte
    return np.sqrt(D2)[0] # all elements should be the same (in practice I have checked the difference is less than 0.1%)

def D_growthfactor_M95(z,pardict):
    """Return normalized growth factor using Matsubara 1995 flat-LCDM formula."""
    # Matsubara 1995 exact formula for flat LCDM
    zp1 = 1.0+z
    Om0 = pardict['Om0']
    a1 = (-1.0 + Om0)/(Om0*zp1**3)
    a2 = (-1.0 + Om0)/Om0
    f1 = hyp2f1(5./6, 1.5, 11./6, a1)
    f2 = hyp2f1(5./6, 1.5, 11./6, a2)
    numer = f1 * np.sqrt(1.0-Om0+Om0*zp1**3)
    denom = f2 * zp1**2.5 # normalise so D0=1
    return numer/denom

def f_growthrate_approx(z,pardict):
    return pow(Om(z,pardict), 0.55)

def H(z,pardict): # km/s/(Mpc/h) so H(0)=100. km/s/(Mpc/h)
    return pardict['H0_100'] * E(z,pardict)

def aH(z,pardict): # conformal Hubble, units km/s/(Mpc/h)
    a = 1./(1.+z)
    return a * H(z,pardict)

def Om(z,pardict): # the fractional matter density (between 0 and 1)
    Om0 = pardict['Om0']
    E2 = E(z,pardict)**2
    return Om0*(1.+z)**3/E2

def E(z,pardict): # H(z)/H0, unitless
    Om0 = pardict['Om0']
    Ok0 = pardict['omk']/pardict['h']**2
    E2 = Om0*(1.+z)**3 + Ok0*(1.+z)**2 + (1.-Om0-Ok0)
    return np.sqrt(E2)

def conformal_distance(z,pardict):
    """Compute line-of-sight conformal distance in Mpc/h by numerical integration."""
    def _integrand(z_):
        return 1./E(z_,pardict) # Unitless
    if np.isclose(z,0.):
        return 0.0
    if np.isfinite(_integrand(z)): # prevent negative square roots
        y,err = quad(_integrand, 0.0, z, epsabs=1e-9)
        return y * CSPEED/pardict['H0_100'] # Mpc/h

def chi_to_z(chi,z,pardict): # redshift as a fn of distance chi
    # using linear approx about observed z
    chi0 = conformal_distance(z,pardict)
    ret = z + H(z,pardict)/CSPEED * (chi-chi0)
    return ret
    
# Smoothing: use Gaussian filter of radius R
def WG(k,R):
    return np.exp(-0.5*(k*R)**2)

# The radius is taken to be the Lagrangian radius R_Lgn of the halo
def R_to_M0(R,z,Om0): # from eqn 3 in Bullock and Boylan-Kolchin:  
    M0 = 1.71e11 * (Om0/0.3) * (1./0.7)**2 * R**3 # units: h^(-1) M_\odot (NB little h)
    return M0*(1.+z)**3

def M_to_R(M,z,Om0):
    y = M / 1.71e11 * (0.3/Om0) * (0.7/1)**2
    return y**(1./3) / (1.+z)

# cross check with colossus
def test_R_Lgn(M,z,pars_colossus):
    try:
        from colossus.cosmology import cosmology
        from colossus.halo import mass_so
    except:
        raise ImportError
    cosmology.addCosmology('myCosmo', **pars_colossus)
    cosmo = cosmology.setCosmology('myCosmo')

    # any sig8 and ns will do since we only require the background parameters
    R = mass_so.M_to_R(M=M, z=z, mdef='1m')/1e3 # Lagrangian radius in Mpc/h at redshift z
    print('R_Lgn (colossus): ', R)
    # print('M: ',mass_so.R_to_M(R=R_colossus*1e3, z=zin, mdef='1m'))

def get_f_and_sigma8():
    import params
    print(f_growthrate(0.,params.get_params('planck')))
    print(sigma8(0.,params.get_params('planck')))
    print(f_growthrate(0.,params.get_params('wmap')))
    print(sigma8(0.,params.get_params('wmap')))
    print(f_growthrate(0.,params.get_params('euclid')))

def plot():
    import matplotlib.pyplot as plt
    zin = 0.341
    M = 1.e13
    print('redshift: ', zin)
    print('halo mass: ', M)

    import params
    pars = params.get_params('planck')

    R_Lgn = M_to_R(M,zin,Om0=pars['Om0'])
    print('R_Lagrangian: ', R_Lgn)    
    test_R_Lgn(M,zin,params.pars_colossus)

    k,P = pk(zin,pars)
    P_filtered = WG(k,R=R_Lgn)**2 * P

    plt.loglog(k, P, 'k', lw=1, label='P_L CAMB')
    plt.loglog(k, P_filtered, 'b', lw=1, label='P_L CAMB * W_G(kR)^2 w/ R={:6.4f} Mpc/h'.format(R_Lgn))
    plt.xlim(2e-5,3)
    plt.ylim(1e1,1e5)
    plt.legend(frameon=False)
    plt.show()

if __name__ == '__main__':
    plot()
