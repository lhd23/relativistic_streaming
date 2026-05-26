import numpy as np
from scipy.special import sici # sine and cosine integral
from scipy.integrate import quad
from scipy.interpolate import interp1d
import fftlog

import os
dir = os.path.dirname(os.path.abspath(__file__))

import params
from colossus.cosmology import cosmology
# cosmo = cosmology.setCosmology('planck18')
from colossus.lss import bias
from colossus.lss import mass_function

# constants
GN = 4.3009172706e-3 * 1.e-6 # Newton's grav constant in Mpc/h / (Msol/h) . (km/s)^2
CSPEED = 2.99792458e5 # km/s

class halomod:

    def __init__(self,z=0.,cosmo_pars='planck18'):
        """Initialize halo-model helpers at redshift z for a Colossus cosmology."""
        # Accept local preset names plus explicit Colossus names.
        if cosmo_pars in params.HALO_COSMO_PAR_MAP:
            cosmo_name = params.get_halo_cosmo_name(cosmo_pars)
        elif cosmo_pars in ('planck18', 'planck18-only', 'WMAP7-only', 'planck15-only'):
            cosmo_name = cosmo_pars
        else:
            raise ValueError('param_set {} not valid'.format(cosmo_pars))

        self.cosmo = cosmology.setCosmology(cosmo_name)
        self.z = z
        self.a = 1./(1.+self.z)
        self.H = self.cosmo.Hz(self.z) # km/s/Mpc
        self.Om = self.cosmo.Om(z)
        self.Delta = self._Delta(self.z)
        self.rhobar = self.cosmo.rho_m(z) * 1.e9 # units Msolar/h (h/Mpc)^3
        self.Dgrowth = self.cosmo.growthFactor(self.z)
        self.calH = self.a*self.H
        self.potfac = 3./2 * self.calH**2 * self.Om # equal to 4\pi G a^2 \bar\rho; units (km/s/Mpc)^2

        # mass integral correction for I11
        def _dI_lowM(M): # correction at low mass
            return M/self.rhobar * self.dndM(M) * self.b1(M)
        self.M1,self.M2 = 1e10,1e17
        I0,err = quad(_dI_lowM,self.M1,self.M2,limit=100)
        self.Ia = 1.0-I0

        self.lnpk_interp = None
        self.xi_interp = None
        self.xi_Psi_delta_interp = None
        
    # M has units Msol / h
    def _Delta(self,z):
        ret = 18.*np.pi**2 + 82.*(self.Om-1.) - 39.*(self.Om-1.)**2 # bullock 2001 (based on bryan and norman)
        ret /= self.Om
        # from colossus.halo import mass_so
        # assert np.isclose(ret,mass_so.deltaVir(self.z) / self.cosmo.Om(self.z))
        return ret

    # mass function
    def dndM(self,M):
        return 1./M * mass_function.massFunction(M, self.z, q_out='dndlnM', mdef='fof', model='sheth99')

    # mass function
    def dndlnM(self,M):
        return M*self.dndM(M)
    
    def rvir(self,M): # Mpc/h
        y = 3.*M / (4.*np.pi) / (self.rhobar*self.Delta)
        return y**(1./3)

    def rhos(self,M,c=9.):
        rs = self.rvir(M)/c
        return M/(4.*np.pi*rs**3) / (np.log(1.+c)-c/(1.+c))

    def M_in(self,r,M,c=9.):
        rs = self.rvir(M)/c
        x = r/rs
        if r < self.rvir(M):
            return 4. * np.pi * self.rhos(M,c) * rs**3 \
                * (np.log(1.+x)-x/(1.+x))
        else: # if r > rvir
            return 4. * np.pi * self.rhos(M,c) * rs**3 \
                * (np.log(1.+c)-c/(1.+c))

    def phi_NFW(self,r,M,c=9.,z=None): # varphi, dimensionless, truncated
        """Truncated NFW potential profile (dimensionless)."""
        if np.isscalar(r):
            a = 1./(1.+self.z) if z is None else 1./(1.+z)
            rv = self.rvir(M)
            rs = rv/c

            if np.isclose(r,0.):
                return -a*a * 4.*np.pi*GN * self.rhos(M,c) * rs**2 * c/(1.+c) / CSPEED**2

            if r > rv:
                return -a*a * GN * self.M_in(r,M,c)/r / CSPEED**2
            else: # r < rv
                p1 = GN * self.M_in(r,M,c)/r / CSPEED**2
                p2 = 4.*np.pi*GN * self.rhos(M,c) * rs**3 \
                    * (1./(r+rs)-1./(rs+rv)) / CSPEED**2
                return -a*a * (p1+p2)
        else:
            return np.array([self.phi_NFW(_,M,c,z) for _ in r])

    def phi_NFW_infinite(self,r,M,c=9.,z=None): # varphi
        if np.isscalar(r):
            a = 1./(1.+self.z) if z is None else 1./(1.+z)
            f = 1./(np.log(1.+c)-c/(1.+c))
            rv = self.rvir(M)
            rs = rv/c
            x = r/rs
            y0 = -a*a * c*f * GN*M/rv / CSPEED**2
            if np.isclose(r,0.):
                return y0
            else:
                return y0 * np.log(1.+x)/x
        else:
            return np.array([self.phi_NFW_infinite(_,M,c,z) for _ in r])

    def phi_NFW_bar(self,r,M1,M2,c=9.,log_integrate=False,z=None):
        """Mass-averaged truncated NFW potential between [M1, M2]."""
        # if z not None, to do this properly we would also need to pass
        # the redshift the mass function
        if np.isscalar(r):
            if log_integrate:
                _fnc = lambda lnM: self.dndlnM(np.exp(lnM)) * self.phi_NFW(r,np.exp(lnM),c,z)
                val,err = quad(_fnc,np.log(M1),np.log(M2),limit=100)                
            else: # linear (slow)
                _fnc = lambda M: self.dndM(M) * self.phi_NFW(r,M,c,z)
                val,err = quad(_fnc,M1,M2,limit=100)
            ret = val/self.nbar(M1,M2,log_integrate)
            return ret
        else:
            return np.array([self.phi_NFW_bar(_,M1,M2,c,log_integrate,z) for _ in r])

    def phi_NFW_infinite_bar(self,r,M1,M2,c=9.,log_integrate=False,z=None):
        # if z not None, to do this properly we would also need to pass
        # the redshift the mass function
        if np.isscalar(r):
            if log_integrate:
                _fnc = lambda lnM: self.dndlnM(np.exp(lnM)) * self.phi_NFW_infinite(r,np.exp(lnM),c,z)
                val,err = quad(_fnc,np.log(M1),np.log(M2),limit=100)
            else: # linear (slow)
                _fnc = lambda M: self.dndM(M) * self.phi_NFW_infinite(r,M,c,z)
                val,err = quad(_fnc,M1,M2,limit=100)
            ret = val/self.nbar(M1,M2,log_integrate)
            return ret
        else:
            return np.array([self.phi_NFW_infinite_bar(_,M1,M2,c,log_integrate,z) for _ in r])

    def Delta_phi_NFW(self,r,M,c=9.): # narrow bin limit, subtracts zero lag
        if np.isscalar(r):
            return self.phi_NFW(r,M,c) - self.phi_NFW(0.,M,c)
        else:
             return np.array([self.Delta_phi_NFW(_,M,c) for _ in r])

    def Delta_Psi_1h(self,r,MA,MB,c=9.,norm=False): # narrow bin limit
        A = self.Delta_phi_NFW(r,MA,c) #- self.phi_NFW(0.,MA,c) # bug spotted 1.3.25
        B = self.Delta_phi_NFW(r,MB,c) #- self.phi_NFW(0.,MB,c)
        D = B-A
        # if norm:
        #     D /= 1.0+self.xi_AB(r,MA,MB)
        return D

    def dphi_dr_neg(self,r,M,c=9.): # -dphi/dr / (c/aH) / c^2 # dimensionless
        if np.isscalar(r):
            a = 1./(1.+self.z)
            H = self.cosmo.Hz(self.z)
            rv = self.rvir(M)
            M = self.M_in(r,M,c) if r<rv else M
            return a*a * GN*M/r**2 * (-CSPEED/self.calH) / CSPEED**2
        else:
            return np.array([self.dphi_dr_neg(_,M,c) for _ in r])

    def b1(self,M):
        # nu = peaks.peakHeight(M, z)
        # b = bias.haloBiasFromNu(nu, model='cole89') # cole89 is pbs
        b = bias.haloBias(M, model='cole89', z=self.z, mdef='vir')
        return b

    def y_NFW(self,k,M,c=9.): # fourier transform of nfw profile
        rv = self.rvir(M)
        rs = rv/c
        x = k*rs
        S1,C1 = sici((1.+c)*x)
        S2,C2 = sici(x)
        y = np.cos(x)*(C1-C2) + np.sin(x)*(S1-S2) - np.sin(c*x)/((1.+c)*x)
        N = np.log(1.+c) - c/(1.+c)
        y /= N
        return y

    def I11(self,k,c=9.): # mass integral
        def _dI(M):
            return M/self.rhobar * self.dndM(M) * self.b1(M) * self.y_NFW(k,M,c)
        Ib,err = quad(_dI,self.M1,self.M2,limit=100)
        return Ib + self.Ia

    def nbar(self,M1,M2,log_integrate=False):
        if log_integrate:
            _fnc = lambda lnM: self.dndlnM(np.exp(lnM))
            val,err = quad(_fnc,np.log(M1),np.log(M2),limit=100)
        else: # linear (slow)
            _fnc = lambda M: self.dndM(M)
            val,err = quad(_fnc,M1,M2,limit=100)
        return val

    def b1bar(self,M1,M2,log_integrate=False):
        if log_integrate:
            _fnc = lambda lnM: self.dndlnM(np.exp(lnM)) * self.b1(np.exp(lnM))
            val,err = quad(_fnc,np.log(M1),np.log(M2),limit=100)
        else: # linear (slow)
            _fnc = lambda M: self.dndM(M) * self.b1(M)
            val,err = quad(_fnc,M1,M2,limit=100)
        ret = val/self.nbar(M1,M2,log_integrate)
        return ret

# two-halo routines

    def Pm(self,k):
        if self.lnpk_interp is None: # create spline
            ks,pk = np.loadtxt(dir+'/tables/Pk_Planck18_large.dat',unpack=True)
            self.lnpk_interp = interp1d(np.log(ks),np.log(pk))
        p0 = np.exp(self.lnpk_interp(np.log(k)))
        return p0 * self.Dgrowth**2

    def xi(self,r): # matter-matter, linear theory
        if self.xi_interp is None:
            k,pk = np.loadtxt(dir+'/tables/Pk_Planck18_large.dat',unpack=True)
            F = fftlog.CorrelationFFTLog(k,pk)
            rs = F.qv
            D = self.Dgrowth
            self.xi_interp = interp1d(rs,F.xi00*D*D)
        return self.xi_interp(r)

    def xi_Psi_delta(self,r): # Psi-matter, linear theory, dimensionless
        if self.xi_Psi_delta_interp is None: # create spline
            k,pk = np.loadtxt(dir+'/tables/Pk_Planck18_large.dat',unpack=True)
            F = fftlog.CorrelationFFTLog(k,pk)
            rs = F.qv
            _xi_Psi_delta = -F.xi0m2 * self.potfac / CSPEED**2
            D =	self.Dgrowth
            self.xi_Psi_delta_interp = interp1d(rs,_xi_Psi_delta*D*D)
        return self.xi_Psi_delta_interp(r)

    def xi_Psi_delta0(self):
        k,pk = np.loadtxt(dir+'/tables/Pk_Planck18_large.dat',unpack=True)
        F = fftlog.CorrelationFFTLog(k,pk)
        rs = F.qv
        _xi_Psi_delta = -F.xi0m2 * self.potfac / CSPEED**2
        D = self.Dgrowth
        return _xi_Psi_delta[0] * D*D

    def xi_AB(self,r,MA,MB): # narrow bins
        return self.b1(MA) * self.b1(MB) * self.xi(r)
        
    def _GIxi(self,c=9.):
        """Return (r, Q) with Q = (4πG a^2 rhobar / c^2) * G * I11 * xi_L."""
        ks = np.geomspace(1e-4,2e2,1000)
        I11 = np.array([self.I11(_,c) for _ in ks])
        qs = I11 * self.Pm(ks)
        F = fftlog.CorrelationFFTLog(ks,qs)
        r = F.qv
        Q = -F.xi0m2 * self.potfac / CSPEED**2 # xi0m2 because -1/k^2 from potential
        return (r,Q)

    def Psi_delta_2h(self,r,M,c=9.): #\langle\Psi(x)\delta_i(x)\rangle_2h, narrow bin limit
        r,Q = self._GIxi(c)
        return (r,self.b1(M)*Q)

    def Delta_Psi_2h(self,MA1,MA2,MB1,MB2,c=9.,norm=False): # subtract zero lag
        bA = self.b1bar(MA1,MA2)
        bB = self.b1bar(MB1,MB2)
        Delta_b = bB - bA
        r,Q = self._GIxi(c)
        D = Q-Q[0]
        if norm:
            D /= 1.0 + bA * bB * self.xi(r)
        return (r,Delta_b*D)

    def Delta_Psi_2h_AB(self,MA,MB,c=9.,norm=False): # narrow bin limit; subtract out zero lag
        Delta_b = self.b1(MB) - self.b1(MA)
        r,Q = self._GIxi(c)
        D = Q-Q[0]
        if norm:
            D /= 1.0+self.xi_AB(r,MA,MB)
        return (r,Delta_b*D)

    def Delta_Psi_total(self,MA,MB,c=9.,norm=False): # narrow bin limit
        r,y2 = self.Delta_Psi_2h_AB(MA,MB,c)
        y1 = self.Delta_Psi_1h(r,MA,MB,c)
        D = y1+y2
        if norm:
            D /= 1.0+self.xi_AB(r,MA,MB)
        return (r,D)

    def dDeltaPsi_dr(self,MA,MB,c=9.,norm=False,which='total'):
        """Return d/dr of one-, two-, or total Delta_Psi profile."""
        if which == '1h':
            r = np.geomspace(0.5,300,1000)
            D = self.Delta_Psi_1h(r,MA,MB,c,norm=False)
            dD_dr = np.gradient(D,r)
            return (r,dD_dr)
        elif which == '2h':
            r,D = self.Delta_Psi_2h_AB(MA,MB,c,norm)
            interp = interp1d(r,D)
            rs = np.geomspace(0.5,300,1000)
            Ds = interp(rs)
            dD_dr = np.gradient(Ds,rs)
            return (rs,dD_dr)            
        elif which == 'total':
            r,D = self.Delta_Psi_total(MA,MB,c,norm)
            interp = interp1d(r,D)
            rs = np.geomspace(0.5,300,1000)
            Ds = interp(rs)
            dD_dr = np.gradient(Ds,rs)
            return (rs,dD_dr)

if __name__ == '__main__':
    M,c = 2.2e13,9.
    z = 0.341
    halo = halomod(z=z,cosmo_pars='planck')
    print('dispersion (linear theory): ', halo.b1(M) * halo.xi_Psi_delta0() * CSPEED / halo.calH )
    print('dispersion (two halo): ', halo.Psi_delta_2h(0.,M,c=c)[1][0] * CSPEED / halo.calH) # slow
    print('dispersion (one halo): ', halo.phi_NFW(0.,M,c=c) * CSPEED / halo.calH )
