import numpy as np
from scipy.interpolate import interp1d,RegularGridInterpolator
from scipy.integrate import quad,dblquad
from scipy.special import legendre

import cosmo
import params
from corrfuncs_halo import HaloSupport
from corrfuncs_io import (
    load_clpt_xi_u_tables,
    load_gravz_table,
    load_linear_corr_table,
    load_psi_corrs_table,
)


# vartheta is the angular separation between n_1 and n_2, i.e. cos(vartheta)=n_1.n_2
# varmu = cos(vartheta)
# theta is the separation between the LOS and the separation vec{s}
# mu is the standard variable by which we decompose into multipoles. It is NOT the same as varmu!
# Note: importing camb messes with parallelization


def _pair_coincides(x1, x2, varmu):
    return np.isclose(x1, x2) and np.isclose(varmu, 1.0)


def _interp_with_cutoff(interpolator, r, rcut):
    """Evaluate an interpolator and zero out values below the cutoff scale."""
    values = interpolator(r)
    if np.isscalar(r):
        return values if r > rcut else 0.0
    r = np.asarray(r)
    values = np.asarray(values)
    return np.where(r > rcut, values, 0.0)


class gsm(HaloSupport):

    def __init__(self,bA=1.0, bB=1.0, with_gravz=True, with_clpt=True, V0_boost=1.0,\
                 MA=None, MB=None, with_rsd=True, z=0.0, param_set='planck', \
                 cA=9.0, cB=9.0, MA1=None, MA2=None, MB1=None, MB2=None, with_lightcone=False,
                 with_lookback=False, with_local=False, cov_connected=False, with_all_cov=False): # remember to update gsm_sdu init
        """Initialize GSM state from lookup tables and optional physical corrections."""
        self._init_background(bA, bB, with_rsd, z, param_set)
        linear_table = load_linear_corr_table(self.z, param_set)
        self._init_linear_interpolators(linear_table)
        self._init_covariance_interpolators(load_psi_corrs_table(self.z, param_set), with_all_cov)
        self._init_lookback(with_lookback)
        self.with_lightcone = with_lightcone
        self.with_local = with_local
        self.cov_connected = cov_connected
        if with_clpt:
            self._apply_clpt_override(param_set)
        self._init_gravz(with_gravz, V0_boost, linear_table, param_set)
        self._init_halo_terms(MA, MB, cA, cB, MA1, MA2, MB1, MB2, param_set)

    def _init_background(self, bA, bB, with_rsd, z, param_set):
        self.pars = params.get_params(param_set)
        self.param_set = param_set
        self.z = z
        self.with_rsd = with_rsd
        self.bA = bA
        self.bB = bB
        self.Om = cosmo.Om(z, self.pars)
        self.aH = cosmo.aH(z, self.pars) # km/s/(Mpc/h)
        self.dH = params.CSPEED/self.aH # Mpc/h
        self.f = cosmo.f_growthrate(z, self.pars)
        self.D_growth = cosmo.D_growthfactor_M95(z, self.pars)

    def _init_linear_interpolators(self, linear_table):
        r = linear_table[:,0]
        self.r_fftlog = r # the r's returned when running fftlog on the pk
        self.sigv = np.sqrt(linear_table[0,2]) # km/s
        self.sigv2 = self.sigv**2
        self.sigu = self.sigv/self.aH # Mpc/h where [aH]=km/s/(Mpc/h)
        self.sigu2 = self.sigu**2
        self.xi_interp = interp1d(r, linear_table[:,1], bounds_error=False, fill_value='extrapolate')
        self.Psi_para_interp = interp1d(r, linear_table[:,2], bounds_error=False, fill_value=(self.sigv**2,0.))
        self.Psi_perp_interp = interp1d(r, linear_table[:,3], bounds_error=False, fill_value=(self.sigv**2,0.))
        self.U_interp = interp1d(r, linear_table[:,4], bounds_error=False, fill_value=(0.,0.))
        self.xis_spline = None

    def _init_covariance_interpolators(self, psi_table, with_all_cov):
        self.with_all_cov = with_all_cov
        self.xi_up_interp = interp1d(psi_table[:,0], psi_table[:,1], bounds_error=False, fill_value='extrapolate')
        self.xi_pp_interp = interp1d(psi_table[:,0], psi_table[:,2], bounds_error=False, fill_value='extrapolate')

    def _init_lookback(self, with_lookback):
        self.with_lookback = with_lookback
        self.Ad0 = self.D_growth
        self.Av0 = self.aH * self.D_growth * self.f # aH*D*f
        self.Au0 = self.Av0 / self.aH # D*f at the fiducial z
        self.Apot0 = self.D_growth * (1.+self.z) # D/a at the fiducial z

        zs = np.linspace(0.5*self.z,1.5*self.z,250)
        chis = np.array([cosmo.conformal_distance(z_, self.pars) for z_ in zs])
        growth = cosmo.D_growthfactor_M95(zs, self.pars)
        growth_rate = cosmo.f_growthrate_approx(zs, self.pars)
        aH = cosmo.aH(zs, self.pars)
        self.chi2z = interp1d(chis, zs, fill_value='extrapolate')
        self.Ad_interp = interp1d(chis, growth/self.Ad0, fill_value='extrapolate')
        self.Au_interp = interp1d(chis, growth*growth_rate/self.Au0, fill_value='extrapolate')
        self.Av_interp = interp1d(chis, aH*growth*growth_rate/self.Av0, fill_value='extrapolate')
        self.Apot_interp = interp1d(chis, growth*(1.+zs)/self.Apot0, fill_value='extrapolate')

    def _apply_clpt_override(self, param_set):
        r, xi_clpt, U_clpt = load_clpt_xi_u_tables(self.z, param_set)
        self.r_clpt = r
        self.xi_interp = interp1d(self.r_clpt, xi_clpt, bounds_error=False, fill_value='extrapolate')
        self.U_interp = interp1d(self.r_clpt, U_clpt, bounds_error=False, fill_value=(0.,0.))

    def _init_gravz(self, with_gravz, V0_boost, linear_table, param_set):
        self.with_gravz = with_gravz
        if not self.with_gravz:
            return
        gravz_table = load_gravz_table(self.z, param_set)
        assert np.all(gravz_table[:,0] == linear_table[:,0])
        self.V_interp = interp1d(gravz_table[:,0], gravz_table[:,2], bounds_error=False, fill_value='extrapolate')
        self.V0 = gravz_table[0,2] * V0_boost

    def _init_halo_terms(self, MA, MB, cA, cB, MA1, MA2, MB1, MB2, param_set):
        halo_cosmo = params.HALO_COSMO_PAR_MAP.get(param_set, param_set)
        self._configure_halo_model(MA, MB, cA, cB, MA1, MA2, MB1, MB2, halo_cosmo_pars=halo_cosmo)

    def _pair_geometry(self, chi1, chi2, varmu):
        r = self.r(chi1, chi2, varmu)
        u1 = (chi1-varmu*chi2)/r
        u2 = (varmu*chi1-chi2)/r
        return r, u1, u2

    def _one_plus_xi(self, r, chi1=None, chi2=None, varmu=None):
        return 1.0 + self.xi(r, chi1, chi2, varmu)

    def _lookback_potential_factor(self, chi):
        if np.isclose(chi, 0.0):
            return 1.0 / self.Apot0
        return self.Apot_interp(chi)

    def _gravz_two_halo_mean(self, chi1, chi2, varmu, xi_norm):
        if self.with_local:
            first = self.bA*(self.V0-self.V2(chi1,0.,varmu)) + self.bB*(self.V1(chi1,chi2,varmu)-self.V1(0.,chi2,varmu))
            second = self.bB*(self.V0-self.V1(0.,chi2,varmu)) + self.bA*(self.V2(chi1,chi2,varmu)-self.V2(chi1,0.,varmu))
        else:
            first = self.bA*self.V0 + self.bB*self.V1(chi1,chi2,varmu)
            second = self.bB*self.V0 + self.bA*self.V2(chi1,chi2,varmu)
        return np.array([first, second]) / xi_norm

    def _gravz_one_halo_mean(self, r, chi1, chi2):
        if not self.use_halo:
            return np.zeros(2)
        if self.with_lookback:
            z1 = self.chi2z(chi1)
            z2 = self.chi2z(chi2)
        else:
            z1, z2 = None, None
        VA_1h, VB_1h = self._halo_pair(r, z1=z1, z2=z2)
        return np.array([self.V0A_1h + VB_1h, self.V0B_1h + VA_1h])

    def varmu(self,vartheta):
        return np.cos(vartheta * np.pi/180.)

    def r(self,chi1,chi2,varmu):
        rsquared = chi1**2 + chi2**2 - 2.*varmu*chi1*chi2
        return np.sqrt(rsquared)

    def xi(self,r,chi1=None,chi2=None,varmu=None):
        ret = self.bA*self.bB*self.xi_interp(r)
        if self.with_lookback:
            ret *= self.Ad_interp(chi1) * self.Ad_interp(chi2)
        if self.with_lightcone:
            _, u1, u2 = self._pair_geometry(chi1, chi2, varmu)
            ret += self.U(r) * (self.bB*u1-self.bA*u2) / self.dH
            ret += self.Psi_12_uu(chi1,chi2,varmu) / self.dH**2
        return ret

    def Psi_para(self,r): # (km/s)^2
        return self.Psi_para_interp(r)

    def Psi_perp(self,r): # (km/s)^2
        return self.Psi_perp_interp(r)
    
    def U(self,r): # Mpc/h
        return self.U_interp(r)

    # def zeta(self,r): # convenience func using linear theory U
    #     return self.U(r) * (2./self.f) / (1.0+self.xi(r))

    def Psi_12(self,chi1,chi2,varmu): # Psi_ij n_1^i n_2^j, (km/s)^2
        r, u1, u2 = self._pair_geometry(chi1, chi2, varmu)
        ret = self.Psi_para(r)*u1*u2 + self.Psi_perp(r)*(varmu-u1*u2)
        if self.with_lookback:
            ret *= self.Av_interp(chi1) * self.Av_interp(chi2)
        return ret

    def rho(self,chi1,chi2,varmu): # unitless
        if _pair_coincides(chi1, chi2, varmu):
            return 1.
        return self.Psi_12(chi1,chi2,varmu) / self.sigv**2

    def Psi_12_uu(self,chi1,chi2,varmu): # Psi_12 / calH^2, (Mpc/h)^2
        # this is the one referred to in paper not Psi_12
        ret = self.Psi_12(chi1,chi2,varmu) / self.aH**2
        if self.with_lookback:
            ret *= self.Au_interp(chi1) * self.Au_interp(chi2)
        return ret

    def U1(self,chi1,chi2,varmu): # vec{U}.n_1, Mpc/h; <u(x1)delta(x2)>
        r, u1, _ = self._pair_geometry(chi1, chi2, varmu)
        ret = self.U(r)*u1
        if self.with_lookback:
            ret *= self.Au_interp(chi1) * self.Ad_interp(chi2)
        return ret
    
    def U2(self,chi1,chi2,varmu): # vec{U}.n_2, Mpc/h; <delta(x1)u(x2)>
        r, _, u2 = self._pair_geometry(chi1, chi2, varmu)
        ret = self.U(r)*u2
        if self.with_lookback:
            ret *= self.Ad_interp(chi1) * self.Au_interp(chi2)
        return ret

    def V(self,r): # Mpc/h
        return self.V_interp(r)

    def V1(self,chi1,chi2,varmu): # <Psi(x1)delta(x2)>
        r = self.r(chi1,chi2,varmu)
        ret = self.V(r)
        if self.with_lookback:
            ret *= self._lookback_potential_factor(chi1) * self.Ad_interp(chi2)
        return ret

    def V2(self,chi1,chi2,varmu): # <delta(x1)Psi(x2)>
        r = self.r(chi1,chi2,varmu)
        ret = self.V(r)
        if self.with_lookback:
            ret *= self.Ad_interp(chi1) * self._lookback_potential_factor(chi2)
        return ret

    def mu_mean(self,chi1,chi2,varmu): # Mpc/h
        """Return pairwise mean LOS displacement vector [mu1, mu2] in Mpc/h."""
        r = self.r(chi1,chi2,varmu)
        xi_norm = self._one_plus_xi(r, chi1, chi2, varmu)
        y_2h = np.zeros(2)
        if self.with_rsd:
            y_2h[0] +=  self.bB * self.U1(chi1,chi2,varmu) / xi_norm
            y_2h[1] += -self.bA * self.U2(chi1,chi2,varmu) / xi_norm
        if self.with_lightcone:
            sigu2_11 = self.sigu2 * self.Au_interp(chi1)**2
            sigu2_22 = self.sigu2 * self.Au_interp(chi2)**2
            psi_uu = self.Psi_12_uu(chi1,chi2,varmu)
            y_2h[0] += 1./self.dH * (sigu2_11*varmu + psi_uu) / xi_norm
            y_2h[1] += 1./self.dH * (sigu2_22*varmu + psi_uu) / xi_norm
        if self.with_gravz:
            y_2h += self._gravz_two_halo_mean(chi1, chi2, varmu, xi_norm)
            # don't need to divide by 1+xi the 1h terms (zero lag and non zero lag) since there is cancellation
        return y_2h + self._gravz_one_halo_mean(r, chi1, chi2)

    def cov(self,chi1,chi2,varmu): # (Mpc/h)^2
        """Build the 2x2 covariance of LOS displacements for a pair configuration."""
        r = self.r(chi1,chi2,varmu)
        xi = self.xi(r,chi1,chi2,varmu)
        sigu11 = self.sigu * self.Au_interp(chi1)
        sigu22 = self.sigu * self.Au_interp(chi2)
        C = np.zeros((2,2))
        C[0,0] = sigu11**2 / (1.+xi)
        C[1,1] = sigu22**2 / (1.+xi)
        C[0,1] = self.rho(chi1,chi2,varmu) * np.sqrt(C[0,0] * C[1,1])
        C[1,0] = C[0,1]
        if self.with_all_cov:
            assert self.with_local # includes psi_O by default
            C += self.C_gravz(chi1,chi2,varmu)
            C += self.C_cross(chi1,chi2,varmu)        
        if self.cov_connected:
            mu = self.mu_mean(chi1,chi2,varmu)
            C += -np.outer(mu,mu)
        return C

    def C_gravz(self,chi1,chi2,varmu): # (Mpc/h)^2
        r = self.r(chi1,chi2,varmu)
        xi = self.xi_pp_interp(r) # \tilde\xi_\psi\psi = -0.5*<(psi1-psi2)^2>
        xi0 = 0.0 # \tilde\xi_\psi\psi(0) = 0
        xi1 = self.xi_pp_interp(chi1)
        xi2 = self.xi_pp_interp(chi2)
        C = np.zeros((2,2))
        C[0,0] = xi0-2.*xi1
        C[1,1] = xi0-2.*xi2
        C[0,1] = xi-xi1-xi2
        C[1,0] = C[0,1]
        C *= 1./(1.+xi)
        return C

    def C_cross(self,chi1,chi2,varmu): # (Mpc/h)^2
        # includes psi_O by default
        r = self.r(chi1,chi2,varmu)
        u1 = (chi1-varmu*chi2)/r
        u2 = (varmu*chi1-chi2)/r
        xi = self.xi_up_interp(r)
        xi0 = 0.0 # <u(x)\psi(x)>=0
        xi1 = self.xi_up_interp(chi1)
        xi2 = self.xi_up_interp(chi2)
        C = np.zeros((2,2))
        C[0,0] = 2.*(xi0-xi1)
        C[1,1] = 2.*(xi0-xi2)
        C[0,1] = u1*xi-xi1 + (-u2*xi-xi2)
        C[1,0] = C[0,1]
        xi = self.xi(r,chi1,chi2,varmu)
        C *= 1./(1.+xi)
        return C

    def det(self,chi1,chi2,varmu):
        C = self.cov(chi1,chi2,varmu)
        return C[0,0]*C[1,1] - C[0,1]**2

    def cov_inv(self,chi1,chi2,varmu):
        C = self.cov(chi1,chi2,varmu)
        inv = np.zeros_like(C)
        inv[0,0] = C[1,1]
        inv[1,1] = C[0,0]
        inv[0,1] = -C[0,1]
        inv[1,0] = -C[1,0]
        inv /= self.det(chi1,chi2,varmu)
        return inv

    def pdf(self,r1,r2,chi1,chi2,varmu):
        # r1,r2 real distances, chi1,chi2 redshift distances
        if _pair_coincides(r1, r2, varmu): # galaxies coincide
            return 0.
        mu1,mu2 = self.mu_mean(r1,r2,varmu)
        Del = np.array([chi1-r1-mu1,chi2-r2-mu2])
        inv = self.cov_inv(r1,r2,varmu)
        y2 = np.dot(Del,np.dot(inv,Del))
        ret = np.exp(-0.5*y2)
        det = self.det(r1,r2,varmu)
        ret /= 2.*np.pi * np.sqrt(det)
        return ret

    def pdf_alt(self,r1,r2,chi1,chi2,varmu): # including the volume factors
        return self.pdf(r1,r2,chi1,chi2,varmu) * (r1/chi1 * r2/chi2)**2

    def __call__(self,chi1,chi2,varmu): # 1+xi_s        
        """Evaluate 1 + xi_s by integrating over real-space pair separations."""
        def _func(r2,r1): # NB the reverse order!!
            if _pair_coincides(r1, r2, varmu): # galaxies coincide
                return 0.
            y = self.pdf_alt(r1,r2,chi1,chi2,varmu)
            r = self.r(r1,r2,varmu)
            y *= (1.+self.xi(r,chi1,chi2,varmu))
            return y
        val,err = dblquad(_func,chi1-15,chi1+15,chi2-15,chi2+15,epsabs=1e-8,epsrel=1e-8)
        return val

    def xis(self,chi1,chi2,varmu):
        y = self.__call__(chi1,chi2,varmu)
        return y-1.0

    def C0(self,chi1,chi2): # angle average of xis (of varmu not mu)
        def _func(varmu):
            ret = 0.5*self.xis(chi1,chi2,varmu)
            return ret
        val,err = quad(_func,-1.0,1.0,epsabs=1e-8,epsrel=1e-8,limit=100)
        return val

    def get_s(self,chi1,chi2,varmu):
        ssquared = chi1**2 + chi2**2 - 2.*varmu*chi1*chi2
        return np.sqrt(ssquared)

    def get_d(self,chi1,chi2,varmu):
        d2 = chi1**2 + chi2**2 + 2.*varmu*chi1*chi2
        return 0.5*np.sqrt(d2)

    def get_mu(self,chi1,chi2,varmu): # the multipole mu (not the same as varmu)
        s = self.get_s(chi1,chi2,varmu)
        d = self.get_d(chi1,chi2,varmu)
        mu = 0.5*(chi1**2-chi2**2)/(s*d)
        return mu

# for computing multipoles
class gsm_sdu(gsm):

    def __init__(self, bA=1., bB=1., with_gravz=True, with_clpt=True, V0_boost=1., \
                 MA=None, MB=None, with_rsd=True, z=0., param_set='planck', \
                 cA=9.0, cB=9.0, MA1=None, MA2=None, MB1=None, MB2=None, \
                 with_lightcone=False, with_lookback=False, with_local=False, cov_connected=False, with_all_cov=False):
        """Specialized GSM wrapper for fixed (s, d, mu) multipole evaluations."""

        super().__init__(bA,bB,with_gravz,with_clpt,V0_boost,MA,MB,with_rsd,z,param_set, \
                         cA,cB,MA1,MA2,MB1,MB2,with_lightcone,with_lookback,with_local,cov_connected, with_all_cov)
        self._init_C0_interpolator(z, param_set)

    def _init_C0_interpolator(self, z, param_set):
        d = cosmo.conformal_distance(z, params.get_params(param_set))
        x1 = np.linspace(d-100., d+100., 1)
        x2 = x1
        X1, X2 = np.meshgrid(x1, x2)
        Z = np.zeros_like(X1)
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                Z[i,j] = self.C0(X1[i,j], X2[i,j])
        self.C0_interp = RegularGridInterpolator((x1, x2), Z, bounds_error=False, fill_value=None)

    def _chi_pair(self, s, d, mu):
        chi1 = self.get_chi1(s, d, mu)
        chi2 = self.get_chi2(s, d, mu)
        return chi1, chi2, self.get_varmu(s, d, mu)
        
    def get_chi1(self,s,d,mu):
        y = (s/2.)**2 + d**2 + s*d*mu
        return np.sqrt(y)

    def get_chi2(self,s,d,mu):
        y = (s/2.)**2 + d**2 - s*d*mu
        return np.sqrt(y)

    def get_varmu(self,s,d,mu):
        chi1 = self.get_chi1(s,d,mu)
        chi2 = self.get_chi2(s,d,mu)
        return (d**2 - (s/2.)**2) / (chi1*chi2)

    def __call__(self,s,d,mu): # 1+xi_s
        chi1, chi2, varmu = self._chi_pair(s, d, mu)
        return super().__call__(chi1,chi2,varmu)

    def multipole(self,ell,s,d):
        def _fcn(mu):
            xis = -1.0 + self.__call__(s,d,mu)
            ret = xis * legendre(ell)(mu)
            ret *= (2.*ell+1)/2.
            return ret
        y,err = quad(_fcn,-1.0,1.0,limit=100)
        return y

    def C0_multipole(self,ell,s,d):
        def _fcn(mu):
            chi1 = self.get_chi1(s,d,mu)
            chi2 = self.get_chi2(s,d,mu)
            ret = self.C0_interp(chi1,chi2) * legendre(ell)(mu)
            ret *= (2.*ell+1)/2.
            return ret
        y,err = quad(_fcn,-1.0,1.0,limit=100)
        return y


class gsm_DOL(HaloSupport):

    def __init__(self,bA=1.0, bB=1.0, Nstd=8., connected=True, \
                 with_gravz=False, with_clpt=True, V0_boost=1.0, \
                 MA=None, MB=None, with_rsd=True, z=0.0, param_set='planck', \
                 cA=9.0, cB=9.0, MA1=None, MA2=None, MB1=None, MB2=None):
        """Initialize DOL GSM ingredients and interpolation tables."""
        self._init_dol_background(bA, bB, Nstd, z, param_set)
        r = self._init_dol_clpt(param_set)
        linear_table = load_linear_corr_table(self.z, param_set)
        self._init_dol_dispersion(linear_table, r, connected)
        self._init_dol_gravz(with_gravz, V0_boost, linear_table, param_set)
        self._init_dol_halo(MA, MB, cA, cB, MA1, MA2, MB1, MB2, param_set)

    def _init_dol_background(self, bA, bB, Nstd, z, param_set):
        self.z = z
        self.pars = params.get_params(param_set)
        self.param_set = param_set
        self.bA = bA
        self.bB = bB
        self.f = cosmo.f_growthrate(z, self.pars)
        self.aH = cosmo.aH(z, self.pars) # km/s/(Mpc/h)
        self.dH = params.CSPEED/self.aH # Mpc/h
        self.Nstd = Nstd

    def _init_dol_clpt(self, param_set):
        r, xi_clpt, U_clpt = load_clpt_xi_u_tables(self.z, param_set)
        self.xi_interp = interp1d(r,xi_clpt,fill_value='extrapolate')
        self.U_interp = interp1d(r,U_clpt,fill_value='extrapolate') # Mpc/h
        return r

    def _init_dol_dispersion(self, linear_table, r_clpt, connected):
        Psi_para = linear_table[:,2]
        sigv2 = Psi_para[0] # (km/s)^2
        sigpsi2 = sigv2 / (self.f*self.aH)**2 # (Mpc/h)^2
        B_para = linear_table[:,5] # A_para/sig_psi^2
        B_perp = linear_table[:,6] # A_perp/sig_psi^2
        A_para = B_para * sigpsi2 # (Mpc/h)^2
        A_perp = B_perp * sigpsi2 # (Mpc/h)^2
        self.A_para_interp = interp1d(linear_table[:,0],A_para)
        self.A_perp_interp = interp1d(linear_table[:,0],A_perp)
        Xi_para = self.A_para(r_clpt)/(1.0+self.xi(r_clpt))
        Xi_para -= self.zeta(r_clpt)**2 if connected else 0.
        Xi_perp = self.A_perp(r_clpt)/(1.0+self.xi(r_clpt))
        self.Xi_para_interp = interp1d(r_clpt,Xi_para,fill_value='extrapolate')
        self.Xi_perp_interp = interp1d(r_clpt,Xi_perp,fill_value='extrapolate')

    def _init_dol_gravz(self, with_gravz, V0_boost, linear_table, param_set):
        self.with_gravz = with_gravz
        if not self.with_gravz:
            return
        D_gz = load_gravz_table(self.z, param_set)
        assert np.all(D_gz[:,0] == linear_table[:,0])
        self.V_interp = interp1d(D_gz[:,0], D_gz[:,2], bounds_error=False, fill_value='extrapolate')
        self.V0 = D_gz[0,2] * V0_boost

    def _init_dol_halo(self, MA, MB, cA, cB, MA1, MA2, MB1, MB2, param_set):
        halo_cosmo = params.HALO_COSMO_PAR_MAP.get(param_set, param_set)
        self._configure_halo_model(MA, MB, cA, cB, MA1, MA2, MB1, MB2, halo_cosmo_pars=halo_cosmo)

    def _interp_cutoff(self, interpolator, r, rcut):
        return _interp_with_cutoff(interpolator, r, rcut)

    def _gravz_shift_difference(self, r):
        if not self.with_gravz:
            return 0.0
        xi_norm = 1.0 + self.xi(r)
        two_halo = np.array([
            (self.bA*self.V0 + self.bB*self.V(r)) / xi_norm,
            (self.bB*self.V0 + self.bA*self.V(r)) / xi_norm,
        ])
        one_halo = np.zeros(2)
        if self.use_halo:
            VA_1h, VB_1h = self._halo_pair(r)
            one_halo[0] = self.V0A_1h + VB_1h
            one_halo[1] = self.V0B_1h + VA_1h
        return (two_halo + one_halo)[0] - (two_halo + one_halo)[1]


    def rpara(self,r,mu_prime):
        return r*mu_prime

    def rperp(self,r,mu_prime):
        return r*np.sqrt(1.0-mu_prime**2)

    def r(self,rpara,rperp):
        r2 = rpara**2 + rperp**2
        return np.sqrt(r2)

    def mu_prime(self,rpara,rperp):
        return rpara/self.r(rpara,rperp)
        
    def xi(self,r):
        return self.bA*self.bB*self.xi_interp(r)

    def A_para(self,r,rcut=0.6):
        return self._interp_cutoff(self.A_para_interp, r, rcut)

    def A_perp(self,r,rcut=0.6):
        return self._interp_cutoff(self.A_perp_interp, r, rcut)

    def U(self,r,rcut=0.6): # Mpc/h
        return self._interp_cutoff(self.U_interp, r, rcut)

    def zeta(self,r,rcut=0.6):
        _zeta = (self.bA+self.bB) / self.f * self.U(r,rcut) / (1.+self.xi(r))
        return _zeta

    def V(self,r): # Mpc/h
        return self.V_interp(r)
        
    def Xi_para(self,r,rcut=0.6):
        return self._interp_cutoff(self.Xi_para_interp, r, rcut)

    def Xi_perp(self,r,rcut=0.6):
        return self._interp_cutoff(self.Xi_perp_interp, r, rcut)

    def u12(self,rpara,rperp):
        """Return mean pairwise LOS velocity/displacement term for DOL GSM."""
        r = self.r(rpara,rperp)
        ret = self.f * self.mu_prime(rpara,rperp) * self.zeta(r)
        return ret + self._gravz_shift_difference(r)

    def sigma2_12(self,rpara,rperp):
        r = self.r(rpara,rperp)
        y = self.mu_prime(rpara,rperp)
        ret = y*y*self.Xi_para(r) + (1.0-y*y)*self.Xi_perp(r)
        ret *= self.f**2
        return ret

    def xis(self,s,mu):
        return -1.0+self.__call__(s,mu)

    def xis_alt(self,spara,sperp):
        s = self.r(spara,sperp)
        mu = self.mu_prime(spara,sperp)
        return self.xis(s,mu)

    def pdf(self,rpara,spara,sperp):
        ry,sy,sx = rpara,spara,sperp
        d = sy-ry-self.u12(ry,sx)
        X2 = d*d/self.sigma2_12(ry,sx)
        ret = np.exp(-0.5*X2)/np.sqrt(2.*np.pi*self.sigma2_12(ry,sx))
        return ret

    def _gsm_integrand(self,rpara,spara,sperp):
        ret = self.pdf(rpara,spara,sperp)
        r = self.r(rpara,sperp)
        ret *= 1.0+self.xi(r)
        return ret

    def __call__(self,s,mu): # 1+xis
        """Evaluate 1 + xi_s(s, mu) via 1D Gaussian streaming integration."""
        spara = self.rpara(s,mu)
        sperp = self.rperp(s,mu)
        sx,sy = sperp,spara
        ry0 = sy - self.u12(sy,sx)
        dy = np.sqrt(self.sigma2_12(sy,sx)) # ~5 Mpc/h

        # set integration limits (note ry1,ry2 can be negative valued)
        ry1 = ry0 - self.Nstd*dy
        ry2 = ry0 + self.Nstd*dy
        # if s > 35.0: # this sometimes leads to a discontinuity at z>0.2
        #     ry1 = ry0 - self.Nstd*dy
        #     ry2 = ry0 + self.Nstd*dy
        # else: # since the pdf becomes narrower as s -> 0 choose narrower bounds
        #     ry1 = ry0 - 5.0*dy
        #     ry2 = ry0 + 5.0*dy

        # determine if we integrate a confguration with r<1
        ry_min = 0.0 if ry1<0.0<ry2 else np.min([ry1,ry2])
        if self.r(ry_min,sx) <= 1.0:
            print('warning: integrating a configuration with r<1)')

        val,err = quad(self._gsm_integrand, ry1, ry2, args=(sy,sx), limit=100)
        return val

    def multipole(self,ell,s):
        """Compute the Legendre multipole xi_ell(s)."""
        def _fcn(mu,s):
            ret = self.xis(s,mu) * legendre(ell)(mu)
            ret *= (2.*ell+1.)/2.
            return ret
        if np.isscalar(s):
            val,err = quad(_fcn,-1.0,1.0,args=(s,),limit=100)
            return val
        else:
            return np.array([self.multipole(ell, _) for _ in s])
