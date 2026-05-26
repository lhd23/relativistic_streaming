from nfw import halomod


class HaloSupport:
    """Shared setup/evaluation for optional one-halo grav-z terms."""

    def _configure_halo_model(self, MA, MB, cA, cB, MA1, MA2, MB1, MB2, halo_cosmo_pars):
        self.use_halo = MA is not None and MB is not None
        self.log_integrate = False
        if not self.use_halo:
            return

        self.MA, self.MB = MA, MB
        self.cA, self.cB = cA, cB
        self.MA1, self.MA2 = MA1, MA2
        self.MB1, self.MB2 = MB1, MB2

        self._halo_binned = all(v is not None for v in (MA1, MA2, MB1, MB2))
        self.log_integrate = self._halo_binned

        self.halo = halomod(z=self.z, cosmo_pars=halo_cosmo_pars)
        self.V0A_1h = self._halo_phi(0.0, tracer='A')
        self.V0B_1h = self._halo_phi(0.0, tracer='B')

    def _halo_phi(self, r, tracer, z=None):
        """Return one-halo potential profile in Mpc/h for tracer A or B."""
        if tracer == 'A':
            M, c, M1, M2 = self.MA, self.cA, self.MA1, self.MA2
        elif tracer == 'B':
            M, c, M1, M2 = self.MB, self.cB, self.MB1, self.MB2
        else:
            raise ValueError(f'Unknown tracer "{tracer}"')

        if self._halo_binned:
            phi = self.halo.phi_NFW_bar(r, M1, M2, c, self.log_integrate, z)
        else:
            phi = self.halo.phi_NFW(r, M, c, z)
        return phi * (-self.dH)

    def _halo_pair(self, r, z1=None, z2=None):
        """Return (VA_1h, VB_1h) at separation r."""
        return self._halo_phi(r, tracer='A', z=z1), self._halo_phi(r, tracer='B', z=z2)
