
from collections import deque
import numpy as np
from scipy.sparse import linalg

from dedalus.tools.config import config
STORE_LU = config['linear algebra'].getboolean('store_LU')
PERMC_SPEC = config['linear algebra']['permc_spec']
USE_UMFPACK = config['linear algebra'].getboolean('use_umfpack')

class MultistepIMEX:
    """
    Base class for implicit-explicit multistep methods.

    Parameters
    ----------
    nfields : int
        Number of fields in problem
    domain : domain object
        Problem domain

    Notes
    -----
    These timesteppers discretize the system
        M.dt(X) + L.X = F
    into the general form
        aj M.X(n-j) + bj L.X(n-j) = cj F(n-j)
    where j runs from {0, 0, 1} to {amax, bmax, cmax}.

    The system is then solved as
        (a0 M + b0 L).X(n) = cj F(n-j) - aj M.X(n-j) - bj L.X(n-j)
    where j runs from {1, 1, 1} to {cmax, amax, bmax}.

    References
    ----------
    D. Wang and S. J. Ruuth, Journal of Computational Mathematics 26, (2008).*

    * Our coefficients are related to those used by Wang as:
        amax = bmax = cmax = s
        aj = α(s-j) / k(n+s-1)
        bj = γ(s-j)
        cj = β(s-j)

    """

    def __init__(self, CoeffSystem, *args):

        self.RHS = CoeffSystem(*args)

        # Create deque for storing recent timesteps
        N = max(self.amax, self.bmax, self.cmax)
        self.dt = deque([0.]*N)

        # Create coefficient systems for multistep history
        self.MX = MX = deque()
        self.LX = LX = deque()
        self.F = F = deque()
        for j in range(self.amax):
            MX.append(CoeffSystem(*args))
        for j in range(self.bmax):
            LX.append(CoeffSystem(*args))
        for j in range(self.cmax):
            F.append(CoeffSystem(*args))

        # Attributes
        self._iteration = 0
        self._LHS_params = None

    def step(self, dt, state_vector, S, L, M, P, NL, LU):
        """Advance solver by one timestep."""

        # References
        MX = self.MX
        LX = self.LX
        F = self.F
        RHS = self.RHS

        # Cycle and compute timesteps
        self.dt.rotate()
        self.dt[0] = dt

        # Compute IMEX coefficients
        a, b, c = self.compute_coefficients(self.dt, self._iteration)
        self._iteration += 1

        # Update RHS components and LHS matrices
        MX.rotate()
        LX.rotate()
        F.rotate()

        MX0 = MX[0]
        LX0 = LX[0]
        F0 = F[0]
        a0 = a[0]
        b0 = b[0]

        if STORE_LU:
            update_LHS = ((a0, b0) != self._LHS_params)
            self._LHS_params = (a0, b0)

        m_start = S.m_min
        m_end = S.m_max
        m_size = m_end - m_start + 1

        for m in range(m_start,m_end+1):
            m_local = m-m_start
            P[m_local] = a0*M[m_local] + b0*L[m_local]
            MX0.data[m_local] = M[m_local].dot(state_vector.data[m_local])
            LX0.data[m_local] = L[m_local].dot(state_vector.data[m_local])
            F0.data[m_local] = NL.data[m_local]

            # Build RHS
            RHS.data[m_local] *= 0.
            for j in range(1, len(c)):
                RHS.data[m_local] += c[j] * F[j-1].data[m_local]
            for j in range(1, len(a)):
                RHS.data[m_local] -= a[j] * MX[j-1].data[m_local]
            for j in range(1, len(b)):
                RHS.data[m_local] -= b[j] * LX[j-1].data[m_local]

            # Solve
            if STORE_LU:
                if update_LHS:
                    LU[m_local] = linalg.splu(P[m_local].tocsc(), permc_spec=PERMC_SPEC)
                pLHS = LU[m_local]
                state_vector.data[m_local] = pLHS.solve(RHS.data[m_local])
            else:
                state_vector.data[m_local] = linalg.spsolve(P[m_local],RHS.data[m_local], use_umfpack=USE_UMFPACK, permc_spec=PERMC_SPEC)

class SBDF1(MultistepIMEX):
    """
    1st-order semi-implicit BDF scheme [Wang 2008 eqn 2.6]

    Implicit: 1st-order BDF (backward Euler)
    Explicit: 1st-order extrapolation (forward Euler)

    """

    amax = 1
    bmax = 1
    cmax = 1

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k0, *rest = timesteps

        a[0] = 1 / k0
        a[1] = -1 / k0
        b[0] = 1
        c[1] = 1

        return a, b, c


class SBDF2(MultistepIMEX):
    """
    2nd-order semi-implicit BDF scheme [Wang 2008 eqn 2.8]

    Implicit: 2nd-order BDF
    Explicit: 2nd-order extrapolation

    """

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return SBDF1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = (1 + 2*w1) / (1 + w1) / k1
        a[1] = -(1 + w1) / k1
        a[2] = w1**2 / (1 + w1) / k1
        b[0] = 1
        c[1] = 1 + w1
        c[2] = -w1

        return a, b, c


class SBDF3(MultistepIMEX):
    """
    3rd-order semi-implicit BDF scheme [Wang 2008 eqn 2.14]

    Implicit: 3rd-order BDF
    Explicit: 3rd-order extrapolation

    """

    amax = 3
    bmax = 3
    cmax = 3

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 2:
            return SBDF2.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k2, k1, k0, *rest = timesteps
        w2 = k2 / k1
        w1 = k1 / k0

        a[0] = (1 + w2/(1 + w2) + w1*w2/(1 + w1*(1 + w2))) / k2
        a[1] = (-1 - w2 - w1*w2*(1 + w2)/(1 + w1)) / k2
        a[2] = w2**2 * (w1 + 1/(1 + w2)) / k2
        a[3] = -w1**3 * w2**2 * (1 + w2) / (1 + w1) / (1 + w1 + w1*w2) / k2
        b[0] = 1
        c[1] = (1 + w2)*(1 + w1*(1 + w2)) / (1 + w1)
        c[2] = -w2*(1 + w1*(1 + w2))
        c[3] = w1*w1*w2*(1 + w2) / (1 + w1)

        return a, b, c


class SBDF4(MultistepIMEX):
    """
    4th-order semi-implicit BDF scheme [Wang 2008 eqn 2.15]

    Implicit: 4th-order BDF
    Explicit: 4th-order extrapolation

    """

    amax = 4
    bmax = 4
    cmax = 4

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 3:
            return SBDF3.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k3, k2, k1, k0, *rest = timesteps
        w3 = k3 / k2
        w2 = k2 / k1
        w1 = k1 / k0

        A1 = 1 + w1*(1 + w2)
        A2 = 1 + w2*(1 + w3)
        A3 = 1 + w1*A2

        a[0] = (1 + w3/(1 + w3) + w2*w3/A2 + w1*w2*w3/A3) / k3
        a[1] = (-1 - w3*(1 + w2*(1 + w3)/(1 + w2)*(1 + w1*A2/A1))) / k3
        a[2] = w3 * (w3/(1 + w3) + w2*w3*(A3 + w1)/(1 + w1)) / k3
        a[3] = -w2**3 * w3**2 * (1 + w3) / (1 + w2) * A3 / A2 / k3
        a[4] = (1 + w3) / (1 + w1) * A2 / A1 * w1**4 * w2**3 * w3**2 / A3 / k3
        b[0] = 1
        c[1] = w2 * (1 + w3) / (1 + w2) * ((1 + w3)*(A3 + w1) + (1 + w1)/w2) / A1
        c[2] = -A2 * A3 * w3 / (1 + w1)
        c[3] = w2**2 * w3 * (1 + w3) / (1 + w2) * A3
        c[4] = -w1**3 * w2**2 * w3 * (1 + w3) / (1 + w1) * A2 / A1

        return a, b, c


class CNAB1(MultistepIMEX):
    """
    1st-order Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.5.3]

    Implicit: 2nd-order Crank-Nicolson
    Explicit: 1st-order Adams-Bashforth (forward Euler)

    """

    amax = 1
    bmax = 1
    cmax = 1

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k0, *rest = timesteps

        a[0] = 1 / k0
        a[1] = -1 / k0
        b[0] = 1 / 2
        b[1] = 1 / 2
        c[1] = 1

        return a, b, c


class CNAB2(MultistepIMEX):
    """
    2nd-order Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.9]

    Implicit: 2nd-order Crank-Nicolson
    Explicit: 2nd-order Adams-Bashforth

    """

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
#            return CNAB1.compute_coefficients(timesteps, iteration)
            return SBDF1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps
        w1 = k1 / k0

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = 1 / 2
        b[1] = 1 / 2
        c[1] = 1 + w1/2
        c[2] = -w1 / 2

        return a, b, c

class MCNAB2(MultistepIMEX):
    """
    2nd-order modified Crank-Nicolson Adams-Bashforth scheme [Wang 2008 eqn 2.10]

    Implicit: 2nd-order modified Crank-Nicolson
    Explicit: 2nd-order Adams-Bashforth

    """

    amax = 2
    bmax = 2
    cmax = 2

    @classmethod
    def compute_coefficients(self, timesteps, iteration):

        if iteration < 1:
            return SBDF1.compute_coefficients(timesteps, iteration)

        a = np.zeros(self.amax+1)
        b = np.zeros(self.bmax+1)
        c = np.zeros(self.cmax+1)

        k1, k0, *rest = timesteps

        w1 = k1 / k0

        a[0] = 1 / k1
        a[1] = -1 / k1
        b[0] = (8 + 1/w1) / 16
        b[1] = (7 - 1/w1) / 16
        b[2] = 1 / 16
        c[1] = 1 + w1/2
        c[2] = -w1 / 2

        return a, b, c

class RungeKuttaIMEX:
    """
    Base class for implicit-explicit multistep methods.

    Parameters
    ----------
    nfields : int
        Number of fields in problem
    domain : domain object
        Problem domain

    Notes
    -----
    These timesteppers discretize the system
        M.dt(X) + L.X = F
    by constructing s stages
        M.X(n,i) - M.X(n,0) + k Hij L.X(n,j) = k Aij F(n,j)
    where j runs from {0, 0} to {i, i-1}, and F(n,i) is evaluated at time
        t(n,i) = t(n,0) + k ci

    The s stages are solved as
        (M + k Hii L).X(n,i) = M.X(n,0) + k Aij F(n,j) - k Hij L.X(n,j)
    where j runs from {0, 0} to {i-1, i-1}.

    The final stage is used as the advanced solution*:
        X(n+1,0) = X(n,s)
        t(n+1,0) = t(n,s) = t(n,0) + k

    * Equivalently the Butcher tableaus must follow
        b_im = H[s, :]
        b_ex = A[s, :]
        c[s] = 1

    References
    ----------
    U. M. Ascher, S. J. Ruuth, and R. J. Spiteri, Applied Numerical Mathematics (1997).

    """

    def __init__(self, CoeffSystem, *args):

        self.RHS = CoeffSystem(*args)


        self.MX0 = CoeffSystem(*args)
        self.LX = LX = [CoeffSystem(*args) for i in range(self.stages)]
        self.NL = NL = [CoeffSystem(*args) for i in range(self.stages)]

        self._iteration = 0
        self._LHS_params = None

    def step(self, dt, state_vector, B, L, M, P, nonlinear, LU):
        """Advance solver by one timestep."""

        # References
        RHS = self.RHS
        MX0 = self.MX0
        LX = self.LX
        NL = self.NL
        A = self.A
        H = self.H
        c = self.c
        k = dt

        ell_start = B.ell_min
        ell_end = B.ell_max
        m_start = B.m_min
        m_end = B.m_max
        m_size = m_end - m_start + 1

        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                index = ell_local*m_size+m_local

                MX0.data[index] = M[ell_local].dot(state_vector.data[index])

        # Compute stages
        # (M + k Hii L).X(n,i) = M.X(n,0) + k Aij F(n,j) - k Hij L.X(n,j)
        for i in range(1, self.stages+1):

            # Compute F(n,i-1)
            nonlinear(state_vector,NL[i-1],0)

            # Compute L.X(n,i-1)
            for ell in range(ell_start,ell_end+1):
                ell_local = ell-ell_start
                P[ell_local] = M[ell_local] + (k*H[i,i])*L[ell_local]
                for m in range(m_start,m_end+1):
                    m_local = m-m_start
                    index = ell_local*m_size+m_local

                    LX[i-1].data[index] = L[ell_local].dot(state_vector.data[index])

            # Construct RHS(n,i)
                    np.copyto(RHS.data[index],MX0.data[index])
                    for j in range(0,i):
                        RHS.data[index] += k * A[i,j] * NL[j].data[index]
                        RHS.data[index] -= k * H[i,j] * LX[j].data[index]

           # Solve
                    state_vector.data[index] = linalg.spsolve(P[ell_local],RHS.data[index], use_umfpack=USE_UMFPACK, permc_spec=PERMC_SPEC)


class RK111(RungeKuttaIMEX):
    """1st-order 1-stage DIRK+ERK scheme [Ascher 1997 sec 2.1]"""

    stages = 1

    c = np.array([0, 1])

    A = np.array([[0, 0],
                  [1, 0]])

    H = np.array([[0, 0],
                  [0, 1]])


class RK222(RungeKuttaIMEX):
    """2nd-order 2-stage DIRK+ERK scheme [Ascher 1997 sec 2.6]"""

    stages = 2

    γ = (2 - np.sqrt(2)) / 2
    δ = 1 - 1 / γ / 2

    c = np.array([0, γ, 1])

    A = np.array([[0,  0 , 0],
                  [γ,  0 , 0],
                  [δ, 1-δ, 0]])

    H = np.array([[0,  0 , 0],
                  [0,  γ , 0],
                  [0, 1-γ, γ]])


class RKHM(RungeKuttaIMEX):
    """2nd-order 2-stage scheme from Hollerbach and Marti"""

    stages = 2

    c = np.array([0, 0.5, 1])

    A = np.array([[  0,  0 , 0],
                  [  1,  0 , 0],
                  [0.5, 0.5, 0]])

    H = np.array([[0   , 0  ,   0],
                  [0.5 , 0.5,   0],
                  [0.5 , 0  , 0.5]])
