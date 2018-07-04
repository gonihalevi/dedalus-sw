"""
Higher-order Navier Stokes on the sphere.
"""

import numpy as np
from scipy.sparse import linalg as spla
import sphere_wrapper as sph
import equations_SW as eq
import os
import dedalus.public as de
import time
import pathlib
from mpi4py import MPI

# Load config options
from dedalus.tools.config import config
STORE_LU = config['linear algebra'].getboolean('store_LU')
PERMC_SPEC = config['linear algebra']['permc_spec']
USE_UMFPACK = config['linear algebra'].getboolean('use_umfpack')

# Discretization parameters
L_max = 255  # spherical harmonic order
S_max = 3  # spin order (leave fixed)

# Physical parameters
H = 1
g = 2
Om2 = 1.885e-3
Om = Om2
period = 2*np.pi/Om2
a = 10e3

pole=True
phi_0 = np.pi/2.
theta_0 = np.pi
w = 0.05 #width of initial perturbation
amp = 1*H

# Integration parameters
dt = period/6000.  # timestep
n_iterations = 200  # total iterations
n_output = 2  # data output cadence
output_folder = 'output_files/pole_rot/'  # data output folder

# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank

L_dealias=3/2

# Make domain
phi_basis   = de.Fourier('phi'  , 2*(L_max+1), interval=(0,2*np.pi), dealias=L_dealias)
theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi), dealias=L_dealias)
domain = de.Domain([phi_basis,theta_basis], grid_dtype=np.float64)

# set up sphere

m_start = domain.distributor.coeff_layout.start(1)[0]
m_len = domain.distributor.coeff_layout.local_shape(1)[0]
m_end = m_start + m_len - 1
N_theta = int((L_max+1)*L_dealias)
S = sph.Sphere(L_max,S_max,N_theta=N_theta,m_min=m_start,m_max=m_end)

phi = domain.grids(L_dealias)[0]
theta_slice = domain.distributor.grid_layout.slices(domain.dealias)[1]
theta_len = domain.local_grid_shape(domain.dealias)[1]
theta_global = S.grid
theta = S.grid[theta_slice].reshape((1,theta_len))

u = sph.TensorField(1,S,domain)
h = sph.TensorField(0,S,domain)
c = sph.TensorField(0,S,domain)

Du = sph.TensorField(2,S,domain)
uh = sph.TensorField(1,S,domain)
Dc = sph.TensorField(1,S,domain)
divuh = sph.TensorField(0,S,domain)

u_rhs = sph.TensorField(1,S,domain)
h_rhs = sph.TensorField(0,S,domain)
c_rhs = sph.TensorField(0,S,domain)

state_vector = eq.StateVector(S,u,h,c)
RHS = eq.StateVector(S,u,h,c)

# Add random perturbations to the spectral coefficients
# rand = np.random.RandomState(seed=42)
# u.layout='c'
# for m in range(0,L_max+1):
#     md = m - m_start
#     (start_index,end_index,spin) = S.tensor_index(m,1)
#     shape = (end_index[-1])
#     noise = rand.standard_normal(shape)
#     phase = rand.uniform(0,2*np.pi,shape)
#     if m>=m_start and m<=m_end:
#         u['c'][md] = 0.0001 * noise*np.exp(1j*phase)

# def hump(i,j,rlat0,rlon0,a):
#     phiamp = 1
#     radius = a/10.
#     rlat0 *= np.pi/180.
#     rlon0 *= np.pi/180.
#     rlat, rlon = np.pi/2. - i, j
#     dist = a*np.acos(np.sin(rlat0)*np.sin(rlat)+np.cos(rlat0) \
#                     *np.cos(rlat)*np.cos(rlon-rlon0))
#     if dist < radius:
#         hum0 = phiamp/2.0 * (1.0+np.cos(pi*dist/radius))
#     else:
#         hump = 0

if pole == False:
    h['g'] = amp*np.exp( -( theta-theta_0)**2/w**2 - (phi-phi_0)**2/w**2)
else:
    h['g'] = amp*np.exp( -( theta-theta_0)**2/w**2)

state_vector.pack(u,h,c)

# build matrices
P,M,L = [],[],[]
for m in range(m_start,m_end+1):
    Mm,Lm = eq.shallow_water(S,m,[g,H,Om])
    M.append(Mm.astype(np.complex128))
    L.append(Lm.astype(np.complex128))
    P.append(0.*Mm.astype(np.complex128))

# calculate RHS nonlinear terms from state_vector
def nonlinear(state_vector,RHS):

    state_vector.unpack(u,h,c)

    Du.layout = 'c'
    Dc.layout = 'c'
    for m in range(m_start,m_end+1):
        md = m - m_start
        S.grad(m,1,u['c'][md],Du['c'][md])
        S.grad(m,0,c['c'][md],Dc['c'][md])

    u_rhs.layout = 'g'
    uh.layout = 'g'
    c_rhs.layout = 'g'
    h_rhs.layout = 'g'
    u_rhs['g'][0] = - (u['g'][0]*Du['g'][0] + u['g'][1]*Du['g'][2]) # add forcing
    u_rhs['g'][1] = - (u['g'][0]*Du['g'][1] + u['g'][1]*Du['g'][3]) # add forcing
    uh['g'][0] = u['g'][0]*h['g'][0]
    uh['g'][1] = u['g'][1]*h['g'][0]
    c_rhs['g'][0] = - (u['g'][0]*Dc['g'][0] + u['g'][1]*Dc['g'][1]) # add heating
    h_rhs['g'][0] = 0* ((H+h['g'][0])**2 - (H+h['g'][0])**4)

    divuh.layout = 'c'
    for m in range(m_start,m_end+1):
        md = m - m_start
        S.div(m,1,uh['c'][md],divuh['c'][md])
        h_rhs['c'][md] -= divuh['c'][md]

    RHS.pack(u_rhs,h_rhs,c_rhs)

# Setup outputs
file_num = 1
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
t = 0

# Combine matrices and perform LU decompositions for constant timestep
for m in range(m_start,m_end+1):
    md = m - m_start
    Pmd = M[md] + dt*L[md]
    if STORE_LU:
        P[md] = spla.splu(Pmd.tocsc(), permc_spec=PERMC_SPEC)
    else:
        P[md] = Pmd

start_time = time.time()
# Main loop
for i in range(n_iterations):

    if i % n_output == 0:

        state_vector.unpack(u,h,c)

#        for m in range(m_start,m_end+1):
#            md = m - m_start
#            (start_index,end_index,spin) = S.tensor_index(m,1)
#            om['c'][md] = 1j*(S.op('k-',m,1).dot(u['c'][md][start_index[0]:end_index[0]]) - S.op('k+',m,-1).dot(u['c'][md][start_index[1]:end_index[1]]))

        # gather full data to output
        uth_global = comm.gather(u['g'][0], root=0)
        uph_global = comm.gather(u['g'][1], root=0)
        h_global = comm.gather(h['g'][0], root=0) 
        c_global = comm.gather(c['g'][0], root=0)
#        om_global = comm.gather(om['g'][0], root=0)

        if rank == 0:
            # Save data
            uph_global = np.hstack(uph_global)
            uth_global = np.hstack(uth_global)
            h_global = np.hstack(h_global)
            c_global = np.hstack(c_global)
#            om_global = np.hstack(om_global)
            np.savez(os.path.join(output_folder, 'output_%i.npz' %file_num),
#                     p=p_global, om=om_global, vph=vph_global, vth=vth_global,
                     h=h_global, c=c_global, uph=uph_global, uth=uth_global,
                     t=np.array([t]), phi=phi[:,0], theta=theta_global)
            file_num += 1

            # Print iteration and maximum vorticity
            print('Iter:', i, 'Time:', t, 'h max:', np.max(np.abs(h_global)))

    nonlinear(state_vector,RHS)
    for m in range(m_start,m_end+1):
        md = m - m_start
        RHS.data[md] = M[md].dot(state_vector.data[md]) + dt*(RHS.data[md])
        if STORE_LU:
            state_vector.data[md] = P[md].solve(RHS.data[md])
        else:
            state_vector.data[md] = spla.spsolve(P[md],RHS.data[md])

    t += dt

#    # imposing that the m=0 mode of u,h,c are purely real
#    if i % 100 == 1:
#        state_vector.unpack(u,h,c)
#        u.require_grid_space()
#        u.require_coeff_space()
#        h.require_grid_space()
#        h.require_coeff_space()
#        c.require_grid_space()
#        c.require_coeff_space()
#        state_vector.pack(u,h,c)


end_time = time.time()
print('total time: ', end_time-start_time)
