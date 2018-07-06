
import numpy     as np
import sphere128 as sph
from dedalus.tools.cache import CachedMethod

class Sphere:
    def __init__(self,L_max,S_max=0,N_theta=None,m_min=None,m_max=None):
        self.L_max, self.S_max  = L_max, S_max
        if N_theta == None: N_theta = L_max+1
        self.N_theta = N_theta

        self.m_min = m_min
        self.m_max = m_max

        if self.m_min == None: self.m_min = -L_max
        if self.m_max == None: self.m_max =  L_max

        # grid and weights for the all transforms
        self.cos_grid,self.weights = sph.quadrature(self.N_theta-1,niter=3,report_error=False)
        self.grid = np.arccos(self.cos_grid)
        self.sin_grid = np.sqrt(1-self.cos_grid**2)
        
        self.pushY, self.pullY = {}, {}
        
        for s in range(-S_max,S_max+1):
            for m in range(self.m_min,self.m_max+1):
                Y = sph.Y(self.L_max,m,s,self.cos_grid)
                self.pushY[(m,s)] = (self.weights*Y).astype(np.float64)
                self.pullY[(m,s)] = (Y.T).astype(np.float64)
    
        # downcast to double precision
        self.grid     = self.grid.astype(np.float64)
        self.weights  = self.weights.astype(np.float64)
        self.sin_grid = self.sin_grid.astype(np.float64)
        self.cos_grid = self.cos_grid.astype(np.float64)
    
    @CachedMethod
    def op(self,op_name,m,s):
        return sph.operator(op_name,self.L_max,m,s).astype(np.float64)

    @CachedMethod
    def L_min(self,m,s):
        return sph.L_min(m,s)

    def zeros(self,m,s_out,s_in):
        return sph.zeros(self.L_max,m,s_out,s_in)

    def forward_spin(self,m,s,data):
        # grid --> coefficients
        return self.pushY[(m,s)].dot(data)
    
    def backward_spin(self,m,s,data):
        # coefficients --> grid
        return self.pullY[(m,s)].dot(data)

    @CachedMethod
    def tensor_index(self,m,rank):
        num = np.arange(2**rank)
        spin = (-1)**(1+num)
        for k in range(2,rank+1):
            spin += ((-1)**(1+num//2**(k-1))).astype(np.int64)

        if rank == 0: spin = [0]

        start_index = [0]
        end_index = []
        for k in range(2**rank):
            end_index.append(start_index[k]+self.L_max-sph.L_min(m,spin[k])+1)
            if k < 2**rank-1:
                start_index.append(end_index[k])

        return (start_index,end_index,spin)

    @CachedMethod
    def unitary(self,rank=1,adjoint=False):
        return sph.unitary(rank=rank,adjoint=adjoint)

    def forward(self,m,rank,data):

        if rank == 0:
            return self.forward_spin(m,0,data[0])

        (start_index,end_index,spin) = self.tensor_index(m,rank)
   
        unitary = self.unitary(rank=rank,adjoint=True)

        data = np.einsum("ij,j...->i...",unitary,data)

        shape = np.array(np.array(data).shape[1:])
        shape[0] = end_index[-1]

        data_c = np.zeros(shape,dtype=np.complex128)

        for i in range(2**rank):
            data_c[start_index[i]:end_index[i]] = self.forward_spin(m,spin[i],data[i])
        return data_c

    def backward(self,m,rank,data,unitary=None):

        if rank == 0:
            return self.backward_spin(m,0,data)

        (start_index,end_index,spin) = self.tensor_index(m,rank)

        unitary = self.unitary(rank=rank,adjoint=False)

        shape = np.array(np.array(data).shape)
        shape = np.concatenate(([2**rank],shape))
        shape[1] = self.N_theta

        data_g = np.zeros(shape,dtype=np.complex128)

        for i in range(2**rank):
            data_g[i] = self.backward_spin(m,spin[i],data[start_index[i]:end_index[i]])
        return np.einsum("ij,j...->i...",unitary,data_g)

    def grad(self,m,rank_in,data_in,data_out):
        # data_in and data_out are in coefficient space

        (start_index_in,end_index_in,spin_in) = self.tensor_index(m,rank_in)
        rank_out = rank_in+1
        (start_index_out,end_index_out,spin_out) = self.tensor_index(m,rank_out)

        half = 2**(rank_out-1)
        for i in range(2**(rank_out)):
            if i//half == 0:
                operator = self.op('k-',m,spin_in[i%half])
            else:
                operator = self.op('k+',m,spin_in[i%half])

            np.copyto( data_out[start_index_out[i]:end_index_out[i]],
                       operator.dot(data_in[start_index_in[i%half]:end_index_in[i%half]]) )

    def div(self,m,rank_in,data_in,data_out):
        # data_in and data_out are in coefficient space

        (start_index_in,end_index_in,spin_in) = self.tensor_index(m,rank_in)
        rank_out = rank_in-1
        (start_index_out,end_index_out,spin_out) = self.tensor_index(m,rank_out)

        for i in range(2**(rank_out)):
            op1 = self.op('k+',m,spin_out[i]-1)
            op2 = self.op('k-',m,spin_out[i]+1)

            np.copyto( data_out[start_index_out[i]:end_index_out[i]],
                       op1.dot(data_in[start_index_in[2*i]:end_index_in[2*i]])
                      +op2.dot(data_in[start_index_in[2*i+1]:end_index_in[2*i+1]]) )

class TensorField:

    def __init__(self,rank,S,domain):
        self.domain = domain
        self.S = S
        self.rank = rank

        self.m_min, self.m_max = S.m_min, S.m_max

        local_grid_shape = self.domain.distributor.layouts[-1].local_shape(scales=domain.dealias)
        grid_shape = np.append(2**rank,np.array(local_grid_shape))
        local_mtheta_shape = self.domain.distributor.layouts[1].local_shape(scales=domain.dealias)
        mtheta_shape = np.append(2**rank,np.array(local_mtheta_shape))

        self.grid_data = np.zeros(grid_shape,dtype=np.float64)
        self.mtheta_data = np.zeros(mtheta_shape,dtype=np.complex128)

        self.fields = domain.new_fields(2**rank)
        for field in self.fields:
            field.set_scales(domain.dealias)
            field.require_grid_space()
        self.coeff_data = []
        for m in range(self.m_min, self.m_max + 1):
            (start_index,end_index,spin) = self.S.tensor_index(m,rank)
            self.coeff_data.append(np.zeros(end_index[-1],dtype=np.complex128))

        self._layout = 'g'
        self.data = self.grid_data

    def __getitem__(self, layout):
        """Return data viewed in specified layout."""

        self.require_layout(layout)
        return self.data

    def __setitem__(self, layout, data):
        """Set data viewed in specified layout."""

        self.layout = layout
        np.copyto(self.data, data)

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, layout):
        self._layout = layout
        if self._layout == 'g':
            self.data = self.grid_data
        elif self._layout == 'c':
            self.data = self.coeff_data

    def require_layout(self, layout):

        if layout == 'g' and self._layout == 'c':
            self.require_grid_space()
        elif layout == 'c' and self._layout == 'g':
            self.require_coeff_space()

    def require_coeff_space(self):
        """Transform from grid space to coeff space"""

        rank = self.rank

        for i,field in enumerate(self.fields):
            field.require_grid_space()
            field.data = self.data[i]
            field.require_layout(self.domain.distributor.layouts[1])

        for m in range(self.m_min, self.m_max+1):
            m_local = m - self.m_min
            self.coeff_data[m_local] = self.S.forward(m,rank,
                                                      [self.fields[i].data[m_local] for i in range(2**rank)])

        self.data = self.coeff_data
        self.layout = 'c'

    def require_grid_space(self):
        """Transform from coeff space to grid space"""

        rank = self.rank

        for m in range(self.m_min, self.m_max + 1):
            m_local = m - self.m_min
            self.mtheta_data[:,m_local,:] = self.S.backward(m,rank,self.coeff_data[m_local])

        for i,field in enumerate(self.fields):
            field.layout = self.domain.distributor.layouts[1]
            field.data = self.mtheta_data[i]
            field.require_grid_space()
            self.grid_data[i] = field.data

        self.data = self.grid_data
        self._layout = 'g'


