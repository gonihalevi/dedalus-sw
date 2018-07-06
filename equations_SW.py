import numpy             as np
import scipy.sparse      as sparse
import matplotlib.pyplot as plt
from scipy.linalg         import eig
#from mpl_toolkits.basemap import Basemap

# parameters:
# g: gravity
# H: equilibrium height
# Om: rotation rate

# equations:

# dt(um) + g*km*h - 2*Om*i*C*um = - (u.grad u)_m
# dt(up) + g*kp*h + 2*Om*i*C*up = - (u.grad u)_p
# dt(h) + H*kp*um + H*km*up = - kp*(h*u)_m - km*(h*u)_p
# dt(c) = - (u.grad c)

# variable order: um up h c

def shallow_water(S,m,params):
    """Defines M, L matrices for advection"""
        
    g,H,Om,a  = params[0],params[1],params[2],params[3]

    # (-,-)
    L00 = -2*Om*1j*S.op('C',m,-1)

    # (-,+)
    L01 = S.zeros(m,-1,1)

    # (-,h)
    L02 = g*S.op('k-',m,0)/a

    # (-,c)
    L03 = S.zeros(m,-1,0)

    # (+,-)
    L10 = S.zeros(m,1,-1)

    # (+,+)
    L11 = 2*Om*1j*S.op('C',m,+1)

    # (+,h)
    L12 = g*S.op('k+',m,0)/a

    # (+,c)
    L13 = S.zeros(m,1,0)

    # (h,-)
    L20 = H*S.op('k+',m,-1)/a

    # (h,+)
    L21 = H*S.op('k-',m,1)/a

    # (h,h)
    L22 = S.zeros(m,0,0)

    # (h,c)
    L23 = S.zeros(m,0,0)

    # no linear dependence for c:
    L30 = S.zeros(m,0, 1)
    L31 = S.zeros(m,0,-1)
    L32 = S.zeros(m,0, 0)
    L33 = S.zeros(m,0, 0)

    L = sparse.bmat([[L00,L01,L02,L03],[L10,L11,L12,L13],[L20,L21,L22,L23],[L30,L31,L32,L33]])

    R00 = S.op('I',m,-1)
    R01 = S.zeros(m,-1,1)
    R02 = S.zeros(m,-1,0)
    R03 = S.zeros(m,-1,0)

    R10 = S.zeros(m,1,-1)
    R11 = S.op('I',m,1)
    R12 = S.zeros(m,1,0)
    R13 = S.zeros(m,1,0)

    R20 = S.zeros(m,0,-1)
    R21 = S.zeros(m,0,1)
    R22 = S.op('I',m,0)
    R23 = S.zeros(m,0,0)

    R30 = S.zeros(m,0,-1)
    R31 = S.zeros(m,0,1)
    R32 = S.zeros(m,0,0)
    R33 = S.op('I',m,0)

    R = sparse.bmat([[R00,R01,R02,R03],[R10,R11,R12,R13],[R20,R21,R22,R23],[R30,R31,R32,R33]])

    return R,L

class StateVector:

    def __init__(self,u,h,c):
        self.data = []
        self.m_min = u.S.m_min
        self.m_max = u.S.m_max
        self.len_u = []
        self.len_h = []
        for m in range(self.m_min,self.m_max+1):
            m_local = m - self.m_min
            self.len_u.append(u['c'][m_local].shape[0])
            self.len_h.append(h['c'][m_local].shape[0])
            self.data.append(np.concatenate((u['c'][m_local],h['c'][m_local],c['c'][m_local])))

    def pack(self,u,h,c):
        for m in range(self.m_min,self.m_max+1):
            m_local = m - self.m_min
            self.data[m_local] = np.concatenate((u['c'][m_local],h['c'][m_local],c['c'][m_local]))

    def unpack(self,u,h,c):
        u.layout='c'
        h.layout='c'
        c.layout='c'
        for m in range(self.m_min,self.m_max+1):
            m_local = m - self.m_min
            len_u = self.len_u[m_local]
            len_h = self.len_h[m_local]
            u['c'][m_local] = self.data[m_local][:len_u]
            h['c'][m_local] = self.data[m_local][len_u:len_u+len_h]
            c['c'][m_local] = self.data[m_local][len_u+len_h:]


def show_ball(S, field, index, longitude=0, latitude=0, mp = None):
    
    if mp == None:
        figure, ax = plt.subplots(1,1)
        figure.set_size_inches(3,3)

    lon = np.linspace(0, 2*np.pi, 2*(S.L_max+1))
    lat = S.grid - np.pi/2
    
    meshed_grid = np.meshgrid(lon, lat)
    lat_grid = meshed_grid[1]
    lon_grid = meshed_grid[0]
    
    if mp == None:
        mp = Basemap(projection='ortho', lat_0=latitude, lon_0=longitude, ax=ax)
        mp.drawmapboundary()
        mp.drawmeridians(np.arange(0, 360, 30))
        mp.drawparallels(np.arange(-90, 90, 30))

    x, y = mp(np.degrees(lon_grid), np.degrees(lat_grid))
    im = mp.pcolor(x, y, np.transpose(field), cmap='RdYlBu_r')
    
    
    plt.savefig('images/om_%05i.png' %index)
    return im,mp
