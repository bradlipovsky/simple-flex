import numpy as np
import matplotlib.pyplot as plt


h = 200
rhosw = 1028
rhoi = 917
g = 9.81
nu = 1e16

alpha = 3 *rhosw* g / nu / h**3
beta = alpha/rhosw / g

sigma = rhoi * g * h / 4
f = -h * sigma 

k = np.logspace(-3,3,1000)

exponent = -1*(alpha / k**4 + beta * f / k**2)

kc = k[exponent>0].min()

fig,ax=plt.subplots()
plt.loglog(2*np.pi/k, exponent, label='stable')
plt.loglog(2*np.pi/k, -exponent, label='unstable')
plt.plot([2*np.pi/kc,2*np.pi/kc],ax.get_ylim(),'--k',label='critical wavelength')
plt.xlabel('Wavelength (m)')
plt.ylabel('Growth/Decay Rate (1/s)')
plt.title(f'Critical wavelength: {(2*np.pi/kc):.2f} m')
plt.legend()
plt.grid()
plt.savefig('buckling-criterion.png')
plt.close()