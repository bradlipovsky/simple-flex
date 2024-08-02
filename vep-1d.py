from firedrake import IntervalMesh, FunctionSpace, TrialFunctions, \
    TestFunctions, SpatialCoordinate, Function, solve, DirichletBC, \
        Constant, grad, dx, conditional, split, project, VTKFile
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

def main():
    t0 = perf_counter()
    nx = 200 #number of nodes
    Lx = 50000 #length of mesh in meters
    H = 200 #thickness of the bar in meters
    mesh = IntervalMesh(nx, Lx)

    Q1 = FunctionSpace(mesh, 'CG', 2) #this will be η's function space
    Q2 = FunctionSpace(mesh, 'CG', 2) #... and M's function space
    Q3 = FunctionSpace(mesh, 'CG', 2) #... and phi_dot's function space
    Q4 = FunctionSpace(mesh, 'CG', 2) #... and phiP_dot's function space

    Q = Q1*Q2*Q3*Q4 #mixed function space for (η, M,phi_dot,phiP_dot)

    u = Function(Q) #deflection η, moment M_xx
    η, M, phi_dot,phip_dot = split(u)
    n, m, p, q = TestFunctions(Q)

    ρ_i = 917 #density of ice (kg/m^3)
    g = 9.81 #gravitational constant (N/kg)
    E = 1e9 #Young's Modulus (Pa)
    μ = 1/3 #Poisson's Ratio
    D = H**3*E/(12*(1 - μ**2))
    Ht = 1 # tidal height (meters)
    sigma = -ρ_i*g*H/4  #membrane stress

    phi_old = Function(Q3) # Viscous curvature in the present previous time step
    phip_old = Function(Q4) # Plastic curvature in the present previous time step

    day = 86400
    year = 365 * day
    dt = 0.1*day #timestep
    viscosity = 1e14 #viscosity (Pa s)
    C = viscosity*H**3 / 3  # "Flexural viscosity"

    bc_η = DirichletBC(Q.sub(0), Constant(0), (1, 2)) #η = 0 on dS
    bc_M = DirichletBC(Q.sub(1), Constant(0), (1, 2)) #M = 0 on dS

    nt = 1000 #number of timesteps
    outfile = VTKFile("vep.pvd")

    for i in range(nt):
        F =  (grad(m)[0]*grad(M)[0])*dx # divergence of moment in force balance
        F += (m*ρ_i*g*η)*dx # force balance
        F += grad(η)[0] * grad(m*H*sigma)[0]*dx # membrane loading in force balance

        F += -(n*M)*dx # Moment in the consitutive equation
        F += (grad(D*n)[0]*grad(η)[0])*dx # curvature in the constitutive equation
        F += n*D*(phi_old  +  phi_dot*dt)*dx # Viscous curvature in the constitutive equation
        F += n*D*(phip_old + phip_dot*dt)*dx # Plastic curvature in the constitutive equation

        F += p*phi_dot*dx + p*M/C*dx # viscous curvature evolution

        '''
        Plastic curvature evolution. This is the important part!
        '''
        # plastic_rate = 1e13 #viscosity (Pa s)
        # Cp = plastic_rate*H**3 / 3  # "Flexural viscosity"
        Cp = C
        M_crit = 1e60 # critical moment
        Mstar = Function(Q2).project(conditional(M > M_crit, M, 0)\
                                 + conditional(M < -M_crit, M, 0) )
        F += q*phip_dot*dx + q*Mstar/Cp*dx # plastic curvature evolution

        # L =   phi_old*n*dx # from consitutive equation
        # L +=  phip_old*n*dx # from consitutive equation
        F += -(ρ_i*g*Ht*m)*dx # from force balance
        

        # u = Function(Q) # q = (η, M, phi_dot, phip_dot)
        # F = a - L
        solve(F == 0, u, bcs = [bc_η, bc_M])
        
        Δphi = Function(Q3)
        Δphi.project(u[2]*dt)
        phi_old.project(phi_old + Δphi)

        Δphip = Function(Q4)
        Δphip.project(u[3]*dt)
        phip_old.project(phip_old + Δphip)

        deflections, moment, phi, phip = u.subfunctions
        deflections.rename("Deflections")
        moment.rename("Moment")
        phi_old.rename("Viscous Curvature")
        phip_old.rename("Plastic Curvature")
        outfile.write (deflections, moment,phi_old,phip_old, time=i*dt)
        print(f'Finished timestep {i+1} of {nt} at {(perf_counter() - t0):.2f} seconds')


if __name__ == '__main__':
    main()