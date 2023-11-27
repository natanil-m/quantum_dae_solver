

# !pip install perceval-quandela
# !pip install tqdm

import perceval as pcvl
import numpy as np
from math import comb
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm as tqdm
import sys
from perceval.simulators import Simulator
from perceval.backends import SLOSBackend

print(pcvl.__version__)

nphotons = int(sys.argv[1])


# Modeling parameters

range_min = 0  # minimum of the interval on which we wish to approximate our function
range_max = int(sys.argv[2])  # maximum of the interval on which we wish to approximate our function
n_grid = int(sys.argv[3])    # number of grid points of the discretized differential equation

print('photons: ',nphotons,' range_min = 0 , range_max =',range_max,' n_grid = ',n_grid)

X = np.linspace(range_min, range_max-range_min, n_grid)  # Optimisation grid



random_seed = np.random.randint(0, 10000)
np.random.seed(random_seed)
print("Random Seed:", random_seed)

# Boundary condition
coefii = 10

# del_0 = 0  #del
# omega_0 = 0     #w
# del_0_v = -1/coefii
# omega_0_v = 383.9911/coefii
del_0_v = float(sys.argv[4])
omega_0_v = float(sys.argv[5])


print('init delta:',del_0_v)
print('init omega:',omega_0_v)


# Differential equation parameters
ws = 376.9911

# ws = 0.376
# coefii = ws/4
K1 = 5/coefii
K2 = 10/coefii
K3 = 1.7/coefii
ws = ws/coefii

def F(u_zegond,u_prime, u, x):       # DE, works with numpy arrays
    # print(len(u_zegond))
    # print(len(u_prime))
    # print(len(u))
    delta = u
    delta_prime = u_prime
    omega = delta_prime+ws
    omega_prime = u_zegond
    return (delta_prime+ws-omega)+omega_prime-K1+K2*np.sin(coefii*delta)+K3*(-ws+omega)*coefii





# Parameters of the quantum machine learning procedure
N = nphotons              # Number of photons
m = nphotons              # Number of modes
eta = 5                   # weight granted to the initial condition
a = 200                   # Approximate boundaries of the interval that the image of the trial function can cover
fock_dim = comb(N + m - 1, N)
# lambda coefficients for all the possible outputs
lambda_random = 2 * a * np.random.rand(fock_dim) - a
# dx serves for the numerical differentiation of f
dx = (range_max-range_min) / (n_grid - 1)

# Input state with N photons and m modes
input_state = pcvl.BasicState([1]*N+[0]*(m-N))
print(input_state)



"Haar unitary parameters"
# number of parameters used for the two universal interferometers (2*m**2 per interferometer)
parameters = np.random.normal(size=4*m**2)

px = pcvl.P("px")
c = pcvl.Unitary(pcvl.Matrix.random_unitary(m, parameters[:2 * m ** 2]), name="W1")\
     // (0, pcvl.PS(px))\
     // pcvl.Unitary(pcvl.Matrix.random_unitary(m, parameters[2 * m ** 2:]), name="W2")


backend = pcvl.BackendFactory().get_backend("SLOS")
backend.set_circuit(pcvl.Unitary(pcvl.Matrix.random_unitary(m)))
backend.preprocess([input_state])

pcvl.pdisplay(c)

def computation(params):
    global current_loss
    global computation_count
    "compute the loss function of a given differential equation in order for it to be optimized"
    computation_count += 1

    loss_i = 0

    delta_0 = None  # boundary condition
    omega_0 = None  # boundary condition

    coefs = lambda_random  # coefficients of the M observable
    # initial condition with the two universal interferometers and the phase shift in the middle
    U_1 = pcvl.Matrix.random_unitary(m, params[:2 * m ** 2])
    U_2 = pcvl.Matrix.random_unitary(m, params[2 * m ** 2:])

    px = pcvl.P("x")
    c = pcvl.Unitary(U_2) // (0, pcvl.PS(px)) // pcvl.Unitary(U_1)

    px.set_value(np.pi * range_min)
    # U = c.compute_unitary(use_symbolic=False)
    # backend.U = U
    backend.set_circuit(c)
    delta_0 = np.sum(np.multiply(backend.all_prob(input_state), coefs))




    # loss=0


    # Y[0] is before the domain we are interested in (used for differentiation), x_0 is at Y[1]
    Y = np.zeros(n_grid + 2)

    # x_0 is at the beginning of the domain, already calculated
    Y[1] = delta_0

    px.set_value(np.pi * (range_min - dx)/range_max)
    # backend.U = c.compute_unitary(use_symbolic=False)
    backend.set_circuit(c)
    Y[0] = np.sum(np.multiply(backend.all_prob(input_state), coefs))


    for i in range(1, n_grid):
        x = X[i]
        px.set_value(np.pi * x/range_max)
        # backend.U = c.compute_unitary(use_symbolic=False)
        backend.set_circuit(c)
        Y[i + 1] = np.sum(np.multiply(backend.all_prob(input_state), coefs))

    px.set_value(np.pi * (range_max + dx)/range_max)
    # backend.U = c.compute_unitary(use_symbolic=False)
    backend.set_circuit(c)
    Y[n_grid + 1] = np.sum(np.multiply(backend.all_prob(input_state), coefs))


    # Differentiation
    Y_prime = (Y[2:] - Y[:-2])/(2*dx)
    omega_0 = Y_prime[0]+ws

    Y_zegond = (Y_prime[2:] - Y_prime[:-2])/(2*dx)

    loss_i += np.sum((F(Y_zegond,Y_prime[1:-1], Y[2:-2], X[1:-1]))**2)

    # boundary condition given a weight eta
    loss_b = eta * (delta_0 - del_0_v) ** 2 * len(X)
    loss_b += eta * (omega_0 - omega_0_v) ** 2 * len(X)



    current_loss = (loss_i+loss_b) / len(X)
    return current_loss


def callbackF(parameters):
    """callback function called by scipy.optimize.minimize allowing to monitor progress"""
    global current_loss
    global computation_count
    global loss_evolution
    global start_time
    global best_param

    now = time.time()
    # pbar.set_description("M= %d Loss: %0.5f #computations: %d elapsed: %0.5f" %
    #                      (m, current_loss, computation_count, now-start_time))
    # pbar.update(1)
    loss_evolution.append((current_loss, now-start_time))
    computation_count = 0
    start_time = now
    best_param = parameters



computation_count = 0
current_loss = 0
start_time = time.time()
loss_evolution = []

pbar = tqdm.tqdm()
res = minimize(computation, parameters, callback=callbackF, method='BFGS', options={'gtol': 1E-2})
print('final_loss = ',current_loss)

# print("Unitary parameters", res.x)
print("Unitary parameters", best_param)
np.save('/u/msoltaninia/DQC_Photonic/store_results/'+'seed='+str(random_seed)+' n_photon='+str(nphotons)+' range='+str(range_min)+'-'+str(range_max)+' '+'ngrid = '+str(n_grid)+'.npy', best_param)

