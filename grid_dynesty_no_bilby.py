from scipy import optimize
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy import optimize
from scipy.special import jv
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import corner
import multiprocessing
import sys
import argparse
import index_calc
import dynesty
from dynesty import utils as dyfunc
from dynesty.utils import resample_equal
from dynesty import plotting as dyplot
import pickle


#CONSTANTS and FUNCTIONS

AU = 1.496e+13
SOLAR_MASS = 1.989e+33
G = 6.67e-8
C_CGS = 3e+10
T_lisa=31556952

def func_to_sol(E, M,e):
        return E - e * np.sin(E) - M

def der_to_func(E, M,e):
        return 1 - e  * np.cos(E)

def double_der_to_func(E, M,e):
        return e * np.sin(E)

def solve_kepler(M,e):
        E = np.pi  
        E = optimize.newton(func_to_sol, E, args=(M,e), fprime=der_to_func,fprime2=double_der_to_func, disp=False)
        return E

def kepler_series_solution(M, e, terms=30):
    E = M
    for n in range(1, terms + 1):
        E += (2 / n) * jv(n, n * e) * np.sin(n * M)
    return E


## Initialize the argument parser
parser = argparse.ArgumentParser(description="To include time series data and indexes")

# Add arguments for 2 input files and 1 name foe output file
parser.add_argument('triggers', type=str, help='Path to the first text file')
parser.add_argument('eccen', type=float, help='eccentricity')
parser.add_argument('semi', type=float, help='semi_major axis')
parser.add_argument('inclin', type=float, help='Inclination')
parser.add_argument('time_inn', type=float)
parser.add_argument('outputs', type=str)

# Parse the arguments
args = parser.parse_args()

data = np.loadtxt(args.triggers)
T_burst=data[:,0]-data[0,0]

e_true = args.eccen 
a_out_true= args.semi 
inclin_true= args.inclin 
time_in_true= args.time_inn
output= args.outputs

mean_i,std_i,index= index_calc.calc_distribution(T_burst)
time_lower= (mean_i-10*std_i)/1e6
time_higher= (mean_i+10*std_i)/1e6
print("index\n",index)

Iter_arr_inner=index

#def model(Iter_arr_inner,M_phase,a_out,w_out,e,time_in_s,i_in,theta_s,phi_s):  
def loglike(theta):  
    """The cost function which should be minimized for
       the correct inference of the orbital parameters
    """
    M_phase = theta[0]
    a_out = theta[1]
    w_out = theta[2]
    e = theta[3]
    time_in_s = theta[4]
    i_in = theta[5]
    theta_s = theta[6]
    phi_s = theta[7]
    M_out = theta[8]

    time_in=time_in_s*1e6
    T_lisa=31556952  # in seconds 
    a = a_out*AU*1000
    M_out = np.pow(10,theta[8])

    t0=Iter_arr_inner*(time_in)
    M = t0 * np.sqrt((G * M_out * SOLAR_MASS / (np.power(a,3))))  + M_phase
    M_range=(M + np.pi) % (2 * np.pi) - np.pi
    E = solve_kepler(M_range,e)  
    beta = e / (1 + np.sqrt(1 - e ** 2))
    F=E + 2 * np.arctan(beta * np.sin(E) / (1 - beta * np.cos(E)))
    R=(a * (1 - e** 2) / (1 + e * np.cos(F))) / C_CGS
    Doppler = R * np.sin(F + w_out) * np.sin(i_in)
    t_calculated = Doppler 
    t_calculated = (time_in*Iter_arr_inner) - t_calculated
  
# Roemer Delay due to the LISA's orbit

    RD_LISA = 499.00478 * np.sin(theta_s) * np.cos((2 * np.pi * (Iter_arr_inner*time_in/T_lisa))-phi_s)
    T_cal = t_calculated - RD_LISA

    T_cal = T_cal - T_cal[0]
    sigma=150
    chi2 = np.sum(((T_cal - T_burst) / sigma)**2)
    cost = -0.5 * (chi2) - 0.5 * len(Iter_arr_inner)* np.log(2 * np.pi * sigma**2)
    return cost

ndim = 9

def ptform(u):

    theta = np.zeros(9)
    theta[0]=u[0]*2*np.pi
    theta[1]=u[1]
    theta[2]=u[2]*np.pi
    theta[3]=u[3]*0.7
    theta[4]=time_lower + u[4]*(time_higher-time_lower)
    theta[5]=u[5]*np.pi/2
    theta[6]=u[6]*np.pi/2
    theta[7]=u[7]*2*np.pi
    theta[8]=6+u[8]
    return theta

# sampler = dynesty.NestedSampler(loglike, ptform, ndim)
# sampler.run_nested()
# sresults = sampler.results

# "Dynamic" nested sampling.

#periodic option 
dsampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim,nlive=1700,sample='rwalk')
dsampler.run_nested()
results = dsampler.results

with open(output + ".pkl", "wb") as f:
    pickle.dump(results, f)

# --- Rescale the samples ---
samples = results.samples.copy()
# Example rescaling - check these are the correct indices!
samples[:, 1] *= 1000        # Semi-major axis to AU*1000?
samples[:, 4] *= 1e6         # Time period to seconds
samples[:, 8] = 10**samples[:, 7]  # Convert log(M) to M

# --- Truth values (use None if unknown) ---
truths = [None, a_out_true, None, e_true, time_in_true, None, None, 4e6]

# --- Labels (must match ndim) ---
labels = [
    "Initial Mean Anomaly",
    "Semi-major axis (au)",
    "Pericenter Angle",
    "Eccentricity",
    "Time Period (s)",
    "Lisa's Polar Angle",
    "Lisa's Azimuthal Angle",
    "Mass of SMBH (M☉)"
]

# --- Plot traceplot with rescaled samples ---
tfig, taxes = dyplot.traceplot(
    results,
    labels=labels,
    trace_cmap='viridis',
    quantiles=None,
    show_titles=True,
    truths=truths,
    truth_color='black'
)

plt.savefig(output+".png", dpi=300, bbox_inches="tight")
plt.close()