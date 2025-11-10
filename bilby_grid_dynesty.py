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
import bilby
from bilby.core.utils.random import rng, seed
from bilby.core.prior import PriorDict, Uniform, Constraint, LogUniform
from bilby.bilby_mcmc.proposals import ProposalCycle, AdaptiveGaussianProposal, PriorProposal
import sys
import argparse
import index_calc
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
# parser.add_argument('triggers', type=str, help='Path to the first text file')
parser.add_argument('eccen', type=float, help='eccentricity')
parser.add_argument('semi', type=float, help='semi_major axis')
parser.add_argument('inclin', type=float, help='Inclination')
parser.add_argument('time_inn', type=float)
parser.add_argument('output', type=str)

# Parse the arguments
args = parser.parse_args()

data = np.loadtxt(args.output+"/"+args.output +"_triggers_refined.dat")
T_burst=data[:,0]-data[0,0]
T_burst-=T_burst[0]
e_true = args.eccen 
a_out_true= args.semi 
inclin_true= args.inclin 
time_in_true= args.time_inn
output= args.output

mean_i,std_i,index= index_calc.calc_distribution(T_burst)
print("index\n",index)

Iter_arr_inner=index

sigma=150

def model(Iter_arr_inner,M_phase,a_out,w_out,e,time_in_s,i_in,theta_s,phi_s,M_out_small):  
    """The cost function which should be minimized for
       the correct inference of the orbital parameters
    """

    M_out=M_out_small*1e7         #SBH mass
    time_in=time_in_s*1e6
    T_lisa=31556952  # in seconds 
    a = a_out*AU*1000

    t0=Iter_arr_inner*(time_in)
    M = t0 * np.sqrt((G * M_out * SOLAR_MASS / (np.power(a,3))))  + M_phase
    M_range=(M + np.pi) % (2 * np.pi) - np.pi
    E = kepler_series_solution(M_range,e)  
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
    return T_cal

# injection_parameters = dict(M_phase=M_0,
#                             a_out=a_out_initial/1000,
#                             w_out=np.pi/4,
#                             e=e_initial,
#                             time_in_s=time_in_initial/1e6,
#                             i_in=i_in_initial,
#                             theta_s=theta_initial,
#                             phi_s=phi_initial)


# prior defination
label = "nested_sampling_for_Roemer_Delay"
outdir = str(args.output)+"/outdir_dynesty"
Iter_arr_inner = index
likelihood = bilby.likelihood.GaussianLikelihood(Iter_arr_inner, T_burst, model, sigma)
priors = dict()
priors['M_phase'] = bilby.prior.Uniform(minimum=0, maximum=2*np.pi, name='M_phase', latex_label='M_phase')
priors['a_out'] = bilby.prior.Uniform(minimum=0.05, maximum=0.65, name='a_out', latex_label='a_out')
priors['w_out'] = bilby.prior.Uniform(minimum=0, maximum=2*np.pi, name='w_out', latex_label='w_out')
priors['e'] = bilby.prior.Uniform(minimum=0, maximum=0.7, name='e', latex_label='e')
priors['time_in_s'] = bilby.prior.Uniform(minimum=(mean_i - 10*std_i)/1e6, maximum=(mean_i + 10*std_i)/1e6, name='time_in_s', latex_label='time_in_s')
priors['i_in'] = bilby.prior.Uniform(minimum=0, maximum=np.pi/2, name='i_in', latex_label='i_in')
priors['theta_s'] = bilby.prior.Uniform(minimum=0, maximum=np.pi/2, name='theta_s', latex_label='theta_s')
priors['phi_s'] = bilby.prior.Uniform(minimum=0, maximum=2*np.pi, name='phi_s', latex_label='phi_s')
priors['M_out_small'] = bilby.prior.Uniform(minimum=0.395, maximum=0.405, name='M_out_small', latex_label='M_out_small')


# And run sampler
result_dynesty = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=2500,
    npool=8,
    sample="rwalk",
    #injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

fig = result_dynesty.plot_corner()
fig.savefig(f"{outdir}/bilby_corner_plot.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# --- Extract and rescale samples ---
samples = result_dynesty.posterior.to_numpy()

# Example rescaling (adjust indices carefully)
samples[:, 1] *= 1000              # Semi-major axis to AU*1000
samples[:, 4] *= 1e6               # Time period to seconds
samples[:, 8] = 10**7 * samples[:, 8]  # Convert log(M) to M

# --- Truth values (use None if unknown) ---
truths = [None, a_out_true, None, e_true, time_in_true, None, None, None, 4e6]

# --- Labels (must match ndim) ---
labels = [
    "Initial Mean Anomaly",
    "Semi-major axis (AU)",
    "Pericenter Angle",
    "Eccentricity",
    "Time Period (s)",
    "Inclination",
    "Lisa's Polar Angle",
    "Lisa's Azimuthal Angle",
    "Mass of SMBH (M☉)"
]

# --- Custom Corner Plot ---
figure = corner.corner(
    samples[:,0:9],
    labels=labels,
    show_titles=True,
    title_kwargs={"fontsize": 12},
)

corner.overplot_lines(figure, truths, color="C1")

plt.savefig(f"{outdir}/{label}_dynesty.png", dpi=300, bbox_inches="tight")
plt.close(figure)
