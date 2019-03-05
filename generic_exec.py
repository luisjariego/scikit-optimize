"""
    Script para optimizar funciones de varias dimensiones.
"""

import numpy as np
from skopt import gp_minimize
from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb, gaussian_acquisition_1D
from skopt.plots import plot_convergence
from skopt.benchmarks import hart6, branin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import json
from time import time

import sys

if len(sys.argv) < 2:
    print ("Por favor, introduce el fichero de errores medios para hacer el plot.")
    print ("\tpython3 generic_exec.py <fichero_de_configuracion>")
    quit()

config_file = sys.argv[1]

#Abrimos el fichero de configuracion
with open(config_file) as file:
    conf = json.load(file)

noise_level = conf['noise_level']
if len(sys.argv) == 2:
    seed = conf['seed']
else:
    seed = int(sys.argv[2])

n_calls = conf['number_of_calls']
#La funcion a optimizar debe llamarse 'f', o de lo contrario cambiar la llamada a la funcion
objf_name = conf['objf_name']
obj_func = conf['objective_function']    
#Cargar los limites para cada dimension
bounds = []
for dim in conf['bounds']:
    bounds.append(conf['bounds'][dim])
bounds = np.array(bounds).astype(float).tolist()

n_grid = conf['grid']
n_random_starts = conf['n_random_starts']
#Opciones para plotear/guardar errores
plot_function = conf['plot_function']
plot_acquisition = conf['plot_acquisition']
plot_conv = conf['plot_convergence']
verbose = conf['verbose']
errors_file = conf['error_file']
save_errors = conf['save_errors']
stds_file = conf['stds_file']
save_stds = conf['save_stds']
n_runs = conf['n_runs']
max_time = conf['max_time'] #En segundos TODO usar?

#Funciones de adquisicion
acquisition_functions = conf['acquisition_functions']
n_acq_func = len(acquisition_functions)
#Parametros opcionales
n_points = conf['n_points_sample_acq']
acquisition_noise = conf['acquisition_noise']
#Parametros para controlar exploracion vs explotacion en las funciones de adquisicion
kappa = conf['kappa'] #Para la funcion de adquisicion LCB
xi = conf['xi'] #Para las funciones de adquisicion EI o PI

#Parametros adicionales para la funcion de adquisicion
acq_func_kwargs = dict()
acq_func_kwargs['acq_noise']=acquisition_noise
for w in conf['weights']:
    acq_func_kwargs[w] = conf['weights'][w]

#Funcion objetivo
exec(obj_func) #TODO definir asi?
#def f(x, noise_level=noise_level):
#    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level)

#Hallo el minimo real de la funcion
oned = len(bounds) < 2
grid = np.empty((n_grid, len(bounds)))
for i, b in enumerate(bounds):
    grid[:,i] = np.linspace(b[0], b[1], n_grid)
    
fx = [f(x_i, noise_level=0.0) for x_i in grid] #Sin ruido para hallar el minimo real

#Minimo dado o grid
try:
    minimum = conf['global_minimum']
    min_x = conf['true_minimum']
except KeyError:
    minimum = min(fx)
    min_x = grid[np.argmin(fx)]

if verbose>0:
    print ("The global minimum of '{}' is \nf(x^*)={}, found in".format(objf_name, minimum))
    try:
        for w in conf['true_minimum']:
            tm = conf['true_minimum'][w]
            print ("\tx^*={}".format(tm))#To save errors for each acquisition function
    except TypeError:
        print ("\tx^*={}".format(min_x))#To save errors for each acquisition function
    print("\n")

if save_errors:
    dump_errors = open(errors_file, 'w')
    #line = "ERROR MADE IN EACH EVALUATION\n"
    #dump_errors.write(line)
    #line = "Number of evaluations: {}\n".format(n_calls)
    #dump_errors.write(line)
    #line = "Errors for each acquisition function:\n"
    #dump_errors.write(line)

if save_stds:
    dump_stds = open(stds_file, 'w')

#Para cada funcion de adquisicion
for k, acq_func in enumerate(acquisition_functions):
    if verbose>0:
        print ("ACQUISITION FUNCTION USED: '{}'".format(acq_func))
    random_state = seed
    np.random.seed(seed) #TODO misma semilla para numpy
    errors = np.empty((n_runs, n_calls))
    #Ejecutamos el optimizador
    for i in range(n_runs):
        print ("'{}': Iteration ".format(acq_func), "%i" % (i+1), end="\r")
        res = gp_minimize(f, bounds, n_calls=n_calls, n_random_starts=n_random_starts,
                            acq_func=acq_func, random_state=random_state, n_points=n_points,
                            kappa=kappa, xi=xi, verbose=(verbose > 1), acq_func_kwargs=acq_func_kwargs)

        #Hallo el error en cada caso
        errors[i] = res.func_vals - minimum #TODO valor absoluto por si hay ruido?

        #Preparamos para la siguiente ejecucion
        #random_state += 1 #TODO ir cambiando semilla?
        
        #El minimo aproximado es
        if verbose>0:
            print ("\nMinimum found:")
            print ("x^* = %.4f, f(x^*)=%.4f" % (res.x[0], res.fun)) if oned else print ("x^*={}, f(x^*)={}".format(res.x, res.fun))

    print ("\n")

    #Media y desviacion tipica de los errores
    error_means = [np.mean(errors[:,j]) for j in range(n_calls)]
    error_std = [np.std(errors[:,j]) for j in range(n_calls)]

    #Guardar los errores
    if save_errors:
        line = "{}:".format(acq_func)
        dump_errors.write(line)
        for e in error_means:
            line = "{},".format(e)
            dump_errors.write(line)
        dump_errors.write("\n")

    #Guardar las desviaciones tipicas de los errores
    if save_stds:
        line = "{}:".format(acq_func)
        dump_stds.write(line)
        for e in error_std:
            line = "{},".format(e)
            dump_stds.write(line)
        dump_stds.write("\n")

    #Plot de la convergencia (medias)
    if plot_conv:
        fig, ax_list = plt.subplots(2,1)
        fig.suptitle("Convergence for '{}'".format(acq_func))

        ax_list[0].set_title("Error in each step")
        x = range(1, len(error_means)+1)
        ax_list[0].bar(x, error_means)

        plot_convergence(res, ax=ax_list[1], true_minimum=minimum)

    #Plot de la funcion + observaciones (ultima ejecucion)
    if plot_function and len(bounds)<2:
        # Plot f(x) + contours
        plt.figure()
        plt.title("Objective function and minimum found for '{}'".format(acq_func))
        plt.plot(grid, fx, "r--", label="True (unknown)")
        plt.fill(np.concatenate([grid, grid[::-1]]),
                 np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
                                 [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
                 alpha=.2, fc="r", ec="None")

        # Plot GP(x) + contours
        x_gp = res.space.transform(grid.tolist())
        gp = res.models[-1]
        y_pred, sigma = gp.predict(x_gp, return_std=True)

        plt.plot(grid, y_pred, "g--", label=r"$\mu_{GP}(x)$")
        plt.fill(np.concatenate([grid, grid[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma, 
                                 (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.2, fc="g", ec="None")

        # Plot sampled points
        plt.plot(res.x_iters, res.func_vals, "r.", markersize=15, label="Observations")
        
        plt.plot(min_x, minimum, "go", label="True minimum")
        plt.plot(res.x[0], res.fun, "bo", label="Minimum found")
        plt.title(objf_name)
        #plt.title(r"$x^* = %.4f, f(x^*) = %.4f$" % (res.x[0], res.fun))
        plt.legend(loc="best", prop={'size': 8}, numpoints=1)
        
        plt.grid()

    elif plot_function and len(bounds)<3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax = Axes3D(fig)

        # Make data.
        X, Y = np.meshgrid(grid[:,0], grid[:,1])
        R = X, Y
        Z = f(R, noise_level=0.0)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.scatter(*min_x, zs=[minimum], c="green", label="True minimum", depthshade=True)
        ax.scatter(*res.x, zs=[res.fun], c="purple", label="Minimum found")
        plt.title(objf_name)
        plt.legend(loc="best", prop={'size': 8}, numpoints=1)
        
        plt.grid()

    elif plot_function:
        print ("It is not possible to plot a function with those dimensions.")
    
    #Plot de la funcion de adquisicion (ultima ejecucion)
    if plot_acquisition and len(bounds)<2:
        fig, ax_list = plt.subplots(5,2)
        fig.suptitle("Function and acquisition for '{}'".format(acq_func))
        
        x_gp = res.space.transform(grid.tolist())
        for n_iter in range(5):
            gp = res.models[n_iter]
            curr_x_iters = res.x_iters[:5+n_iter]
            curr_func_vals = res.func_vals[:5+n_iter]

            # Plot true function.
            plt.subplot(5, 2, 2*n_iter+1)
            plt.plot(grid, fx, "r--", label="True (unknown)")
            plt.fill(np.concatenate([grid, grid[::-1]]),
                 np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
                                 [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
                 alpha=.2, fc="r", ec="None")

            # Plot GP(x) + contours
            y_pred, sigma = gp.predict(x_gp, return_std=True)
            plt.plot(grid, y_pred, "g--", label=r"$\mu_{GP}(x)$")
            plt.fill(np.concatenate([grid, grid[::-1]]),
                     np.concatenate([y_pred - 1.9600 * sigma, 
                                     (y_pred + 1.9600 * sigma)[::-1]]),
                     alpha=.2, fc="g", ec="None")

            # Plot sampled points
            plt.plot(curr_x_iters, curr_func_vals,
                     "r.", markersize=8, label="Observations")

            # Adjust plot layout
            plt.grid()

            if n_iter == 0:
                plt.legend(loc="best", prop={'size': 6}, numpoints=1)

            if n_iter != 4:
                plt.tick_params(axis='x', which='both', bottom='off', 
                                top='off', labelbottom='off') 

            # Plot EI(x)
            plt.subplot(5, 2, 2*n_iter+2)
            
            acq1 = gaussian_ei(x_gp, gp, y_opt=np.min(curr_func_vals))
            plt.plot(grid, acq1, "b", label="EI(x)")
            plt.fill_between(grid.ravel(), -2.0, acq1.ravel(), alpha=0.3, color='blue')
            acq2 = gaussian_pi(x_gp, gp, y_opt=np.min(curr_func_vals))
            plt.plot(grid, acq2, "r", label="PI(x)")
            plt.fill_between(grid.ravel(), -2.0, acq2.ravel(), alpha=0.3, color='red')
            acq3 = -gaussian_lcb(x_gp, gp, kappa=kappa)
            plt.plot(grid, acq3, "g", label="LCB(x)")
            plt.fill_between(grid.ravel(), -2.0, acq3.ravel(), alpha=0.3, color='green')
            
            if acq_func == "weighted":
                n_candidates=3
                lcb_weight = acq_func_kwargs.get("lcb_w", 1./n_candidates)
                ei_weight = acq_func_kwargs.get("ei_w", 1./n_candidates)
                pi_weight = acq_func_kwargs.get("pi_w", 1./n_candidates)

                weights_sum =lcb_weight + ei_weight + pi_weight
                
                if weights_sum != 1: #The sum must be 1
                    lcb_weight = 1. * lcb_weight / weights_sum
                    ei_weight = 1. * ei_weight / weights_sum
                    pi_weight = 1. * pi_weight / weights_sum
                acq = ei_weight * acq1 + pi_weight * acq2 + lcb_weight * acq3
                plt.plot(grid, acq, "y", label="weighted(x)")
                plt.fill_between(grid.ravel(), -2.0, acq.ravel(), alpha=0.3, color='yellow')
    
            next_x = res.x_iters[n_random_starts+n_iter]
            plt.axvline(next_x, linestyle="-.", color="black")
            #next_acq = gaussian_ei(res.space.transform([next_x]), gp, y_opt=np.min(curr_func_vals))
            #plt.plot(next_x, next_acq, "bo", markersize=6, label="Next query point")

            # Adjust plot layout
            plt.ylim(0, max(max(acq1), max(acq2), max(acq3))+0.1)
            plt.grid()

            if n_iter == 0:
                plt.legend(loc="best", prop={'size': 6}, numpoints=1)

            if n_iter != 4:
                plt.tick_params(axis='x', which='both', bottom='off', 
                                top='off', labelbottom='off') 
    elif plot_acquisition:
        print ("It is not possible to plot the acquisition of a function with those dimensions.")
    
    plt.show()

if save_errors:
    dump_errors.close()

