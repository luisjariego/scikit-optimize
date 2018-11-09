
import numpy as np
from skopt import gp_minimize
from skopt.acquisition import gaussian_ei, gaussian_pi, gaussian_lcb, gaussian_acquisition_1D
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import json

#Abrimos el fichero de configuración
with open('configuration.json') as file:
    conf = json.load(file)

noise_level = conf['noise_level']
seed = conf['seed']
n_calls = conf['number_of_calls']
#La funcion a optimizar debe llamarse 'f', o de lo contrario cambiar la llamada a la funcion
obj_func = conf['objective_function']
#Cargar los limites para cada dimension
bounds = []
for dim in conf['bounds']:
    bounds.append(conf['bounds'][dim])

n_grid = conf['grid']
n_random_starts = conf['n_random_starts']
#Opciones para plotear/guardar errores
plot_function = conf['plot_function']
plot_conv = conf['plot_convergence']
verbose = conf['verbose']
errors_file = conf['error_file']
save_errors = conf['save_errors']
n_runs = conf['n_runs']
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
exec(obj_func) #TODO definir asi o de otra forma?
#def f(x, noise_level=noise_level):
#    return (np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level)

#Hallo el minimo real de la funcion
grid = np.linspace(bounds[0][0], bounds[0][1], n_grid).reshape(-1, 1)
fx = [f(x_i, noise_level=0.0) for x_i in grid] #Sin ruido para hallar el minimo real
minimum = min(fx)
min_x = grid[np.argmin(fx)]

if verbose>0:
    print ("True minimum is:")
    print ("x^* = %.4f, f(x^*)=%.4f" % (min_x, minimum))

#To save errors for each acquisition function
if save_errors:
    dump_errors = open(errors_file, 'w')
    #line = "ERROR MADE IN EACH EVALUATION\n"
    #dump_errors.write(line)
    #line = "Number of evaluations: {}\n".format(n_calls)
    #dump_errors.write(line)
    #line = "Errors for each acquisition function:\n"
    #dump_errors.write(line)

#Para cada funcion de adquisicion
for k, acq_func in enumerate(acquisition_functions):
    if verbose>0:
        print ("ACQUISITION FUNCTION USED: '{}'".format(acq_func))
    random_state = seed
    errors = np.empty((n_runs, n_calls))
    #Ejecutamos el optimizador
    for i in range(n_runs):
        print ("'{}': Iteration ".format(acq_func), "%i" % (i+1), end="\r")
        res = gp_minimize(f, bounds, n_calls=n_calls, n_random_starts=n_random_starts,
                            acq_func=acq_func, random_state=random_state, n_points=n_points,
                            kappa=kappa, xi=xi, acq_func_kwargs=acq_func_kwargs)

        #El minimo aproximado es
        if verbose>0:
            print ("\nMinimum found:")
            print ("x^* = %.4f, f(x^*)=%.4f" % (res.x[0], res.fun))

        #Hallo el error en cada caso
        errors[i] = res.func_vals - minimum

        #Preparamos para la siguiente ejecucion
        #random_state += 1 #TODO ir cambiando semilla?
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

    #Plot de la convergencia
    if plot_conv:
        fig, ax_list = plt.subplots(2,1)
        fig.suptitle("Convergence for '{}'".format(acq_func))

        ax_list[0].set_title("Error in each step")
        x = range(1, len(error_means)+1)
        ax_list[0].bar(x, error_means)

        plot_convergence(res, ax=ax_list[1], true_minimum=minimum)

    # Plot f(x) + contours
    if plot_function:
        plt.figure()
        plt.title("Objective function and minimum found for '{}'".format(acq_func))
        plt.plot(grid, fx, "r--", label="True (unknown)")
        plt.fill(np.concatenate([grid, grid[::-1]]),
                 np.concatenate(([fx_i - 1.9600 * noise_level for fx_i in fx],
                                 [fx_i + 1.9600 * noise_level for fx_i in fx[::-1]])),
                 alpha=.2, fc="r", ec="None")
        plt.plot(min_x, minimum, "go", label="True minimum")
        plt.plot(res.x[0], res.fun, "bo", label="Minimum found")
        plt.legend()
        plt.grid()
        plt.show()

if save_errors:
    dump_errors.close()
