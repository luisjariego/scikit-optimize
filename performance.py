
import numpy as np
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

error_file = 'ev_errors.txt'
plot_conv = True
#minimum = -0.9094

#Preparamos el plot
if plot_conv:
    fig, ax = plt.subplots(2,1)
    fig.suptitle("Comparing convergence")
    ax[0].set_title("Error in each step")
    ax[1].set_title("Convergence")

#Leemos el fichero de errores
f = open(error_file, 'r')
n_acq_functions = 0

line = f.readline()
while line != "":
    n_acq_functions += 1
    line = line.split(':')
    acq_func = line[0]
    errors = [float(y) for y in line[1].split(',')[:-1]]
    #Calculamos convergencia
    conv = np.empty(len(errors))
    conv[0] = errors[0]
    i=0
    for e in errors[1:]:
        if e<conv[i]:
            conv[i+1] = e
        else:
            conv[i+1] = conv[i]
        i+=1
    #conv = minimum + conv

    #Plot de la convergencia
    if plot_conv:
        x = range(1, len(errors)+1)
        #Errores bars
        #ax[0].bar(x, errors, alpha=0.5, label=acq_func)
        #Plot errores
        ax[0].plot(x, errors, label=acq_func)
        #Plot convergencia
        ax[1].plot(x, conv, label=acq_func)
        ax[1].axhline(0.0, linestyle='dashed', color='red', linewidth=1)
    
    line = f.readline()

if plot_conv:
    plt.legend()
    plt.show()

