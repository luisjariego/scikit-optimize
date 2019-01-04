#!/usr/bin/python

import sys

import numpy as np
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print ("Por favor, introduce el fichero de errores medios para hacer el plot.")
    print ("\tpython3 performance.py <fichero_medias> <<opt:fichero_stds>> <<opt:n_runs>>")
    quit()

error_file = sys.argv[1]
try:
    stds_file = sys.argv[2]
    stds = 1
except:
    stds = 0

try:
    n_runs = int(np.round(sys.argv[3]))
except:
    n_runs = 1
    
plot_conv = True

#Preparamos el plot
if plot_conv:
    fig, ax = plt.subplots(2,1)
    fig.suptitle("Comparing convergence")
    #ax[0].set_prop_cycle(color=plt.cm.gist_heat(np.linspace(0,0.9, 7)))
    #ax[1].set_prop_cycle(color=plt.cm.gist_heat(np.linspace(0,0.9, 7)))
    ax[0].set_title("Error in each step")
    ax[1].set_title("Convergence")

#Leemos el fichero de errores
f = open(error_file, 'r')
if stds:
    g = open(stds_file, 'r')
n_acq_functions = 0

line = f.readline()
while line != "":
    if stds:
        ls = g.readline()
        ls = ls.split(':')
        error_stds = [float(y) for y in ls[1].split(',')[:-1]]
        error_stds = np.array(error_stds)
    
    n_acq_functions += 1
    line = line.split(':')
    acq_func = line[0]
    errors = [float(y) for y in line[1].split(',')[:-1]]
    errors = np.array(errors)
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
        st_error = error_stds/np.sqrt(n_runs) #TODO plot con standard error o desviacion tipica?
        if stds:
            #Plot errores
            y1 = np.maximum(0, errors - st_error)
            y2 = errors + st_error
            ax[0].fill_between(x, y1=y1, y2=y2, alpha=0.2)
            ax[0].errorbar(x, errors, yerr = st_error , fmt = '-', label=acq_func)

            #Plot convergencia
            y1 = np.maximum(0, conv - st_error)
            y2 = conv + st_error
            ax[1].fill_between(x, y1=y1, y2=y2, alpha=0.2)
            ax[1].errorbar(x, conv, yerr = st_error , fmt = '-', label=acq_func)
        else:
            ax[0].plot(x, errors, label=acq_func)
            ax[1].plot(x, conv, label=acq_func)
        ax[1].axhline(0.0, linestyle='dashed', color='red', linewidth=1)
    
    line = f.readline()

if plot_conv:
    plt.legend()
    plt.show()

