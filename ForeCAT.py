import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
# give it the path wherever the ForeCAT files are stored
sys.path.append(os.path.abspath('/Users/ckay/OSPREI/ForeCAT'))
import ForeCAT_functions as FC
import CME_class as CC
import ForceFields as FF

#---------------------------------------------------------------------------------------|
# Read in the filename from the command line and load the parameters -(No Touching!)----|
#---------------------------------------------------------------------------------------|
input_values = FC.readinputfile()
CME_params, ipos, rmax, tprint, Ntor, Npol = FC.getInps(input_values)

#---------------------------------------------------------------------------------------|
# Simulation set up --(No Touching!)----------------------------------------------------|
#---------------------------------------------------------------------------------------|

# Initialize GPU arrays and load magnetic field data
if FC.useGPU:
    FF.init_GPU(FC.CR, Ntor, Npol)
else:
    FF.init_CPU(FC.CR, Ntor, Npol)


# Initialize CME
CME = CC.CME(ipos, CME_params, Ntor, Npol, FC.user_vr, FC.user_exp, FC.user_mass, FC.user_Bexp, FC.rsun)

# Initialize magnetic field and distance pickles
FC.initdefpickle(FC.CR)

# Open save files and initialize any plotting
if FC.saveData: FC.openfile(CME)
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|


#---------------------------------------------------------------------------------------|
# Simulation Main Loop --(No Touching!)-------------------------------------------------|
#---------------------------------------------------------------------------------------|

dtprint = CME.dt # initiate print counter
# Run until nose hits rmax
while CME.points[CC.idcent][1,0] <= rmax:
    CME.update_CME(FC.user_vr, FC.user_exp, FC.user_mass, FC.user_Bexp)
    dtprint += CME.dt
    if (dtprint > tprint):          # if not printing every time step, this 
        FC.printstep(CME)
        dtprint = CME.dt	      # the counter
FC.printstep(CME) # print final position
if FC.saveData: FC.outfile.close()    # close the output files
if FC.useGPU: FF.clear_GPU() # reset GPU if needed
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|


