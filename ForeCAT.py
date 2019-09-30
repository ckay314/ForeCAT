import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
# give it the path wherever the ForeCAT files are stored
sys.path.append(os.path.abspath('/home/cdkay/ForeCAT'))
import ForeCAT_functions as FC
import CME_class as CC
import ForceFields as FF

#---------------------------------------------------------------------------------------|
# Read in the filename from the command line and load the parameters -(No Touching!)----|
#---------------------------------------------------------------------------------------|

# Get CME parameters
fname = str(sys.argv[1])

# Determine the prefix used for saving files
fprefix = fname[:-4]

# Get CME parameters
CME_params, ipos, rmax, tprint, Ntor, Npol = FC.read_in_params(fname)
print 'Use GPU', FC.useGPU


#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|








#---------------------------------------------------------------------------------------|
# User Models (Only change things in this section)--------------------------------------|
#---------------------------------------------------------------------------------------|

# Stellar/Solar Parameters -------------------------------------------------------------|
rsun    = 7e10   # stellar radius [cm]
rotrate = 2.8e-6 # rotation rate  [s^-1]
Rss     = 2.5    # source surface radius [Rsun]

FC.set_star_params(rsun, rotrate)

# Radial Velocity ----------------------------------------------------------------------|
# This can be any function user_vr that takes the CME nose distance (in Rsun)
# as input and returns the radial velocity in cm/s

# Three phase 
rga  = 1.3
rap  = 4.5
vmin = 70. * 1e5
vmax = 514. * 1e5
a_prop = (vmax**2 - vmin **2) / 2. / (rap - rga) # [ cm/s ^2 / rsun]

def user_vr(R_nose, rhat):
	if R_nose  <= rga: vtemp = vmin
	elif R_nose > rap: vtemp = vmax
	else: vtemp = np.sqrt(vmin**2 + 2. * a_prop * (R_nose - rga))
	return vtemp, vtemp*rhat

# Expansion Model ----------------------------------------------------------------------|
# This can be any function user_exp that takes the CME nose distance (in Rsun)
# as input and returns the angular width in degrees

# Exponential Parameters
aw0 =8.      # initial width (deg)
awM = 20.    # AW gained as R-> inf (deg)
awR = 1.5    # rate at which AW changes (rsun)
awmax = 22.  # maximum AW (if want less than aw0+awM) (deg)

# Exponential Expansion
def user_exp(R_nose):
	alpha = np.min([aw0 + awM*(1. - np.exp(-(R_nose-1.)/awR)), awmax])
	return alpha
#---------------------------------------------------------------------------------------|

#---------------------------------------------------------------------------------------|
# Mass Model ---------------------------------------------------------------------------|
# This can be any function user_mass that takes the CME nose distance (in Rsun)
# as input and returns the mass in g

max_M = 3e15 #[g]
rmaxM = 10 # rsun
# Constant mass
def user_mass(R_nose):
	M = max_M / 2. * (1 + (R_nose-CME_params[2])/(rmaxM - CME_params[2]))
	return np.min([M, max_M])
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|












#---------------------------------------------------------------------------------------|
# Simulation set up --(No Touching!)----------------------------------------------------|
#---------------------------------------------------------------------------------------|

# Initialize GPU arrays and load magnetic field data
if FC.useGPU:
	FF.init_GPU(FC.CR, Ntor, Npol, rsun, Rss)
else:
	FF.init_CPU(FC.CR, Ntor, Npol, rsun, Rss)

# Initialize CME
CME = CC.CME(ipos, CME_params, Ntor, Npol, user_vr, user_exp, user_mass, rsun)

# Initialize magnetic field and distance pickles
FC.initdefpickle(FC.CR)

# Open save files and initialize any plotting
FC.init_files(fprefix, CME)
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|


#---------------------------------------------------------------------------------------|
# Simulation Main Loop --(No Touching!)-------------------------------------------------|
#---------------------------------------------------------------------------------------|

dtprint = 0. # initiate print counter
# Run until nose hits rmax
while CME.points[CC.idcent][1,0] <= rmax:
	#CME.calc_forces()
	CME.update_CME(user_vr, user_exp, user_mass)
	dtprint += CME.dt
	if (dtprint > tprint):          # if not printing every time step, this 
		FC.print_status(CME)  # determines when to print and resets
		dtprint = 0. 	      # the counter
FC.print_status(CME) # print final position
FC.close_files()     # close the output files
if FC.useGPU: FF.clear_GPU() # reset GPU if needed
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|


