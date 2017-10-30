import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
import random
import pickle
# give it the path wherever the ForeCAT files are stored
sys.path.append(os.path.abspath('/home/cdkay/ForeCAT'))
import ForeCAT_functions as FC
import CME_class as CC
import GPU_functions as GF


#---------------------------------------------------------------------------------------|
# Read in the filename from the command line and load the parameters -(No Touching!)----|
#---------------------------------------------------------------------------------------|

# Get CME parameters
fname = str(sys.argv[1])

# Determine the prefix used for saving files
fprefix = fname[:-4]

# Get CME parameters
CME_params, ipos, rmax, tprint, Ntor, Npol = FC.read_in_params(fname)

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

# Radial Velocity ----------------------------------------------------------------------|
# This can be any function user_vr that takes the CME nose distance (in Rsun)
# as input and returns the radial velocity in cm/s

# Three phase 
rga  = 1.5
rap  = 15.
vmin = 50. * 1e5
vmax = 950. * 1e5
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
aw0 =5.      # initial width (deg)
awM = 45.    # AW gained as R-> inf (deg)
awR = 1.5    # rate at which AW changes (rsun)
awmax = 55.  # maximum AW (if want less than aw0+awM) (deg)

# Exponential Expansion
def user_exp(R_nose):
	alpha = np.min([aw0 + awM*(1. - np.exp(-(R_nose-1.)/awR)), awmax])
	return alpha
#---------------------------------------------------------------------------------------|

#---------------------------------------------------------------------------------------|
# Mass Model ---------------------------------------------------------------------------|
# This can be any function user_mass that takes the CME nose distance (in Rsun)
# as input and returns the mass in g

init_M = 9.2e15 #[g]
# Constant mass
def user_mass(R_nose):
	M = init_M / 2. * (1 + (R_nose-CME_params[2])/(10. - CME_params[2]))
	return np.min([M, init_M])

#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|












#---------------------------------------------------------------------------------------|
# Simulation set up --(No Touching!)----------------------------------------------------|
#---------------------------------------------------------------------------------------|

# Initialize GPU arrays and load magnetic field data
#GF.init_GPU(FC.CR, Ntor, Npol, rsun, Rss)

# Initialize CME
#CME = CC.CME(ipos, CME_params, Ntor, Npol, user_vr, user_exp, user_mass, rsun)

# Initialize magnetic field and distance pickles
FC.initdefpickle(FC.CR)


#---------------------------------------------------------------------------------------|
# Simulation Main Loop --(No Touching!)-------------------------------------------------|
#---------------------------------------------------------------------------------------|
print ipos
Npoints = Ntor * Npol
idcent = int(Npoints / 2)
mypos = np.zeros(3)
mypos[:] = ipos[:]

random.seed(142)
Nruns = 100
all_runs=np.zeros([Nruns, 39,3]) # 39 for init + 0.5 btwn 1.5 and 20
for j in range(Nruns):
	# this works but inefficient but have to transfer low B pickle anyway so not that bad
	GF.init_GPU(FC.CR, Ntor, Npol, rsun, Rss)
	CME = CC.CME(mypos, CME_params, Ntor, Npol, user_vr, user_exp, user_mass, rsun)

	distcounter = 1

	mylats  = []
	mylons  = []
	mytilts = []
	# Run until nose hits rmax
	while CME.points[CC.idcent][1,0] <= rmax:
	#while dtprint==0:
		CME.update_CME(user_vr, user_exp, user_mass)
		#if dtprint > tprint:          # if not printing every time step, this 
		if CME.points[idcent][1,0] > distcounter:
			print j, CME.points[idcent][1,0], CME.points[idcent][1,1], CME.points[idcent][1,2]+rotrate * 60 * 180 / 3.14159  * CME.t, CME.tilt
			mylats.append(CME.points[idcent][1,1])
			mylons.append(CME.points[idcent][1,2]+rotrate * 60 * 180 / 3.14159  * CME.t)
			mytilts.append(CME.tilt)
			distcounter += 0.5
	all_runs[j,:,0] = mylats
	all_runs[j,:,1] = mylons
	all_runs[j,:,2] = mytilts
	mypos[0] = ipos[0] + 2.*(random.random() - 0.5)
	mypos[1] = ipos[1] + 2.*(random.random() - 0.5)
	mypos[2] = ipos[2] + 10.*(random.random() - 0.5)
	#print mypos, ipos
	GF.clear_GPU()


# save pickle
pickle_file = 'ensemble_'+fprefix+'.pkl'
pickle.dump(all_runs, open(pickle_file, 'wb'))


#GF.clear_GPU()	     # reset GPU 
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|
#---------------------------------------------------------------------------------------|


