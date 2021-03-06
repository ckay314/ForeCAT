from __future__ import division
from matplotlib.patches import Patch
from pylab import *
import numpy as np
import math
import sys
import pickle
import CME_class as CC


global dtor, radeg

dtor  = 0.0174532925  # degrees to radians
radeg = 57.29577951    # radians to degrees
global picklejar 
picklejar = '/Users/ckay/PickleJar/'

def readinputfile():
    # Get the CME number
    global fprefix
    if len(sys.argv) < 2: 
        #sys.exit("Need an input file")
        print('No input file given!')
        sys.exit()
    else:
        input_file = sys.argv[1]
        inputs = np.genfromtxt(input_file, dtype=str)
        fprefix = input_file[:-4]
        input_values = get_inputs(inputs)
    return input_values

def get_inputs(inputs):
    possible_vars = ['ilat', 'ilon', 'tilt', 'CR', 'Cd', 'rstart', 'shapeA', 'shapeB', 'tprint', 'rmax', 'rotCME', 'Ntor', 'Npol', 'L0', 'useGPU', 'raccel1', 'raccel2', 'vrmin', 'vrmax', 'AWmin', 'AWmax', 'AWr', 'maxM', 'rmaxM', 'rsun', 'rotrate', 'Rss', 'saveData', 'printData', 'shapeB0']
    # if matches add to dictionary
    input_values = {}
    for i in range(len(inputs)):
        temp = inputs[i]
        if temp[0][:-1] in possible_vars:
            input_values[temp[0][:-1]] = temp[1]
        else:
            print temp[0][:-1], ' not a valid input '
    return input_values

def getInps(input_values):
    global rsun, rotrate, kmRs, Rss
    # assume solar defaults
    rsun = 7e10
    rotrate = 2.8e-6
    Rss = 2.5
    if 'rsun' in input_values:  rsun = float(input_values['rsun'])
    if 'rotrate' in input_values:  rotrate = float(input_values['rotrate'])
    if 'Rss' in input_values:  Rss = float(input_values['Rss'])
    kmRs  = 1.0e5 / rsun 
    
    
    # pull parameters for initial position
    try:
        ilat = float(input_values['ilat'])
        ilon = float(input_values['ilon'])
        origtilt = float(input_values['tilt'])
    except:
        print('Missing at least one of ilat, ilon, tilt.  Cannot run without :(')
        sys.exit()
    # code written orig for tilt clockwise from north but take input
    # as counterclockwise from west to match observations now
    origtilt = (origtilt +3600)%360. # force to be positive
    tilt = 90 - origtilt 
    
    init_pos = [ilat, ilon, tilt]
    
    # pull Carrington Rotation (or other ID for magnetogram)
    global CR
    try: 
        CR = int(input_values['CR'])
    except:
        print('Missing Carrington Rotation number (or other magnetogram ID).  Cannot run without :(')
        sys.exit()
        
    # check for drag coefficient
    global Cd
    try: 
        Cd = float(input_values['Cd'])
    except:
        print('Assuming Cd = 1')
        Cd = 1.
    
    # check for CME shape and initial nose distance
    try: 
        rstart = float(input_values['rstart'])
    except:
        print('Assuming CME nose starts at 1.1')
        rstart = 1.1
    try: 
        shapeA = float(input_values['shapeA'])
    except:
        print('Assuming A = 1')
        shapeA = 1.
    try: 
        shapeB = float(input_values['shapeB'])
    except:
        print('Assuming B = 0.15')
        shapeB = 0.15
    CME_params = [shapeA, shapeB, rstart]
    
    # get distance where we stop the simulation
    try: 
        rmax = float(input_values['rmax'])
    except:
        print('Assuming simulation stops at 10 Rs')
        rmax = 10.
    
    # determine frequency to print to screen
    tprint = 1. # default value
    if 'tprint' in input_values: tprint = float(input_values['tprint'])
        
    # determine if including rotation
    global rotCME
    rotCME = True
    if 'rotCME' in input_values: 
        if input_values['rotCME'] == 'False': 
            rotCME = False
        
    # determine torus grid resolution
    Ntor = 15
    Npol = 13
    if 'Ntor' in input_values:  Ntor = int(input_values['Ntor'])
    if 'Npol' in input_values:  Npol = int(input_values['Npol'])
    
    # determine L0 parameter
    global lon0
    lon0 = 0.
    if 'L0' in input_values:  lon0 = float(input_values['L0'])
    
    # determine if using GPU
    global useGPU
    useGPU = False
    if 'useGPU' in input_values: 
        if input_values['useGPU'] == 'True': 
            useGPU = True
            
    # get radial propagation model params
    global rga, rap, vmin, vmax, a_prop
    rga = 1.3
    rap = 4.0
    vmin = 70. * 1e5
    if 'raccel1' in input_values:  rga = float(input_values['raccel1'])
    if 'raccel2' in input_values:  rap = float(input_values['raccel2'])
    if 'vrmin' in input_values:  vmin = float(input_values['vrmin']) * 1e5
    try:
        vmax = float(input_values['vrmax']) *1e5
    except:
        print('Need final CME speed vrmax')
        sys.exit()
    a_prop = (vmax**2 - vmin **2) / 2. / (rap - rga) # [ cm/s ^2 / rsun]
    
    # get expansion model params
    aw0 = 8.
    awR = 1.5
    if 'AWmin' in input_values:  aw0 = float(input_values['AWmin'])
    if 'AWr' in input_values:  awR = float(input_values['AWr'])
    try:
        awM = float(input_values['AWmax']) - aw0
    except:
        print('Need final CME angular width AWmax')  
        sys.exit()          
    global user_exp 
    user_exp = lambda R_nose: aw0 + awM*(1. - np.exp(-(R_nose-1.)/awR))
    
    # check if given a B0, if so let shape B evolve at same rate as full AW
    shapeB0 = shapeB # set default to no change in shape if not specified
    if 'shapeB0' in input_values:  shapeB0 = float(input_values['shapeB0'])
    global user_Bexp
    user_Bexp = lambda R_nose: shapeB0 + (shapeB-shapeB0)*(1. - np.exp(-(R_nose-1.)/awR))
    
    # mass
    rmaxM = 10.
    if 'rmaxM' in input_values: rmaxM = float(input_values['rmaxM']) 
    try:
        max_M = float(input_values['maxM']) * 1e15
    except:
        print('Assuming 1e15 g CME')            
        max_M = 1e15
    global user_mass
    user_mass = lambda R_nose: np.min([max_M / 2. * (1 + (R_nose-CME_params[2])/(rmaxM - CME_params[2])), max_M])
    
    global saveData
    saveData = False
    if 'saveData' in input_values: 
        if input_values['saveData'] == 'True': 
            saveData = True
    
    global printData
    printData = True
    if 'printData' in input_values: 
        if input_values['printData'] == 'False': 
            saveData = False
            
    return CME_params, init_pos, rmax, tprint, Ntor, Npol
    

def initdefpickle(CR):
	# moved from pickle B since now use GPU_funcs for mag field stuff instead
	# opens the pickle holding the angular distance from the HCS
	global dists
	# get pickle name
	fname = 'CR' + str(CR) 
	# distance pickle [inCH, fromHCS, fromCH, fromPS(calc later maybe)]
	f1 = open(picklejar+fname+'_dists.pkl', 'rb')

	#print "loading distance pickle ..."
	dists = pickle.load(f1)
	f1.close()
	# make arrays not lists
	dists = np.array(dists)


def user_vr(R_nose, rhat):
    if R_nose  <= rga: vtemp = vmin
    elif R_nose > rap: vtemp = vmax
    else: vtemp = np.sqrt(vmin**2 + 2. * a_prop * (R_nose - rga))
    return vtemp, vtemp*rhat


def openfile(CME):
    global outfile
    outfile = open(fprefix + ".dat", "w")
    printstep(CME)
    
def printstep(CME):
    thislon = CME.points[CC.idcent][1,2]
    if lon0 > -998:
        thislon -= lon0
    # convert tilt from clockwise from N to counter from W
    tilt = (90-CME.tilt+3600.) % 360. # in between 0 and 360
    if tilt > 180: tilt -=360.
    vCME = np.sqrt(np.sum(CME.vels[0,:]**2))/1e5
    vdef = np.sqrt(np.sum((CME.vdefLL+CME.vdragLL)**2))/1e5
    # outdata is [t, lat, lon, tilt, vCME, vDef, AW, A, B]
    outdata = [CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], thislon, tilt, vCME, vdef, CME.ang_width*radeg, CME.shape_ratios[0], CME.shape_ratios[1]]
    outprint = ''
    for i in outdata:
        outprint = outprint +'{:7.3f}'.format(i) + ' '  
    if printData: print outprint  
    if saveData: outfile.write(outprint+'\n')

def calc_drag(CME):
#only calculate nonradial drag (ignore CME propagation)

	# need to calculate SW density use Guhathakurta 2006 model 
	# which depends on angular distance from HCS
	HCSdist = calc_dist(CME.cone[1,1], CME.cone[1,2], 1) 

	# determine SW speed
	global SW_v
	SW_rho, SW_v = calc_SW(CME.cone[1,0], HCSdist)

	# get total CME velocity vector (def+drag, not propagation)
	vr = np.sqrt(np.sum(CME.vels[0,:]**2))
	#CME_v = CME.vels[1,:] + CME.vels[2,:]
	colat = (90. - CME.cone[1,1]) * dtor
	lon = CME.cone[1,2] * dtor
	colathat = np.array([np.cos(lon) * np.cos(colat), np.sin(lon) * np.cos(colat), -np.sin(colat)]) 
	lonhat = np.array([-np.sin(lon), np.cos(lon), 0.])
	# scale the value to the new timestep
	CME_v = (CME.vdefLL[0] * colathat + CME.vdefLL[1] * lonhat + CME.vdragLL[0] * colathat + CME.vdragLL[1] * lonhat)  / (CME.points[CC.idcent][1,0] + vr * CME.dt * 60 / rsun) 

	# remove any radial component
	CMEv_nr = CME_v - np.dot(CME_v, CME.rhat) * CME.rhat	
	magdifvec = np.sqrt(CMEv_nr[0]**2 + CMEv_nr[1]**2 + CMEv_nr[2]**2)

	# use a variable form of Cd = tanh(beta)
	# this means using some approx for beta -> fit to Aschwanden Physics of Solar Corona figure 
	H = np.maximum(CME.cone[1,0] - 1., 0.01)
	#print H
	beta = 2.515 * np.power(H, 1.382) 
	varCd = Cd * math.tanh(beta)
	# determine drag force
	Fd = - (2. * varCd * SW_rho / CME.shape[1] / rsun / math.pi) * CMEv_nr * magdifvec
	return Fd

def calc_dist(lat, lon, didx):
# copied over from pickleB

# Use a similar slerping method as calcB but for the distance pickle.  We use the distances
# determined at the source surface height so that we must perform three slerps but no linterp.
# The distance pickle has dimensions [180, 360, 3], not the half degree resolution used in
# the B pickle
	# determine the nearest grid indices
	latidx = int(lat) + 89 # lower index for lat (upper is just + 1)
	lonidx = int(lon) % 360      # lower index for lon
	lonidx2 = (lonidx + 1) % 360 # make wrap at 0/360
	p1 = dists[latidx+1, lonidx, didx]    
	p2 = dists[latidx+1, lonidx2, didx]   
	p3 = dists[latidx, lonidx, didx]      
	p4 = dists[latidx, lonidx2, didx]     
	angdist = trislerp(lat, lon, p1, p2, p3, p4, 1.)
	return angdist


def trislerp(lat_in, lon_in, q1, q2, q3, q4, delta):
# copied over from pickleB

# This function assumes the spacing between points is delta. It uses the standard slerp formula 
#to slerp twice in longitude and then uses those results for one slerp in latitude.  This 
#function works fine for either scalar or vector qs.
	f_lat = (lat_in % delta) / delta  # contribution of first point in lat (0->1)
	f_lon = (lon_in % delta) / delta  # contribution of first point in lon (0->1)
	omega = delta * 3.14159 / 180.  # angular spacing
	# two lon slerps
	qa = (q1 * np.sin((1-f_lon) * omega) + q2 * np.sin(f_lon * omega)) / np.sin(omega) 
	qb = (q3 * np.sin((1-f_lon) * omega) + q4 * np.sin(f_lon * omega)) / np.sin(omega) 
	# one lat slerp
	qf = (qb * np.sin((1-f_lat) * omega) + qa * np.sin(f_lat * omega)) / np.sin(omega) 
	return qf



def calc_SW(r_in, HCSang):
	# Guhathakurta values
	fluxes = [2.5e3, 1.6e3] # SB and CH flux
	width_coeffs = [64.6106, -29.5795, 5.68860, 2.5, 26.3156]
	Ncs_coeffs = [2.6e5, 5.5986, 5.4155, 0.82902, -5.6654, 3.9784]
	Np_coeffs  = [8.6e4, 4.5915, 2.4406, -0.95714, -3.4846, 5.6630]
	
	# V74 Peg Alf values
	#fluxes = [56667. * 2.5e3, 56667. * 1.6e3] # SB and CH flux
	#width_coeffs = [40.2475, -14.2685, 1.59382, 4.5, 8.31402]
	#Ncs_coeffs = [2.81563e10, 5.79564, -5.04916, 4.54462, -13.2614, 9.30484]
	#Np_coeffs  = [3.16234e10, -2.0708, 5.2549, -1.99208, 4.21678, -3.19237]

	# V74 Peg Therm values
	#fluxes = [36000. * 2.5e3, 36000. * 1.6e3] # SB and CH flux
	#width_coeffs = [35.6838, -10.1296, 1.01284, 4.5, 10.6106]
	#Ncs_coeffs = [6.18079e9, 8.86609, -6.80094, 9.65199, -25.1931, 16.770]
	#Np_coeffs  = [3.30502e9, 2.96568, -0.426645, 13.6215, -27.4705, 14.9313]

	# 2029 Alfven
	#fluxes = [1.1467*2.5e3, 1.1467*1.6e3] # SB and CH flux
	#width_coeffs = [32.9821, -0.155529, -0.549913, 4.5, 22.5462]
	#Ncs_coeffs = [7.8474e5, 3.63769, 2.58435,-4.26632, 11.1436,-7.60057]
	#Np_coeffs  = [5.20256e5, 3.81369, 2.90452, -4.78581, 11.3536, -7.3248]

	# 2029 Thermal
	#fluxes = [0.0062*2.5e3, 0.0062*1.6e3] # SB and CH flux
	#width_coeffs = [37.4301, -4.88806, 0.105226, 4.5, 17.5647]
	#Ncs_coeffs = [988.305, 11.3272, 1.02168, 1.89289, -6.45504, 4.57029]
	#Np_coeffs  = [259.985, 13.9709, 0.915617,-1.31153, 0.666318, -0.111793]

	# 2029 Thermal2
	#fluxes = [1.86*2.5e3, 1.86*1.6e3] # SB and CH flux
	#width_coeffs = [41.0306, -9.11843, 0.832520, 4.5, 16.8562]
	#Ncs_coeffs = [377948, 7.48092, -0.219622, -0.0849005, -0.800739, 0.312819 ]
	#Np_coeffs  = [433389, 7.01102, 0.195413, -2.008087, 4.91103,-3.56231]

	#  mass flux at 1 AU
	scale = 1.
	SBflux = scale * fluxes[0] * 215.**2 #v in km; 1 Au nv = 2500*1e8 /s or Mdot= 1.86e-14 Msun/yr
	CHflux = scale * fluxes[1] * 215.**2 # Mdot = 1.19e-14
	## determine width of SB (from MHD simulation) !probably changes w/solar cycle... explore later
	my_w = width_coeffs[0] - width_coeffs[1] * r_in + width_coeffs[2] * r_in **2
	if r_in > width_coeffs[3]: my_w =  width_coeffs[4] # value at maximum
	## calculate CS and CH polynomial values
	ri = 1. / r_in
	ri = np.min([1., ri])
	#print ri
	##multiplied in the mysterious 1e8 G06 doesnt mention but includes
	Ncs = Ncs_coeffs[0] * np.exp(Ncs_coeffs[1] *ri +Ncs_coeffs[2] *ri**2) * ri**2 * (1. + Ncs_coeffs[3]  * ri + Ncs_coeffs[4]  * ri**2 + Ncs_coeffs[5]  * ri**3)
	Np  = Np_coeffs[0] * np.exp(Np_coeffs[1]*ri + Np_coeffs[2]*ri**2) * ri**2 * (1. + Np_coeffs[3] * ri + Np_coeffs[4] * ri**2 + Np_coeffs[5] * ri**3)

	# determine relative contributions of SB and CH polys
	exp_factor = np.exp(-HCSang**2 / my_w**2 / 2.) #2 from method of getting my_w
	my_dens = (Np + (Ncs - Np) * exp_factor)  # cm^-3

	# Chen density
	#my_dens = 3.99e8 * (3. * ri**12 + ri**4) + 2.3e5 * ri**2

	# determine velocity from flux and density
	my_vel  = 1.e5 * (CHflux + (SBflux - CHflux) * exp_factor)/ my_dens / r_in**2  #cm/s

	return my_dens * 1.6727e-24, my_vel

# Geometry programs
def SPH2CART(sph_in):
	r = sph_in[0]
	colat = (90. - sph_in[1]) * dtor
	lon = sph_in[2] * dtor
	x = r * np.sin(colat) * np.cos(lon)
	y = r * np.sin(colat) * np.sin(lon)
	z = r * np.cos(colat)
	return [x, y, z]

def CART2SPH(x_in):
# calcuate spherical coords from 3D cartesian
# output lat not colat
	r_out = np.sqrt(x_in[0]**2 + x_in[1]**2 + x_in[2]**2)
	colat = np.arccos(x_in[2] / r_out) * 57.29577951
	lon_out = np.arctan(x_in[1] / x_in[0]) * 57.29577951
	if lon_out < 0:
		if x_in[0] < 0:
			lon_out += 180.
		elif x_in[0] > 0:
			lon_out += 360. 
	elif lon_out > 0.:
		if x_in[0] < 0:
			lon_out += 180. 
	return [r_out, 90. - colat, lon_out]

def rotx(vec, ang):
# Rotate a 3D vector by ang (input in degrees) about the x-axis
	ang *= dtor
	yout = np.cos(ang) * vec[1] - np.sin(ang) * vec[2]
	zout = np.sin(ang) * vec[1] + np.cos(ang) * vec[2]
	return [vec[0], yout, zout]

def roty(vec, ang):
# Rotate a 3D vector by ang (input in degrees) about the y-axis
	ang *= dtor
	xout = np.cos(ang) * vec[0] + np.sin(ang) * vec[2]
	zout =-np.sin(ang) * vec[0] + np.cos(ang) * vec[2]
	return [xout, vec[1], zout]

def rotz(vec, ang):
# Rotate a 3D vector by ang (input in degrees) about the y-axis
	ang *= dtor
	xout = np.cos(ang) * vec[0] - np.sin(ang) * vec[1]
	yout = np.sin(ang) * vec[0] + np.cos(ang) * vec[1]
	return [xout, yout, vec[2]]


