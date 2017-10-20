from __future__ import division
from matplotlib.patches import Patch
from pylab import *
import numpy as np
import math
import sys
import pickle
import CME_class as CC


global rsun, dtor, radeg, kmRs, rotrate

#MAKE SURE THESE MATCH THE STAR
rsun  =  7e10		 # set at solar for now but will be
rotrate = 2.8e-6         # updated to correct values later

dtor  = 0.0174532925  # degrees to radians
radeg = 57.29577951    # radians to degrees
kmRs  = 1.0e5 / rsun # km (/s) divided by rsun (in cm)

def read_in_params(file):

# get rid of ins and use a checker that pulls var names from a dictionary!!!!!!!!!!!!
	

	#read in the parameters from a text file
	a = np.genfromtxt(file, dtype=None)
	ins = []
	myvar_names = []
	myvar_vals  = []
	needed_vars = ['ilat', 'ilon', 'tilt', 'CR', 'Cd', 'rstart', 'shapeA', 'shapeB', 'tprint', 'rmax', 'rotCME', 'Ntor', 'Npol', 'L0']
	ordered_var_vals = np.zeros(len(needed_vars))
	for i in range(len(a)):
		temp = a[i]
		myvar_names.append(temp[0][:-1])
		myvar_vals.append(temp[1])
		ins.append(temp[1])

	# check for each of the needed variables by name
	any_missing = False
	idx_counter = 0
	for a_var in needed_vars:
		if a_var in myvar_names:
			in_idx = myvar_names.index(a_var)
			ordered_var_vals[idx_counter] = myvar_vals[in_idx]
			print a_var,  myvar_vals[in_idx]
		else:
			any_missing = True
			print "Add ", a_var+': into input file' 
		idx_counter += 1

	# set variables to the read in values 
	global vmax, tprint, CR, Cd, rotCME, lon0
	ilat    = ordered_var_vals[0]      # [deg]
	ilon    = ordered_var_vals[1]      # [deg]
	tilt    = ordered_var_vals[2]	   # [deg]
	init_pos   = [ilat, ilon, tilt]

	CR		  = int(ordered_var_vals[3])
	Cd		  = ordered_var_vals[4]

	# CME_params = [A, B, rstart]
	CME_params = np.array([ordered_var_vals[6], ordered_var_vals[7], ordered_var_vals[5]], dtype=float)

	tprint  = ordered_var_vals[8]      # [mins]
	rmax    = ordered_var_vals[9]      # [Rstar]
	rotCME 	  = ordered_var_vals[10]
	Ntor = int(ordered_var_vals[11])
	Npol = int(ordered_var_vals[12])
	lon0 = ordered_var_vals[13]

	return CME_params, init_pos, rmax, tprint, Ntor, Npol


def initdefpickle(CR):
	# moved from pickle B since now use GPU_funcs for mag field stuff instead
	# opens the pickle holding the angular distance from the HCS
	global dists
	# get pickle name
	fname = 'CR' + str(CR) 
	# distance pickle [inCH, fromHCS, fromCH, fromPS(calc later maybe)]
	f1 = open('/home/cdkay/MagnetoPickles/'+fname+'_dists.pkl', 'rb')

	#print "loading distance pickle ..."
	dists = pickle.load(f1)
	f1.close()
	# make arrays not lists
	dists = np.array(dists)


def init_files(fname, CME):

	global f1, fname2
	fname2 = fname #useless but lets me print to screen so I can monitor progress
		       #while running a bash script of ForeCATs

	# Initialize the files where the output values are stored
	f1 = open(fname + ".dat", "w")
	#CME.init_CMEplot()
	#if makemovie == 1: CME.take_selfie()

	# if set lon0 < 400 then doing simulation in Carrington lon (CME will slip with respect to CR lon due to rotation)
	if lon0 < -400:
		print CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
	# assume space craft is at some initial lon0 and determine motion in that inertial frame
	else:
		print CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2] - lon0

	# same as above but for writing to file instead of printing to screen
	# include more values too
	if lon0 < -400:
		f1.write('%10.2f %15f %15f %15f %15f %15f %15f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f \n' % 
			(CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2], 
			 CME.shape[0], CME.shape[1], CME.shape[2], CME.vels[0,0]/1.e5, CME.vels[0,1]/1.e5, CME.vels[0,2]/1.e5,
			 CME.vels[1,0]/1.e5, CME.vels[1,1]/1.e5, CME.vels[1,2]/1.e5, CME.vels[2,0]/1.e5, CME.vels[2,1]/1.e5, CME.vels[2,2]/1.e5, CME.tilt, CME.ang_width*radeg))
	else:
		f1.write('%10.2f %15f %15f %15f %15f %15f %15f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f \n' % 
			(CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2] - lon0, 
			 CME.shape[0], CME.shape[1], CME.shape[2], CME.vels[0,0]/1.e5, CME.vels[0,1]/1.e5, CME.vels[0,2]/1.e5,
			 CME.vels[1,0]/1.e5, CME.vels[1,1]/1.e5, CME.vels[1,2]/1.e5, CME.vels[2,0]/1.e5, CME.vels[2,1]/1.e5, CME.vels[2,2]/1.e5, CME.tilt, CME.ang_width*radeg))


def print_status(CME):
	# This is mostly the same as the printing in the init files, it just doesn't init files
	# print to screen vector mags, not individual components
	v1 = np.sqrt(np.sum(CME.vels[0,:]**2)) # vr
	vdefLL = CME.vdefLL
	vdragLL = CME.vdragLL
	colat = (90. - CME.cone[1,1]) * dtor
	lon = CME.cone[1,2] * dtor
	colathat = np.array([np.cos(lon) * np.cos(colat), np.sin(lon) * np.cos(colat), -np.sin(colat)]) 
	lonhat = np.array([-np.sin(lon), np.cos(lon), 0.])
	vdef = vdefLL[0] * colathat + vdefLL[1] * lonhat
	vdrag = vdragLL[0] * colathat + vdragLL[1] * lonhat
	v2 = np.sqrt(np.sum(vdef**2)) # vdef
	v3 = np.sqrt(np.sum(vdrag**2)) # vdrag
	calc_width = np.arctan((CME.shape[1] + CME.shape[2]) / (CME.points[CC.idcent][1,0] - CME.shape[0] - CME.shape[1])) * radeg
	# co-rotating frame
	if lon0 < -400:
		print CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2], CME.tilt, CME.ang_width * radeg
		f1.write('%10.2f %15f %15f %15f %15f %15f %15f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f \n' % 
		(CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2], 
		 CME.shape[0], CME.shape[1], CME.shape[2], CME.vels[0,0]/1.e5, CME.vels[0,1]/1.e5, CME.vels[0,2]/1.e5,
		 vdef[0]/1.e5, vdef[1]/1.e5, vdef[2]/1.e5, vdrag[0]/1.e5, vdrag[1]/1.e5, vdrag[2]/1.e5, CME.tilt, CME.ang_width*radeg))

	# inertial frame
	else:
		print CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], (CME.points[CC.idcent][1,2] - lon0 + rotrate * 60 * radeg * CME.t) %360., CME.tilt, CME.ang_width * radeg, CME.M
		#print CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], (CME.points[CC.idcent][1,2] - lon0 + rotrate * 60 * radeg * CME.t) %360. , v1/1e5, v2/1e5,v3/1e5, CME.shape[0], CME.shape[1], CME.shape[2], CME.tilt
		f1.write('%10.2f %15f %15f %15f %15f %15f %15f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f \n' % 
		(CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], (CME.points[CC.idcent][1,2] -lon0 + rotrate * 60 * radeg * CME.t) % 360., 
		 CME.shape[0], CME.shape[1], CME.shape[2], CME.vels[0,0]/1.e5, CME.vels[0,1]/1.e5, CME.vels[0,2]/1.e5,
		 vdef[0]/1.e5, vdef[1]/1.e5, vdef[2]/1.e5, vdrag[0]/1.e5, vdrag[1]/1.e5, vdrag[2]/1.e5, CME.tilt, CME.ang_width*radeg))

def close_files():
	f1.close()

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


