import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import math
import pickle
import CME_class as CC
import ForeCAT_functions as FC


mod = SourceModule("""
	 #include <math.h>

    __device__ void rotx(float ang, float *y, float *z)
		{
		float newy = cos(ang) * *y - sin(ang) * *z;
		float newz = sin(ang) * *y + cos(ang) * *z;
		*y = newy;
		*z = newz;
		}

    __device__ void roty(float ang, float *x, float *z)
		{
		float newx =  cos(ang) * *x + sin(ang) * *z;
		float newz = -sin(ang) * *x + cos(ang) * *z;
		*x = newx;
		*z = newz;
		}

    __device__ void rotz(float ang, float *x, float *y)
		{
		float newx = cos(ang) * *x - sin(ang) * *y;
		float newy = sin(ang) * *x + cos(ang) * *y;
		*x = newx;
		*y = newy;
		}

    __global__ void calc_pos_GPU(float *tangs, float *pangs,  float *XYZpos, float *SPHpos, float *shape, float *RLLT)
		{
		// calculate the position of each grid point
		int myid = threadIdx.x + blockIdx.x * blockDim.x;
		float thetaT = tangs[myid];
		float thetaP = pangs[myid];

		// calculate the xyz in the unrotated CME frame
		float x = RLLT[0] + (shape[0] + shape[1] * cos(thetaP)) * cos(thetaT);
		float y = shape[1] * sin(thetaP);
		float z = (shape[2] + shape[1] * cos(thetaP)) * sin(thetaT);

		// convert from the CME fram to the Sun frame (CART2CART)
		// 1. rot by negative tilt about x
		rotx(-RLLT[3], &y, &z);
		// 2. rot by negative lat about y
		roty(-RLLT[1], &x, &z);
		// 3. rot by lon about z 
		rotz(RLLT[2], &x, &y);
		XYZpos[myid*3] = x;
		XYZpos[myid*3 + 1] = y;
		XYZpos[myid*3 + 2] = z;

		// Determine new spherical position
		float R = sqrt(powf(x,2.) + powf(y,2.) + powf(z,2.));
		SPHpos[myid*3] = R;
		SPHpos[myid*3 + 1] = 90. - acos(z / R) *  57.29577951;
		float lon = atan(y / x)*  57.29577951;
		if (x < 0. ) {lon += 180.;}
		else if (lon < 0.) {lon += 360.;}
		SPHpos[myid*3 + 2] = lon;
		}


    __device__ float getB(float R, float lat, float lon, int Bid, float *BL, float Rdel, float vsw, float rotrate, float RsR, float RSS)
		{
		// The magnetic field data is saved a long 1D array, some math is required to 
		// determine the appropriate index from the position

		// check if above SS or not
		float newR = min(RSS - 0.000001, R);

		// mag field component id
		int newBid = Bid;  
		if (R >= RSS){newBid = 3;}

		// determine R spacing assuming RSS and 150 points
		float scale = 150. / (RSS - 1.);

		// calculate ids for the given position
		int Rid = (int)((newR - Rdel) * scale);
		int latid = (int) (lat*2) + 179;
		int lonid = (int)(lon*2);
		int id = 1039680 * Rid + 2880 * latid + 4 * lonid + newBid; 
		
		// change between consecutive lons and R vals
		int dlon = 4;	
		int dR = 1039680;
		if (lonid == 720){dlon = -2876;}  // wrapping at edges

		// determine the 8 adjacent points needed for interpolation
		float B1 = BL[id + 2880]; // top left
		float B2 = BL[id + 2880 + dlon]; // top right
		float B3 = BL[id]; //bottom left
		float B4 = BL[id + dlon ]; // bottom right
		float B5 = BL[id + 2880 + dR]; // top left, up R
		float B6 = BL[id + 2880 + dlon + dR]; // top right, up R
		float B7 = BL[id + dR]; //bottom left, up R
		float B8 = BL[id + dlon + dR]; // bottom right, up R

		// Determine weighting of the interpolation points
		// assuming 0.5 degree and 1 /scale Rs resolution
		float flat = lat*2 - (int)(lat*2);  // weight of first point
		float flon = lon*2 - (int)(lon*2); // really just modulus of 0.5  
		float om = 0.5 * 3.14159 / 180.; // angular spacing

		// lon slerps (spherical linear interpolation)
		float Ba = (B1 * sin((1 - flon) * om) + B2 * sin(flon * om)) / sin(om);  
		float Bb = (B3 * sin((1 - flon) * om) + B4 * sin(flon * om)) / sin(om); 
		float Bc = (B5 * sin((1 - flon) * om) + B6 * sin(flon * om)) / sin(om); 
		float Bd = (B7 * sin((1 - flon) * om) + B8 * sin(flon * om)) / sin(om); 

		// lat slerps
		float Baa = (Bb * sin((1 - flat) * om) + Ba * sin(flat * om)) / sin(om);
		float Bbb = (Bd * sin((1 - flat) * om) + Bc * sin(flat * om)) / sin(om);

		// radial linterp
		float fR = newR*scale - (int)(newR*scale); 
		float Bmag = (1 - fR) * Baa + fR * Bbb;

		// done if R < Rss
		float Bout = Bmag; 

		// scale appropriately and add Parker spiral if R > Rss
		if (R > RSS){	
			float Br = Bmag * newR * newR / R / R;
			float sc = sin((90. - lat) * 0.0174532925); // sin colat
			float cc = cos((90. - lat) * 0.0174532925); // cos colat
			float sl = sin(lon * 0.0174532925);
			float cl = cos(lon * 0.0174532925);
			float Bphi = - Br * (R - RSS) * RsR * 7e10 * rotrate * sc / vsw; //set to 0 to turn off Parker
			if (Bid == 3){Bout = sqrt( Br * Br + Bphi * Bphi);}  // magnitude
			if (Bid == 0){Bout = sc * cl * Br - sl * Bphi;}  // x component
			if (Bid == 1){Bout = sc * sl * Br + cl * Bphi;}  // y component
			if (Bid == 2){Bout = cc * Br;}                   // z component
			}

		return Bout;

		// uncomment one of these and comment above to scale the PFSS model
		//return Bout * min(RSS, R); // scale by extra R
		//return Bout * min(RSS, R) *min(RSS, R); // scale by extra R2		
		}

    __global__ void calc_forces_GPU(float *SPHpos, float *BL, float *tangs, float *pangs, float *shape, float * RLLT, float * Rdelta, float *outs, float * vsw_in, float * rotrate_in, float * RsR_in, float * RSS_in)
		{
		// determine grid id
		int myid = threadIdx.x + blockIdx.x * blockDim.x;

		// other params
		float Rdel = Rdelta[0]; // R offset (needed for higher pickles which start at R=Rdel)
		float vsw = vsw_in[0];  // solar wind speed (needed to get Parker B)
		float rotrate = rotrate_in[0];  // rotation rate (also for Parker B)
		float RsR = RsR_in[0];  // ratio of Rstar to Rsun 
		float RSS = RSS_in[0];  // source surface radius

		// grid point coords
		float myr = SPHpos[myid*3];
		float lat = SPHpos[myid*3 + 1];
		float lon = SPHpos[myid*3 + 2];
		float colat = (90. - lat) * 0.0174532925;

		// will repeatedly need sin and cos of colat and lon, calc only once
		float sc = sin(colat);
		float cc = cos(colat);
		float sl = sin(lon * 0.0174532925);
		float cl = cos(lon * 0.0174532925);

		// radial unit vector
		float rhat[3] = {sc * cl, sc * sl, cc};

		// Determine magnetic field vector
		float Bmag = getB(myr, lat, lon, 3, BL, Rdel, vsw, rotrate, RsR, RSS);
		float Bhat[3] = {getB(myr, lat, lon, 0, BL, Rdel, vsw, rotrate, RsR, RSS)/Bmag, getB(myr, lat, lon, 1, BL, Rdel, vsw, rotrate, RsR, RSS)/Bmag, getB(myr, lat, lon, 2, BL, Rdel, vsw, rotrate, RsR, RSS)/Bmag};
		// can replace with a different vector for debugging purposes
		//float Bhat[3] = {sc * cl, sc * sl, cc};  //radial B
		//float Bhat[3] = {0., 0., 1.}; //vertical B

		// calculate the gradient in B2 for the pressure force
		float Blat1 = getB(myr, lat + 0.5, lon, 3, BL, Rdel, vsw, rotrate, RsR, RSS);
		float Blat2 = getB(myr, lat - 0.5, lon, 3, BL, Rdel, vsw, rotrate, RsR, RSS);
		float dB2dlat = (powf(Blat1,2) - powf(Blat2, 2)) / 2. / (0.5 * 0.0174532925 * RsR * 7e10 * myr);
		float Blon1 = getB(myr, lat, lon + 0.5, 3, BL, Rdel, vsw, rotrate, RsR, RSS);
		float Blon2 = getB(myr, lat, lon - 0.5, 3, BL, Rdel, vsw, rotrate, RsR, RSS);		
		float dB2dlon = (powf(Blon1,2) - powf(Blon2, 2)) / 2. / (0.5 * 0.0174532925 * RsR * 7e10 * myr * sc);
		// convert to the NEGATIVE Cartesian coordinate
		float dB2[3] = {cc * cl * dB2dlat + sl * dB2dlon, cc * sl * dB2dlat - cl * dB2dlon, -sc * dB2dlat};


		// calculate the magnetic tension force
	
		// need the poloidal and toroidal angles of the grid point
		float myTP = pangs[myid];
		float myTT = tangs[myid];

		// fake compression on bottom side
		if (RLLT[0] >= 99999991.3){
			if (myTP >= 0.){  //.523 for 30 deg
					float scaleup = (RLLT[0]-1.2)/0.15;
					if (scaleup > 10){scaleup = 10;}
					Bmag = getB(myr, lat, lon, 3, BL, Rdel, vsw, rotrate, RsR, RSS)*scaleup;
					Blat1 = Blat1 * scaleup;
					Blat2 = Blat2 * scaleup;
					Blon1 = Blon1 * scaleup;
					Blon2 = Blon2 * scaleup;
					dB2dlat = (powf(Blat1,2) - powf(Blat2, 2)) / 2. / (0.5 * 0.0174532925 * RsR * 7e10 * myr);
					dB2dlon = (powf(Blon1,2) - powf(Blon2, 2)) / 2. / (0.5 * 0.0174532925 * RsR * 7e10 * myr * sc);
			}		float dB2[3] = {cc * cl * dB2dlat + sl * dB2dlon, cc * sl * dB2dlat - cl * dB2dlon, -sc * dB2dlat};
				
		}

		// calc sin and cos once
		float sp = sin(myTP);
		float cp = cos(myTP);
		float st = sin(myTT);
		float ct = cos(myTT);

		// copy shape parameters to short variables
		float a = shape[0];
		float b = shape[1];
		float c = shape[2];

		// toroidal tangent vector
		float xt[3] = {-(a + b * cp) * st, 0., (c + b * cp) * ct};
		float xtmag = sqrt(powf(xt[0], 2) + powf(xt[1], 2) + powf(xt[2], 2));
		float tgt_t[3] = {xt[0] / xtmag, xt[1] / xtmag, xt[2] / xtmag}; 

		// poloidal tangent vector
		float xp[3] = {-b * sp * ct, b * cp, -b * sp * st};
		float xpmag = sqrt(powf(xp[0], 2) + powf(xp[1], 2) + powf(xp[2], 2));
		float tgt_p[3] = {xp[0] / xpmag, xp[1] / xpmag, xp[2] / xpmag}; 

		// normal vector - cross product of tangents
		float norm[3] = {tgt_p[1] * tgt_t[2] - tgt_p[2] * tgt_t[1], tgt_p[2] * tgt_t[0] - tgt_p[0] * tgt_t[2], tgt_p[0] * tgt_t[1] - tgt_p[1] * tgt_t[0]};

		// calculate second derivatives
		float xpp[3] = {-b * cp * ct, -b *sp, -b * cp * st};
		float xtt[3] = {-(a + b * cp) * ct, 0., -(c + b *cp) * st};
		float xpt[3] = {b * sp * st, 0., - b * sp * ct};

		// coefficients of first fundamental form
		float E1FF = xp[0] * xp[0] + xp[1] * xp[1] + xp[2] * xp[2];
		float F1FF = xp[0] * xt[0] + xp[1] * xt[1] + xp[2] * xt[2];
		float G1FF = xt[0] * xt[0] + xt[1] * xt[1] + xt[2] * xt[2];

		// coefficients of second fundamental form
		float e2FF = norm[0] * xpp[0] + norm[1] * xpp[1] + norm[2] * xpp[2];
		float f2FF = norm[0] * xpt[0] + norm[1] * xpt[1] + norm[2] * xpt[2];
		float g2FF = norm[0] * xtt[0] + norm[1] * xtt[1] + norm[2] * xtt[2];

		// convert vectors to Sun frame
		rotx(-RLLT[3], &tgt_t[1], &tgt_t[2]);
		roty(-RLLT[1], &tgt_t[0], &tgt_t[2]);
		rotz(RLLT[2], &tgt_t[0], &tgt_t[1]);
		rotx(-RLLT[3], &tgt_p[1], &tgt_p[2]);
		roty(-RLLT[1], &tgt_p[0], &tgt_p[2]);
		rotz(RLLT[2], &tgt_p[0], &tgt_p[1]);
		rotx(-RLLT[3], &norm[1], &norm[2]);
		roty(-RLLT[1], &norm[0], &norm[2]);
		rotz(RLLT[2], &norm[0], &norm[1]);

		// solar magnetic field components in pol/tor directions
		float Bt_u = Bhat[0] * tgt_t[0] + Bhat[1] * tgt_t[1] + Bhat[2] * tgt_t[2]; 
		float Bp_u = Bhat[0] * tgt_p[0] + Bhat[1] * tgt_p[1] + Bhat[2] * tgt_p[2]; 


		// unit tgt vect in direction of draping on the surface
		float DD[2] = {Bp_u / abs(Bp_u) * sqrt(1. - powf(Bt_u, 2)), Bt_u};  // drape normal into poloidal dir (normal option)
		//float DD[2] = {Bp_u, Bt_u / abs(Bt_u) * sqrt(1. - powf(Bp_u, 2))};  // drape normal into toroidal dir
		//float DD[2] = {Bp_u, Bt_u}; // no draping of Br into Bp

		// determine geodesic curvature and tension force
		float k = (e2FF * DD[0] * DD[0] + f2FF * DD[0] * DD[1] + g2FF * DD[1] * DD[1]) / (E1FF * DD[0] * DD[0] + F1FF * DD[0] * DD[1] + G1FF * DD[1] * DD[1]);
		float Ft_mag = abs(k) * Bmag * Bmag / 4. / 3.14159 / (RsR * 7e10);

		// remove radial component
		float ndr = norm[0] * rhat[0] + norm[1] * rhat[1] + norm[2] * rhat[2]; // norm dot rhat
		float n_nr[3] = {norm[0] - ndr * rhat[0], norm[1] - ndr * rhat[1], norm[2] - ndr * rhat[2]};
		float Ftens[3] = {-Ft_mag * n_nr[0], -Ft_mag * n_nr[1], -Ft_mag * n_nr[2]}; // points in negative nr


		// determine magnetic pressure gradient force which is easy since we already have the grads
		// remove component of F_Pgrad parallel to the draped B
		float drBhat[3]  = {DD[0] * tgt_p[0] + DD[1] * tgt_t[0], DD[0] * tgt_p[1] + DD[1] * tgt_t[1], DD[0] * tgt_p[2] + DD[1] * tgt_t[2]};
		float dB2parmag = dB2[0] * drBhat[0] + dB2[1] * drBhat[1] + dB2[2] * drBhat[2];
		float dB2perp[3] = {dB2[0] - dB2parmag * drBhat[0], dB2[1] - dB2parmag * drBhat[1], dB2[2] - dB2parmag * drBhat[2]};
		float FPgrad[3] = {dB2perp[0]/8./3.14159, dB2perp[1]/8./3.14159, dB2perp[2]/8./3.14159}  ;

		// put in the output vectors for transfer from the GPU
		int j;
		// this is making them overwrite each other - > 6 * (myid+1) not 7*
		//for (j = 6*myid; j<6*(myid+1); j++){outs[j] = 0;};
		outs[6*myid]   = Bmag;
		outs[6*myid+1]   = FPgrad[1];
		outs[6*myid+2]   = FPgrad[2];
		outs[6*myid+3]   = Ftens[0];
		outs[6*myid+4]   = Ftens[1];
		outs[6*myid+5]   = Ftens[2];
		}

    __global__ void calc_torque_GPU(float *SPHpos, float *XYZpos, float *tangs, float *pangs, float *shape, float * RLLT, float *forces_gpu, float *tottor_in)
		{
		// determine grid id
		int myid = threadIdx.x + blockIdx.x * blockDim.x;

		// grid point coords
		float myr = SPHpos[myid*3];
		float lat = SPHpos[myid*3 + 1];
		float lon = SPHpos[myid*3 + 2];
		float colat = (90. - lat) * 0.0174532925;
		float myXYZ[3] = {XYZpos[myid*3], XYZpos[myid*3+1],XYZpos[myid*3+2]};

	
		// need the poloidal and toroidal angles of the grid point
		float myTP = pangs[myid];
		float myTT = tangs[myid];

		// calc sin and cos once
		float sp = sin(myTP);
		float cp = cos(myTP);
		float st = sin(myTT);
		float ct = cos(myTT);

		// calc distance along x-axis for each toroidal angle, which we subtract
		// to get the distance perp to the x-axis/axis of rotation
		float toraxx[3] = {RLLT[0] + (shape[0] + shape[1] * cp) * ct, 0., 0.};
		// convert vectors to Sun frame
		rotx(-RLLT[3], &toraxx[1], &toraxx[2]);
		roty(-RLLT[1], &toraxx[0], &toraxx[2]);
		rotz(RLLT[2], &toraxx[0], &toraxx[1]);
		// determine the perpendicular distance AKA lever arm length
		float dist[3] = {XYZpos[myid*3] - toraxx[0], XYZpos[myid*3+1] - toraxx[1], XYZpos[myid*3+2] - toraxx[2]};
		// get the total deflection force at the point
		float myFdef[3] = {forces_gpu[6*myid] + forces_gpu[6*myid+3],forces_gpu[6*myid+1] + forces_gpu[6*myid+4],forces_gpu[6*myid+2] + forces_gpu[6*myid+5]};
		// torque is cross of dist and myFdef
		float Fcrossd[3] = {dist[1]*myFdef[2]-dist[2]*myFdef[1], dist[2]*myFdef[0]-dist[0]*myFdef[2], dist[0]*myFdef[1]-dist[1]*myFdef[0]};
		// dot with rhat
		float lonnose = RLLT[2];
		float colatnose = 3.14159/2. - RLLT[1];
		float rhat[3] = {sin(colatnose) * cos(lonnose), sin(colatnose) * sin(lonnose), cos(colatnose)};		
		float tottor =  (Fcrossd[0] * rhat[0] + Fcrossd[1] * rhat[1] + Fcrossd[2] * rhat[2]);
		// take only component of torque in radial direction
		tottor_in[myid] = tottor;
	}

""")

def init_GPU(CR, Ntor, Npol, rsun_in, RSS_in):
	global rsun, dtor, radeg, kmRs, RSS, dR
	rsun  = rsun_in	
	RSS   = RSS_in 

	dtor  = 0.0174532925   # degrees to radians
	radeg = 57.29577951    # radians to degrees
	kmRs  = 1.0e5 / rsun   # km (/s) divided by rsun (in cm)
	dR = (RSS - 1.0) / 150.

	# allocate memory on the GPU
	global CMEpos, CMEposxyz_gpu, CMEposSPH_gpu, forces, forces_gpu, CMEpos2_gpu
	# position of the CME points
	Npoints = Ntor * Npol 
	#CMEpos = [[]] * Npoints
	CMEposxyz = np.zeros([Npoints * 3], dtype=np.float32)
	CMEposxyz_gpu = cuda.mem_alloc(CMEposxyz.size * CMEposxyz.dtype.itemsize)
	CMEposxyz_gpu = gpuarray.to_gpu(CMEposxyz)
	CMEposSPH = np.zeros([Npoints * 3], dtype=np.float32)
	CMEposSPH_gpu = cuda.mem_alloc(CMEposSPH.size * CMEposSPH.dtype.itemsize)
	CMEposSPH_gpu = gpuarray.to_gpu(CMEposSPH)
	# other holders
	forces = np.zeros(Npoints * 6)
	forces = forces.astype(np.float32)
	forces_gpu = cuda.mem_alloc(forces.size * forces.dtype.itemsize)
	forces_gpu = gpuarray.to_gpu(forces)

	# Determine and copy the poloidal and toroidal spacing onto the GPU
	if Ntor !=1: delta_maj = 120. / (Ntor - 1.) * dtor  # point spacing along toroidal
	if Ntor ==1: delta_maj = 0.
	delta_min = 120. / (Npol - 1.) * dtor  # point spacing along poloidal
	t_angs = np.array([delta_maj * (int(Ntor / 2) - int(i/Npol)) for i in range(Npoints)], dtype=np.float32)
	p_angs = np.array([ delta_min * (i - int(Npol / 2)) for i in range(Npol)] * Ntor, dtype=np.float32)
	global tangs_gpu, pangs_gpu, shape_gpu
	tangs_gpu = cuda.mem_alloc(t_angs.size * t_angs.dtype.itemsize)
	tangs_gpu = gpuarray.to_gpu(t_angs)
	pangs_gpu = cuda.mem_alloc(p_angs.size * p_angs.dtype.itemsize)
	pangs_gpu = gpuarray.to_gpu(p_angs)

	# Set yo arrays for the CME shape and position
	global shape_gpu, RLLT_gpu
	temp  = np.zeros([3], dtype = np.float32)
	shape_gpu = cuda.mem_alloc(temp.size * temp.dtype.itemsize)
	temp  = np.zeros([4], dtype = np.float32)
	RLLT_gpu = cuda.mem_alloc(temp.size * temp.dtype.itemsize) #R, lat lon and tilt


	# load the pickles which hold the mag field data
	#f1 = open('/gevalt4/PickleJar/CR' + str(FC.CR) +'a.pkl', 'rb') 
	f1 = open('/home/cdkay/MagnetoPickles/CR'+str(FC.CR)+'a.pkl', 'rb')
	#print "loading low pickle ..."
	B_low = pickle.load(f1)
	f1.close()
	#f1 = open('/gevalt4/PickleJar/CR' + str(FC.CR) +'b.pkl', 'rb')
	f1 = open('/home/cdkay/MagnetoPickles/CR'+str(FC.CR)+'b.pkl', 'rb')
	#print "loading high pickle ..."
	B_high = pickle.load(f1)
	f1.close() 

	# convert to 1D array
	global B_low1D, B_high1D, B_mid1D
	B_low1D = np.reshape(B_low, [-1])
	B_high1D = np.reshape(B_high, [-1]) 

	# create a mid with half of low and high since GPU has tiny memory
	Ba = np.reshape(B_low[40:-1, :, :,:], [-1])
	Bb = np.reshape(B_high[:45, :,:,:], [-1])
	B_mid1D = np.concatenate((Ba, Bb))

	# Get ready for the GPU and add the first pickle
	B_low1D = B_low1D.astype(np.float32)
	B_high1D = B_high1D.astype(np.float32)
	B_mid1D = B_mid1D.astype(np.float32) # is the longest of the three
	global BL_gpu
	BL_gpu = cuda.mem_alloc(B_mid1D.size * B_mid1D.dtype.itemsize)
	BL_gpu = gpuarray.to_gpu(B_low1D)
	global R_offset_gpu
	R_offset = np.array([1.0])
	R_offset = R_offset.astype(np.float32)
	R_offset_gpu = cuda.mem_alloc(R_offset.size * R_offset.dtype.itemsize)
	R_offset_gpu = gpuarray.to_gpu(R_offset)
	global Bzone
	Bzone = 0 # use to indicate if in low, mid, high

	# More things for the GPU (solar wind v, rotation rate, R star ratio, RSS)
	global vsw_gpu
	vsw = np.zeros(1)
	vsw = vsw.astype(np.float32)
	vsw_gpu = cuda.mem_alloc(vsw.size * vsw.dtype.itemsize)
	global rotrate_gpu
	rotrate = np.array(FC.rotrate, dtype=np.float32)
	rotrate_gpu = cuda.mem_alloc(R_offset.size * R_offset.dtype.itemsize)
	rotrate_gpu = gpuarray.to_gpu(rotrate)
	global RsR_gpu 
	RsR = np.array(rsun / 7e10, dtype=np.float32)
	RsR_gpu = cuda.mem_alloc(R_offset.size * R_offset.dtype.itemsize)
	RsR_gpu = gpuarray.to_gpu(RsR)
	global RSS_gpu 
	RSSar = np.array(RSS, dtype=np.float32)
	RSS_gpu = cuda.mem_alloc(R_offset.size * R_offset.dtype.itemsize)
	RSS_gpu = gpuarray.to_gpu(RSSar)
	
	# total torque variable
	global tottor_gpu
	tottor = np.zeros(Npoints)
	tottor = forces.astype(np.float32)
	tottor_gpu = cuda.mem_alloc(tottor.size * tottor.dtype.itemsize)
	tottor_gpu = gpuarray.to_gpu(tottor)
		
	# set up ability to call GPU functions
	global GPU_calcpos
	GPU_calcpos = mod.get_function("calc_pos_GPU")
	global GPU_calcforces
	GPU_calcforces = mod.get_function("calc_forces_GPU")
	global GPU_calctorque
	GPU_calctorque = mod.get_function("calc_torque_GPU")

def init_CPU(CR, Ntor, Npol, rsun_in, RSS_in):
	global rsun, dtor, radeg, kmRs, RSS, dR, RsR
	rsun  = rsun_in	
	RSS   = RSS_in 
	RsR = rsun/7e10 # star radius in solar radii

	dtor  = 0.0174532925   # degrees to radians
	radeg = 57.29577951    # radians to degrees
	kmRs  = 1.0e5 / rsun   # km (/s) divided by rsun (in cm)
	dR = (RSS - 1.0) / 150.

	# Determine the poloidal and toroidal spacing (already in CME class???)
	#if Ntor !=1: delta_maj = 120. / (Ntor - 1.) * dtor  # point spacing along toroidal
	#if Ntor ==1: delta_maj = 0.
	#delta_min = 120. / (Npol - 1.) * dtor  # point spacing along poloidal
	#t_angs = np.array([delta_maj * (int(Ntor / 2) - int(i/Npol)) for i in range(Npoints)], dtype=np.float32)
	#p_angs = np.array([ delta_min * (i - int(Npol / 2)) for i in range(Npol)] * Ntor, dtype=np.float32)

	global B_low, B_high
	# load the pickles which hold the mag field data
	f1 = open('/home/cdkay/MagnetoPickles/CR'+str(FC.CR)+'a.pkl', 'rb')
	#print "loading low pickle ..."
	B_low = pickle.load(f1)
	f1.close()
	f1 = open('/home/cdkay/MagnetoPickles/CR'+str(FC.CR)+'b.pkl', 'rb')
	#print "loading high pickle ..."
	B_high = pickle.load(f1)
	f1.close() 

	# make B mid here?

	# flag for in low/high pickle
	Bzone = 0

def calc_pos(CME):
	# put the current shape and position on the GPU
	global shape, shape_gpu, RLLT, RLLT_gpu
	shape = np.array([CME.shape[0], CME.shape[1], CME.shape[2]], dtype=np.float32)
	shape_gpu = gpuarray.to_gpu(shape)	
	RLLT = np.array([CME.cone[1,0], CME.cone[1,1] * dtor, CME.cone[1,2] * dtor, CME.tilt * dtor], dtype=np.float32)
	RLLT_gpu = gpuarray.to_gpu(RLLT)
	# Call GPU kernel which calculates the shape
	GPU_calcpos(tangs_gpu, pangs_gpu, CMEposxyz_gpu, CMEposSPH_gpu, shape_gpu, RLLT_gpu, block=(CC.Npol, 1, 1), grid=(CC.Ntor,1,1))

	# is this necessary, faster if just leave on GPU?
	CMEposxyz = CMEposxyz_gpu.get()
	CMEposSPH = CMEposSPH_gpu.get()
	for i in range(CC.Npoints):
		CME.points[i][0,:] = CMEposxyz[3*i : 3*(i+1)]
		CME.points[i][1,:] = CMEposSPH[3*i : 3*(i+1)]


def calc_posCPU(CME):
	# serial version of GPU calcpos
	RLLT = [CME.cone[1,0], CME.cone[1,1] * dtor, CME.cone[1,2] * dtor, CME.tilt * dtor]
	xs = RLLT[0] + (CME.shape[0] + CME.shape[1]*np.cos(CME.p_angs))*np.cos(CME.t_angs)
	ys = CME.shape[1] * np.sin(CME.p_angs)
	zs = (CME.shape[2] + CME.shape[1] * np.cos(CME.p_angs))*np.sin(CME.t_angs)
	# make a vector for position in CME coordinates
	CCvec = [xs,ys,zs]
	CCvec1 = FC.rotx(CCvec,-CME.tilt)
	CCvec2 = FC.roty(CCvec1,-CME.cone[1,1])
	xyzpos = FC.rotz(CCvec2, CME.cone[1,2])
	CMErs  = np.sqrt(xyzpos[0]**2+xyzpos[1]**2+xyzpos[2]**2)
	CMElats = 90. - np.arccos(xyzpos[2]/CMErs) / dtor
	CMElons = np.arctan(xyzpos[1]/xyzpos[0]) / dtor
	negxs = np.where(xyzpos[0]<0)
	CMElons[negxs] +=180.
	CMElons[np.where(CMElons<0)] += 360.
	SPHpos = [CMErs, CMElats, CMElons]
	for i in range(CC.Npoints):
		CME.points[i][0,:] = [xyzpos[0][i], xyzpos[1][i], xyzpos[2][i]]
		CME.points[i][1,:] = [SPHpos[0][i], SPHpos[1][i], SPHpos[2][i]]
	# convenient to store these in class as arrays for serial version
	CME.rs = CMErs
	CME.lats = CMElats
	CME.lons = CMElons

def calc_forces(CME):
	global BL_gpu, R_offset_gpu, Bzone, rotrate_gpu, RsR_gpu, RSS_gpu
	# put new solar wind velocity on GPU
	temp = np.array([FC.SW_v.astype(np.float32)])
	vsw_gpu = gpuarray.to_gpu(temp)
	# updated positions already on GPU

	# call force calculation kernel
	GPU_calcforces(CMEposSPH_gpu, BL_gpu, tangs_gpu, pangs_gpu, shape_gpu, RLLT_gpu, R_offset_gpu, forces_gpu, vsw_gpu, rotrate_gpu, RsR_gpu, RSS_gpu, block=(CC.Npol, 1, 1), grid=(CC.Ntor,1,1))

	# check if need to switch to next magnetic field zone and if so upload to GPU
	# check to switch to mid
	if (Bzone==0 and CME.points[CC.idcent][1,0] > (1. + dR * 70)):
		Bzone = 1
		BL_gpu = gpuarray.to_gpu(B_mid1D)
		R_offset = np.array([1 + 40 * dR])
		R_offset = R_offset.astype(np.float32)
		R_offset_gpu = gpuarray.to_gpu(R_offset)
	# check to switch to high
	if (Bzone==1 and CME.points[CC.idcent][1,0] > (1. + 115 * dR) ):
		Bzone = 2
		BL_gpu = gpuarray.to_gpu(B_high1D)
		R_offset = np.array([1. + 75 * dR])
		R_offset = R_offset.astype(np.float32)
		R_offset_gpu = gpuarray.to_gpu(R_offset)

	# get forces from GPU 
	forces =  forces_gpu.get()
	print 'from GPU', forces[0:6]
	# clean out NaNs that randomly show up at a few points on occasion (weird angle I'm guessing)
	forces[np.where(np.isnan(forces) == True)] = 0.
	for i in range(CC.Npoints):
		# set the forces equal to the values from GPU
		CME.defforces[i][0,:] = forces[6*i + 3: 6*(i+1)]
		CME.defforces[i][1,:] = forces[6*i : 6*i +3]

def calc_forcesCPU(CME):
	# will repeatedly need sin/cos of lon/colat -> calc once here
	sc = np.sin((90.-CME.lats)*dtor)
	cc = np.cos((90.-CME.lats)*dtor)
	sl = np.sin(CME.lons*dtor)
	cl = np.cos(CME.lons*dtor)
	# unit vec in radial direction
	rhat = [sc*cl, sc*sl, cc]
	
	# calculate mag field at points AKA the reall fun part
	getBCPU(CME.rs, CME.lats, CME.lons)


def getBCPU(R, lat, lon):
	# doing all vec at once? is wise?

	
	# relvant globals FC.SW_v, FC.rotrate, RsR, RSS
	Rids = (R-1.)/dR
	Rids = Rids.astype(int)
	# if Rids < 75 then in low pickle, else < 150 in high pickle, else above source surface
	lowIDs  = np.where(Rids<75)
	highIDs = np.where((Rids>74) & (Rids<150))
	ssIDs   = np.where(Rids >=150)
	latIDs = lat*2 + 179
	latIDs = latIDs.astype(int)
	lonIDs = lon*2
	lonIDs = lonIDs.astype(int)
	
	# determine the B at the 8 adjacent points
	# assuming B low for now!!!!!
	B1 = B_low[Rids, latIDs+1, lonIDs,:]
	B2 = B_low[Rids, latIDs+1, (lonIDs+1)%720,:]
	B3 = B_low[Rids, latIDs, lonIDs,:]
	B4 = B_low[Rids, latIDs, (lonIDs+1)%720,:]
	B5 = B_low[Rids+1, latIDs+1, lonIDs,:]
	B6 = B_low[Rids+1, latIDs+1, (lonIDs+1)%720,:]
	B7 = B_low[Rids+1, latIDs, lonIDs,:]
	B8 = B_low[Rids+1, latIDs, (lonIDs+1)%720,:]

	# determine weighting of interpolation points
	flat = lat*2 - latIDs + 179
	flon = lon*2 - lonIDs
	om = 0.5 * dtor # assuming half deg spacing
	som = np.sin(om)	

	# lon slerps
	Npoints = len(flat)
	sflon1 = np.zeros([Npoints, 4])
	sflon2 = np.zeros([Npoints, 4])
	for i in range(4):
		sflon1[:,i] = np.sin((1.-flon)*om) 
		sflon2[:,i] = np.sin(flon*om)
	Ba = (B1 * sflon1 + B2 * sflon2) / som
	Bb = (B3 * sflon1 + B4 * sflon2) / som
	Bc = (B5 * sflon1 + B6 * sflon2) / som
	Bd = (B7 * sflon1 + B8 * sflon2) / som

	# lat slerps
	sflat1 = np.zeros([Npoints, 4])
	sflat2 = np.zeros([Npoints, 4])
	for i in range(4):
		sflat1[:,i] = np.sin((1.-flat)*om) 
		sflat2[:,i] = np.sin(flat*om)
	Baa = (Bb * sflat1 + Ba * sflat2) / som
	Bbb = (Bd * sflat1 + Bc * sflat2) / som
	
	# radial linterp
	fR = ((R-1.)/dR - Rids)
	fR1 = np.zeros([Npoints,4])
	for i in range(4):
		fR1[:,i] = fR
	Bmag = (1-fR1) * Baa + fR1*Bbb
	return Bmag


	# FORCES CURRENTLY SET TO OUTPUT B IN SPOT 0

def calc_torque():
	# call torque calculation kernel
	GPU_calctorque(CMEposSPH_gpu, CMEposxyz_gpu, tangs_gpu, pangs_gpu, shape_gpu, RLLT_gpu, forces_gpu, tottor_gpu, block=(CC.Npol, 1, 1), grid=(CC.Ntor,1,1))
	tottor = tottor_gpu.get() * rsun 
	# more NaN cleaning
	tottor[np.where(np.isnan(tottor) == True)] = 0.
	return np.sum(tottor)
	
def clear_GPU():

	# allocate memory on the GPU
	(free,total)=cuda.mem_get_info()
	#print("Global memory occupancy:%f%% free"%(free*100/total))
	global CMEposxyz_gpu, CMEposSPH_gpu, forces_gpu
	global tangs_gpu, pangs_gpu, shape_gpu
	global shape_gpu, RLLT_gpu
	global BL_gpu
	global R_offset_gpu
	global vsw_gpu
	global rotrate_gpu
	global RsR_gpu 
	global RSS_gpu 
	del CMEposxyz_gpu
	CMEposSPH_gpu.gpudata.free() 
	forces_gpu.gpudata.free()
	tangs_gpu.gpudata.free()
	pangs_gpu.gpudata.free()
	shape_gpu.gpudata.free()
	RLLT_gpu.gpudata.free()
	BL_gpu.gpudata.free()
	R_offset_gpu.gpudata.free()
	del vsw_gpu
	del rotrate_gpu
	del RsR_gpu
	del RSS_gpu
	(free,total)=cuda.mem_get_info()
	#print("Global memory occupancy:%f%% free"%(free*100/total))

