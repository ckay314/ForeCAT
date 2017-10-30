from pylab import *
import matplotlib.pyplot  as pyplot
import pickle
import numpy as np
from matplotlib import gridspec

matplotlib.rcParams.update({'font.size':12})

idstr='07'
data = pickle.load(open('ensemble_CME'+idstr+'.pkl', 'rb'))
data2 = pickle.load(open('ensemble_CME'+idstr+'.pkl', 'rb'))
data3 = np.genfromtxt(open('CME'+idstr+'.dat'))

data[:,:,2][np.where(data[:,:,2] > 90)] = 360 - data[:,:,2][np.where(data[:,:,2] > 90)]
data2[:,:,2][np.where(data2[:,:,2] > 90)] = 360 - data2[:,:,2][np.where(data2[:,:,2] > 90)]
data3[:,-2][np.where(data3[:,-2] > 90)] = 360 - data3[:,-2][np.where(data3[:,-2] > 90)]
data[:,:,2] = 90. - data[:,:,2]
data2[:,:,2] = 90. - data2[:,:,2]
data3[:,-2] = 90. - data3[:,-2]

fig = figure()
gs = gridspec.GridSpec(3,2)
ax1a = fig.add_subplot(gs[0])
ax2a = fig.add_subplot(gs[2], sharex=ax1a)
ax3a = fig.add_subplot(gs[4], sharex=ax1a)
ax1b = fig.add_subplot(gs[1], sharey=ax1a)
ax2b = fig.add_subplot(gs[3], sharey=ax2a, sharex=ax1b)
ax3b = fig.add_subplot(gs[5], sharey=ax3a, sharex=ax1b)

setp(ax1b.get_yticklabels(), visible=False)
setp(ax2b.get_yticklabels(), visible=False)
setp(ax3b.get_yticklabels(), visible=False)
setp(ax1a.get_xticklabels(), visible=False)
setp(ax1b.get_xticklabels(), visible=False)
setp(ax2a.get_xticklabels(), visible=False)
setp(ax2b.get_xticklabels(), visible=False)

degree = unichr(176)
ax1a.set_title('Synoptic')
ax1b.set_title('Synchronic')
ax1a.set_ylabel('Latitude ('+degree+')')
ax2a.set_ylabel('Longitude ('+degree+')')
ax3a.set_ylabel('Tilt ('+degree+')')
ax3a.set_xlabel('Distance (R$_S$)')
ax3b.set_xlabel('Distance (R$_S$)')


Nsims =  data.shape[0]
Rs = [1.15, 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11., 11.5, 12., 12.5, 13., 13.5, 14., 14.5, 15., 15.5, 16., 16.5, 17., 17.5, 18., 18.5, 19., 19.5, 20.]

lat_stda = np.zeros(39)
lon_stda = np.zeros(39)
tilt_stda = np.zeros(39)
lat_stdb = np.zeros(39)
lon_stdb = np.zeros(39)
tilt_stdb = np.zeros(39)
lat_mina = np.zeros(39)
lon_mina = np.zeros(39)
tilt_mina = np.zeros(39)
lat_minb = np.zeros(39)
lon_minb = np.zeros(39)
tilt_minb = np.zeros(39)
lat_maxa = np.zeros(39)
lon_maxa = np.zeros(39)
tilt_maxa = np.zeros(39)
lat_maxb = np.zeros(39)
lon_maxb = np.zeros(39)
tilt_maxb = np.zeros(39)
lat_avga = np.zeros(39)
lon_avga = np.zeros(39)
tilt_avga = np.zeros(39)
lat_avgb = np.zeros(39)
lon_avgb = np.zeros(39)
tilt_avgb = np.zeros(39)
lat_meda = np.zeros(39)
lon_meda = np.zeros(39)
tilt_meda = np.zeros(39)
lat_medb = np.zeros(39)
lon_medb = np.zeros(39)
tilt_medb = np.zeros(39)

for i in range(39):
	lat_stda[i] = np.std(data[:,i,0])
	lon_stda[i] = np.std(data[:,i,1])
	tilt_stda[i] = np.std(data[:,i,2])
	lat_stdb[i] = np.std(data2[:,i,0])
	lon_stdb[i] = np.std(data2[:,i,1])
	tilt_stdb[i] = np.std(data2[:,i,2])
	lat_mina[i] = np.min(data[:,i,0])
	lon_mina[i] = np.min(data[:,i,1])
	tilt_mina[i] = np.min(data[:,i,2])
	lat_minb[i] = np.min(data2[:,i,0])
	lon_minb[i] = np.min(data2[:,i,1])
	tilt_minb[i] = np.min(data2[:,i,2])
	lat_maxa[i] = np.max(data[:,i,0])
	lon_maxa[i] = np.max(data[:,i,1])
	tilt_maxa[i] = np.max(data[:,i,2])
	lat_maxb[i] = np.max(data2[:,i,0])
	lon_maxb[i] = np.max(data2[:,i,1])
	tilt_maxb[i] = np.max(data2[:,i,2])
	lat_avga[i] = np.mean(data[:,i,0])
	lon_avga[i] = np.mean(data[:,i,1])
	tilt_avga[i] = np.mean(data[:,i,2])
	lat_avgb[i] = np.mean(data2[:,i,0])
	lon_avgb[i] = np.mean(data2[:,i,1])
	tilt_avgb[i] = np.mean(data2[:,i,2])
	lat_meda[i] = np.median(data[:,i,0])
	lon_meda[i] = np.median(data[:,i,1])
	tilt_meda[i] = np.median(data[:,i,2])
	lat_medb[i] = np.median(data2[:,i,0])
	lon_medb[i] = np.median(data2[:,i,1])
	tilt_medb[i] = np.median(data2[:,i,2])


for i in range(Nsims):
	# the center LLT simulation (starting point)
	ax1a.fill_between(Rs, lat_mina, lat_maxa, color='Silver')
	ax2a.fill_between(Rs, lon_mina, lon_maxa, color='Silver')
	ax3a.fill_between(Rs, tilt_mina, tilt_maxa, color='Silver')
	ax1b.fill_between(Rs, lat_minb, lat_maxb, color='Silver')
	ax2b.fill_between(Rs, lon_minb, lon_maxb, color='Silver')
	ax3b.fill_between(Rs, tilt_minb, tilt_maxb, color='Silver')
	ax1a.fill_between(Rs, lat_avga-lat_stda, lat_avga+lat_stda, color='Gray')
	ax2a.fill_between(Rs, lon_avga-lon_stda, lon_avga+lon_stda, color='Gray')
	ax3a.fill_between(Rs, tilt_avga-tilt_stda, tilt_avga+tilt_stda, color='Gray')
	ax1b.fill_between(Rs, lat_avgb-lat_stdb, lat_avgb+lat_stdb, color='Gray')
	ax2b.fill_between(Rs, lon_avgb-lon_stdb, lon_avgb+lon_stdb, color='Gray')
	ax3b.fill_between(Rs, tilt_avgb-tilt_stdb, tilt_avgb+tilt_stdb, color='Gray')
	ax1a.plot(Rs, lat_avga, 'k--', linewidth=2.5)
	ax2a.plot(Rs, lon_avga, 'k--', linewidth=2.5)
	ax3a.plot(Rs, tilt_avga, 'k--', linewidth=2.5)
	ax1b.plot(Rs, lat_avgb, 'k--', linewidth=2.5)
	ax2b.plot(Rs, lon_avgb, 'k--', linewidth=2.5)
	ax3b.plot(Rs, tilt_avgb, 'k--', linewidth=2.5)
	ax1a.plot(Rs, lat_meda, 'k', linewidth=2.5)
	ax2a.plot(Rs, lon_meda, 'k', linewidth=2.5)
	ax3a.plot(Rs, tilt_meda, 'k', linewidth=2.5)
	ax1b.plot(Rs, lat_medb, 'k', linewidth=2.5)
	ax2b.plot(Rs, lon_medb, 'k', linewidth=2.5)
	ax3b.plot(Rs, tilt_medb, 'k', linewidth=2.5)
	ax1a.plot(Rs, data[0,:,0], color='b', linewidth=2.5)
	ax2a.plot(Rs, data[0,:,1], color='b', linewidth=2.5)
	ax3a.plot(Rs, data[0,:,2], color='b', linewidth=2.5)
	ax1b.plot(data3[:,1], data3[:,2], color='#88CCEE', linewidth=2.5)
	ax2b.plot(data3[:,1], data3[:,3], color='#88CCEE', linewidth=2.5)
	ax3b.plot(data3[:,1], data3[:,-2], color='#88CCEE', linewidth=2.5)
	ax1b.plot(Rs, data[0,:,0], color='b', linewidth=2.5)
	ax2b.plot(Rs, data[0,:,1], color='b', linewidth=2.5)
	ax3b.plot(Rs, data[0,:,2], color='b', linewidth=2.5)
	ax1b.plot(Rs, data2[0,:,0], color='r', linewidth=2.5)
	ax2b.plot(Rs, data2[0,:,1], color='r', linewidth=2.5)
	ax3b.plot(Rs, data2[0,:,2], color='r', linewidth=2.5)


setp(xticks)

subplots_adjust(hspace=0.001, wspace=0.001)
savefig('sig_ensemble_CME'+idstr+'.png')

