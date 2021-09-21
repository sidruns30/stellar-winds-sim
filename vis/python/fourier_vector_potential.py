"""
Read Athena++ output data files.
"""

# Python modules

import numpy as np
from numpy import *
import glob
import os
import sys
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

#=========================================================================================

nx,ny,nz = (256,256,256)

L = 2.0
box_radius = L/2.0
i_dump = 120

yt_load(i_dump)
region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,(-box_radius,'pc'):(box_radius,'pc'):256j,
  (-box_radius,'pc'):(box_radius,'pc'):256j]

x = np.array(region['x'])
y = np.array(region['y'])
z = np.array(region['z'])

Bcc1 = np.array(region['Bcc1'])
Bcc2 = np.array(region['Bcc2'])
Bcc3 = np.array(region['Bcc3'])

# Bcc1 = np.sin(2.0*pi*y)
# Bcc2 = np.cos(2.0*pi*z)
# Bcc3 = np.sin(2.0*pi*x)

kx = arange(0,nx)*np.pi/L
ky = arange(0,nx)*np.pi/L
kz = arange(0,nx)*np.pi/L



KX,KY,KZ = np.meshgrid(kx,ky,kz,indexing='ij')

Bx_tilde = np.fft.fftn(Bcc1)
By_tilde = np.fft.fftn(Bcc2)
Bz_tilde = np.fft.fftn(Bcc3)

Ksq = KX**2 + KY**2 + KZ**2

khat = [KX/sqrt(Ksq+1e-20),KY/sqrt(Ksq+1e-20),KZ/sqrt(Ksq+1e-20)]

khatdotb = dot_product([Bx_tilde,By_tilde,Bz_tilde],khat)

B_par = [khatdotb*khat[0], khatdotb*khat[1],khatdotb*khat[2] ]

B_perp = [Bx_tilde - B_par[0], By_tilde - B_par[1], Bz_tilde - B_par[2]]

B_new = [np.fft.ifftn(B_perp[0]).real,np.fft.ifftn(B_perp[1]).real,np.fft.ifftn(B_perp[2]).real]
bsq_new = B_new[0]**2 + B_new[1]**2 + B_new[2]**2

KxB = cross_product([KX,KY,KZ],[Bx_tilde,By_tilde,Bz_tilde])

#Ax_tilde = 1j * KxB[0]/(Ksq+1e-20) * -1j
#Ay_tilde = 1j * KxB[1]/(Ksq+1e-20) * -1j
#Az_tilde = 1j * KxB[2]/(Ksq+1e-20) * -1j


curlB = curl([Bcc1,Bcc2,Bcc3])

curlBfft_x  = np.fft.fftn(curlB[0])
curlBfft_y  = np.fft.fftn(curlB[1])
curlBfft_z  = np.fft.fftn(curlB[2])

Ax_tilde = curlBfft_x /(Ksq+1e-20) #1j * KxB[0]/(Ksq+1e-20)
Ay_tilde = curlBfft_y /(Ksq+1e-20)  #1j * KxB[1]/(Ksq+1e-20)
Az_tilde = curlBfft_z /(Ksq+1e-20) #1j * KxB[2]/(Ksq+1e-20)



# Ax_tilde = 1j * KxB[0]/(Ksq+1e-20)
# Ay_tilde = 1j * KxB[1]/(Ksq+1e-20)
# Az_tilde = 1j * KxB[2]/(Ksq+1e-20)
>>>>>>> 899ba23594cf0536ad2987e1f57f0a74b88c0cf0

Ax = np.fft.ifftn(Ax_tilde).real
Ay = np.fft.ifftn(Ay_tilde).real
Az = np.fft.ifftn(Az_tilde).real



B_check = curl([Ax,Ay,Az])


k x A = B
