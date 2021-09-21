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

nx,ny,nz = (100,100,1)

number_of_iterations = 2000

NG = 0 #ghost zones

x = np.linspace(-1,1,nx)
y = np.linspace(-1,1,ny)
z = np.linspace(-1,1,nz)

dx = np.diff(x)[0]

X,Y,Z = np.meshgrid(x,y,z,indexing='ij')

Bx = 2.0*pi * np.sin(2.0*pi*Y)
By = 2.0*pi * np.cos(2.0*pi*X)
Bz = 0.0*X

Bx = 2.0*np.exp(-(X**2 + Y**2)*5) * (5.0*Y*np.cos(2.0*pi*X) + pi * np.sin(2.0*pi*Y) + 5.0*Y*np.cos(2.0*pi*Y))
By = -2.0*np.exp(-(X**2 + Y**2)*5) * (5.0*X*np.cos(2.0*pi*Y) + pi * np.sin(2.0*pi*X) + 5.0*X*np.cos(2.0*pi*X))
Bz = 0.0*X

# Bx =  2.0  * Y
# By = -2.0  * X
# Bz = 0*X

#A_an = X**2 + Y**2 
A_an = (-np.cos(2.0*pi*Y) - np.cos(2.0*pi*X))* np.exp(-(X**2 + Y**2 )*5)
def curl(a):
  x_tmp = x
  y_tmp = y
  z_tmp = z
  if (nz>2): return [ gradient(a[2],y_tmp,axis=1) - gradient(a[1],z_tmp,axis=2), gradient(a[0],z_tmp,axis=2) - gradient(a[2],x_tmp,axis=0), gradient(a[1],x_tmp,axis=0) - gradient(a[0],y_tmp,axis=1)]
  else: return [ gradient(a[2],y_tmp,axis=1) , - gradient(a[2],x_tmp,axis=0), gradient(a[1],x_tmp,axis=0) - gradient(a[0],y_tmp,axis=1)]

curlb = np.array(curl([Bx,By,Bz]))


A1 = X*0
A2 = X*0
A3 = X*0
rho_err = X*0

boundary = 'no_gradient'

def apply_boundary_conditions(A,type = 'outflow'):
	## X Boundary Condition **

	if (type == 'outflow'):
		g=2
		# for j in arange(0,ny):
		# 	for k in arange(0,nz):
		# 		A[1,j,k] = A[2,j,k]
		# 		A[0,j,k] = A[2,j,k]
		# 		A[nx-2,j,k] = A[nx-3,j,k]
		# 		A[nx-1,j,k] = A[nx-3,j,k]

		# ## Y Boundary Condition **

		# for i in arange(0,nx):
		# 	for k in arange(0,nz):
		# 		A[i,0,k] = A[i,2,k]
		# 		A[i,1,k] = A[i,2,k]
		# 		A[i,ny-2,k] = A[i,ny-3,k]
		# 		A[i,ny-1,k] = A[i,ny-3,k]



		# ## Z Boundary Condition **

		# if (nz > 2):
		# 	for i in arange(0,nx):
		# 		for j in arange(0,ny):
		# 			A[i,j,0] = A[i,j,2]ki
		# 		A[nx-2,j,k] = 2.0* A[nx-3,j,k] - A[nx-4,j,k]
		# 		A[nx-1,j,k] = 2.0* A[nx-2,j,k] - A[nx-3,j,k]
		# ## Y Boundary Condition **

		# for i in arange(0,nx):
		# 	for k in arange(0,nz):
		# 		A[i,1,k] = 2.0 * A[i,2,k] - A[i,3,k]
		# 		A[i,0,k] = 2.0 * A[i,1,k] - A[i,2,k]
		# 		A[i,ny-2,k] = 2.0 * A[i,ny-3,k] - A[i,ny-4,k]
		# 		A[i,ny-1,k] = 2.0 * A[i,ny-2,k] - A[i,ny-3,k]


	elif (type== "quadratic_gradient"):
		#third derivative = 0 
		# d/dx^3 f = -0.5 * f_i-2 + f_i-1 - f_i+1 + 0.5 f_i+2/ dx^3
		# f_i-2 = 2 f_i-1 - 2 f_i+1 + f_i+2
		# f_i+2 = f_i-2  - 2 f_i-1 + 2 f_i+1
		for j in arange(0,ny):
			for k in arange(0,nz):
				A[1,j,k] = 2.0 * A[2,j,k] - 2.0*A[4,j,k] + A[5,j,k]
				A[0,j,k] = 2.0 * A[1,j,k] - 2.0*A[3,j,k] + A[4,j,k]
				A[nx-2,j,k] = 2.0* A[nx-3,j,k] - 2.0 * A[nx-5,j,k] + A[nx-6,j,k]
				A[nx-1,j,k] = 2.0* A[nx-2,j,k] - 2.0 * A[nx-4,j,k] + A[nx-5,j,k]
		## Y Boundary Condition **

		for i in arange(0,nx):
			for k in arange(0,nz):
				A[i,1,k] = 2.0 * A[i,2,k] - 2.0*A[i,4,k] + A[i,5,k]
				A[i,0,k] = 2.0 * A[i,1,k] - 2.0*A[i,3,k] + A[i,4,k]
				A[i,ny-2,k] = 2.0* A[i,ny-3,k] - 2.0 * A[i,ny-5,k] + A[i,ny-6,k]
				A[i,ny-1,k] = 2.0* A[i,ny-2,k] - 2.0 * A[i,ny-4,k] + A[i,ny-5,k]
	else:
		for j in arange(0,ny):
			for k in arange(0,nz):
				A[1,j,k] = A_an[1,j,k]
				A[0,j,k] = A_an[0,j,k]
				A[nx-2,j,k] = A_an[nx-2,j,k]
				A[nx-1,j,k] = A_an[nx-1,j,k]

		## Y Boundary Condition **

		for i in arange(0,nx):
			for k in arange(0,nz):
				A[i,0,k] = A_an[i,0,k]
				A[i,1,k] = A_an[i,1,k]
				A[i,ny-2,k] = A_an[i,ny-2,k]
				A[i,ny-1,k] = A_an[i,ny-1,k]



		## Z Boundary Condition **

		if (nz > 2):
			for i in arange(0,nx):
				for j in arange(0,ny):
					A[i,j,0] = A_an[i,j,0]
					A[i,j,1] = A_an[i,j,1]
					A[i,j,nz-2] = A_an[i,j,nz-2]
					A[i,j,nz-1] = A_an[i,j,nz-1]


# for i in arange(0,nx):
# 	for j in arange(0,ny):
# 		for k in arange(0,nz):
# 			if ((i<NG or i>nx-NG-1) or (j<NG or j>ny-NG-1) or (k<NG or k>ny-NG-1) ): 
# 				A3[i,j,k] = A_an[i,j,k]
			

Astar1 = np.zeros((nx,ny,nz)) #A1*1.0 + 0.0
Astar2 = np.zeros((nx,ny,nz)) #A2*1.0 + 0.0
Astar3 = np.zeros((nx,ny,nz)) #A3*1.0 + 0.0
omega = 1.5


for iter in arange(number_of_iterations):
	print (iter)
	for i in arange(NG,nx-NG):
		for j in arange(NG,ny-NG):
			if (nz>2):
				for k in arange(NG,nz-NG):
					Astar1[i,j,k] = A1[i,j,k] * (1-omega) + omega/6.0 * ( A1[i+1,j,k] + A1[i-1,j,k] + A1[i,j+1,k] + A1[i,j-1,k] + A1[i,j,k+1] + A1[i,j,k-1] + 6.0*curlb[0,i,j,k] * dx**2.0 )
					Astar2[i,j,k] = A2[i,j,k] * (1-omega) + omega/6.0 * ( A2[i+1,j,k] + A2[i-1,j,k] + A2[i,j+1,k] + A2[i,j-1,k] + A2[i,j,k+1] + A2[i,j,k-1] + 6.0*curlb[1,i,j,k] * dx**2.0 )
					Astar3[i,j,k] = A3[i,j,k] * (1-omega) + omega/6.0 * ( A3[i+1,j,k] + A3[i-1,j,k] + A3[i,j+1,k] + A3[i,j-1,k] + A3[i,j,k+1] + A3[i,j,k-1] + 6.0*curlb[2,i,j,k] * dx**2.0 )
			
			else: 
				if (boundary == 'no_gradient'):
					if (i==NG and j==NG):
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A1[i+1,j,0] + A1[i,j+1,0]  + curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A2[i+1,j,0] + A2[i,j+1,0]  + curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A3[i+1,j,0] + A3[i,j+1,0]  + curlb[2,i,j,0] * dx**2.0 )
					elif (i==NG and j==ny-NG-1):
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A1[i+1,j,0] + A1[i,j-1,0]  + curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A2[i+1,j,0] + A2[i,j-1,0]  + curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A3[i+1,j,0] + A3[i,j-1,0]  + curlb[2,i,j,0] * dx**2.0 )
					elif (i==nx-NG-1 and j==NG):
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A1[i-1,j,0] + A1[i,j+1,0]  + curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A2[i-1,j,0] + A2[i,j+1,0]  + curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A3[i-1,j,0] + A3[i,j+1,0]  + curlb[2,i,j,0] * dx**2.0 )
					elif (i==nx-NG-1 and j==ny-NG-1):
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A1[i-1,j,0] + A1[i,j-1,0]  + curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A2[i-1,j,0] + A2[i,j-1,0]  + curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A3[i-1,j,0] + A3[i,j-1,0]  + curlb[2,i,j,0] * dx**2.0 )
					elif (i==NG):				
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A1[i+1,j,0] + A1[i,j+1,0] + A1[i,j-1,0]  + curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A2[i+1,j,0] + A2[i,j+1,0] + A2[i,j-1,0]  + curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A3[i+1,j,0] + A3[i,j+1,0] + A3[i,j-1,0]  + curlb[2,i,j,0] * dx**2.0 )
					elif (j==NG):
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A1[i+1,j,0] + A1[i-1,j,0] + A1[i,j+1,0]  + curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A2[i+1,j,0] + A2[i-1,j,0] + A2[i,j+1,0]  + curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A3[i+1,j,0] + A3[i-1,j,0] + A3[i,j+1,0]  + curlb[2,i,j,0] * dx**2.0 )	
					elif (i==nx-NG-1):
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A1[i-1,j,0] + A1[i,j+1,0] + A1[i,j-1,0]  + curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A2[i-1,j,0] + A2[i,j+1,0] + A2[i,j-1,0]  + curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A3[i-1,j,0] + A3[i,j+1,0] + A3[i,j-1,0]  + curlb[2,i,j,0] * dx**2.0 )
					elif (j==ny-NG-1):
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A1[i+1,j,0] + A1[i-1,j,0] + A1[i,j-1,0]  + curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A2[i+1,j,0] + A2[i-1,j,0] + A2[i,j-1,0]  + curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-omega) + omega/(4.0-omega) * ( A3[i+1,j,0] + A3[i-1,j,0] + A3[i,j-1,0]  + curlb[2,i,j,0] * dx**2.0 )			
					else:
						A1[i,j,0] = A1[i,j,0] * (1-omega) + omega/4.0 * ( A1[i+1,j,0] + A1[i-1,j,0] + A1[i,j+1,0] + A1[i,j-1,0]  + curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) + omega/4.0 * ( A2[i+1,j,0] + A2[i-1,j,0] + A2[i,j+1,0] + A2[i,j-1,0]  + curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) + omega/4.0 * ( A3[i+1,j,0] + A3[i-1,j,0] + A3[i,j+1,0] + A3[i,j-1,0]  + curlb[2,i,j,0] * dx**2.0 )
				elif (boundary == "linear_gradient"):
					if (i==NG and j==NG):  #corner 
						#1 - 2 omega/(4-2*omega)  = (4-2 omega - 4)
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A1[i,j+1,0] + A1[i,j-1,0]  + 4.0*curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A2[i,j+1,0] + A2[i,j-1,0]  + 4.0*curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A3[i,j+1,0] + A3[i,j-1,0]  + 4.0*curlb[2,i,j,0] * dx**2.0 )

					if (i==NG):				
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A1[i,j+1,0] + A1[i,j-1,0]  + 4.0*curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A2[i,j+1,0] + A2[i,j-1,0]  + 4.0*curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A3[i,j+1,0] + A3[i,j-1,0]  + 4.0*curlb[2,i,j,0] * dx**2.0 )
					elif (j==NG):
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A1[i+1,j,0] + A1[i-1,j,0] + 4.0*curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A2[i+1,j,0] + A2[i-1,j,0] + 4.0*curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A3[i+1,j,0] + A3[i-1,j,0] + 4.0*curlb[2,i,j,0] * dx**2.0 )	
					elif (i==nx-NG-1):
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A1[i,j+1,0] + A1[i,j-1,0]  + 4.0*curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A2[i,j+1,0] + A2[i,j-1,0]  + 4.0*curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A3[i,j+1,0] + A3[i,j-1,0]  + 4.0*curlb[2,i,j,0] * dx**2.0 )
					elif (j==ny-NG-1):
						A1[i,j,0] = A1[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A1[i+1,j,0] + A1[i-1,j,0] + 4.0*curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A2[i+1,j,0] + A2[i-1,j,0] + 4.0*curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) * 4.0/(4.0-2.0*omega) + omega/(4.0-2.0*omega) * ( A3[i+1,j,0] + A3[i-1,j,0] + 4.0*curlb[2,i,j,0] * dx**2.0 )			
					else:
						A1[i,j,0] = A1[i,j,0] * (1-omega) + omega/4.0 * ( A1[i+1,j,0] + A1[i-1,j,0] + A1[i,j+1,0] + A1[i,j-1,0]  + 4.0*curlb[0,i,j,0] * dx**2.0 )
						A2[i,j,0] = A2[i,j,0] * (1-omega) + omega/4.0 * ( A2[i+1,j,0] + A2[i-1,j,0] + A2[i,j+1,0] + A2[i,j-1,0]  + 4.0*curlb[1,i,j,0] * dx**2.0 )
						A3[i,j,0] = A3[i,j,0] * (1-omega) + omega/4.0 * ( A3[i+1,j,0] + A3[i-1,j,0] + A3[i,j+1,0] + A3[i,j-1,0]  + 4.0*curlb[2,i,j,0] * dx**2.0 )
				
	if (iter>1): max_diff = np.mean(abs(A3_prev-A3))
	if (iter>1): print (max_diff.mean())

	# for i in arange(NG+1,nx-NG-1):
	# 	for j in arange(NG+1,ny-NG-1):
	# 		rho_err[i,j,0] = abs(A3[i+1,j,0] + A3[i-1,j,0] + A3[i,j+1,0] + A3[i,j-1,0]  - 4.0*A3[i,j,0] + 4.0*curlb[0,i,j,0] * dx**2.0)

	#if (iter>1): print (rho_err.mean())
	# plt.clf()
	# plt.plot(X[:,ny//2,0])
	# plt.show()

	#print(A3[:,ny//2,0])

	A1_prev = A1*1.0
	A2_prev = A2*1.0 
	A3_prev = A3*1.0

	# apply_boundary_conditions(A1,type = 'outflow')
	# apply_boundary_conditions(A2,type = 'outflow')
	# apply_boundary_conditions(A3,type = 'quadratic_gradient')


B_check = curl([A1,A2,A3])

