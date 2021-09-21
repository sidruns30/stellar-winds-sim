import matplotlib
#from matplotlib import rc
from matplotlib.patches import Ellipse
import re
#Uncomment the following if you want to use LaTeX in figures
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('mathtext',fontset='cm')
#rc('mathtext',rm='stix')
#rc('text', usetex=True)
# #add amsmath to the preamble
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"]

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import scipy
from scipy.optimize import fsolve
import sys
from scipy.optimize import brentq
from scipy.optimize import minimize
import random


class Star:
    name = ''
    eccentricity = 0.
    mean_angular_motion = 0.
    alpha = 0.
    beta = 0.
    tau = 0.
    gamma = 0.
    x1 = 0.
    x2 = 0.
    x3 = 0.
    v1 = 0.
    v2 = 0.
    v3 = 0.
    Mdot = 0.
    vwind = 0.
    orbit_array = 0.
    a = 0.
    period = 0.
    is_in_disk = False

def cross_product(a,b):
    return [ a[1]*b[2] - b[1]*a[2], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] ]
def matrix_vec_mult(A,b):
    result = [0,0,0]
    for i in range(3):
        for j in range(3):
            result[i] += A[i,j]*b[j]

    return result

def transpose(A):
    result = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            result[i,j] = A[j,i]
    return result
def get_rotation_matrix(alpha,beta,gamma=0):
    X_rot = np.zeros((3,3))
    Z_rot = np.zeros((3,3))
    Z_rot2 = np.zeros((3,3))
    rot = np.zeros((3,3))
    rot_tmp = np.zeros((3,3))


    Z_rot2[0,0] = np.cos(gamma)
    Z_rot2[0,1] = -np.sin(gamma)
    Z_rot2[0,2] = 0.
    Z_rot2[1,0] = np.sin(gamma)
    Z_rot2[1,1] = np.cos(gamma)
    Z_rot2[1,2] = 0.
    Z_rot2[2,0] = 0.
    Z_rot2[2,1] = 0.
    Z_rot2[2,2] = 1.

    X_rot[0,0] = 1.
    X_rot[0,1] = 0.
    X_rot[0,2] = 0.
    X_rot[1,0] = 0.
    X_rot[1,1] = np.cos(beta)
    X_rot[1,2] = -np.sin(beta)
    X_rot[2,0] = 0.
    X_rot[2,1] = np.sin(beta)
    X_rot[2,2] = np.cos(beta)

    Z_rot[0,0] = np.cos(alpha)
    Z_rot[0,1] = -np.sin(alpha)
    Z_rot[0,2] = 0.
    Z_rot[1,0] = np.sin(alpha)
    Z_rot[1,1] = np.cos(alpha)
    Z_rot[1,2] = 0.
    Z_rot[2,0] = 0.
    Z_rot[2,1] = 0.
    Z_rot[2,2] = 1.

    for i in range(3):
        for j in range(3):
            for k in range(3):
                rot_tmp[i,j] += X_rot[i,k] * Z_rot[k,j]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                rot[i,j] += Z_rot2[i,k] * rot_tmp[k,j]


    return rot

def convert_E_to_S_notation(Ename):
    E_list = ['E7', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21',
       'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30',
       'E31', 'E32', 'E33', 'E34', 'E35', 'E36','E0']
    S_list = ['S31', 'S26', 'S65', 'S67', 'S83', 'S66', 'S72', 'S95', 'S96 ',
       'S87 ', 'R14 ', 'S97', 'S140', 'R15', 'R26', 'R36', 'R28', 'R1',
       'R29', 'R19', 'R30', 'R25', 'R27', 'R20', 'R45', 'S2']

    if (Ename in E_list):
        index = E_list.index(Ename)
        return S_list[index]
    else:
        return ""


def get_euler_angles(x1,x2,x3,star):
    r_vec = np.array([x1,x2,x3])
    v_vec = np.array([v1,v2,v3])
    h_vec = cross_product(r_vec,v_vec)
    ecc_vec = np.array(cross_product(v_vec,h_vec))/gm_ - r_vec/np.linalg.norm(r_vec)
    ecc_lu = np.linalg.norm(ecc_vec)
    
    a_lu = 1./(2./np.linalg.norm(r_vec) - np.linalg.norm(v_vec)**2./gm_)
    
    new_zhat = cross_product([x1,x2,x3],[v1,v2,v3])
    
    #new_zhat = [1.,-1.,0.]
    new_zhat = np.array(new_zhat)/np.sqrt(new_zhat[0]**2. + new_zhat[1]**2. + new_zhat[2]**2.)
    z1 = new_zhat[0]
    z2 = new_zhat[1]
    z3 = new_zhat[2]
    beta = np.arccos(z3)
    #beta = pi - i  in the notation of lu?????
    #n = (r x v) x z
    
    z2 -z1
    n1 = z2 / sqrt(z1**2.+z2**2.)
    n2 = -z1 / np.sqrt(z1**2. + z2**2.)
    n3 = 0
    alpha = -np.arctan2(n2,n1)   #(-z1/(np.sqrt(1.-z3**2.)+1e-20 ))
    
    Omega_lu = np.arctan2(n1,n2)
    #alpha = Omega - pi/2.in notation of Lu
    
    rotation_matrix = get_rotation_matrix(alpha,beta)
    
    ecc_rotated = matrix_vec_mult(rotation_matrix,ecc_vec)
    gamma = - np.arctan2(ecc_rotated[1],ecc_rotated[0])
    #gamma = pi - omega in notation of Lu
    
    omega_lu = np.arccos(-(n1*ecc_vec[0] + n2*ecc_vec[1])/(np.sqrt(n1**2.+n2**2.)* np.sqrt(ecc_vec[0]**2.+ecc_vec[1]**2.+ecc_vec[2]**2.) ))
    if (ecc_vec[2]<0):
        omega_lu = np.pi*2. - omega_lu

    star.alpha = alpha
    star.beta = beta
    star.gamma = gamma

#    print "alpha, beta, gamma", alpha, beta, gamma
#    print "Omega, i, omega", Omega,i, omega_lu

#given r and v, get orbital parameters assuming that we have already rotated
#into the plane of the disk
def get_orbital_parameters(x1,x2,v1,v2,star):
    r_0 = np.sqrt(x1**2. + x2**2.)
    phi_0 = np.arctan2(x2,x1)
    
    v_r_star = v1 * np.cos(phi_0) + v2 * np.sin(phi_0);
    v_t_star = v1 * - np.sin(phi_0) + v2 * np.cos(phi_0);
    mu = gm_
    p_star = (r_0 * v_t_star)**2./mu;
    V_0 = sqrt(mu/p_star);
    
    eccentricity = np.sqrt((v_t_star/V_0 -1.)**2. + (v_r_star/V_0)**2. );
    
    true_anomaly_0 = np.arctan2(v_r_star/V_0,v_t_star/V_0 -1.)
    if eccentricity <1 :
        a = p_star/(1. - (eccentricity)**2.)
        b = a * np.sqrt(1. - eccentricity**2. )
    else:
        a = p_star/(eccentricity**2.-1.)
        b = a * np.sqrt(eccentricity**2.-1.)

    mean_angular_motion = np.sqrt(mu/(a*a*a));
    period = np.sqrt(4.*np.pi**2. / (gm_) * a**3.)

    if eccentricity <1 :
        eccentric_anomaly_0 = np.arctan2(np.sqrt(1.-eccentricity**2.)*np.sin(true_anomaly_0),eccentricity + np.cos(true_anomaly_0))
        eccentric_anomaly_0_alt = np.arctan2(x2/b,x1/a+eccentricity)
        mean_anomaly_0 = eccentric_anomaly_0 - eccentricity * np.sin(eccentric_anomaly_0)
    else:
        tmp = np.tan(true_anomaly_0/2.) * np.sqrt(eccentricity-1.)/np.sqrt(eccentricity+1.)
        eccentric_anomaly_0 = np.log((1.+tmp)/(1.-tmp))
        mean_anomaly_0 = eccentricity * np.sinh(eccentric_anomaly_0) - eccentric_anomaly_0


    #mean_anomaly = mean_angular_motion * (t - tau)   t = -simulation_start_time
    tau =  -simulation_start_time - mean_anomaly_0/mean_angular_motion  #t_paumard
    star.eccentricity = eccentricity
    star.mean_angular_motion = mean_angular_motion
    star.tau = tau
    star.a = a
    star.period = period


#given orbital parameters, evolve orbit
def get_orbit(star,t_vals):
    period = 2.*np.pi/star.mean_angular_motion
    a = (gm_/star.mean_angular_motion**2.)**(1./3.)
    
    if star.eccentricity <1 :
        b = a * np.sqrt(1. - star.eccentricity**2. )
    else:
        b = a * np.sqrt(star.eccentricity**2.-1.)

    def eqn(e_anomaly,m_anamoly):
        if (star.eccentricity<1):
            return m_anamoly - e_anomaly + star.eccentricity * np.sin(e_anomaly)
        else:
            return m_anamoly + e_anomaly - star.eccentricity * np.sinh(e_anomaly)

    mean_anomaly = star.mean_angular_motion * (t_vals - star.tau)

    # = mean_angular_motion * (t + simulation_start_time + mean_anomaly_0/mean_angular_motion)

    eccentric_anomaly =  fsolve(eqn,mean_anomaly,args = (mean_anomaly,))


    if (star.eccentricity<1):
        x1_t= a * (np.cos(eccentric_anomaly) - star.eccentricity)
        x2_t= b * np.sin(eccentric_anomaly)
        Edot = star.mean_angular_motion/ (1.-star.eccentricity * np.cos(eccentric_anomaly))
        v1_t = - a * np.sin(eccentric_anomaly) * Edot
        v2_t = b * np.cos(eccentric_anomaly) * Edot
    else:
        x1_t = a * ( star.eccentricity - np.cosh(eccentric_anomaly) )
        x2_t = b * np.sinh(eccentric_anomaly)
        Edot = -star.mean_angular_motion/ (1. - star.eccentricity * np.cosh(eccentric_anomaly))
        v1_t = a * (- np.sinh(eccentric_anomaly) * Edot)
        v2_t = b * np.cosh(eccentric_anomaly) * Edot

    return [x1_t,x2_t,0.], [v1_t,v2_t,0.]


def ecc_mag_func(z,x,y,vx,vy,vz):
    r_vec = np.array([x,y,z])
    v_vec = np.array([vx,vy,vz])
    h_vec = cross_product(r_vec,v_vec)
    ecc_vec = np.array(cross_product(v_vec,h_vec))/gm_ - r_vec/np.linalg.norm(r_vec)
    return np.linalg.norm(ecc_vec)
def minimum_ecc_z(x,y,vx,vy,vz):
    res = scipy.optimize.minimize(ecc_mag_func,0,args=(x,y,vx,vy,vz))
    return res['x'][0]