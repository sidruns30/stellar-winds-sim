# Python modules
import numpy as np
from numpy import *
import glob
import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

sys.path.append("/global/scratch/smressle/star_cluster/restart_grmhd/vis/python")
from athena_script import *
import athena_script as asc


def time_average(idump_min = 100,idump_max = 110,th_l=1.5,phi_l=2.7,is_magnetic = False,disk_frame=False):
    new_dic = {}
    idumps  = arange(idump_min,idump_max+1)

    #dump_name = glob.glob("dump_spher_%d_th_*.npz" %idumps[0])[0]
    if (disk_frame==False): dump_name = glob.glob("dump_spher_%d_th_%g_phi_%g*.npz" %(idumps[0],th_l,phi_l))[0]
    else: dump_name = glob.glob("dump_spher_disk_frame_%04d*.npz" %(idumps[0]))[0]
    asc.rdnpz(dump_name)
    dic = np.load(dump_name)

    for key in dic.keys():
        exec("globals()['%s_tavg'] = asc.%s - asc.%s" % (key,key,key))
    global Lx_avg, Ly_avg,Lz_avg,Lxdot_avg,Lydot_avg,Lzdot_avg,mdot_avg, vr_avg,vth_avg,vphi_avg
    global vrsq_avg, vphisq_avg,vthsq_avg,vsq_avg, Br_avg,Bth_avg,Bphi_avg,bsq_avg,Br_abs_avg,alpham_avg,alphar_avg,Fm_avg,Fr_avg,FJ_avg
    global Bth_abs_avg
    Lx_avg = 0
    Ly_avg = 0
    Lz_avg = 0
    Lxdot_avg = 0
    Lydot_avg = 0
    Lzdot_avg = 0
    Lzdot_in = 0
    Lxdot_in = 0
    Lydot_in = 0
    Ldotr_avg = 0
    Ldotth_avg = 0
    Ldotph_avg = 0
    Edotr_avg = 0
    Edotth_avg = 0
    Edotph_avg = 0
    Edot_in = 0
    Edot_avg = 0
    Edot_B_avg = 0
    Edot_B_in = 0
    mdot_avg = 0
    mdot_in = 0
    rho_tavg_in = 0
    press_tavg_in = 0
    bsq_avg_in = 0
    vr_avg = 0
    vth_avg = 0
    vphi_avg = 0
    vrsq_avg = 0
    vphisq_avg = 0
    vthsq_avg = 0
    vsq_avg = 0
    Br_avg = 0
    Bth_avg = 0
    Bphi_avg = 0
    bsq_avg = 0
    Br_abs_avg = 0
    Bth_abs_avg = 0
    Br_Bphi_avg = 0
    Bphi_msinth_avg = 0
    alpham_avg = 0
    alphar_avg = 0
    alpham_avg_2d = 0
    alphar_avg_2d = 0
    Fm_avg = 0
    Fr_avg = 0
    FJ_avg = 0
    rho_vr_avg = 0
    rho_vr_vph_sinth_avg = 0
    vphi_sinth_avg = 0
    Br_Bphi_msinth_avg = 0
    Br_Bphi_in = 0

    rho_vr_avg_2d = 0
    rho_vr_vph_sinth_avg_2d = 0
    vphi_sinth_avg_2d = 0
    Br_Bphi_msinth_avg_2d = 0

    rho_vr_vph_sinth_avg_3d = 0
    lambda_mri_th_avg_num = 0
    lambda_mri_th_avg_den = 0



######      GEOMETRY   #########
			 #    ct*cp   -sp  st*cp   vx_new    vx
    # ct*sp    cp  st*sp   vy_new  = vy
    # -st      0   ct      vz_new    vz

    # inverse 

    # ct*cp ct*sp -st  
    # -sp   cp    0
    # st*cp st*sp ct

    # x_hat_prime = [np.cos(th)*np.cos(ph),-np.sin(ph),np.sin(th)*np.cos(ph)]
    # y_hat_prime = [np.cos(th)*np.sin(ph),np.cos(ph),np.sin(th)*np.sin(ph)]
    # z_hat_prime = [-np.sin(th),0,np.cos(th)]

    # v_x_new * x_hat + v_y_new * y_hat + v_z_new * z_hat = vx xhat + vy yhat + vz zhat 

    # vx = v_x_new x_hat[0] + vy_new y_hat[0] + vz_new z_hat[0]
##########################################################
    for i in idumps:
        print ("reading dump: ", i)
        #dump_name = glob.glob("dump_spher_%d_th_*.npz" %i)[0]
        if (disk_frame==True): dump_name = glob.glob("dump_spher_disk_frame_%04d*.npz" %(i))[0]
        else: dump_name = glob.glob("dump_spher_%d_th_%g_phi_%g*.npz" %(i,th_l,phi_l))[0]
        asc.rdnpz(dump_name)
        # z_hat = np.array([np.sin(asc.th_tilt)*np.cos(asc.phi_tilt),np.sin(asc.th_tilt)*np.sin(asc.phi_tilt),np.cos(asc.th_tilt)])
        # x_hat = np.array([np.cos(asc.th_tilt)*np.cos(asc.phi_tilt),np.cos(asc.th_tilt)*np.sin(asc.phi_tilt),-np.sin(asc.th_tilt)])
        # y_hat = np.array([-np.sin(asc.phi_tilt),np.cos(asc.phi_tilt),0])

        # x_hat_prime = [np.cos(asc.th_tilt)*np.cos(asc.phi_tilt),-np.sin(asc.phi_tilt),np.sin(asc.th_tilt)*np.cos(asc.phi_tilt)]
        # y_hat_prime = [np.cos(asc.th_tilt)*np.sin(asc.phi_tilt),np.cos(asc.phi_tilt),np.sin(asc.th_tilt)*np.sin(asc.phi_tilt)]
        # z_hat_prime = [-np.sin(asc.th_tilt),0,np.cos(asc.th_tilt)]

        # vx_tmp = asc.vel1
        # vy_tmp = asc.vel2
        # vz_tmp = asc.vel3 

        # asc.vel1 = vx_tmp*x_hat_prime[0] + vy_tmp*y_hat_prime[0] + vz_tmp*z_hat_prime[0]
        # asc.vel2 = vx_tmp*x_hat_prime[1] + vy_tmp*y_hat_prime[1] + vz_tmp*z_hat_prime[1]
        # asc.vel3 = vx_tmp*x_hat_prime[2] + vy_tmp*y_hat_prime[2] + vz_tmp*z_hat_prime[2]

        if (is_magnetic==True): asc.B1 = asc.Bcc1 
        if (is_magnetic==True): asc.B2 = asc.Bcc2
        if (is_magnetic==True): asc.B3 = asc.Bcc3
        asc.get_mdot(mhd=is_magnetic)
        asc.get_stress_cart(mhd = is_magnetic)
        for key in dic.keys():
            exec("globals()['%s_tavg'] += asc.%s/len(idumps)" % (key,key))
        Lx_avg += asc.rho*asc.l_x/len(idumps)
        Ly_avg += asc.rho*asc.l_y/len(idumps)
        Lz_avg += asc.rho*asc.l_z/len(idumps)
        Lxdot_avg += asc.Lx_dot/len(idumps)
        Lydot_avg += asc.Ly_dot/len(idumps)
        Lzdot_avg += asc.Lz_dot/len(idumps)

        Lxdot_in += asc.Lx_dot * (asc.vr<0)/len(idumps)
        Lydot_in += asc.Ly_dot * (asc.vr<0)/len(idumps)
        Lzdot_in += asc.Lz_dot * (asc.vr<0)/len(idumps)

        Ldot_r  = (4.0 * np.pi * asc.r**2.0 * asc.rho *asc.vr  ) * asc.l_z 
        Ldot_th = (4.0 * np.pi * asc.r**2.0 * asc.rho *asc.vth ) * asc.l_z 
        Ldot_ph = (4.0 * np.pi * asc.r**2.0 * asc.rho *asc.vphi) * asc.l_z 

        Edot_r  = (4.0 * np.pi * asc.r**2.0 * asc.rho *asc.vr   ) * asc.bernoulli 
        Edot_th = (4.0 * np.pi * asc.r**2.0 * asc.rho *asc.vth  ) * asc.bernoulli 
        Edot_ph = (4.0 * np.pi * asc.r**2.0 * asc.rho *asc.vphi ) * asc.bernoulli 

        Edot_in += asc.mdot * asc.bernoulli * (asc.vr<0)/len(idumps)
        Edot_avg += asc.mdot * asc.bernoulli/len(idumps)

        Ldotr_avg  += Ldot_r/len(idumps)
        Ldotth_avg += Ldot_th/len(idumps)
        Ldotph_avg += Ldot_ph/len(idumps)

        Edotr_avg  += Edot_r/len(idumps)
        Edotth_avg += Edot_th/len(idumps)
        Edotph_avg += Edot_ph/len(idumps)

        mdot_avg += asc.mdot/len(idumps)
        mdot_in += asc.mdot * (asc.vr<0)/len(idumps)
        vr_avg += asc.vr*asc.rho/len(idumps)
        vth_avg += asc.vth*asc.rho/len(idumps)
        vphi_avg += asc.vphi*asc.rho/len(idumps)
        vrsq_avg += asc.vr**2.*asc.rho/len(idumps)
        vthsq_avg += asc.vth**2.*asc.rho/len(idumps)
        vphisq_avg += asc.vphi**2.*asc.rho/len(idumps)
        vsq_avg += (asc.vel1**2.0 + asc.vel2**2.0 + asc.vel3**2.0)*asc.rho/len(idumps)

        rho_tavg_in += asc.rho * (asc.vr<0)/len(idumps)
        press_tavg_in += asc.press* (asc.vr<0)/len(idumps)


        Fr_avg += asc.F_reynolds/len(idumps)
        FJ_avg += asc.F_J/len(idumps)
        def angle_average(arr,weight=None):
            dx3 = np.diff(asc.ph[0,0,:])[0]
            dx2 = np.diff(asc.th[0,:,0])[0]
            dOmega = (np.sin(asc.th)*dx2*dx3)
            if weight is None: weight = 1.0
            return (arr * weight * dOmega).mean(-1).mean(-1)/(dOmega*weight).mean(-1).mean(-1)
        def phi_average(arr):
            dx3 = np.diff(asc.ph[0,0,:])[0]
            return (arr).mean(-1)
        rho_vr_vph_sinth_avg += angle_average(asc.rho*asc.vr*asc.vphi*np.sin(asc.th)) /len(idumps)
        rho_vr_avg += angle_average(asc.rho*asc.vr)/len(idumps)
        vphi_sinth_avg += angle_average(asc.vphi*np.sin(asc.th))/len(idumps)

        rho_vr_vph_sinth_avg_2d += phi_average(asc.rho*asc.vr*asc.vphi*np.sin(asc.th)) /len(idumps)
        rho_vr_avg_2d += phi_average(asc.rho*asc.vr)/len(idumps)
        vphi_sinth_avg_2d += phi_average(asc.vphi*np.sin(asc.th))/len(idumps)

        rho_vr_vph_sinth_avg_3d += (asc.rho*asc.vr*asc.vphi*np.sin(asc.th)) /len(idumps)


        if (is_magnetic==True):
            bsq = asc.B1**2 + asc.B2**2 + asc.B3**2

            Bdotv = asc.B1*asc.vel1 + asc.B2*asc.vel2 + asc.B3*asc.vel3
            Edot_B_avg += -(4.0 * np.pi * asc.r**2) * asc.Br * Bdotv/len(idumps)    #Stone+ Athena paper
            Edot_B_in  += -(4.0 * np.pi * asc.r**2) * asc.Br * Bdotv * (asc.vr<0)/len(idumps) 

            Br_avg += asc.Br/len(idumps)
            Bth_avg += asc.Bth/len(idumps)
            Bphi_avg += asc.Bphi/len(idumps)
            bsq_avg += (bsq)/len(idumps)
            bsq_avg_in += bsq * (asc.vr<0)/len(idumps)
            Br_abs_avg += np.abs(asc.Br)/len(idumps)
            Bth_abs_avg += np.abs(asc.Bth)/len(idumps)
            Br_Bphi_msinth_avg += angle_average(asc.Br*asc.Bphi*-1.0*np.sin(asc.th))/len(idumps)
            Br_Bphi_msinth_avg_2d += phi_average(asc.Br*asc.Bphi*-1.0*np.sin(asc.th))/len(idumps)
            Br_Bphi_avg += asc.Br*asc.Bphi/len(idumps)
            alpham_avg += asc.alpha_m/len(idumps)
            #alphar_avg += asc.alpha_h/len(idumps)
            Fm_avg += asc.F_maxwell/len(idumps)
            Fr_avg += asc.F_reynolds/len(idumps)
            FJ_avg += asc.F_J/len(idumps)

            lambda_mri_th_avg_num += np.abs(asc.Bth)/len(idumps)
            lambda_mri_th_avg_den += np.abs(np.sqrt(asc.rho)*asc.vphi/(asc.r*np.sin(asc.th)))/len(idumps)

            Br_Bphi_in += asc.Br*asc.Bphi * (asc.vr<0)/len(idumps)


    if (is_magnetic==True): alphar_avg = (rho_vr_vph_sinth_avg - rho_vr_avg*vphi_sinth_avg)/angle_average(press_tavg + bsq_avg/2.0)
    else:  alphar_avg = (rho_vr_vph_sinth_avg - rho_vr_avg*vphi_sinth_avg)/angle_average(press_tavg )
    if (is_magnetic==True): alpham_avg = Br_Bphi_msinth_avg/angle_average(press_tavg + bsq_avg/2.0)

    if (is_magnetic==True): alphar_avg_2d = (rho_vr_vph_sinth_avg_2d - rho_vr_avg_2d*vphi_sinth_avg_2d)/phi_average(press_tavg + bsq_avg/2.0)
    else:  alphar_avg_2d = (rho_vr_vph_sinth_avg_2d - rho_vr_avg_2d*vphi_sinth_avg_2d)/phi_average(press_tavg )
    if (is_magnetic==True): alpham_avg_2d = Br_Bphi_msinth_avg_2d/phi_average(press_tavg + bsq_avg/2.0)




    for key in dic.keys():
        exec("new_dic['%s_tavg'] = %s_tavg" %(key,key))
    new_dic['Lx_avg'] = Lx_avg
    new_dic['Ly_avg'] = Ly_avg
    new_dic['Lz_avg'] = Lz_avg
    new_dic['Lxdot_avg'] = Lxdot_avg
    new_dic['Lydot_avg'] = Lydot_avg
    new_dic['Lzdot_avg'] = Lzdot_avg
    new_dic['Lxdot_in'] = Lxdot_in
    new_dic['Lydot_in'] = Lydot_in
    new_dic['Lzdot_in'] = Lzdot_in
    new_dic['Ldotr_avg'] = Ldotr_avg
    new_dic['Ldotth_avg'] = Ldotth_avg
    new_dic['Ldotph_avg'] = Ldotph_avg
    new_dic['Edotr_avg'] = Edotr_avg
    new_dic['Edotth_avg'] = Edotth_avg
    new_dic['Edotph_avg'] = Edotph_avg
    new_dic['Edot_in'] = Edot_in
    new_dic['Edot_avg'] = Edot_avg
    new_dic['mdot_avg'] = mdot_avg
    new_dic['mdot_in'] = mdot_in
    new_dic['vr_avg'] = vr_avg/rho_tavg
    new_dic['vth_avg'] = vth_avg/rho_tavg
    new_dic['vphi_avg'] = vphi_avg/rho_tavg
    new_dic['vrsq_avg'] = vrsq_avg/rho_tavg
    new_dic['vthsq_avg'] = vthsq_avg/rho_tavg
    new_dic['vphisq_avg'] = vphisq_avg/rho_tavg
    new_dic['vsq_avg'] = vsq_avg/rho_tavg
    new_dic['rho_tavg_in'] = rho_tavg_in
    new_dic['press_tavg_in'] = press_tavg_in


    if (is_magnetic==True):
        new_dic["Edot_B_avg"] = Edot_B_avg
        new_dic["Edot_B_in"] = Edot_B_in   
        new_dic['Br_avg'] = Br_avg
        new_dic['Bth_avg'] = Bth_avg
        new_dic['Bphi_avg'] = Bphi_avg
        new_dic['bsq_avg'] = bsq_avg
        new_dic['bsq_avg_in'] = bsq_avg_in
        new_dic['Br_abs_avg'] = Br_abs_avg
        new_dic['Bth_abs_avg'] = Bth_abs_avg
        new_dic['alpham_avg'] = alpham_avg
        #new_dic['alpham_avg_2d'] = alpham_avg_2d
        new_dic['Br_Bphi_avg'] = Br_Bphi_avg
        new_dic['Br_Bphi_in'] = Br_Bphi_in
        new_dic['Fm_avg'] = Fm_avg
        new_dic['lambda_mri_th_avg'] = lambda_mri_th_avg_num/(lambda_mri_th_avg_den + 1e-15)


    new_dic['rho_vr_vph_sinth_avg'] = rho_vr_vph_sinth_avg_3d
    new_dic['alphar_avg'] = alphar_avg
    #new_dic['alphar_avg_2d'] = alphar_avg_2d
    new_dic['Fr_avg'] = Fr_avg
    new_dic['FJ_avg'] = FJ_avg

    np.savez("dump_spher_avg_%d_%d.npz" %(idumps[0],idumps[-1]),**new_dic)


def time_average_gr(idump_min = 100,idump_max = 200,istep = 1,a = 0,is_magnetic=False):
    new_dic = {}
    idumps  = arange(idump_min,idump_max+1,istep)


    #dump_name = glob.glob("dump_spher_%d_th_*.npz" %idumps[0])[0]
    asc.rdhdf5(idumps[0],ndim=3,coord="spherical",user_x2=True,gr=True,a=a)

    rho_avg = asc.rho*0
    press_avg = asc.press*0
    mdot_avg = asc.rho*0
    uu_avg = asc.uu*0 
    if (is_magnetic ==True):
        bu_avg = asc.bu*0
        bsq_avg = asc.bsq*0
        Bcc1_avg = asc.Bcc1*0
        Bcc2_avg = asc.Bcc2*0
        Bcc3_avg = asc.Bcc3*0
        alpha_avg = 0*asc.rho 
        alpha_r_avg = 0 *asc.rho 
        alpha_m_avg = 0*asc.rho
        alpha_m_avg_delta = 0 * asc.rho


    for i in idumps:
        print ("reading dump", i)
        asc.rdhdf5(i,ndim=3,coord="spherical",user_x2=True,gr=True,a=a)
        asc.get_gr_stress(a=a)
        rho_avg = rho_avg + asc.rho/len(idumps)
        press_avg = press_avg + asc.press/len(idumps)
        uu_avg = uu_avg + asc.uu*asc.rho[None,:,:,:]/len(idumps)
        mdot_avg = mdot_avg +  asc.rho*asc.uu[1]*asc.gdet/len(idumps)
        if (is_magnetic ==True):
            asc.get_gr_stress(a=a)
            bu_avg = bu_avg + asc.bu*asc.rho[None,:,:,:]/len(idumps)
            bsq_avg = bsq_avg + asc.bsq/len(idumps)
            alpha_avg = alpha_avg + asc.alpha/len(idumps)
            alpha_m_avg = alpha_m_avg +asc.alpha_m/len(idumps)
            alpha_m_avg_delta = alpha_m_avg +asc.alpha_m/len(idumps)
            alpha_r_avg = alpha_r_avg +asc.alpha_r/len(idumps)




    dic = {"rho_avg":rho_avg,"press_avg":press_avg,"uu_avg":uu_avg/rho_avg,"mdot_avg":mdot_avg,"a":a,"i_start":idump_min,"i_end":idump_max,"r":asc.r,"th":asc.th,"ph":asc.ph,"gdet":asc.gdet,"g":asc.g,
           "x1f": x1f,"x2f": x2f, "x3f":x3f}
    if (is_magnetic==True):
        dic['bu_avg'] = bu_avg
        dic['bsq_avg'] = bsq_avg 
        dic['alpha_avg'] = alpha_avg 
        dic['alpha_m_avg'] =alpha_m_avg
        dic['alpha_r_avg'] = alpha_r_avg

    np.savez("dump_avg_%d_%d.npz" %(idump_min,idump_max),**dic)

def time_average_gr_cartesian(idump_min = 100,idump_max = 200,istep = 1,a = 0,gam=5.0/3.0,is_magnetic=False):
    th_l = 0
    phi_l = 0
    new_dic = {}
    idumps  = arange(idump_min,idump_max+1,istep)


    #dump_name = glob.glob("dump_spher_%d_th_*.npz" %idumps[0])[0]
    dump_name = glob.glob("dump_spher_0_th_%g_phi_%g*.npz" %(th_l,phi_l))[0]
    rdnpz(dump_name)

    rho_avg = asc.rho*0
    press_avg = asc.press*0
    mdot_avg = asc.rho*0
    uu_avg = asc.uu*0 
    uu_ks_avg = asc.uu*0
    ud_ks_avg = asc.uu*0
    Edot_avg = asc.rho*0
    EdotEM_avg = asc.rho*0
    EdotKE_avg = asc.rho*0
    EdotUint_avg = asc.rho*0
    gravity_force_average = asc.rho*0
    gravity_force_average_sq = asc.rho*0
    pressure_force_average = asc.rho*0
    EM_force_average = asc.rho*0
    advection_force_average  = asc.rho*0
    bernoulli_average = asc.rho*0
    mdot_out_avg = asc.rho*0
    s_avg = asc.rho*0
    Jdot_avg = asc.rho*0
    JdotEM_avg = asc.rho*0
    l_h_avg = asc.rho*0
    Pflux_MA_avg = asc.rho*0
    Pflux_EM_avg = asc.rho*0
    Pflux_MA_rho_avg = asc.rho*0
    Pflux_EM_rho_avg = asc.rho*0
    Pdot_avg = asc.rho*0
    PdotEM_avg = asc.rho*0

    if (is_magnetic ==True):
        bu_avg = asc.bu*0
        bu_ks_avg = asc.bu*0
        bsq_avg = asc.bsq*0
        Bcc1_avg = asc.Bcc1*0
        Bcc2_avg = asc.Bcc2*0
        Bcc3_avg = asc.Bcc3*0
        beta_avg = asc.rho*0
        brbphi_avg = asc.rho*0
        Bz_avg = asc.rho*0
        Bx_avg = asc.rho*0
        By_avg = asc.rho*0
        brsq_avg  = asc.rho*0
        bthsq_avg = asc.rho*0
        bphsq_avg = asc.rho*0
        bxsq_avg =asc.rho*0
        bysq_avg = asc.rho*0
        bzsq_avg = asc.rho*0



    i_dumps_temp = []   
    for i in idumps:
        dump_name = glob.glob("dump_spher_%d_th_%g_phi_%g*.npz" %(i,th_l,phi_l))[0]
        if (os.path.isfile(dump_name)==False): continue
        i_dumps_temp.append(i)
    i_dumps_temp = np.array(i_dumps_temp)

    for i in i_dumps_temp:
        print ("reading dump", i)
        dump_name = glob.glob("dump_spher_%d_th_%g_phi_%g*.npz" %(i,th_l,phi_l))[0]
        if (os.path.isfile(dump_name)): asc.rdnpz(dump_name)
        else: continue

        get_mdot(mhd=is_magnetic,gr=True,a=a)
        rho_avg = rho_avg + asc.rho/len(i_dumps_temp)
        press_avg = press_avg + asc.press/len(i_dumps_temp)
        uu_avg = uu_avg + asc.uu*asc.rho[None,:,:,:]/len(i_dumps_temp)
        uu_ks_avg = uu_ks_avg + asc.uu_ks*asc.rho[None,:,:,:]/len(i_dumps_temp)
        mdot_avg = mdot_avg +  asc.mdot/len(i_dumps_temp)
        mdot_out_avg = mdot_out_avg + asc.mdot * (asc.mdot>0)/len(i_dumps_temp)
        s_avg = s_avg + 1.0/(gam-1.0)*np.log(asc.press/asc.rho**gam)*asc.rho/len(i_dumps_temp)

        asc.cks_metric(asc.x,asc.y,asc.z,a)
        bd = Lower(asc.bu,asc.g)
        asc.ks_metric(asc.r,asc.th,a)

        uu_ks = nan_to_num(asc.uu_ks)
        bu_ks = nan_to_num(asc.bu_ks)
        ud_ks = nan_to_num(asc.Lower(asc.uu_ks,asc.g))
        bd_ks = nan_to_num(asc.Lower(asc.bu_ks,asc.g))

        ud_ks_avg = ud_ks_avg + ud_ks*asc.rho[None,:,:,:]/len(i_dumps_temp)

        asc.Tud_calc(asc.uu_ks,ud_ks,asc.bu_ks,bd_ks,is_magnetic= is_magnetic,gam=gam)
        Edot = - (asc.Tud[1][0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2) + asc.mdot )
        EdotEM = -(asc.TudEM[1][0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2))
        EdotKE =  -(asc.rho*asc.uu_ks[1]*ud_ks[0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2) + asc.mdot )
        EdotMA = Edot-EdotEM
        EdotUint = EdotMA-EdotKE

        Edot_avg = Edot_avg + Edot/len(i_dumps_temp)
        EdotEM_avg = EdotEM_avg + EdotEM/len(i_dumps_temp)
        EdotKE_avg= EdotKE_avg + EdotKE/len(i_dumps_temp)
        EdotUint_avg= EdotUint_avg + EdotUint/len(i_dumps_temp)


        Jdot = (asc.Tud[1][3]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2))
        Jdot_EM = (asc.TudEM[1][3]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2))

        Jdot_avg = Jdot_avg + Jdot/len(i_dumps_temp)
        JdotEM_avg = JdotEM_avg + Jdot_EM/len(i_dumps_temp)

        Pdot = (asc.Tud[1][1]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2))
        Pdot_EM = (asc.TudEM[1][1]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2))

        Pdot_avg = Pdot_avg + Pdot/len(i_dumps_temp)
        PdotEM_avg = PdotEM_avg + Pdot_EM/len(i_dumps_temp)

        Pflux_MA = (asc.rho + asc.press * gam/(gam-1.0) ) * asc.uu_ks[1]*ud_ks[1]
        Pflux_EM = (asc.bsq) * asc.uu_ks[1]*ud_ks[1] - asc.bu_ks[1]*bd_ks[1]

        Pflux_MA_avg = Pflux_MA_avg + Pflux_MA/len(i_dumps_temp)
        Pflux_EM_avg = Pflux_EM_avg + Pflux_EM/len(i_dumps_temp)

        Pflux_MA_rho_avg = Pflux_MA_avg + Pflux_MA*asc.rho/len(i_dumps_temp)
        Pflux_EM_rho_avg = Pflux_EM_avg + Pflux_EM*asc.rho/len(i_dumps_temp)

        l_h_avg = l_h_avg + (1.0 + asc.press/asc.rho * gam/(gam-1.0) + asc.bsq/asc.rho) * ud_ks[3]/len(i_dumps_temp)
        if (is_magnetic ==True):
            bu_avg = bu_avg + asc.bu/len(i_dumps_temp)
            bu_ks_avg = bu_ks_avg + asc.bu_ks/len(i_dumps_temp)
            bsq_avg = bsq_avg + asc.bsq/len(i_dumps_temp)
            beta_avg  = beta_avg + (asc.press/asc.bsq*2.0)**-1.0 * asc.rho /len(i_dumps_temp)
            brbphi_avg = brbphi_avg + bu_ks[1]*bd_ks[3]/len(i_dumps_temp)
            Bz_avg = Bz_avg + asc.Bcc3/len(i_dumps_temp)
            Bx_avg = Bx_avg + asc.Bcc1/len(i_dumps_temp)
            By_avg = By_avg + asc.Bcc2/len(i_dumps_temp)

            brsq_avg = brsq_avg + asc.bu_ks[1]*bd_ks[1]/len(i_dumps_temp)
            bthsq_avg = bthsq_avg + asc.bu_ks[2]*bd_ks[2]/len(i_dumps_temp)
            bphsq_avg = bphsq_avg + asc.bu_ks[3]*bd_ks[3]/len(i_dumps_temp)

            bxsq_avg = bxsq_avg + asc.bu[1]*bd[1]/len(i_dumps_temp)
            bysq_avg = bysq_avg + asc.bu[2]*bd[2]/len(i_dumps_temp)
            bzsq_avg = bzsq_avg + asc.bu[3]*bd[3]/len(i_dumps_temp)


        asc.gravity_term_gr(asc.r,asc.th,a,m=1)

        w = 1.0 + asc.press/asc.rho * gam/(gam-1.0)
        Be = - ud_ks[0]*w -1.0

        gravity_force_average = gravity_force_average + asc.aterm/len(i_dumps_temp)
        gravity_force_average_sq = gravity_force_average_sq + asc.aterm**2.0/len(i_dumps_temp)
        pressure_force_average = pressure_force_average + asc.pressterm/len(i_dumps_temp)
        EM_force_average = EM_force_average + asc.EMterm/len(i_dumps_temp)
        advection_force_average = advection_force_average + asc.advection_term/len(i_dumps_temp)
        bernoulli_average = bernoulli_average + Be/len(i_dumps_temp)






    dic = {"rho_avg":rho_avg,"press_avg":press_avg,"uu_avg":uu_avg/rho_avg,"uu_ks_avg":uu_ks_avg/rho_avg,
    "mdot_avg":mdot_avg,"a":a,"i_start":idump_min,"i_end":idump_max,"r":asc.r,"th":asc.th,"ph":asc.ph,
           "x": asc.x,"y": asc.y, "z":asc.z,"Edot_avg": Edot_avg,"EdotEM_avg": EdotEM_avg, "ud_ks_avg":ud_ks_avg/rho_avg}
    if (is_magnetic==True):
        dic['bu_avg'] = bu_avg
        dic['bu_ks_avg'] = bu_ks_avg
        dic['bsq_avg'] = bsq_avg 
        dic['beta_avg'] = (beta_avg/rho_avg)**-1.0
        dic['brbphi_avg'] = brbphi_avg
        dic['Bz_avg'] = Bz_avg
        dic['Bx_avg'] = Bx_avg
        dic['By_avg'] = By_avg

        dic['brsq_avg'] = brsq_avg
        dic['bphsq_avg'] = bphsq_avg
        dic['bthsq_avg'] = bthsq_avg
        dic['bxsq_avg'] = bxsq_avg
        dic['bysq_avg'] = bysq_avg
        dic['bzsq_avg'] = bzsq_avg
    dic['EdotKE_avg'] = EdotKE_avg
    dic['EdotUint_avg'] = EdotUint_avg
    dic['gravity_force_avg'] = gravity_force_average
    dic['gravity_force_avg_sq'] = gravity_force_average_sq
    dic['press_force_avg'] = pressure_force_average
    dic['EM_force_avg'] = EM_force_average
    dic['advection_force_avg'] = advection_force_average
    dic['Be_avg'] = bernoulli_average
    dic['mdot_out_avg'] = mdot_out_avg
    dic['s_avg'] = s_avg/rho_avg
    dic['Jdot_avg'] = Jdot_avg
    dic['JdotEM_avg'] = JdotEM_avg
    dic['lh_avg'] = l_h_avg
    dic['Pflux_EM_avg'] = Pflux_EM_avg
    dic['Pflux_MA_avg'] = Pflux_MA_avg
    dic['Pflux_EM_rho_avg'] = Pflux_EM_rho_avg/rho_avg
    dic['Pflux_MA_rho_avg'] = Pflux_MA_rho_avg/rho_avg
    dic['Pdot_avg']  = Pdot_avg
    dic['PdotEM_avg'] = PdotEM_avg
    

    np.savez("dump_avg_%d_%d.npz" %(idump_min,idump_max),**dic)


# def mk_1d_avg():
#     dic = {}
#     idumps  = arange(idump_min,idump_max+1)
#     for i in idumps:
#         asc.rdhdf5(i,ndim=3,coord="spherical",user_x2=True,gr=True,a=a)
#         mdot.append angle_average
#     #for 


if __name__ == "__main__":
    if len(sys.argv)>1:
        if sys.argv[1] == "time_average_mhd":
            imin = np.int(sys.argv[2])
            imax = np.int(sys.argv[3])
            time_average(imin,imax,is_magnetic =True)
        elif sys.argv[1] == "time_average_hydro":
            imin = np.int(sys.argv[2])
            imax = np.int(sys.argv[3])
            time_average(imin,imax,is_magnetic =False)
        elif sys.argv[1] == "time_average_mhd_disk_frame":
            imin = np.int(sys.argv[2])
            imax = np.int(sys.argv[3])
            time_average(imin,imax,is_magnetic =True,disk_frame=True)
        elif sys.argv[1] == "time_average_hydro_disk_frame":
            imin = np.int(sys.argv[2])
            imax = np.int(sys.argv[3])
            time_average(imin,imax,is_magnetic =False,disk_frame=True)
        elif sys.argv[1] == "time_average_gr":
            imin = np.int(sys.argv[2])
            imax = np.int(sys.argv[3])
            istep = np.int(sys.argv[4])
            a = np.float64(sys.argv[5])
            is_magnetic=False
            if (np.int(sys.argv[6])==1): is_magnetic= True
            time_average_gr(imin,imax,istep,a=a,is_magnetic=is_magnetic)
        elif sys.argv[1] == "time_average_gr_cartesian":
            imin = np.int(sys.argv[2])
            imax = np.int(sys.argv[3])
            istep = np.int(sys.argv[4])
            a = np.float64(sys.argv[5])
            is_magnetic= True
            time_average_gr_cartesian(imin,imax,istep,a=a,is_magnetic=is_magnetic)


