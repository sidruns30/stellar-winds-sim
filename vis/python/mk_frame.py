# Python modules
import numpy as np
from numpy import *
import glob
import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()


parser.add_argument('--a',
                    default='0',
                    help='spin of black hole for GR')

parser.add_argument('--gamma',
                    default='1.666667',
                    help='adiabatic index')


parser.add_argument('--th',
                    default='0',
                    help='theta coordinate of new axes')

parser.add_argument('--phi',
                    default='0',
                    help='phi coordinate of new axes')

parser.add_argument('--i_glob',
                    default='',
                    help='index of processor')
parser.add_argument('--n_processors',
                    default='',
                    help='number of processors')

parser.add_argument('--mtype',
                    default='',
                    choices =[
                    	"mk_frame_inner",
						"mk_frame_outer",
						"mk_frame_outer_slice",
						"mk_frame_outer_slice_mhd",
						"mk_frame_inner_slice_mhd",
						"mk_frame_inner_slice",
						"convert_dumps",
						"convert_dumps_mhd",
						"convert_dumps_disk_frame",
						"convert_dumps_disk_frame_mhd",
						"Lx_calc",
						"mk_3Dframe",
						"mk_grmhdframe",
						"mk_grframe",
						"mk_grframe_cartesian",
						"mk_frame_grmhd_restart_cartesian",
						"mk_grframe_magnetically_frustrated",
						"mk_1d",
						"mk_1d_cartesian",
						'mk_RM',
						'mk_RM_moving',
						"RM_movie",
						"mk_frame_outer_cold",
						"mk_frame_inner_cold",
						"mk_frame_disk",
						"mk_frame_L_aligned",
						"mk_frame_L_aligned_fieldlines",
						"mk_grframe_magnetically_frustrated_cartesian"],
                    help='type of analysis')

parser.add_argument('--i_start',
                    default='0',
                    help='index of first dump to process')
parser.add_argument('--i_end',
                    default='',
                    help='index of last dump to process')
parser.add_argument('--i_increment',
                    default='1',
                    help='spacing between dumps to process')


args = vars(parser.parse_args())

if (args['mtype'] == ''):
	raise SystemExit('### No analysis selected...please specify using --mtype')
if (args['n_processors'] == ''):
	raise SystemExit('### Need to know number of processors...set using --n_processors')	
if (args['i_glob'] == ''):
	raise SystemExit('### Need to know index of processor...set using --i_glob')	

a = np.double(args['a'])
gam = np.double(args['gamma'])
th_tilt = np.double(args['th'])
phi_tilt = np.double(args['phi'])
i_glob = np.int(args['i_glob'])
n_processors = np.int(args['n_processors'])
m_type = args['mtype']
i_increment = np.int(args['i_increment'])
i_start_glob = np.int(args['i_start'])





sys.path.append("/global/scratch/smressle/star_cluster/restart_grmhd/vis/python")
from athena_script import *
import athena_script as asc


Z_o_X_solar = 0.0177
Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
X_solar = 0.7491
Z_solar = 1.0-X_solar - Y_solar

muH_solar = 1./X_solar
Z = 3. * Z_solar
X = 0.
mue = 2. /(1.+X)
mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
mp_over_kev = 9.994827
keV_to_Kelvin = 1.16045e7
box_radius_inner = 0.05
box_radius_outer = 0.5
def set_limits(isslice = False):
	global den_min_inner,den_min_outer,den_max_inner,den_max_outer,T_min_inner,T_min_outer,T_max_outer,T_max_inner
	if (isslice==False):
		den_min_inner = -1
		den_max_inner = 1.5
		den_min_outer = -2
		den_max_outer = 1
		T_min_inner = -3
		T_max_inner =  0
		T_min_outer = -3
		T_max_outer = 0
	else:
		den_min_outer = -1.5 -0.5 +0.75
		den_max_outer= 0.5 +1.0 -0.25
		den_min_inner = den_min_outer + 1.5 +0.5 - 0.75
		den_max_inner = den_max_outer + 1.5 - 1.0 +0.25
		T_min_outer = -1.5- 1.0+1.0
		T_max_outer = -0.25 + 0.5-0.25
		T_min_inner = T_min_outer + 0.5
		T_max_inner = T_max_outer + 0.5

# box_radius_outer = 2e-3
# box_radius_inner = 2e-4

# den_min_inner = 2
# den_max_inner = 3.5
# den_min_outer = 1.
# den_max_outer = 3.
# T_min_inner = 1.5
# T_max_inner =  3.
# T_min_outer = 0.5
# T_max_outer = 2.5


def mk_frame(var_num,var_den=None,projection_axis = 2,file_suffix ='rho',cb_label = r"$\log_{10}\left(\rho\right)$",i_frame = 0,var_min = 0,var_max = 2.5,cmap = 'ocean' ,isslice=False,fieldlines=False,
	B1=None,B2=None,B3=None,box_radius = 1.0):
    plt.figure(1,figsize = (6,6))
    plt.clf()
    plt.style.use('dark_background')

    nx = var_num.shape[0]
    ny = var_num.shape[1]
    nz = var_num.shape[2]

    if (projection_axis ==2):
        x_plot = region['x'][:,:,nz//2]
        y_plot = region['y'][:,:,nz//2]
        proj_label = 'z'
        if (isslice ==True):
            var_num = var_num[:,:,nz//2]
            if (var_den != None): var_den = var_den[:,:,nz//2]

    elif (projection_axis==1):
        x_plot = region['z'][:,ny//2,:]
        y_plot = region['x'][:,ny//2,:]
        proj_label = 'y'
        if (isslice ==True):
            var_num = var_num[:,ny//2,:]
            if (var_den !=None): var_den = var_den[:,ny//2,:]
    else:
        x_plot = region['y'][nx//2,:,:]
        y_plot = region['z'][nx//2,:,:]
        proj_label = 'x'
        if (isslice ==True):
            var_num = var_num[nx//2,:,:]
            if (var_den != None): var_den = var_den[nx//2,:,:]

    if (var_den ==None):
        if (isslice==False):
            c = plt.contourf(x_plot,y_plot,np.log10(var_num.mean(projection_axis)),levels = np.linspace(var_min,var_max,200),cmap = cmap,extend = 'both')
        else:
            c = plt.contourf(x_plot,y_plot,np.log10(var_num),levels = np.linspace(var_min,var_max,200),cmap = cmap,extend = 'both')
    else:
        if (isslice==False):
            c = plt.contourf(x_plot,y_plot,np.log10(var_num.mean(projection_axis)/var_den.mean(projection_axis)),levels = np.linspace(var_min,var_max,200),
            cmap = cmap,extend = 'both')
        else:
            c = plt.contourf(x_plot,y_plot,np.log10(var_num/var_den),levels = np.linspace(var_min,var_max,200),
            cmap = cmap,extend = 'both')

    plt.xlim(box_radius,-box_radius)
    plt.ylim(-box_radius,box_radius)
    if (fieldlines==True and projection_axis==2): plt.streamplot(np.array(x_plot.transpose()),np.array(y_plot.transpose()),np.array(B1[:,:,nz//2].transpose()),np.array(B2[:,:,nz//2].transpose()),color = 'white')

    # if (projection_axis ==2):
    # 	plt.xlabel(r'$x$ (pc)',fontsize = 25)
    # 	plt.ylabel(r'$y$ (pc)',fontsize = 25)
    # elif (projection_axis==1):
    # 	plt.xlabel(r'$z$ (pc)',fontsize = 25)
    # 	plt.ylabel(r'$x$ (pc)',fontsize = 25)
    # else:
    # 	plt.xlabel(r'$y$ (pc)',fontsize = 25)
    # 	plt.ylabel(r'$z$ (pc)',fontsize = 25)

    # cb = plt.colorbar(c,ax=plt.gca())
    # cb.set_ticks(np.arange(var_min,var_max+.5,.5))
    # cb.set_label(cb_label,fontsize=25)

    # ax = plt.gca()
    # for label in ax.get_xticklabels() + ax.get_yticklabels()+cb.ax.get_yticklabels():
    #     label.set_fontsize(20)

    plt.axis('off')
    plt.tight_layout()

    os.system("mkdir -p frames")
    if (isslice == False):
        plt.savefig("frames/frame_%s_%s_%d.png" % (file_suffix,proj_label,i_frame))
    else: 
        plt.savefig("frames/frame_slice_%s_%s_%d.png" % (file_suffix,proj_label,i_frame))



def set_dump_range():
	global i_start,i_end
	# if len(sys.argv[2:])>=2 and sys.argv[2].isdigit() and sys.argv[3].isdigit():
	# 	i_glob = int(sys.argv[2])
	# 	n_processors = int(sys.argv[3])
	# 	print("i_glob, n_processors:",i_glob,n_processors)
	# else:
	# 	print("Syntax error")
	# 	exit()


	dump_list = glob.glob("*.athdf")
	dump_list.sort()
	n_dumps = len(dump_list)

	if (args['i_end']==''): i_end_glob = n_dumps-1
	else: i_end_glob = np.int(args['i_end'])

	n_dumps = len(arange(i_start_glob,i_end_glob+1,i_increment)) 


	i_0 = i_start_glob #np.int(dump_list[0][15:-6])

	n_dumps_per_processor = np.int ( np.round((n_dumps*1.)/(n_processors*1.)+0.5) )
	i_start = i_0 + i_glob*n_dumps_per_processor
	i_end = i_start + n_dumps_per_processor

	if (i_end>(i_0 + n_dumps-1)):
		i_end = i_0 + n_dumps -1 

	print("i_start, i_end",i_start,i_end)
def set_dump_range_gr():
	global i_start,i_end
	# if len(sys.argv[2:])>=2 and sys.argv[2].isdigit() and sys.argv[3].isdigit():
	#     i_glob = int(sys.argv[2])
	#     n_processors = int(sys.argv[3])
	# else:
	# 	print("Syntax error")
	# 	exit()

	dump_list = glob.glob("*.athdf")
	dump_list.sort()
	n_dumps = len(dump_list) ##//2

	i_0 = i_start_glob #np.int(dump_list[0][15:-6])

	if (args['i_end']==''): i_end_glob = n_dumps-1
	else: i_end_glob = np.int(args['i_end'])

	n_dumps = len(arange(i_start_glob,i_end_glob+1,i_increment)) 


	n_dumps_per_processor = np.int ( (n_dumps*1.)/(n_processors*1.)+0.5 )
	i_start = i_0 + i_glob*n_dumps_per_processor
	i_end = i_start + n_dumps_per_processor

	if (i_end>(n_dumps-1)):
		i_end = n_dumps -1 

def mk_frame_inner(isslice=False,iscold=False,mhd=False):
	set_dump_range()
	set_limits(isslice= isslice)
	global region

	for i_dump in range(i_start,i_end):
		fcheck = "frames/frame_T_inner_z_%d.png" %i_dump
		if (isslice==True): fcheck = "frames/frame_slice_T_inner_z_%d.png" %i_dump
		if not os.path.isfile(fcheck): #asc.rdhdf5(i_dump,ndim=3,block_level=6,x1min=-.05,x1max=.05,x2min=-.05,x2max=.05,x3min=-.05,x3max=.05)
			asc.yt_load(i_dump)
			region = asc.ds.r[(-box_radius_inner,'pc'):(box_radius_inner,'pc'):128j,(-box_radius_inner,'pc'):(box_radius_inner,'pc'):128j,
			(-box_radius_inner,'pc'):(box_radius_inner,'pc'):128j]
			T_kelvin = (region['press']/region['rho']*mu_highT*mp_over_kev*keV_to_Kelvin)
			#from athena_script import *
			for i_proj in [2]: #range(3):
				if (mhd==False): mk_frame(var_num = region['rho'],var_min=den_min_inner,var_max=den_max_inner,file_suffix="rho_inner",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,box_radius = box_radius_inner)
				else:  mk_frame(var_num = region['rho'],var_min=den_min_inner,var_max=den_max_inner,file_suffix="rho_inner",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,fieldlines=True, B1=region['Bcc1'],B2=region['Bcc2'],B3=region['Bcc3'],box_radius = box_radius_inner)
				mk_frame(var_num = region['press'],var_den = region['rho'],file_suffix="T_inner",cb_label = r"$\log_{10}\left(T\right)$",
					i_frame=i_dump,var_min=T_min_inner,var_max=T_max_inner,cmap = 'gist_heat',projection_axis = i_proj,isslice=isslice,box_radius = box_radius_inner)
				if (iscold==True): mk_frame(var_num = region['rho']*(T_kelvin<12.5e3),var_min =den_min_inner,var_max = den_max_inner,file_suffix ="rho_cold_outer",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,fieldlines=False,box_radius = box_radius_inner)

def mk_frame_outer(isslice=False,mhd=False,iscold=False):
	set_dump_range()
	set_limits(isslice= isslice)
	global region
	for i_dump in range(i_start,i_end):
		if not os.path.isfile("frames/frame_T_outer_z_%d.png" %i_dump): #asc.rdhdf5(i_dump,ndim=3,block_level=3,x1min=-.5,x1max=.5,x2min=-.5,x2max=.5,x3min=-.5,x3max=.5)
			asc.yt_load(i_dump)
			region = asc.ds.r[(-box_radius_outer,'pc'):(box_radius_outer,'pc'):128j,(-box_radius_outer,'pc'):(box_radius_outer,'pc'):128j,
			(-box_radius_outer,'pc'):(box_radius_outer,'pc'):128j]
			T_kelvin = (region['press']/region['rho']*mu_highT*mp_over_kev*keV_to_Kelvin)
			#from athena_script import *
			for i_proj in [2]: #range(3):
			    if (mhd==False): mk_frame(var_num = region['rho'],var_min =den_min_outer,var_max = den_max_outer,file_suffix ="rho_outer",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,box_radius = box_radius_outer)
			    else: mk_frame(var_num = region['rho'],var_min =den_min_outer,var_max = den_max_outer,file_suffix ="rho_outer",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,fieldlines=True,B1=region['Bcc1'],B2=region['Bcc2'],B3=region['Bcc3'],box_radius = box_radius_outer)
			    if (iscold==True): mk_frame(var_num = region['rho']*(T_kelvin<12.5e3),var_min =den_min_outer,var_max = den_max_outer,file_suffix ="rho_cold_outer",i_frame = i_dump,projection_axis = i_proj,isslice=isslice,fieldlines=False,box_radius = box_radius_outer)
			    mk_frame(var_num = region['press'],var_den = region['rho'],file_suffix="T_outer",cb_label = r"$\log_{10}\left(T\right)$",
			        i_frame=i_dump,var_min=T_min_inner,var_max=T_max_outer,cmap = 'gist_heat',projection_axis = i_proj,isslice=isslice,box_radius = box_radius_outer)


def convert_dumps_to_spher(MHD=False):
	set_dump_range()
	omega_phi = None
	# if len(sys.argv)>6:
	# 	omega_phi = np.float(sys.argv[6])
	# else:
	# 	omega_phi = None
	for idump in range(i_start,i_end):
		asc.rd_yt_convert_to_spherical(idump,th=th_tilt,ph=phi_tilt,omega_phi = omega_phi,MHD=MHD)

def convert_dumps_disk_frame(mhd=False):
	set_dump_range()
	asc.set_constants()


	def get_l_angles_slice(idump,levels = 8):
		global th_l,phi_l
		asc.rd_hst('star_wind.hst',is_magnetic=mhd)
		L_tot = np.sqrt(asc.Lx_avg**2. + asc.Ly_avg**2. + asc.Lz_avg**2.) + 1e-15
		r_in = 2.*2./2.**levels/128.

		x_rat = (asc.Lx_avg/L_tot)[:,asc.r_to_ir(r_in*10):r_to_ir(0.8*asc.arc_secs)].mean(-1)[idump]
		y_rat = (asc.Ly_avg/L_tot)[:,asc.r_to_ir(r_in*10):r_to_ir(0.8*asc.arc_secs)].mean(-1)[idump]
		z_rat = (asc.Lz_avg/L_tot)[:,asc.r_to_ir(r_in*10):r_to_ir(0.8*asc.arc_secs)].mean(-1)[idump]
		th_l = np.arccos(z_rat)
		phi_l = np.arctan2(y_rat,x_rat)

	# def get_l_angles_slice(idump,levels = 8):
	# 	global th_l,phi_l
	# 	rd_hst('star_wind.hst',is_magnetic=mhd)
	# 	L_tot = np.sqrt(Lx_avg**2. + Ly_avg**2. + Lz_avg**2.) + 1e-15
	# 	r_in = 2.*2./2.**levels/128.

	# 	x_rat = (Lx_avg/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)[idump]
	# 	y_rat = (Ly_avg/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)[idump]
	# 	z_rat = (Lz_avg/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)[idump]
	# 	th_l = np.arccos(z_rat)
	# 	phi_l = np.arctan2(y_rat,x_rat)

	for idump in range(i_start,i_end):
		get_l_angles_slice(idump,levels = 8)
		dump_name="dump_spher_disk_frame_%04d.npz" %idump
		asc.rd_yt_convert_to_spherical(idump,th=th_l,ph=phi_l,MHD=mhd,dump_name=dump_name)
def merge_frames():
    n_frames = len(glob.glob("frame_rho_outer_x_*.png"))
    for iframe in range(n_frames):
        plt.figure(figsize=(8,2))
        plt.subplot(131)
        plt.imshow(plt.imread('frame_rho_outer_x_%d.png' %iframe))
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(plt.imread('frame_rho_outer_y_%d.png' %iframe))
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(plt.imread('frame_rho_outer_z_%d.png' %iframe))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('frame_rho_outer_all_%d.png' %iframe,bbox_inches='tight')
        plt.close()

def calculate_L_X():
	set_dump_range()
	D_BH = 8.3e3 #in parsecs
	#tan(theta) ~ theta ~ x_pc /D_BH
	arc_secs = 4.84814e-6 * D_BH
	for i_dump in range(i_start,i_end):
		asc.yt_load(i_dump)
		L_x_1_5 = asc.get_Xray_Lum("Lam_spex_Z_solar_2_10_kev",1.5*arc_secs)
		L_x_10 = asc.get_Xray_Lum("Lam_spex_Z_solar_2_10_kev",10.0*arc_secs)
		dic = {"t": asc.ds.current_time, "Lx_1_5": L_x_1_5,"Lx_10": L_x_10}
		np.savez("Lx_%04d.npz" %i_dump,**dic)

def column_density():
	set_dump_range()
	global region
	box_radius = 1.0 #pc 
	for i_dump in range(i_start,i_end):
		fname = "npz_files/column_density_z_%d.npz" % (i_dump)
		if not os.path.isfile(fname):
			asc.yt_load(i_dump)
			Lz =(asc.ds.domain_right_edge-asc.ds.domain_left_edge)[2]  #pc 

			region = asc.ds.r[(-box_radius,'pc'):(box_radius,'pc'):128j,(-box_radius,'pc'):(box_radius,'pc'):128j,
			(-Lz/2.0,'pc'):(Lz/2.0,'pc'):128j ]

			column_density = region['rho'].mean(-1)*Lz

			dic = {"column_density": column_density, "x": region['x'].mean(-1), "y": region['y'].mean(-1), "t":asc.ds.current_time}
			os.system("mkdir -p npz_files")
			np.savez(fname,**dic)

def Xray_image():
	set_dump_range()
	D_BH = 8.3e3 #in parsecs
	#tan(theta) ~ theta ~ x_pc /D_BH
	arc_secs = 4.84814e-6 * D_BH

def mk_frame_3D():
    set_dump_range()
    for i_dump in range(i_start,i_end):
        fname = "frames/3Dframe_smooth_%d.png" % (i_dump)
        if isfile(fname):
            continue
        else:
            asc.yt_load(i_dump)
            from yt.visualization.volume_rendering.api import Scene, VolumeSource 
            import numpy as np
            sc  = Scene()
            vol = VolumeSource(asc.ds, field="density")
            bounds = (1e-2, 10.**1.5)
            tf = yt.ColorTransferFunction(np.log10(bounds))
            def linramp(vals, minval, maxval):
                return (vals - vals.min())/(vals.max() - vals.min())
            #tf.add_layers(8, colormap='ocean')
            tf.map_to_colormap(np.log10(bounds[0]),np.log10(bounds[1]),colormap='ocean',scale_func=linramp)
            #tf.add_layers(8, colormap='ocean')
            tf.grey_opacity = False
            vol.transfer_function = tf
            vol.tfh.tf = tf
            vol.tfh.bounds = bounds
            vol.tfh.plot('transfer_function.png', profile_field="density")
            cam = sc.add_camera(asc.ds, lens_type='plane-parallel')
            cam.resolution = [512,512]
            # cam.position = ds.arr(np.array([282.*np.cos(theta)*np.sin(phi),282.*np.cos(theta)*np.cos(phi),282.*np.sin(theta)]), 'code_length')
            # cam.switch_orientation(normal_vector=normal_vector,
            #                        north_vector=north_vector)
            cam.set_width(asc.ds.domain_width*0.25)

            cam.position = asc.ds.arr(np.array([0,0,-0.5]), 'code_length')
            normal_vector = [0,0,-1]  #camera to focus
            north_vector = [0,1,0]  #up direction
            cam.switch_orientation(normal_vector=normal_vector,north_vector=north_vector)
            sc.add_source(vol)
            sc.render()
            # sc.save('tmp2.png',sigma_clip = 6.0)
            # sc = yt.create_scene(asc.ds,lens_type = 'perspective')
            # sc.camera.zoom(2.0)
            # sc[0].tfh.set_bounds([1e-4,1e2])
            os.system("mkdir -p frames")
            sc.save(fname,sigma_clip = 6.0)

def mk_frame_3D_uniform_grid():
	set_dump_range()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
			fname_rho = "frames/3Dframe_uniform_grid_rho_%d.png" % (i_dump)
			fname_T = "frames/3Dframe_uniform_grid_T_%d.png" % (i_dump)
			if os.path.isfile(fname_rho):
				continue
			else:
				asc.yt_load(i_dump)
				box_radius = 1.0 
				region = asc.ds.r[(-box_radius,'pc'):(box_radius,'pc'):512j,
				(-box_radius,'pc'):(box_radius,'pc'):512j,
				(-box_radius,'pc'):(box_radius,'pc'):512j ]
				x,y,z = region['x'],region['y'],region['z']
				import numpy as np

				bbox = np.array([[-box_radius,box_radius],[-box_radius,box_radius],[-box_radius,box_radius]])
				rho = region['density']
				press = region['press']
				T  = press/rho *  mu_highT*mp_over_kev*keV_to_Kelvin
				data =  dict(density = (np.array(rho),"Msun/pc**3"),temperature = (np.array(T),"K"),x = (np.array(x),"pc"), y = (np.array(y),"pc"),z = (np.array(z),"pc"))
				ds = yt.load_uniform_grid(data,rho.shape,length_unit="pc",bbox=bbox)

				#phi = np.linspace(0,2*pi,100)
				#		for iphi in range(100):
				from yt.visualization.volume_rendering.api import Scene, VolumeSource
				sc  = Scene()
				vol = VolumeSource(ds, field="density")
				bound_min = ds.arr(1e-2,"Msun/pc**3.").in_cgs()
				bound_max = ds.arr(10.**1.5,"Msun/pc**3.").in_cgs()
				tf_min = ds.arr(1e-2,"Msun/pc**3.").in_cgs()
				tf_max = ds.arr(10.**0.5,"Msun/pc**3.").in_cgs()
				bounds = (bound_min, bound_max)

				tf = yt.ColorTransferFunction(np.log10(bounds))
				def linramp(vals, minval, maxval):
					return (vals - vals.min())/(vals.max() - vals.min())
				#tf.map_to_colormap(np.log10(bounds[0]),np.log10(bounds[1]),colormap='ocean',scale_func=linramp)
				tf.add_layers(8, colormap='ocean',mi = np.log10(tf_min),ma = np.log10(tf_max),col_bounds=([np.log10(tf_min),np.log10(tf_max)])) #,w = 0.01,alpha = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])  #ds_highcontrast
				tf.add_step(np.log10(tf_max*2),np.log10(bound_max), [0.5,0.5,0.5,1.0])
				tf.grey_opacity = False
				vol.transfer_function = tf
				vol.tfh.tf = tf
				vol.tfh.bounds = bounds
				vol.tfh.plot('transfer_function.png', profile_field="density")
				cam = sc.add_camera(ds, lens_type='perspective')
				cam.resolution = [512,512]
				# cam.position = ds.arr(np.array([282.*np.cos(theta)*np.sin(phi),282.*np.cos(theta)*np.cos(phi),282.*np.sin(theta)]), 'code_length')
				# cam.switch_orientation(normal_vector=normal_vector,
				#                        north_vector=north_vector)
				cam.set_width(ds.domain_width*0.25)

				#cam.position = ds.arr(np.array([0.5*np.sin(phi),,0.5*np.cos(phi)]),'code_length')
				cam.position = ds.arr(np.array([0,0,-0.5]), 'code_length')   #CHANGE WITH PHI
				normal_vector = [0,0,-1]  #camera to focus  #CHANGE WITH PHI
				north_vector = [0,1,0]  #up direction   
				cam.switch_orientation(normal_vector=normal_vector,north_vector=north_vector)
				sc.add_source(vol)
				sc.render()
				# sc.save('tmp2.png',sigma_clip = 6.0)
				# sc = yt.create_scene(asc.ds,lens_type = 'perspective')
				# sc.camera.zoom(2.0)
				# sc[0].tfh.set_bounds([1e-4,1e2])
				os.system("mkdir -p frames")
				sc.save(fname_rho,sigma_clip = 6.0)


				############  TEMPERATURE ##############
				sc  = Scene()
				vol = VolumeSource(ds, field="temperature")
				bound_min = ds.arr(1e5,"K").in_cgs()
				bound_max = ds.arr(1e9,"K").in_cgs()
				bounds = (bound_min, bound_max)

				tf = yt.ColorTransferFunction(np.log10(bounds))
				def linramp(vals, minval, maxval):
					return (vals - vals.min())/(vals.max() - vals.min())
				#tf.map_to_colormap(np.log10(bounds[0]),np.log10(bounds[1]),colormap='ocean',scale_func=linramp)
				tf.add_step(np.log10(3e7),np.log10(1e9),[1.0,0.0,0.0,0.5])
				tf.add_step(np.log10(1e6),np.log10(3e7),[0.5,0.0,0.5,0.2])
				tf.add_step(np.log10(1e5),np.log10(1e6),[0.0,0.0,1.0,1.0])
				tf.grey_opacity = False
				vol.transfer_function = tf
				vol.tfh.tf = tf
				vol.tfh.bounds = bounds
				vol.tfh.plot('transfer_function.png', profile_field="temperature")
				cam = sc.add_camera(ds, lens_type='perspective')
				cam.resolution = [512,512]
				# cam.position = ds.arr(np.array([282.*np.cos(theta)*np.sin(phi),282.*np.cos(theta)*np.cos(phi),282.*np.sin(theta)]), 'code_length')
				# cam.switch_orientation(normal_vector=normal_vector,
				#                        north_vector=north_vector)
				cam.set_width(ds.domain_width*0.25)

				#cam.position = ds.arr(np.array([0.5*np.sin(phi),,0.5*np.cos(phi)]),'code_length')
				cam.position = ds.arr(np.array([0,0,-0.5]), 'code_length')   #CHANGE WITH PHI
				normal_vector = [0,0,-1]  #camera to focus  #CHANGE WITH PHI
				north_vector = [0,1,0]  #up direction   
				cam.switch_orientation(normal_vector=normal_vector,north_vector=north_vector)
				sc.add_source(vol)
				sc.render()
				# sc.save('tmp2.png',sigma_clip = 6.0)
				# sc = yt.create_scene(asc.ds,lens_type = 'perspective')
				# sc.camera.zoom(2.0)
				# sc[0].tfh.set_bounds([1e-4,1e2])
				os.system("mkdir -p frames")
				sc.save(fname_T,sigma_clip = 6.0)

def mk_frame_grmhd(is_magnetic=True):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if os.path.isfile(fname):
			dummy = i_dump
		else:
			plt.figure(1)
			plt.clf()
			asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2=True,gr=True,a=a)
			c1 = plt.pcolormesh((asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho)[:,:,0],vmin=-2,vmax=2,cmap="ocean")
			c2 = plt.pcolormesh(-(asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho)[:,:,asc.nz//2],vmin=-2,vmax=2,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-100,100)
			plt.xlim(-100,100)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %g M$' %(asc.t),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			os.system("mkdir -p frames")
			plt.savefig(fname)

		fname = "frames/frame_midplane_%04d.png" % (i_dump)
		if os.path.isfile(fname):
			continue
		else: 
			plt.figure(2)
			plt.clf()
			asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2=True,gr=True,a=a)
			c1 = plt.pcolormesh((asc.x)[:,asc.ny//2,:], (asc.y)[:,asc.ny//2,:],np.log10(asc.rho)[:,asc.ny//2,:],vmin=-2,vmax=2,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			plt.ylim(-100,100)
			plt.xlim(-100,100)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %g M$' %(asc.t),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			os.system("mkdir -p frames")
			plt.savefig(fname)

def mk_frame_gr_magnetically_frustrated(is_magnetic=True):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if os.path.isfile(fname):
			dummy = i_dump
		else:
			plt.figure(1)
			plt.clf()
			asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2=False,gr=True,a=a)
			c1 = plt.pcolormesh((asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho)[:,:,0],vmin=-2,vmax=2,cmap="ocean")
			c2 = plt.pcolormesh(-(asc.r*np.sin(asc.th))[:,:,0], (asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.rho)[:,:,asc.nz//2],vmin=-2,vmax=2,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(asc.psicalc_slice(gr=True,iphi=0)*(r[:,:,0]<200)),30,colors='white')
			# # if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(-asc.psicalc_slice(gr=True,iphi=0)*(r[:,:,0]<200)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(asc.psicalc_slice(gr=True,iphi=asc.nz//2)*(r[:,:,0]<200)),30,colors='white')
			# # if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)*(r[:,:,0]<200)),30,linestyles='--',colors='white')

			plt.ylim(-100,100)
			plt.xlim(-100,100)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %np.int(asc.t),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			os.system("mkdir -p frames")
			plt.savefig(fname)

			plt.xlim(-30,30)
			plt.ylim(-30,30)

			plt.savefig("frames/frame_zoom_in_%04d.png" % (i_dump))

			plt.xlim(-100,100)
			plt.ylim(-100,100)

			asc.x = -asc.x 
			asc.plot_fieldlines_gr(100)

			#if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(asc.psicalc_slice(gr=True,iphi=0)*(r[:,:,0]<200)),30,colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(-asc.psicalc_slice(gr=True,iphi=0)*(r[:,:,0]<200)),30,linestyles='--',colors='white')
			#if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(asc.psicalc_slice(gr=True,iphi=asc.nz//2)*(r[:,:,0]<200)),30,colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)*(r[:,:,0]<200)),30,linestyles='--',colors='white')
			plt.savefig("frames/frame_fieldlines_%04d.png" % (i_dump))

		# fname = "frames/frame_midplane_%04d.png" % (i_dump)
		# if os.path.isfile(fname):
		# 	continue
		# else: 
		# 	plt.figure(2)
		# 	plt.clf()
		# 	asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2=True,gr=True,a=a)
		# 	c1 = plt.pcolormesh((asc.x)[:,asc.ny//2,:], (asc.y)[:,asc.ny//2,:],np.log10(asc.rho)[:,asc.ny//2,:],vmin=-2,vmax=2,cmap="ocean")
		# 	cb1 = plt.colorbar(c1) 
		# 	plt.ylim(-100,100)
		# 	plt.xlim(-100,100)
		# 	plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
		# 	plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
		# 	plt.title(r'$t = %g M$' %(asc.t),fontsize = 20)

		# 	cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

		# 	for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
		# 	    label.set_fontsize(10)

		# 	os.system("mkdir -p frames")
		# 	plt.savefig(fname)
def mk_frame_grmhd_cartesian(is_magnetic=False):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()
			asc.yt_extract_box(i_dump,30,mhd=is_magnetic,gr=True,a=a)
			ny = asc.rho.shape[1]
			c1 = plt.pcolormesh((asc.x)[:,0,:], (asc.z)[:,0,:],np.log10(asc.rho)[:,ny//2,:],vmin=-3,vmax=0,cmap="ocean")
			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-30,30)
			plt.xlim(-30,30)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %np.int(np.array(asc.ds.current_time)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole()
			plt.axis('off')

			plt.axes().set_aspect('equal')

			plt.tight_layout()

			os.system("mkdir -p frames")
			plt.savefig(fname)

def mk_frame_grmhd_restart_cartesian(is_magnetic=False):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()
			asc.rd_yt_convert_to_spherical(i_dump,MHD=True,gr=True,a=a,th=th_tilt,ph=phi_tilt)
			ny = asc.rho.shape[1]
			c1 = plt.pcolormesh((asc.x)[:,:,0], (asc.z)[:,:,0],np.log10(asc.rho)[:,:,0],vmin=-3,vmax=0,cmap="ocean")
			plt.pcolormesh(-(asc.x)[:,:,0], (asc.z)[:,:,0],np.log10(asc.rho)[:,:,ny//2],vmin=-3,vmax=0,cmap="ocean")

			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-50,50)
			plt.xlim(-50,50)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %np.int(np.array(asc.t)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole()
			plt.axis('off')

			plt.axes().set_aspect('equal')

			#plt.tight_layout()

			os.system("mkdir -p frames")
			plt.savefig(fname)

			asc.cks_coord_to_ks(asc.x,asc.y,asc.z,a=a)
			asc.plot_fieldlines_gr(50,a=a,npz=True)
			fname = "frames/frame_fieldlines_%04d.png" % (i_dump)

			plt.clf()

			c1 = plt.pcolormesh((asc.x)[:,ny//2,:], (asc.y)[:,ny//2,:],np.log10(asc.rho)[:,ny//2,:],vmin=-2,vmax=0.5,cmap="ocean")

			cb1 = plt.colorbar(c1) 

			plt.ylim(-50,50)
			plt.xlim(-50,50)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %np.int(np.array(asc.t)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole()
			plt.axis('off')

			plt.axes().set_aspect('equal')

			#plt.tight_layout()

			os.system("mkdir -p frames")
			fname = "frames/frame_midplane_%04d.png" % (i_dump)
			plt.savefig(fname)

def mk_frame_gr_magnetically_frustrated_cartesian(is_magnetic=False):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("framing dump %d" %i_dump)
		fname = "frames/frame_%04d.png" % (i_dump)
		if False: ##os.path.isfile(fname):
			continue
		else:
			plt.figure(1)
			plt.clf()
			asc.rd_yt_convert_to_spherical(i_dump,MHD=True,gr=True,a=a,th=th_tilt,ph=phi_tilt)
			ny = asc.rho.shape[1]
			c1 = plt.pcolormesh((asc.x)[:,:,0], (asc.z)[:,:,0],np.log10(asc.rho)[:,:,0],vmin=-2,vmax=2,cmap="ocean")
			plt.pcolormesh(-(asc.x)[:,:,0], (asc.z)[:,:,0],np.log10(asc.rho)[:,:,ny//2],vmin=-2,vmax=2,cmap="ocean")

			cb1 = plt.colorbar(c1) 
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour((asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=0)),30,linestyles='--',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='-',colors='white')
			# if (is_magnetic == True): plt.contour(-(asc.r*np.sin(asc.th))[:,:,0],(asc.r*np.cos(asc.th))[:,:,0],np.log10(-asc.psicalc_slice(gr=True,iphi=asc.nz//2)),30,linestyles='--',colors='white')

			plt.ylim(-50,50)
			plt.xlim(-50,50)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$z$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %np.int(np.array(asc.t)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole()
			plt.axis('off')

			plt.axes().set_aspect('equal')

			#plt.tight_layout()

			os.system("mkdir -p frames")
			plt.savefig(fname)

			plt.xlim(-500,500)
			plt.ylim(-500,500)
			fname = "frames/frame_jet_%04d.png" % (i_dump)
			plt.savefig(fname)

			plt.ylim(-50,50)
			plt.xlim(-50,50)


			asc.cks_coord_to_ks(asc.x,asc.y,asc.z,a=a)
			asc.plot_fieldlines_gr(50,a=a,npz=True)
			fname = "frames/frame_fieldlines_%04d.png" % (i_dump)

			plt.savefig(fname)



			plt.clf()

			c1 = plt.pcolormesh((asc.x)[:,ny//2,:], (asc.y)[:,ny//2,:],np.log10(asc.rho)[:,ny//2,:],vmin=-2,vmax=0.5,cmap="ocean")

			cb1 = plt.colorbar(c1) 

			plt.ylim(-50,50)
			plt.xlim(-50,50)
			plt.xlabel(r'$x$ ($r_G$)',fontsize = 20)
			plt.ylabel(r'$y$ ($r_G$)',fontsize = 20)
			plt.title(r'$t = %d M$' %np.int(np.array(asc.t)),fontsize = 20)

			cb1.set_label(r"$\log_{10}(\rho)$",fontsize=17)

			for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb1.ax.get_yticklabels():
			    label.set_fontsize(10)

			asc.bhole()
			plt.axis('off')

			plt.axes().set_aspect('equal')

			#plt.tight_layout()

			os.system("mkdir -p frames")
			fname = "frames/frame_midplane_%04d.png" % (i_dump)
			plt.savefig(fname)

def mk_1d_quantities(is_magnetic =True):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		asc.rdhdf5(i_dump,ndim=3,coord="spherical",user_x2=True,gr=True,a=a)
		mdot =  asc.angle_average(asc.rho*asc.uu[1]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2), gr=True)
		Phibh = asc.phibh()*np.sqrt(4*np.pi)
		ud = asc.Lower(asc.uu,asc.g)
		bd = asc.Lower(asc.bu,asc.g)
		asc.Tud_calc(asc.uu,ud,asc.bu,bd,is_magnetic= is_magnetic)
		Jdot = asc.angle_average(asc.Tud[1][3]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True)
		Edot = - (asc.angle_average(asc.Tud[1][0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True) +mdot )
		EdotEM = -asc.angle_average(asc.TudEM[1][0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True)


		dic = {}
		dic['t'] = asc.t
		dic['r'] = asc.r[::4,0,0]
		dic['mdot'] = mdot[::4]
		dic['Jdot'] = Jdot[::4]
		dic['Edot'] = Edot[::4]
		dic['Phibh'] = Phibh[::4]
		dic['EdotEM'] = EdotEM[::4]

		np.savez("1d_dump_%04d.npz" %i_dump,**dic)
def mk_1d_quantities_cartesian(is_magnetic =True):
	set_dump_range_gr()
	print ("Processing dumps ",i_start," to ",i_end)
	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		dump_npz = "dump_spher_%d_th_%.2g_phi_%.2g.npz" %(i_dump,th_tilt,phi_tilt)
		if glob.glob("*out2*athdf") != []:
			dump_file_prefix = glob.glob("*out2*.athdf")[0][:-11]
			dump_hdf5 = dump_file_prefix + "%05d.athdf" %i_dump
		else: dump_hdf5 = ""
		if (os.path.isfile(dump_npz) or os.path.isfile(dump_hdf5) ): asc.rd_yt_convert_to_spherical(i_dump,MHD=True,gr=True,a=a,th=th_tilt,ph=phi_tilt)
		else: 
			print("Skipping dump: ", i_dump)
			continue 
		asc.get_mdot(mhd=True,gr=True,a=a)
		asc.th = np.arccos(asc.z/asc.r)
		asc.ph = np.arctan2(asc.y,asc.x)
		mdot = asc.angle_average_npz(nan_to_num(asc.mdot), gr=True,a=a)
		Br = (asc.bu_ks[1] * asc.uu_ks[0] - asc.bu_ks[0]* asc.uu_ks[1])

		Phibh = asc.angle_average_npz(0.5*np.fabs(Br)*np.sqrt(4.0*pi)*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a) 


		asc.ks_metric(asc.r,asc.th,a)

		uu_ks = nan_to_num(asc.uu_ks)
		bu_ks = nan_to_num(asc.bu_ks)
		ud_ks = nan_to_num(asc.Lower(asc.uu_ks,asc.g))
		bd_ks = nan_to_num(asc.Lower(asc.bu_ks,asc.g))


		asc.Tud_calc(asc.uu_ks,ud_ks,asc.bu_ks,bd_ks,is_magnetic= is_magnetic,gam=gam)
		Jdot = asc.angle_average_npz(asc.Tud[1][3]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a)
		Edot = - (asc.angle_average_npz(asc.Tud[1][0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a) +mdot )
		EdotEM = -asc.angle_average_npz(asc.TudEM[1][0]*4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2),gr=True,a=a)

		Lx = asc.angle_average_npz(asc.rho* (asc.y*asc.uu[3] - asc.z*asc.uu[2]),gr=True,a=a)
		Ly = asc.angle_average_npz(asc.rho* (asc.z*asc.uu[1] - asc.x*asc.uu[3]),gr=True,a=a)
		Lz = asc.angle_average_npz(asc.rho* (asc.x*asc.uu[2] - asc.y*asc.uu[1]),gr=True,a=a)

		Bx = asc.angle_average_npz(asc.Bcc1,gr=True,a=a)
		By = asc.angle_average_npz(asc.Bcc2,gr=True,a=a)
		Bz = asc.angle_average_npz(asc.Bcc3,gr=True,a=a)

		A_jet_p =  asc.angle_average_npz( 4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2) * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th<np.pi/2.0) ,gr=True,a = a)
		A_jet_m =  asc.angle_average_npz( 4.0 * np.pi/3.0 * (3.0*asc.r**2 + a**2) * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th>=np.pi/2.0) ,gr=True,a = a)

		rjet_max_p= (np.amax(asc.r[A_jet_p>0]))
		rjet_max_m= (np.amax(asc.r[A_jet_m>0]))
		wgt = asc.bsq/2.0 * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th<np.pi/2.0)

		x_jet_p = angle_average_npz( asc.x,weight=wgt ,gr=True,a = a)
		y_jet_p = angle_average_npz( asc.y,weight=wgt ,gr=True,a = a)
		z_jet_p = angle_average_npz( asc.z,weight=wgt ,gr=True,a = a)

		#asc.cks_metric(asc.x,asc.y,asc.z,a)
		asc.cks_inverse_metric(asc.x,asc.y,asc.z,a)
		alpha = np.sqrt(-1.0/asc.gi[0,0])
		gamma = asc.uu[0] * alpha

		gamma_jet_p = angle_average_npz(gamma,weight= ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th<np.pi/2.0),gr=True,a=a)
		gamma_jet_m = angle_average_npz(gamma,weight= ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th>np.pi/2.0),gr=True,a=a)


		wgt = asc.bsq/2.0 * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) * (asc.th>=np.pi/2.0)

		x_jet_m = asc.angle_average_npz( asc.x,weight=wgt ,gr=True,a = a)
		y_jet_m = asc.angle_average_npz( asc.y,weight=wgt ,gr=True,a = a)
		z_jet_m = asc.angle_average_npz( asc.z,weight=wgt ,gr=True,a = a)

		wgt = asc.bsq/2.0 * ((asc.bsq/2.0 - 1.5 * asc.rho)>0) 
		vr_jet = asc.angle_average_npz(asc.uu_ks[1]/asc.uu_ks[0],weight=wgt,gr=True,a=a)

		dic = {}
		dic['t'] = np.array(asc.t)
		dic['r'] = asc.r[::4,0,0]
		dic['mdot'] = mdot[::4]
		dic['Jdot'] = Jdot[::4]
		dic['Edot'] = Edot[::4]
		dic['Phibh'] = Phibh[::4]
		dic['EdotEM'] = EdotEM[::4]
		dic['Lx'] = Lx[::4]
		dic['Ly'] = Ly[::4]
		dic['Lz'] = Lz[::4]
		dic['Bx'] = Bx[::4]
		dic['By'] = By[::4]
		dic['Bz'] = Bz[::4]
		dic['A_jet_p'] = A_jet_p[::4]
		dic['A_jet_m'] = A_jet_m[::4]
		dic['x_jet_p'] = x_jet_p[::4]
		dic['y_jet_p'] = y_jet_p[::4]
		dic['z_jet_p'] = z_jet_p[::4]
		dic['x_jet_m'] = x_jet_m[::4]
		dic['y_jet_m'] = y_jet_m[::4]
		dic['z_jet_m'] = z_jet_m[::4]
		dic['rjet_max_p'] = rjet_max_p
		dic['rjet_max_m'] = rjet_max_m
		dic['gamma_jet_m'] = gamma_jet_m[::4]
		dic['gamma_jet_p'] = gamma_jet_p[::4]
		dic['vr_jet'] = vr_jet[::4]

		np.savez("1d_dump_%04d.npz" %i_dump,**dic)
		dic_torus = {}

		thmin = pi/3.0
		thmax = 2.0*pi/3.0

		dic_torus['t'] = np.array(asc.t)
		dic_torus['r'] = asc.r[:,0,0]
		dic_torus['rho'] = asc.angle_average_npz(asc.rho,weight = (asc.th>thmin)*(asc.th<thmax),gr=True,a=a)
		dic_torus['press'] = asc.angle_average_npz(asc.press,weight = (asc.th>thmin)*(asc.th<thmax),gr=True,a=a)
		dic_torus['beta_inv'] =asc.angle_average_npz(asc.bsq/asc.press/2.0,weight = (asc.th>thmin)*(asc.th<thmax),gr=True,a=a)
		dic_torus['pmag'] = asc.angle_average_npz(asc.bsq/2.0,weight = (asc.th>thmin)*(asc.th<thmax),gr=True,a=a)


		np.savez("1d_torus_dump_%04d.npz" %i_dump,**dic_torus)

def mk_RM(moving=False):
	set_dump_range()
	os.system("cp /global/scratch/smressle/star_cluster/mhd_runs/star_inputs/PSR.dat ./")
	asc.PSR_pos()
	print ("Processing dumps ",i_start," to ",i_end)
	angle_array = np.linspace(0,2.0*np.pi,10)
	radius = np.sqrt(asc.dalpha[0]**2 + asc.ddelta[0]**2)
	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		asc.yt_load(i_dump)
		sgra_RM,sgra_DM = asc.get_RM()

		# RM_rand = []
		# DM_rand = []
		# # for angle in angle_array:
		# # 	x_ = radius * np.cos(angle)
		# # 	y_ = radius * np.sin(angle)
		# # 	RM,DM = asc.get_RM(x_,y_,cum = True)[-1]
		# # 	RM_rand.append(RM)
		# # 	DM_rand.append(DM)
		# RM_rand = np.array(RM_rand)
		# DM_rand = np.array(DM_rand)

		simulation_start_time = -1.1
		if (moving == True):
			xp  = asc.dalpha[0] + (np.array(asc.ds.current_time) - (asc.t_yr[0]-2017)/1e3+simulation_start_time)*asc.valpha
			yp  = asc.ddelta[0] + (np.array(asc.ds.current_time) - (asc.t_yr[0]-2017)/1e3+simulation_start_time)*asc.vdelta
		else:
			xp = asc.dalpha[0]
			yp = asc.ddelta[0]
		pulsar_RM,pulsar_DM = asc.get_RM(xp,yp,cum=True)

		dic =  {"t":asc.ds.current_time,"pulsar_RM":pulsar_RM,"pulsar_DM": pulsar_DM,"sgra_RM":sgra_RM,"sgra_DM": sgra_DM,"z_los":asc.z_los} #,"pulsar_RM_rand":RM_rand,"pulsar_DM_rand":DM_rand}
		
		if (moving==True): np.savez("RM_dump_moving_%04d.npz" %i_dump,**dic)
		else: np.savez("RM_dump_%04d.npz" %i_dump,**dic)


def RM_movie():
	from matplotlib.offsetbox import AnchoredText

	set_dump_range()

	e_charge = 4.803e-10
	me = 9.109e-28
	cl = 2.997924e10
	mp = 1.6726e-24
	pc = 3.086e18
	kyr = 3.154e10
	msun = 1.989e33
	R_sun = 6.955e10
	km_per_s = 1e5

	Z_o_X_solar = 0.0177
	Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
	X_solar = 0.7491
	Z_solar = 1.0-X_solar - Y_solar
	from yt.units.yt_array import YTArray

	muH_solar = 1./X_solar
	Z = 3. * Z_solar
	X = 0.
	mue = 2. /(1.+X)



	def _RM_integrand(field,data):
		ne= data["rho"].in_cgs()/mp/mue
		B_par = data["Bcc3"].in_cgs()
		return YTArray(np.array(ne * B_par * e_charge**3/(2.0*np.pi * me**2 * cl**4)),'cm**-3')

	#fig = plt.figure(figsize=(10,10))
	#fig.patch.set_facecolor('black')

	for i_dump in range(i_start,i_end):
		print ("Calculating from dump %d" %i_dump)
		asc.yt_load(i_dump)
		asc.ds.add_field(("gas","RM_integrand"),function = _RM_integrand,units="cm**-3",particle_type = False,sampling_type="cell",force_override=True)


		box_radius = 0.2
		region = asc.ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,  #512
		    (-box_radius,'pc'):(box_radius,'pc'):256j,   #512
		    (-1,'pc'):(1,'pc'):1028j ]
		RM_map = np.array(region['RM_integrand'].mean(-1).in_cgs()) * 2 * pc

		for sat in [True,False]:
			plt.clf()
			plt.style.use('dark_background')
			max_RM = 1.5
			min_RM = -1.5
			#c = matplotlib.pyplot.pcolormesh(region['x'].mean(-1),region['y'].mean(-1),log10(abs(RM_map)),cmap = "RdBu",vmin=np.log10(6.6)-1.5,vmax=np.log10(6.6)+1.5)
			if sat == True : c = matplotlib.pyplot.contourf(region['x'].mean(-1),region['y'].mean(-1),log10(abs(RM_map)),cmap = "autumn",levels = np.linspace(np.log10(6.6)-.01,max_RM,200),extend = "max")
			else: c = matplotlib.pyplot.contourf(region['x'].mean(-1),region['y'].mean(-1),log10(abs(RM_map)),cmap = "cubehelix",levels = np.linspace(min_RM,max_RM,200),extend = "both")
			plt.xlim(box_radius,-box_radius)
			plt.ylim(-box_radius,box_radius)

			plt.axis('off')
			#plt.axes().set_aspect('equal')


			# text_box = AnchoredText(r'$t$ = %g yr' %(np.int(np.round(asc.t*1000-1100))), frameon=True, loc=4, pad=0.5)
			# text_box = AnchoredText(r'$t$ = %g yr' %(np.int(np.round(t*1000-1100))), frameon=True, loc=4, pad=0.5)
			# plt.setp(text_box.patch, facecolor='black', alpha=.9)
			# plt.gca().add_artist(text_box)

			plt.tight_layout()

			os.system("mkdir -p frames")
			if (sat == True):
			    plt.savefig("frames/frame_RM_sat_%d.png" % (i_dump))
			else: 
			    plt.savefig("frames/frame_RM_%d.png" % (i_dump))


def mk_Xray_frame():
	from matplotlib.colors import LinearSegmentedColormap

	set_dump_range()

	for i_dump in range(i_start,i_end):
		asc.yt_load(i_dump)
		plt.clf()
		asc.get_Xray_Lum('Lam_spex_Z_solar_2_8_kev',1.0,make_image=True)

		colors = ["black","#004851","#8D9093","#FFFFFF" ] #"#000000","#54585A" ,"#8D9093","#FFFFFF", "#004851"]  # R -> G -> B

		#colors = ["#000000", "#4cbb17","#FFFFFF" ] #"#000000","#54585A" ,"#8D9093","#FFFFFF", "#004851"]  # R -> G -> B
		#colors = ["#4cbb17", "#C0C0C0"]
		cmap_name = 'my_list'
		cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

		cm = 'magma'


		x_im = np.array(x_im.tolist())
		y_im = np.array(y_im.tolist())
		image = np.array(image.tolist())
		fac = np.int(image.shape[0]/50.)+1
		new_x = np.linspace(-0.492*50,0.492*50,fac*100)  #np.arange(-50*fac,50*fac)*0.492
		new_y = np.linspace(-0.492*50,0.492*50,fac*100)
		new_x,new_y = meshgrid(new_x,new_y,indexing='ij')
		from scipy.interpolate import griddata
		new_image = griddata((x_im.flatten()/arc_secs,y_im.flatten()/arc_secs), image.flatten(), (new_x.flatten(), new_y.flatten()), method='nearest')
		new_image = new_image.reshape(new_x.shape[0],new_y.shape[1])

		coarseness = fac
		temp = new_image.reshape((new_image.shape[0] // coarseness, coarseness,
		            new_image.shape[1] // coarseness, coarseness))
		coarse_new_image = np.mean(temp, axis=(1,3))
		temp = new_x.reshape((new_x.shape[0] // coarseness, coarseness,
		            new_x.shape[1] // coarseness, coarseness))
		coarse_new_x = np.mean(temp, axis=(1,3))
		temp = new_y.reshape((new_y.shape[0] // coarseness, coarseness,
		            new_y.shape[1] // coarseness, coarseness))
		coarse_new_y = np.mean(temp, axis=(1,3))

		#c1 = plt.pcolormesh(x_im/arc_secs,y_im/arc_secs,log10(image),levels = np.linspace(-5,0,200),cmap = 'magma') #extend= 'both',cmap = cm) #"magma")
		c1 = plt.pcolormesh(coarse_new_x,coarse_new_y,log10(coarse_new_image*3.575e-93), cmap = cm,vmin = -3.5,vmax=-0.5) #vmin = -3,vmax = 0) #levels = np.linspace(-3.5,0,200),cmap = 'magma') #extend= 'both',cmap = cm) #"magma")

		#c1 = plt.pcolormesh(coarse_new_x,coarse_new_y,log10(coarse_new_image*3.575e-93), cmap = cm,vmin = -2.5,vmax=1) #vmin = -3,vmax = 0) #levels = np.linspace(-3.5,0,200),cmap = 'magma') #extend= 'both',cmap = cm) #"magma")

		#c1 = plt.pcolormesh(new_x,new_y,log10(new_image*3.575e-93), cmap = cm,vmin = -3.5,vmax=-0.5)
		plt.gca().invert_xaxis()
		#c1 = plt.pcolormesh(x_im[::3,::3]/arc_secs,y_im[::3,::3]/arc_secs,log10(image[::3,::3]), cmap = 'magma',vmin = -2,vmax = 0)
		#
		cb1 = plt.colorbar(c1)
		cb1.set_ticks(np.arange(-10,10,.5))
		#    cb1.set_label(r"$\rho/\langle \rho \rangle - 1 $",fontsize=17)
		#    

		for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels() +cb1.ax.get_yticklabels():
			label.set_fontsize(10)


		plt.xlabel(r'$\Delta$RA Offset from Sgr A* (arcsec)',fontsize=17)
		plt.ylabel(r'$\Delta$Dec Offset from Sgr A* (arcsec)',fontsize=17)
		cb1.set_label(r"X-ray Sfc. Brt. (erg/cm$^2$/s)",fontsize=17)


		plt.axes().set_aspect('equal')
		xlim(10,-10)
		ylim(-10,10)
		plt.savefig("frame_X_ray_%03d.png")
    #circ = matplotlib.patches.Circle((dalpha[0]/arc_secs,ddelta[0]/arc_secs),radius = .5,fill=False,ls='--',lw=3,color='yellow')
    #matplotlib.pyplot.gca().add_artist(circ)
    # xlim(5,-5)
    # ylim(-5,5)


def mk_frame_disk():
	set_dump_range()

	os.system("mkdir -p frames")
	for idump in range(i_start,i_end):
		asc.rdnpz("dump_spher_disk_frame_%04d.npz" %idump)
		asc.get_mdot()
		nx = asc.x.shape[0]
		ny = asc.x.shape[1]
		nz = asc.x.shape[2]
		x = asc.x*1e3 #x_tavg*1e3
		y = asc.y*1e3
		z = asc.z*1e3 #z_tavg*1e3
		plt.figure(1)
		plt.clf()
		r_tmp = np.sqrt(x**2. + y**2. + z**2.)
		c = plt.contourf(x[:,:,0],z[:,:,0],np.log10((asc.rho*r_tmp/1e3)[:,:,0]),levels = np.linspace(-2,2,200),extend='both',cmap = 'bds_highcontrast')
		plt.contourf(-x[:,:,0],z[:,:,0],np.log10((asc.rho*r_tmp/1e3)[:,:,nz//2]),levels = np.linspace(-2,2,200),extend='both',cmap = 'bds_highcontrast')

		plt.xlabel(r'$x$ (mpc)',fontsize = 20)
		plt.ylabel(r'$z$ (mpc)',fontsize = 20)

		cb = plt.colorbar(c,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
		cb.set_label(r"$\langle \rho r \rangle$ $M_\odot/$pc$^2$",fontsize=17)


		for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels():
			label.set_fontsize(10)
		plt.tight_layout()

		fac = 1
		plt.xlim(-0.003*3.*1e3*fac,0.003*3.*1e3*fac)
		plt.ylim(-0.003*3.*1e3*fac,0.003*3.*1e3*fac)
		plt.savefig('frames/frame_rho_disk_%04d.png' %idump)


		plt.figure(2)
		plt.clf()

		c = plt.pcolormesh(x[:,ny//2,:],y[:,ny//2,:], np.log10(asc.rho[:,ny//2,:]),cmap = 'bds_highcontrast',vmin=1,vmax=3.5)
		#plt.pcolormesh(x[:,ny//2,:],y[:,ny//2,:], np.log10(Bth*Br)[:,ny//2,:],cmap = 'cubehelix',vmin=-1,vmax=4.25)

		cb = plt.colorbar(c,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
		cb.set_label(r"$\langle \rho \rangle$ $M_\odot/$pc$^3$",fontsize=17)


		fac = 1 #30 #1 ##30
		plt.xlim(-0.003*3*1e3*fac,.003*3*1e3*fac)
		plt.ylim(-0.003*3*1e3*fac,.003*3*1e3*fac)


		for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels():
			label.set_fontsize(10)
		plt.tight_layout()
		plt.savefig('frames/frame_rho_disk_midplane_%04d.png' %idump)


def mk_frame_L_aligned(fieldlines=False):
	set_dump_range()

	if len(sys.argv)>4:
		th_l = np.float(sys.argv[4])
		phi_l = np.float(sys.argv[5])
	else:
		th_l = 1.3
		phi_l = -1.8

	os.system("mkdir -p frames")
	for idump in range(i_start,i_end):
		asc.rdnpz("dump_spher_%d_th_%g_phi_%g.npz" %(idump,th_l,phi_l))
		nx = asc.x.shape[0]
		ny = asc.x.shape[1]
		nz = asc.x.shape[2]
		x = asc.x*1e3 #x_tavg*1e3
		y = asc.y*1e3
		z = asc.z*1e3 #z_tavg*1e3

		asc.x = x
		asc.y = y
		asc.z = z

		if (fieldlines==True): asc.get_mdot(True)

		for fac in [0.33,1,10]:
			plt.figure(1)
			plt.clf()
			r_tmp = np.sqrt(x**2. + y**2. + z**2.)
			c = plt.contourf(x[:,:,0],z[:,:,0],np.log10((asc.rho*r_tmp/1e3)[:,:,0]),levels = np.linspace(-2,2,200),extend='both',cmap = 'viridis',vmin=-1.75,vmax=-.25)
			plt.contourf(-x[:,:,0],z[:,:,0],np.log10((asc.rho*r_tmp/1e3)[:,:,nz//2]),levels = np.linspace(-2,2,200),extend='both',cmap = 'viridis',vmin=-1.75,vmax=-.25)

			if (fieldlines==True): asc.plot_fieldlines_slice(9*fac)

			plt.xlabel(r'$x$ (mpc)',fontsize = 20)
			plt.ylabel(r'$z$ (mpc)',fontsize = 20)

			# cb = plt.colorbar(c,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
			# cb.set_label(r"$\langle \rho r \rangle$ $M_\odot/$pc$^2$",fontsize=17)


			# for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels():
			# 	label.set_fontsize(10)
			plt.tight_layout()

			plt.xlim(-9*fac,9*fac)
			plt.ylim(-9*fac,9*fac)

			plt.axis('off')
			plt.axes().set_aspect('equal')
			if (fieldlines==True):plt.savefig('frames/frame_rho_phi_slice_fieldlines_fac_%g_%04d.png' %(fac,idump))
			else: plt.savefig('frames/frame_rho_phi_slice_fac_%g_%04d.png' %(fac,idump))


			plt.figure(2)
			plt.clf()

			if (fac ==1 or fac ==0.33): c = plt.pcolormesh(x[:,ny//2,:],y[:,ny//2,:], np.log10((asc.rho*r_tmp/1e3)[:,ny//2,:]),cmap = 'viridis',vmin=-1.5,vmax=-0.5)
			else: c = plt.pcolormesh(x[:,ny//2,:],y[:,ny//2,:], np.log10((asc.rho*r_tmp/1e3)[:,ny//2,:]),cmap = 'viridis',vmin=-1,vmax=1)
			
			if (fieldlines==True): asc.plot_fieldlines_midplane(9*fac)
			#plt.pcolormesh(x[:,ny//2,:],y[:,ny//2,:], np.log10(Bth*Br)[:,ny//2,:],cmap = 'cubehelix',vmin=-1,vmax=4.25)

			# cb = plt.colorbar(c,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
			# cb.set_label(r"$ \rho r $",fontsize=17)


			plt.xlim(-9*fac,9*fac)
			plt.ylim(-9*fac,9*fac)


			# for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels():
			# 	label.set_fontsize(10)

			plt.axis('off')
			plt.axes().set_aspect('equal')
			plt.tight_layout()
			if (fieldlines==True): plt.savefig('frames/frame_rho_midplane_fieldlines_fac_%g_%04d.png' %(fac,idump))
			else: plt.savefig('frames/frame_rho_midplane_fac_%g_%04d.png' %(fac,idump))


if __name__ == "__main__":
	if m_type == "mk_frame_inner":
		mk_frame_inner()
	elif m_type =="mk_frame_outer":
		mk_frame_outer()
	elif m_type == "mk_frame_outer_slice":
		mk_frame_outer(isslice=True)
	elif m_type =="mk_frame_outer_slice_mhd":
		mk_frame_outer(isslice=True,mhd=True)
	elif m_type == "mk_frame_inner_slice_mhd":
		mk_frame_inner(isslice=True,mhd=True)
	elif m_type == "mk_frame_inner_slice":
		mk_frame_inner(isslice=True)
	elif m_type == "convert_dumps":
		convert_dumps_to_spher()
	elif m_type == "convert_dumps_mhd":
		convert_dumps_to_spher(MHD=True)
	elif m_type == "convert_dumps_disk_frame":
		convert_dumps_disk_frame()
	elif m_type == "convert_dumps_disk_frame_mhd":
		convert_dumps_disk_frame(mhd=True)
	elif m_type == "Lx_calc":
		calculate_L_X()
	elif m_type == "mk_3Dframe":
		mk_frame_3D_uniform_grid()
	elif m_type == "mk_grmhdframe":
		mk_frame_grmhd()
	elif m_type == "mk_grframe":
		mk_frame_grmhd(is_magnetic = False)
	elif m_type == "mk_grframe_cartesian":
		mk_frame_grmhd_cartesian(is_magnetic=False)
	elif m_type == "mk_frame_grmhd_restart_cartesian":
		mk_frame_grmhd_restart_cartesian(is_magnetic=False)
	elif m_type == "mk_grframe_magnetically_frustrated":
		mk_frame_gr_magnetically_frustrated(is_magnetic=False)
	elif m_type == "mk_grframe_magnetically_frustrated_cartesian":
		mk_frame_gr_magnetically_frustrated_cartesian(is_magnetic=False)
	elif m_type == "mk_1d":
		mk_1d_quantities(is_magnetic=True)
	elif m_type == "mk_1d_cartesian":
		mk_1d_quantities_cartesian(is_magnetic=True)
	elif m_type == 'mk_RM':
		mk_RM()
	elif m_type == 'mk_RM_moving':
		mk_RM(moving=True)
	elif m_type == "RM_movie":
		RM_movie()
	elif m_type == "mk_frame_outer_cold":
		mk_frame_outer(iscold=True)
	elif m_type == "mk_frame_inner_cold":
		mk_frame_inner(iscold=True)
	elif m_type == "mk_frame_disk":
		mk_frame_disk()
	elif m_type == "mk_frame_L_aligned":
		mk_frame_L_aligned()
	elif m_type == "mk_frame_L_aligned_fieldlines":
		mk_frame_L_aligned(fieldlines=True)
