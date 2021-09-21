# Python modules
import numpy as np
from numpy import *
import glob
import os
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import yt

sys.path.append("/global/scratch/smressle/star_cluster/cuadra_comp/vis/python")
from athena_script import *
import athena_script as asc


def mk_frame(var = 'density',region = 'outer',projection_axis = 2,file_suffix ='rho',i_frame = 0,var_min = 0,var_max = 2.5,cmap = 'ocean' ):
    if (projection_axis ==2):
        proj_label = 'z'
    elif (projection_axis==1):
        proj_label = 'y'
    else:
        proj_label = 'x'

    if (var == 'density'):
        prj = yt.ProjectionPlot(asc.ds,proj_label,"density",width = (1.,"pc"))
        prj.set_zlim('density',10**(var_min),10**(var_max))
        plot_instance = prj.plots['density']
    elif (var == 'temperature'):
        prj = yt.ProjectionPlot(asc.ds,proj_label,"temperature",width = (1.,"pc"),weight_field="density")
        prj.set_zlim('temperature',10**(var_min),10**(var_max))
        plot_instance = prj.plots['temperature']
        prj.set_cmap(field="temperature", cmap='hot')

    if (region =='inner'):
        prj.zoom(10)

#    cb = plot_instance.cax
#    cb.set_ticks(np.arange(var_min,var_max+.5,.5))
    #cb.set_label(cb_label,fontsize=25)

    ax = plot_instance.axes
    for label in ax.get_xticklabels() + ax.get_yticklabels(): #+cb.ax.get_yticklabels():
        label.set_fontsize(20)

#plt.tight_layout()

	os.system("mkdir -p frames")
	prj.save("frames/frame_%s_%s_%d.png" % (file_suffix,proj_label,i_frame))



def set_dump_range():
	global i_start,i_end
	if len(sys.argv[2:])>=2 and sys.argv[2].isdigit() and sys.argv[3].isdigit():
	    i_glob = int(sys.argv[2])
	    n_processors = int(sys.argv[3])
	else:
		print("Syntax error")
		exit()
	n_dumps = len(glob.glob("*.athdf"))

	n_dumps_per_processor = n_dumps/n_processors
	i_start = i_glob*n_dumps_per_processor
	i_end = i_start + n_dumps_per_processor

def mk_frame_inner():
    set_dump_range()

    for i_dump in range(i_start,i_end):
        print "Making frame for dump ", i_dump
        if not os.path.isfile("frames/frame_T_inner_z_%d.png" %i_dump):
            asc.yt_load(i_dump)
            #from athena_script import *
            for i_proj in range(3):
                mk_frame(var = "density",file_suffix="rho_inner",region="inner",i_frame = i_dump,projection_axis = i_proj)
                mk_frame(var = "temperature",file_suffix="T_inner",region="inner",i_frame = i_dump,projection_axis = i_proj,var_min=6,var_max=8)

def mk_frame_outer():
    set_dump_range()
    for i_dump in range(i_start,i_end):
        print "Making frame for dump ", i_dump
        if not os.path.isfile("frames/frame_T_outer_z_%d.png" %i_dump):
            asc.yt_load(i_dump)
            #from athena_script import *
            for i_proj in range(3):
                mk_frame(var='density',var_min =-2,var_max = 1,file_suffix ="rho_outer",i_frame = i_dump,projection_axis = i_proj)
                mk_frame(var='temperature',file_suffix="T_outer",
                    i_frame=i_dump,var_min=6,var_max=8,projection_axis = i_proj)

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

def mk_frame_3D():
    set_dump_range()
    for i_dump in range(i_start,i_end):
        asc.yt_load(i_dump)
        sc = sc = yt.create_scene(asc.ds,lens_type = 'perspective')
        sc.camera.zoom(2.0)
        sc[0].tfh.set_bounds([1e-4,1e2])
        os.system("mkdir -p frames")
        sc.save("frames/3Dframe_%d.png" % (i_frame))


if __name__ == "__main__":
    if len(sys.argv)>1:
        if sys.argv[1] == "mk_frame_inner":
            mk_frame_inner()
        elif sys.argv[1].startswith("mk_frame_outer"):
        	mk_frame_outer()
        elif sys.argv[1] == "mk_3Dframe":
            mk_frame_3D()
        else:
            print( "Unknown command %s" % sys.argv[1] )