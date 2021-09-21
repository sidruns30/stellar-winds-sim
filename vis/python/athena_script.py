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
from matplotlib.patches import Ellipse


#=========================================================================================


def tab(filename, headings=None, dimensions=1):
  """Read .tab files and return dict or array."""

  # Check for valid number of dimensions
  if dimensions != 1 and dimensions !=2 and dimensions != 3:
    raise AthenaError('Improper number of dimensions')

  # Read raw data
  with open(filename, 'r') as data_file:
    raw_data = data_file.readlines()

  # Organize data into array of numbers
  data_array = []
  first_line = True
  last_line_number = len(raw_data)
  line_number = 0
  for line in raw_data:
    line_number += 1
    if line.split()[0][0] == '#':  # comment line
      continue
    row = []
    col = 0
    for val in line.split():
      col += 1
      if col == 1:
        if first_line:
          i_min = int(val)
        if line_number == last_line_number:
          i_max = int(val)
      elif col == 3 and dimensions >= 2:
        if first_line:
          j_min = int(val)
        if line_number == last_line_number:
          j_max = int(val)
      elif col == 5 and dimensions == 3:
        if first_line:
          k_min = int(val)
        if line_number == last_line_number:
          j_max = int(val)
      else:
        row.append(float(val))
    first_line = False
    data_array.append(row)

  # Reshape array based on number of dimensions
  if dimensions == 1:
    j_min = j_max = 0
  if dimensions <= 2:
    k_min = k_max = 0
  array_shape = (k_max-k_min+1,j_max-j_min+1,i_max-i_min+1,len(row))
  data_array = np.reshape(data_array, array_shape)

  # Store separate variables as dictionary entries if headings given
  if headings is not None:
    data_dict = {}
    for n in range(len(headings)):
      data_dict[headings[n]] = data_array[:,:,:,n]
    return data_dict
  else:
    return data_array

#=========================================================================================

def vtk(filename):
  """Read .vtk files and return dict of arrays of data."""

  # Python module
  import struct

  # Read raw data
  with open(filename, 'r') as data_file:
    raw_data = data_file.read()

  # Skip header
  current_index = 0
  current_char = raw_data[current_index]
  while current_char == '#':
    while current_char != '\n':
      current_index += 1
      current_char = raw_data[current_index]
    current_index += 1
    current_char = raw_data[current_index]

  # Function for skipping though the file
  def skip_string(expected_string):
    expected_string_len = len(expected_string)
    if raw_data[current_index:current_index+expected_string_len] != expected_string:
      raise AthenaError('File not formatted as expected')
    return current_index+expected_string_len

  # Read metadata
  current_index = skip_string('BINARY\nDATASET RECTILINEAR_GRID\nDIMENSIONS ')
  end_of_line_index = current_index + 1
  while raw_data[end_of_line_index] != '\n':
    end_of_line_index += 1
  face_dimensions = map(int, raw_data[current_index:end_of_line_index].split(' '))
  current_index = end_of_line_index + 1

  # Function for reading interface locations
  def read_faces(letter, num_faces):
    identifier_string = '{0}_COORDINATES {1} float\n'.format(letter,num_faces)
    begin_index = skip_string(identifier_string)
    format_string = '>' + 'f'*num_faces
    end_index = begin_index + 4*num_faces
    vals = np.array(struct.unpack(format_string, raw_data[begin_index:end_index]))
    return vals,end_index+1

  # Read interface locations
  x_faces,current_index = read_faces('X', face_dimensions[0])
  y_faces,current_index = read_faces('Y', face_dimensions[1])
  z_faces,current_index = read_faces('Z', face_dimensions[2])

  # Prepare to read quantities defined on grid
  cell_dimensions = np.array([max(dim-1,1)
      for dim in face_dimensions])
  num_cells = cell_dimensions.prod()
  current_index = skip_string('CELL_DATA {0}\n'.format(num_cells))
  if raw_data[current_index:current_index+1] == '\n':
    current_index = skip_string('\n')  # extra newline inserted by join script
  data = {}

  # Function for reading scalar data
  def read_cell_scalars():
    begin_index = skip_string('SCALARS ')
    end_of_word_index = begin_index + 1
    while raw_data[end_of_word_index] != ' ':
      end_of_word_index += 1
    array_name = raw_data[begin_index:end_of_word_index]
    string_to_skip = 'SCALARS {0} float\nLOOKUP_TABLE default\n'.format(array_name)
    begin_index = skip_string(string_to_skip)
    format_string = '>' + 'f'*num_cells
    end_index = begin_index + 4*num_cells
    data[array_name] = struct.unpack(format_string, raw_data[begin_index:end_index])
    dimensions = tuple(cell_dimensions[::-1])
    data[array_name] = np.array(data[array_name]).reshape(dimensions)
    return end_index+1

  # Function for reading vector data
  def read_cell_vectors():
    begin_index = skip_string('VECTORS ')
    end_of_word_index = begin_index + 1
    while raw_data[end_of_word_index] != '\n':
      end_of_word_index += 1
    array_name = raw_data[begin_index:end_of_word_index]
    string_to_skip = 'VECTORS {0}\n'.format(array_name)
    array_name = array_name[:-6]  # remove ' float'
    begin_index = skip_string(string_to_skip)
    format_string = '>' + 'f'*num_cells*3
    end_index = begin_index + 4*num_cells*3
    data[array_name] = struct.unpack(format_string, raw_data[begin_index:end_index])
    dimensions = tuple(np.append(cell_dimensions[::-1],3))
    data[array_name] = np.array(data[array_name]).reshape(dimensions)
    return end_index+1

  # Read quantities defined on grid
  while current_index < len(raw_data):
    expected_string = 'SCALARS'
    expected_string_len = len(expected_string)
    if raw_data[current_index:current_index+expected_string_len] == expected_string:
      current_index = read_cell_scalars()
      continue
    expected_string = 'VECTORS'
    expected_string_len = len(expected_string)
    if raw_data[current_index:current_index+expected_string_len] == expected_string:
      current_index = read_cell_vectors()
      continue
    raise AthenaError('File not formatted as expected')
  return x_faces,y_faces,z_faces,data

#=========================================================================================

def athdf(filename, data=None, quantities=None, dtype=np.float32, level=None,
    subsample=False, fast_restrict=False, x1_min=None, x1_max=None, x2_min=None,
    x2_max=None, x3_min=None, x3_max=None, vol_func=None, vol_params=None,
    face_func_1=None, face_func_2=None, face_func_3=None,center_func_1=None, center_func_2=None, center_func_3=None):
  """Read .athdf files and populate dict of arrays of data."""

  # Python modules
  import sys
  import warnings
  import h5py
  global t 

  # Open file
  with h5py.File(filename, 'r') as f:

    # Extract size information
    t = f.attrs['Time']
    max_level = f.attrs['MaxLevel']
    if level is None:
      level = max_level
    block_size = f.attrs['MeshBlockSize']
    root_grid_size = f.attrs['RootGridSize']
    levels = f['Levels'][:]
    logical_locations = f['LogicalLocations'][:]
    nx_vals = []
    for d in range(3):
      if block_size[d] == 1 and root_grid_size[d] > 1:  # sum or slice
        other_locations = [location for location in \
            zip(levels, logical_locations[:,(d+1)%3], logical_locations[:,(d+2)%3])]
        if len(set(other_locations)) == len(other_locations):  # effective slice
          nx_vals.append(1)
        else:  # nontrivial sum
          nx_vals.append(2**level)
      elif block_size[d] == 1:  # singleton dimension
        nx_vals.append(1)
      else:  # normal case
        nx_vals.append(root_grid_size[d] * 2**level)
    nx1 = nx_vals[0]
    nx2 = nx_vals[1]
    nx3 = nx_vals[2]
    lx1 = nx1 / block_size[0]
    lx2 = nx2 / block_size[1]
    lx3 = nx3 / block_size[2]
    num_extended_dims = 0
    for nx in nx_vals:
      if nx > 1:
        num_extended_dims += 1

    # Set volume function for preset coordinates if needed
    coord = f.attrs['Coordinates']
    if level < max_level and not subsample and not fast_restrict and vol_func is None:
      x1_rat = f.attrs['RootGridX1'][2]
      x2_rat = f.attrs['RootGridX2'][2]
      x3_rat = f.attrs['RootGridX3'][2]
      if coord == b'cartesian' or coord == b'minkowski' or coord == b'tilted' \
          or coord == b'sinusoidal':
        if (nx1 == 1 or x1_rat == 1.0) and (nx2 == 1 or x2_rat == 1.0) and \
            (nx3 == 1 or x3_rat == 1.0):
          fast_restrict = True
        else:
          vol_func = lambda xm,xp,ym,yp,zm,zp: (xp-xm) * (yp-ym) * (zp-zm)
      elif coord == b'cylindrical':
        if nx1 == 1 and (nx2 == 1 or x2_rat == 1.0) and (nx3 == 1 or x3_rat == 1.0):
          fast_restrict = True
        else:
          vol_func = lambda rm,rp,phim,phip,zm,zp: (rp**2-rm**2) * (phip-phim) * (zp-zm)
      elif coord == b'spherical_polar' or coord == b'schwarzschild':
        if nx1 == 1 and nx2 == 1 and (nx3 == 1 or x3_rat == 1.0):
          fast_restrict = True
        else:
          vol_func = lambda rm,rp,thetam,thetap,phim,phip: \
              (rp**3-rm**3) * abs(np.cos(thetam)-np.cos(thetap)) * (phip-phim)
      elif coord == b'kerr-schild':
        if nx1 == 1 and nx2 == 1 and (nx3 == 3 or x3_rat == 1.0):
          fast_restrict = True
        else:
          a = vol_params[0]
          def vol_func(rm, rp, thetam, thetap, phim, phip):
            cosm = np.cos(thetam)
            cosp = np.cos(thetap)
            return \
                ((rp**3-rm**3) * abs(cosm-cosp) + a**2 * (rp-rm) * abs(cosm**3-cosp**3)) \
                * (phip-phim)
      elif coord ==b'gr_user':
        if (nx1 == 1 or x1_rat == 1.0) and (nx2 == 1 or x2_rat == 1.0) and \
            (nx3 == 1 or x3_rat == 1.0):
          fast_restrict = True
        else:
          vol_func = lambda xm,xp,ym,yp,zm,zp: (xp-xm) * (yp-ym) * (zp-zm)
      else:
        raise AthenaError('Coordinates not recognized, %s' %coord)

    # Set cell center functions for preset coordinates
    if center_func_1 is None:
      if coord == b'cartesian' or coord == b'minkowski' or coord == b'tilted' \
          or coord == b'sinusoidal' or coord == b'kerr-schild':
        center_func_1 = lambda xm,xp : 0.5*(xm+xp)
      elif coord == b'cylindrical':
        center_func_1 = lambda xm,xp : 2.0/3.0 * (xp**3-xm**3) / (xp**2-xm**2)
      elif coord == b'spherical_polar':
        center_func_1 = lambda xm,xp : 3.0/4.0 * (xp**4-xm**4) / (xp**3-xm**3)
      elif coord == b'schwarzschild':
        center_func_1 = lambda xm,xp : (0.5*(xm**3+xp**3)) ** 1.0/3.0
      elif coord ==b'gr_user':
        center_func_1 = lambda xm,xp : 0.5*(xm+xp)
      else:
        raise AthenaError('Coordinates not recognized, %s' %coord)
    if center_func_2 is None:
      if coord == b'cartesian' or coord == b'cylindrical' or coord == b'minkowski' \
          or coord == b'tilted' or coord == b'sinusoidal' or coord == b'kerr-schild' or coord==b'gr_user':
        center_func_2 = lambda xm,xp : 0.5*(xm+xp)
      elif coord == b'spherical_polar':
        def center_func_2(xm, xp):
          sm = np.sin(xm)
          cm = np.cos(xm)
          sp = np.sin(xp)
          cp = np.cos(xp)
          return (sp-xp*cp - sm+xm*cm) / (cm-cp)
      elif coord == b'schwarzschild':
        center_func_2 = lambda xm,xp : np.arccos(0.5*(np.cos(xm)+np.cos(xp)))
      else:
        raise AthenaError('Coordinates not recognized, %s' %coord)
    if center_func_3 is None:
      if coord == b'cartesian' or coord == b'cylindrical' or coord == b'spherical_polar' \
          or coord == b'minkowski' or coord == b'tilted' or coord == b'sinusoidal' \
          or coord == b'schwarzschild' or coord == b'kerr-schild' or coord==b'gr_user':
        center_func_3 = lambda xm,xp : 0.5*(xm+xp)
      else:
        raise AthenaError('Coordinates not recognized, %s' %coord)

    # Check output level compared to max level in file
    if level < max_level and not subsample and not fast_restrict:
      warnings.warn('Exact restriction being used: performance severely affected; see' \
          + ' documentation', AthenaWarning)
      sys.stderr.flush()
    if level > max_level:
      warnings.warn('Requested refinement level higher than maximum level in file: all' \
          + ' cells will be prolongated', AthenaWarning)
      sys.stderr.flush()

    # Check that subsampling and/or fast restriction will work if needed
    if level < max_level and (subsample or fast_restrict):
      max_restrict_factor = 2**(max_level-level)
      for current_block_size in block_size:
        if current_block_size != 1 and current_block_size%max_restrict_factor != 0:
          raise AthenaError('Block boundaries at finest level must be cell boundaries' \
              + ' at desired level for subsampling or fast restriction to work')

    # Create list of all quantities if none given
    file_quantities = f.attrs['VariableNames'][:]
    coord_quantities = ('x1f', 'x2f', 'x3f', 'x1v', 'x2v', 'x3v')
    if data is not None:
      quantities = data.values()
    elif quantities is None:
      quantities = file_quantities
    else:
      for q in quantities:
        if q not in file_quantities and q not in coord_quantities:
          possibilities = '", "'.join(file_quantities)
          possibilities = '"' + possibilities + '"'
          error_string = 'Quantity not recognized: file does not include "{0}" but does' \
              + ' include {1}'
          raise AthenaError(error_string.format(q, possibilities))
    quantities = [str(q.decode('utf-8')) for q in quantities if q not in coord_quantities]

    # Get metadata describing file layout
    num_blocks = f.attrs['NumMeshBlocks']
    #dataset_names = f.attrs['DatasetNames'][:]
    dataset_names = []
    for dset_name in f.attrs['DatasetNames'][:]:
      dataset_names.append(dset_name.decode('utf-8'))
    dataset_names = np.array(dataset_names)
    dataset_sizes = f.attrs['NumVariables'][:]
    dataset_sizes_cumulative = np.cumsum(dataset_sizes)
    variable_names = []
    for var_name in f.attrs['VariableNames'][:]:
      variable_names.append(var_name.decode('utf-8'))
    variable_names = np.array(variable_names)
    #variable_names = f.attrs['VariableNames'][:]

    quantity_datasets = []
    quantity_indices = []
    for q in quantities:
      var_num = np.where(variable_names == q)[0][0]
      dataset_num = np.where(dataset_sizes_cumulative > var_num)[0][0]
      if dataset_num == 0:
        dataset_index = var_num
      else:
        dataset_index = var_num - dataset_sizes_cumulative[dataset_num-1]
      quantity_datasets.append(dataset_names[dataset_num])
      quantity_indices.append(dataset_index)

    # Locate fine block for coordinates in case of slice
    fine_block = np.where(levels == max_level)[0][0]
    x1m = f['x1f'][fine_block,0]
    x1p = f['x1f'][fine_block,1]
    x2m = f['x2f'][fine_block,0]
    x2p = f['x2f'][fine_block,1]
    x3m = f['x3f'][fine_block,0]
    x3p = f['x3f'][fine_block,1]

    # Prepare dictionary for results
    if data is None:
      data = {}
      new_data = True

    # Populate coordinate arrays
    face_funcs = (face_func_1, face_func_2, face_func_3)
    center_funcs = (center_func_1, center_func_2, center_func_3)
    for d,nx,face_func,center_func in zip(range(1, 4), nx_vals, face_funcs,center_funcs):
      if nx == 1:
        xm = (x1m, x2m, x3m)[d-1]
        xp = (x1p, x2p, x3p)[d-1]
        data['x'+repr(d)+'f'] = np.array([xm, xp])
      else:
        xmin = f.attrs['RootGridX'+repr(d)][0]
        xmax = f.attrs['RootGridX'+repr(d)][1]
        xrat_root = f.attrs['RootGridX'+repr(d)][2]
        if face_func is not None:
          data['x'+repr(d)+'f'] = face_func(xmin, xmax, xrat_root, nx+1)
        elif (xrat_root == 1.0):
          data['x'+repr(d)+'f'] = np.linspace(xmin, xmax, nx+1)
        else:
          xrat = xrat_root ** (1.0 / 2**level)
          data['x'+repr(d)+'f'] = \
              xmin + (1.0-xrat**np.arange(nx+1)) / (1.0-xrat**nx) * (xmax-xmin)
      data['x'+repr(d)+'v'] = np.empty(nx)
      for i in range(nx):
        data['x'+repr(d)+'v'][i] = \
            center_func(data['x'+repr(d)+'f'][i], data['x'+repr(d)+'f'][i+1])

    # Account for selection
    x1_select = False
    x2_select = False
    x3_select = False
    i_min = j_min = k_min = 0
    i_max = nx1
    j_max = nx2
    k_max = nx3
    error_string = '{0} must be {1} than {2} in order to intersect data range'
    if x1_min is not None and x1_min >= data['x1f'][1]:
      if x1_min >= data['x1f'][-1]:
        raise AthenaError(error_string.format('x1_min', 'less', data['x1f'][-1]))
      x1_select = True
      i_min = np.where(data['x1f'] <= x1_min)[0][-1]
    if x1_max is not None and x1_max <= data['x1f'][-2]:
      if x1_max <= data['x1f'][0]:
        raise AthenaError(error_string.format('x1_max', 'greater', data['x1f'][0]))
      x1_select = True
      i_max = np.where(data['x1f'] >= x1_max)[0][0]
    if x2_min is not None and x2_min >= data['x2f'][1]:
      if x2_min >= data['x2f'][-1]:
        raise AthenaError(error_string.format('x2_min', 'less', data['x2f'][-1]))
      x2_select = True
      j_min = np.where(data['x2f'] <= x2_min)[0][-1]
    if x2_max is not None and x2_max <= data['x2f'][-2]:
      if x2_max <= data['x2f'][0]:
        raise AthenaError(error_string.format('x2_max', 'greater', data['x2f'][0]))
      x2_select = True
      j_max = np.where(data['x2f'] >= x2_max)[0][0]
    if x3_min is not None and x3_min >= data['x3f'][1]:
      if x3_min >= data['x3f'][-1]:
        raise AthenaError(error_string.format('x3_min', 'less', data['x3f'][-1]))
      x3_select = True
      k_min = np.where(data['x3f'] <= x3_min)[0][-1]
    if x3_max is not None and x3_max <= data['x3f'][-2]:
      if x3_max <= data['x3f'][0]:
        raise AthenaError(error_string.format('x3_max', 'greater', data['x3f'][0]))
      x3_select = True
      k_max = np.where(data['x3f'] >= x3_max)[0][0]

    # Adjust coordinates if selection made
    if x1_select:
      data['x1f'] = data['x1f'][i_min:i_max+1]
      data['x1v'] = data['x1v'][i_min:i_max]
    if x2_select:
      data['x2f'] = data['x2f'][j_min:j_max+1]
      data['x2v'] = data['x2v'][j_min:j_max]
    if x3_select:
      data['x3f'] = data['x3f'][k_min:k_max+1]
      data['x3v'] = data['x3v'][k_min:k_max]

    # Prepare arrays for data and bookkeeping
    if new_data:
      for q in quantities:
        data[q] = np.zeros((k_max-k_min, j_max-j_min, i_max-i_min), dtype=dtype)
    else:
      for q in quantities:
        data[q].fill(0.0)
    if not subsample and not fast_restrict and max_level > level:
      restricted_data = np.zeros((lx3, lx2, lx1), dtype=bool)

    # Go through blocks in data file
    for block_num in range(num_blocks):

      # Extract location information
      block_level = levels[block_num]
      block_location = logical_locations[block_num,:]

      # Prolongate coarse data and copy same-level data
      if block_level <= level:

        # Calculate scale (number of copies per dimension)
        s = 2 ** (level - block_level)

        # Calculate destination indices, without selection
        il_d = block_location[0] * block_size[0] * s if nx1 > 1 else 0
        jl_d = block_location[1] * block_size[1] * s if nx2 > 1 else 0
        kl_d = block_location[2] * block_size[2] * s if nx3 > 1 else 0
        iu_d = il_d + block_size[0] * s if nx1 > 1 else 1
        ju_d = jl_d + block_size[1] * s if nx2 > 1 else 1
        ku_d = kl_d + block_size[2] * s if nx3 > 1 else 1

        # Calculate (prolongated) source indices, with selection
        il_s = max(il_d, i_min) - il_d
        jl_s = max(jl_d, j_min) - jl_d
        kl_s = max(kl_d, k_min) - kl_d
        iu_s = min(iu_d, i_max) - il_d
        ju_s = min(ju_d, j_max) - jl_d
        ku_s = min(ku_d, k_max) - kl_d
        if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
          continue

        # Account for selection in destination indices
        il_d = max(il_d, i_min) - i_min
        jl_d = max(jl_d, j_min) - j_min
        kl_d = max(kl_d, k_min) - k_min
        iu_d = min(iu_d, i_max) - i_min
        ju_d = min(ju_d, j_max) - j_min
        ku_d = min(ku_d, k_max) - k_min

        # Assign values
        for q,dataset,index in zip(quantities, quantity_datasets, quantity_indices):
          block_data = f[dataset][index,block_num,:]
          if s > 1:
            if nx1 > 1:
              block_data = np.repeat(block_data, s, axis=2)
            if nx2 > 1:
              block_data = np.repeat(block_data, s, axis=1)
            if nx3 > 1:
              block_data = np.repeat(block_data, s, axis=0)
          data[q][kl_d:ku_d,jl_d:ju_d,il_d:iu_d] = \
              block_data[kl_s:ku_s,jl_s:ju_s,il_s:iu_s]

      # Restrict fine data
      else:

        # Calculate scale
        s = 2 ** (block_level - level)

        # Calculate destination indices, without selection
        il_d = block_location[0] * block_size[0] // s if nx1 > 1 else 0
        jl_d = block_location[1] * block_size[1] // s if nx2 > 1 else 0
        kl_d = block_location[2] * block_size[2] // s if nx3 > 1 else 0
        iu_d = il_d + block_size[0] // s if nx1 > 1 else 1
        ju_d = jl_d + block_size[1] // s if nx2 > 1 else 1
        ku_d = kl_d + block_size[2] // s if nx3 > 1 else 1

        # Calculate (restricted) source indices, with selection
        il_s = max(il_d, i_min) - il_d
        jl_s = max(jl_d, j_min) - jl_d
        kl_s = max(kl_d, k_min) - kl_d
        iu_s = min(iu_d, i_max) - il_d
        ju_s = min(ju_d, j_max) - jl_d
        ku_s = min(ku_d, k_max) - kl_d
        if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
          continue

        # Account for selection in destination indices
        il_d = max(il_d, i_min) - i_min
        jl_d = max(jl_d, j_min) - j_min
        kl_d = max(kl_d, k_min) - k_min
        iu_d = min(iu_d, i_max) - i_min
        ju_d = min(ju_d, j_max) - j_min
        ku_d = min(ku_d, k_max) - k_min

        # Account for restriction in source indices
        if nx1 > 1:
          il_s *= s
          iu_s *= s
        if nx2 > 1:
          jl_s *= s
          ju_s *= s
        if nx3 > 1:
          kl_s *= s
          ku_s *= s

        # Apply subsampling
        if subsample:

          # Calculate fine-level offsets (nearest cell at or below center)
          o1 = s/2 - 1 if nx1 > 1 else 0
          o2 = s/2 - 1 if nx2 > 1 else 0
          o3 = s/2 - 1 if nx3 > 1 else 0

          # Assign values
          for q,dataset,index in zip(quantities, quantity_datasets, quantity_indices):
            data[q][kl_d:ku_d,jl_d:ju_d,il_d:iu_d] = \
                f[dataset][index,block_num,kl_s+o3:ku_s:s,jl_s+o2:ju_s:s,il_s+o1:iu_s:s]

        # Apply fast (uniform Cartesian) restriction
        elif fast_restrict:

          # Calculate fine-level offsets
          io_vals = range(s) if nx1 > 1 else (0,)
          jo_vals = range(s) if nx2 > 1 else (0,)
          ko_vals = range(s) if nx3 > 1 else (0,)

          # Assign values
          for q,dataset,index in zip(quantities, quantity_datasets, quantity_indices):
            for ko in ko_vals:
              for jo in jo_vals:
                for io in io_vals:
                  data[q][kl_d:ku_d,jl_d:ju_d,il_d:iu_d] += \
                      f[dataset]\
                      [index,block_num,kl_s+ko:ku_s:s,jl_s+jo:ju_s:s,il_s+io:iu_s:s]
            data[q][kl_d:ku_d,jl_d:ju_d,il_d:iu_d] /= s ** num_extended_dims

        # Apply exact (volume-weighted) restriction
        else:

          # Calculate sets of indices
          i_s_vals = range(il_s, iu_s)
          j_s_vals = range(jl_s, ju_s)
          k_s_vals = range(kl_s, ku_s)
          i_d_vals = range(il_d, iu_d)
          j_d_vals = range(jl_d, ju_d)
          k_d_vals = range(kl_d, ku_d)
          if nx1 > 1:
            i_d_vals = np.repeat(i_d_vals, s)
          if nx2 > 1:
            j_d_vals = np.repeat(j_d_vals, s)
          if nx3 > 1:
            k_d_vals = np.repeat(k_d_vals, s)

          # Accumulate values
          for k_s,k_d in zip(k_s_vals, k_d_vals):
            if nx3 > 1:
              x3m = f['x3f'][block_num,k_s]
              x3p = f['x3f'][block_num,k_s+1]
            for j_s,j_d in zip(j_s_vals, j_d_vals):
              if nx2 > 1:
                x2m = f['x2f'][block_num,j_s]
                x2p = f['x2f'][block_num,j_s+1]
              for i_s,i_d in zip(i_s_vals, i_d_vals):
                if nx1 > 1:
                  x1m = f['x1f'][block_num,i_s]
                  x1p = f['x1f'][block_num,i_s+1]
                vol = vol_func(x1m, x1p, x2m, x2p, x3m, x3p)
                for q,dataset,index in \
                    zip(quantities, quantity_datasets, quantity_indices):
                  data[q][k_d,j_d,i_d] += f[dataset][index,block_num,k_s,j_s,i_s] * vol
          loc1 = (nx1 > 1 ) * block_location[0] / s
          loc2 = (nx2 > 1 ) * block_location[1] / s
          loc3 = (nx3 > 1 ) * block_location[2] / s
          restricted_data[loc3,loc2,loc1] = True

    # Remove volume factors from restricted data
    if level < max_level and not subsample and not fast_restrict:
      for loc3 in range(lx3):
        for loc2 in range(lx2):
          for loc1 in range(lx1):
            if restricted_data[loc3,loc2,loc1]:
              il = loc1 * block_size[0]
              jl = loc2 * block_size[1]
              kl = loc3 * block_size[2]
              iu = il + block_size[0]
              ju = jl + block_size[1]
              ku = kl + block_size[2]
              il = max(il, i_min) - i_min
              jl = max(jl, j_min) - j_min
              kl = max(kl, k_min) - k_min
              iu = min(iu, i_max) - i_min
              ju = min(ju, j_max) - j_min
              ku = min(ku, k_max) - k_min
              for k in range(kl, ku):
                if nx3 > 1:
                  x3m = data['x3f'][k]
                  x3p = data['x3f'][k+1]
                for j in range(jl, ju):
                  if nx2 > 1:
                    x2m = data['x2f'][j]
                    x2p = data['x2f'][j+1]
                  for i in range(il, iu):
                    if nx1 > 1:
                      x1m = data['x1f'][i]
                      x1p = data['x1f'][i+1]
                    vol = vol_func(x1m, x1p, x2m, x2p, x3m, x3p)
                    for q in quantities:
                      data[q][k,j,i] /= vol

  # Return dictionary containing requested data arrays
  return data

#=========================================================================================

def rdhdf5(ifile, ndim = 2,coord ='xy',block_level = 0,x1min=None, x1max=None, x2min=None,
    x2max=None, x3min=None, x3max=None,box_radius = None,user_x1 = False, user_x2=False,gr = False,a = 0,rbr = 200.0,
    x1_harm_max=2.0,npow2=4.0,cpow2=1.0,h =0.3):
  global x,y,z,nx,ny,nz,r,th,ph
  global dic


  def theta_func(xmin, xmax, xrat, nf):
    x_vals = np.linspace(0,1.0,nf)
    t_vals = 2.0*x_vals -1.0
    w_vals = 0.25*(t_vals*(t_vals*t_vals+1.0))+0.5
    return w_vals*xmax + (1.0-w_vals)*xmin

  def mks_theta_func(xmin,xmax,xrat,nf):
    x_vals = np.linspace(0,1.0,nf)
    th_max = xmax
    th_min = xmin
    return (th_max-th_min) * x_vals + (th_min) + 0.5*(1.0-h)*np.sin(2.0*((th_max-th_min) * x_vals + (th_min)))
    ##return np.pi*x_vals + 0.5*(1.0-h) * np.sin(2.0*np.pi*x_vals)

  def hyper_exponetial_r_func(xmin,xmax,xrat,nf):
    logrmin = np.log(xmin) 
    xbr = log(rbr)
    x_scaled = logrmin + x* (x1_harm_max - logrmin); 

    return np.exp(x_scaled + (cpow2*(x_scaled-xbr))**npow2 * (x_scaled>xbr));


  face_func_1 = None
  face_func_2 = None
  if (user_x2) ==True : face_func_2 = theta_func
  if (user_x2  =="mks" or user_x2 =="MKS"): face_func_2 = mks_theta_func 
  if (user_x1  =="hyper_exp" or user_x1==True): face_func_1 = hyper_exponetial_r_func
  if (box_radius is not None):
    x1min = -box_radius
    x1max = box_radius
    x2min = -box_radius
    x2max = box_radius
    x3min = -box_radius  
    x3max = box_radius

  file_prefix = glob.glob("*.athdf")[0][:-11]
  if (gr==True): file_prefix = glob.glob("*out2*.athdf")[0][:-11]
  dic = athdf(file_prefix + "%05d.athdf" %ifile,level = block_level,x1_min=x1min, x1_max=x1max, x2_min=x2min,
    x2_max=x2max, x3_min=x3min, x3_max=x3max , face_func_1=face_func_1, face_func_2 = face_func_2)

  global x1f,x2f,x3f,x1v,x3v,x2v,rho, vel1, vel2, vel3, press, Bcc1,Bcc2,Bcc3,k_ent,ke_ent,ke_ent2,ke_ent3

  x1f = dic['x1f']
  x2f = dic['x2f']
  x3f = dic['x3f']
  x1v = dic['x1v']
  x2v = dic['x2v']
  x3v = dic['x3v']
  rho = dic['rho'].transpose()
  vel1 = dic['vel1'].transpose()
  vel2 = dic['vel2'].transpose()
  vel3 = dic['vel3'].transpose()
  if ('press' in dic.keys()): press = dic['press'].transpose()

  if ('Bcc1' in dic.keys()):
    Bcc1 = dic['Bcc1'].transpose()
    Bcc2 = dic['Bcc2'].transpose()
    Bcc3 = dic['Bcc3'].transpose()
  if ('r0' in dic.keys()): k_ent = dic['r0'].transpose()
  if ('r1' in dic.keys()): ke_ent = dic['r1'].transpose()
  if ('r2' in dic.keys()): ke_ent2 = dic['r2'].transpose()
  if ('r3' in dic.keys()): ke_ent3 = dic['r3'].transpose()
  # for key in dic.keys():
  #   exec("globals()['%s'] = dic['%s']" % (key,key),globals(),locals())
  #   exec("if type(globals()['%s']) is np.ndarray: globals()['%s'] = globals()['%s'].transpose()" %(key,key,key),globals(),locals())

  if (ndim ==2 and coord =="cyl"):
    x = x1v[:,None] * np.cos(x2v)[None,:]
    y = x1v[:,None] * np.sin(x2v)[None,:]
    r = x1v[:,None] * np.cos(x2v*0)[None,:]
    ph =x2v[None,:] * np.cos(x1v*0)[:,None]
  # elif (ndim==1 and coord=="spherical"):
  #   r = x1v[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * np.cos(x3v*0.)[None,None,:]
  #   th = (x1v/x1v)[:,None,None] * x2v[None,:,None] * np.cos(x3v*0.)[None,None,:]
  #   ph = x2v
  #   x = r
  #   y = r*0
  elif ((ndim==3 or ndim==1) and coord == "spherical"):
    x = x1v[:,None,None] * np.sin(x2v)[None,:,None] * np.cos(x3v)[None,None,:]
    y = x1v[:,None,None] * np.sin(x2v)[None,:,None] * np.sin(x3v)[None,None,:]
    z = x1v[:,None,None] * np.cos(x2v)[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    r = x1v[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * np.cos(x3v*0.)[None,None,:]
    th = (x1v/x1v)[:,None,None] * x2v[None,:,None] * np.cos(x3v*0.)[None,None,:]
    ph = (x1v/x1v)[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * x3v[None,None,:]
  elif (coord=="xy" and ndim==2):
    x = x1v[:,None,None] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    y = x2v[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    z = x3v[None,None,:] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None]
  elif (coord =="xy" and ndim ==3):
    x = x1v[:,None,None] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    y = x2v[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    z = x3v[None,None,:] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None]
  elif (coord=='xy' and ndim ==1):
    x = x1v
  else: 
    print ("coordinates not determined in ", ndim, "dimensions")
  nx = x.shape[0]
  if (ndim==1): ny = 1
  else: ny = y.shape[1]
  if (ndim ==2 or ndim==1):
    nz = 1
  else:
    nz = z.shape[2]

  global bsq
  if ("Bcc1" in dic.keys() and gr==False): 
    bsq = Bcc1**2 + Bcc2**2 + Bcc3**2
  
  global uu,gamma,gdet,bu
  if (gr==True and coord=="spherical"):
    gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )
    ks_metric(r,th,a)
    tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
    gamma = np.sqrt(1.0 + tmp);

    # Calculate 4-velocity
    ks_inverse_metric(r,th,a)
    alpha = np.sqrt(-1.0/gi[0,0]);
    uu = np.zeros((4,nx,ny,nz))
    uu[0] = gamma/alpha;
    uu[1] = vel1 - alpha * gamma * gi[0,1];
    uu[2] = vel2 - alpha * gamma * gi[0,2];
    uu[3] = vel3 - alpha * gamma * gi[0,3];
    if ("Bcc1" in dic.keys()):
      B_vec = np.zeros(uu.shape)
      bu = np.zeros(uu.shape)
      B_vec[1] = Bcc1 
      B_vec[2] = Bcc2 
      B_vec[3] = Bcc3      
      for i in range(1,4):
        for mu in range(0,4):
          bu[0] += g[i,mu]*uu[mu]*B_vec[i]
      bu[1] = 1.0/uu[0] * (Bcc1 + bu[0]*uu[1])
      bu[2] = 1.0/uu[0] * (Bcc2 + bu[0]*uu[2])
      bu[3] = 1.0/uu[0] * (Bcc3 + bu[0]*uu[3])
      bu = np.array(bu)
      bu_tmp = bu* 1.0

      bsq = 0
      for i in range(4):
        for j in range(4):
          bsq += g[i,j] * bu[i] * bu[j]

  if (gr==True and glob.glob("*out3*athdf") != []):
    uu = np.zeros((4,nx,ny,nz))
    file_prefix = glob.glob("*out3*athdf")[0][:-11]
    dic2 = athdf(file_prefix + "%05d.athdf" %ifile,level = block_level,x1_min=x1min, x1_max=x1max, x2_min=x2min,
      x2_max=x2max, x3_min=x3min, x3_max=x3max , face_func_2 = face_func_2)
    gamma = dic2['user_out_var0'].transpose()
    uu[0] = dic2['user_out_var1'].transpose()
    uu[1] = dic2['user_out_var2'].transpose()
    uu[2] = dic2['user_out_var3'].transpose()
    uu[3] = dic2['user_out_var4'].transpose()
    if (coord=="spherical"): gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )
    else: gdet = x*0+1
    if (coord=="spherical"): ks_metric(r,th,a)
    else: cks_metric(x,y,z,a)
    #global ud,bu,bd
   
    ud = Lower(uu,g)
    if ('user_out_var5' in dic2.keys() and "Bcc1" in dic.keys()):
      bsq = dic2['user_out_var5'].transpose()*2.0
      B_vec = np.zeros(uu.shape)
      bu = np.zeros(uu.shape)
      B_vec[1] = Bcc1 
      B_vec[2] = Bcc2 
      B_vec[3] = Bcc3
      for i in range(1,4):
        for mu in range(0,4):
          bu[0] += g[i,mu]*uu[mu]*B_vec[i]
      bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
      bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
      bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
      bd = Lower(bu,g)

    global A1,A2,A3,divb_array
    if ('user_out_var6') in dic2.keys():
      A1 = dic2['user_out_var6'].transpose()
      A2 = dic2['user_out_var7'].transpose()
      A3 = dic2['user_out_var8'].transpose()
    if ('user_out_var9') in dic2.keys():
      divb_array = dic2['user_out_var9'].transpose()

def rdhdf5_chris(ifile, ndim = 2,coord ='xy',block_level = 0,x1min=None, x1max=None, x2min=None,
    x2max=None, x3min=None, x3max=None,box_radius = None,user_x1 = False, user_x2=False,gr = False,a = 0,rbr = 200.0,
    x1_harm_max=2.0,npow2=4.0,cpow2=1.0):
  global x,y,z,nx,ny,nz,r,th,ph
  global dic


  def theta_func(xmin, xmax, xrat, nf):
    x_vals = np.linspace(0,1.0,nf)
    t_vals = 2.0*x_vals -1.0
    w_vals = 0.25*(t_vals*(t_vals*t_vals+1.0))+0.5
    return w_vals*xmax + (1.0-w_vals)*xmin

  def mks_theta_func(xmin,xmax,xrat,nf):
    x_vals = np.linspace(0,1.0,nf)
    return np.pi*x_vals + 0.5*(1.0-h) * np.sin(2.0*np.pi*x_vals)

  def hyper_exponetial_r_func(xmin,xmax,xrat,nf):
    logrmin = np.log(xmin) 
    xbr = log(rbr)
    x_scaled = logrmin + x* (x1_harm_max - logrmin); 

    return np.exp(x_scaled + (cpow2*(x_scaled-xbr))**npow2 * (x_scaled>xbr));


  face_func_1 = None
  face_func_2 = None
  if (user_x2) ==True : face_func_2 = theta_func
  if (user_x2  =="mks" or user_x2 =="MKS"): face_func_2 = mks_theta_func 
  if (user_x1  =="hyper_exp" or user_x1==True): face_func_1 = hyper_exponetial_r_func
  if (box_radius is not None):
    x1min = -box_radius
    x1max = box_radius
    x2min = -box_radius
    x2max = box_radius
    x3min = -box_radius  
    x3max = box_radius

  file_prefix = glob.glob("*.athdf")[0][:-11]
  if (gr==True): file_prefix = glob.glob("*out2*.athdf")[0][:-11]
  dic = athdf(file_prefix + "%05d.athdf" %ifile,level = block_level,x1_min=x1min, x1_max=x1max, x2_min=x2min,
    x2_max=x2max, x3_min=x3min, x3_max=x3max , face_func_1=face_func_1, face_func_2 = face_func_2)

  global x1f,x2f,x3f,x1v,x3v,x2v,rho, vel1, vel2, vel3, press, Bcc1,Bcc2,Bcc3

  x1f = dic['x1f']
  x2f = dic['x2f']
  x3f = dic['x3f']
  x1v = dic['x1v']
  x2v = dic['x2v']
  x3v = dic['x3v']
  rho = dic['rho'].transpose()
  vel1 = dic['vel1'].transpose()
  vel2 = dic['vel2'].transpose()
  vel3 = dic['vel3'].transpose()
  if ('press' in dic.keys()): press = dic['press'].transpose()

  if ('Bcc1' in dic.keys()):
    Bcc1 = dic['Bcc1'].transpose()
    Bcc2 = dic['Bcc2'].transpose()
    Bcc3 = dic['Bcc3'].transpose()
  # for key in dic.keys():
  #   exec("globals()['%s'] = dic['%s']" % (key,key),globals(),locals())
  #   exec("if type(globals()['%s']) is np.ndarray: globals()['%s'] = globals()['%s'].transpose()" %(key,key,key),globals(),locals())

  if (ndim ==2 and coord =="cyl"):
    x = x1v[:,None] * np.cos(x2v)[None,:]
    y = x1v[:,None] * np.sin(x2v)[None,:]
  # elif (ndim==1 and coord=="spherical"):
  #   r = x1v[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * np.cos(x3v*0.)[None,None,:]
  #   th = (x1v/x1v)[:,None,None] * x2v[None,:,None] * np.cos(x3v*0.)[None,None,:]
  #   ph = x2v
  #   x = r
  #   y = r*0
  elif ((ndim==3 or ndim==1) and coord == "spherical"):
    x = x1v[:,None,None] * np.sin(x2v)[None,:,None] * np.cos(x3v)[None,None,:]
    y = x1v[:,None,None] * np.sin(x2v)[None,:,None] * np.sin(x3v)[None,None,:]
    z = x1v[:,None,None] * np.cos(x2v)[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    r = x1v[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * np.cos(x3v*0.)[None,None,:]
    th = (x1v/x1v)[:,None,None] * x2v[None,:,None] * np.cos(x3v*0.)[None,None,:]
    ph = (x1v/x1v)[:,None,None] * np.cos(x2v/x2v*0)[None,:,None] * x3v[None,None,:]

  elif (coord=="xy" and ndim==2):
    x = x1v[:,None,None] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    y = x2v[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    z = x3v[None,None,:] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None]
  elif (coord =="xy" and ndim ==3):
    x = x1v[:,None,None] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    y = x2v[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None] * ((x3v+1e20)/(x3v+1e20))[None,None,:]
    z = x3v[None,None,:] * ((x2v+1e20)/(x2v+1e20))[None,:,None] * ((x1v+1e20)/(x1v+1e20))[:,None,None]
  else: 
    print ("coordinates not determined in ", ndim, "dimensions")
  nx = x.shape[0]
  if (ndim==1): ny = 1
  else: ny = y.shape[1]
  if (ndim ==2 or ndim==1):
    nz = 1
  else:
    nz = z.shape[2]

  global bsq
  if ("Bcc1" in dic.keys() ): 
    bsq = Bcc1**2 + Bcc2**2 + Bcc3**2
  
  global uu,gamma,gdet
  if (gr==True and glob.glob("*out3*athdf") != []):
    uu = np.zeros((4,nx,ny,nz))
    file_prefix = glob.glob("*out3*athdf")[0][:-11]
    dic2 = athdf(file_prefix + "%05d.athdf" %ifile,level = block_level,x1_min=x1min, x1_max=x1max, x2_min=x2min,
      x2_max=x2max, x3_min=x3min, x3_max=x3max , face_func_2 = face_func_2)
    gamma = dic2['user_out_var0'].transpose()
    # uu[0] = dic2['user_out_var1'].transpose()
    # uu[1] = dic2['user_out_var2'].transpose()
    # uu[2] = dic2['user_out_var3'].transpose()
    # uu[3] = dic2['user_out_var4'].transpose()
    if (coord=="spherical"): gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )
    else: gdet = x*0+1
    if (coord=="spherical"): ks_metric(r,th,a)
    else: cks_metric(x,y,z,a)
    global ud,bu,bd
   
    ud = Lower(uu,g)
    if True: ##('user_out_var5' in dic2.keys() and "Bcc1" in dic.keys()):
      bsq = dic2['user_out_var5'].transpose()*2.0




# def plot_projection():
#   #do something
def rd_hst_entropy(file,is_magnetic=False,entropy = True):
  global t,dt,M,M1,M2,M3,KE1,KE2,KE3,E,Ktot,Ke,Ue
  hst = np.loadtxt(file)
  n = 0;
  t = hst[:,n]; n+=1
  dt = hst[:,n]; n+=1
  M = hst[:,n]; n+=1
  M1 = hst[:,n]; n+=1
  M2 = hst[:,n]; n+=1
  M3 = hst[:,n]; n+=1
  KE1 = hst[:,n]; n+=1
  KE2 = hst[:,n]; n+=1
  KE3 = hst[:,n]; n+=1
  E   = hst[:,n]; n+=1
  if (is_magnetic==True):
    ME1 = hst[:,n]; n+=1
    ME2 = hst[:,n]; n+=1
    ME3 = hst[:,n]; n+=1
  if (entropy==True):
    Ktot = hst[:,n]; n+=1
    Ke = hst[:,n]; n+=1
    if (n<hst.shape[1]): Ue = hst[:,n]; n+=1


def rd_hst_basic(file,is_magnetic=False):
  global t,dt,M,M1,M2,M3,KE1,KE2,KE3,E
  global ME1,ME2,ME3
  hst = np.loadtxt(file)
  n = 0;
  t = hst[:,n]; n+=1
  dt = hst[:,n]; n+=1
  M = hst[:,n]; n+=1
  M1 = hst[:,n]; n+=1
  M2 = hst[:,n]; n+=1
  M3 = hst[:,n]; n+=1
  KE1 = hst[:,n]; n+=1
  KE2 = hst[:,n]; n+=1
  KE3 = hst[:,n]; n+=1
  E   = hst[:,n]; n+=1
  if (is_magnetic==True):
    ME1 = hst[:,n]; n+=1
    ME2 = hst[:,n]; n+=1
    ME3 = hst[:,n]; n+=1

def rd_hst(file,is_magnetic=False):
  global t,dt,M,M1,M2,M3,KE1,KE2,KE3,E
  global rho_avg,v1_avg,v2_avg,v3_avg,p_avg,r,n
  global mdot_avg, vr_avg,Lx_avg,Ly_avg,Lz_avg,L_avg,ME1,ME2,ME3
  global Phi
  N_r = 128
  hst = np.loadtxt(file)
  n = 0;
  t = hst[:,n]; n+=1
  dt = hst[:,n]; n+=1
  M = hst[:,n]; n+=1
  M1 = hst[:,n]; n+=1
  M2 = hst[:,n]; n+=1
  M3 = hst[:,n]; n+=1
  KE1 = hst[:,n]; n+=1
  KE2 = hst[:,n]; n+=1
  KE3 = hst[:,n]; n+=1
  E   = hst[:,n]; n+=1
  if (is_magnetic==True):
    ME1 = hst[:,n]; n+=1
    ME2 = hst[:,n]; n+=1
    ME3 = hst[:,n]; n+=1
  rho_avg = hst[:,n:n+N_r]; n+= N_r
  v1_avg = hst[:,n:n+N_r]; n+= N_r
  v2_avg = hst[:,n:n+N_r]; n+= N_r
  v3_avg = hst[:,n:n+N_r]; n+= N_r
  p_avg = hst[:,n:n+N_r]; n+= N_r
  r = hst[:,n:n+N_r]; n+= N_r
  mdot_avg = hst[:,n:n+N_r]; n+= N_r
  vr_avg = hst[:,n:n+N_r]; n+= N_r
  Lx_avg = hst[:,n:n+N_r]; n+= N_r
  Ly_avg = hst[:,n:n+N_r]; n+= N_r
  Lz_avg = hst[:,n:n+N_r]; n+= N_r
  L_avg = np.sqrt(Lx_avg**2. + Ly_avg**2. + Lz_avg**2.)
  set_constants()
  global v_kep, l_kep,l_theta,l_phi
  v_kep = np.sqrt(gm_/r)
  l_kep = v_kep * r
  l_theta  = np.arccos(Lz_avg/L_avg)
  l_phi  = np.arctan2(Ly_avg,Lx_avg)

  global mdot_out,mdot_in,M_boundary, M_removed, Edot_avg,Edot_in,Edot_out
  global Lxdot_avg,Lydot_avg,Lzdot_avg,Ldot_avg,kappa_dot_avg
  global Kappa_avg, v_phi_x_avg,v_phi_y_avg,v_phi_z_avg
  global Lxdot_in,Lydot_in,Lzdot_in,Lxdot_out,Lydot_out,Lzdot_out
  global Q_cool,t_cool, bsq_avg,Bx_avg,By_avg,Bz_avg,Br_avg,Bphi_avg,divb, rho_added,Br_abs_avg
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    mdot_out = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    mdot_in = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Edot_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Edot_out = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Edot_in = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lxdot_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lydot_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lzdot_avg = hst[:,n:n+N_r]; n+= N_r
    Ldot_avg = np.sqrt(Lxdot_avg**2. + Lydot_avg**2. + Lzdot_avg**2.)
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lxdot_out = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lydot_out = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lzdot_out = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    rho_added = hst[:,n:n+N_r]; n+= N_r #Lxdot_in = hst[:,n:n+N_r]; n+= N_r
    tmp= np.gradient(rho_added,axis=0)
    tmp[tmp<0] = rho_added[tmp<0]
    rho_added = np.cumsum(tmp,axis=0)
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    if (is_magnetic == False): Lydot_in = hst[:,n:n+N_r]; n+= N_r
    else: Br_abs_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Lzdot_in = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    v_phi_x_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    v_phi_y_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    v_phi_z_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    kappa_dot_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Kappa_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Q_cool = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    t_cool = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Bx_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    By_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Bz_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    bsq_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Br_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1] and n+N_r-1 < hst.shape[1]):
    Bphi_avg = hst[:,n:n+N_r]; n+= N_r
  if (n<hst.shape[1]):
    M_removed = hst[:,n]; n+=1
  if (n<hst.shape[1]):
    M_boundary = hst[:,n]; n+=1
  if (n<hst.shape[1]):
    divb = hst[:,n]; n+=1

  if (is_magnetic==True):
    v_ff = np.sqrt(2.0*gm_/r)
    Phi = Br_abs_avg * 2.0 * np.pi * r /np.sqrt(np.fabs(mdot_avg) * v_ff)
def rd_gr_hst(file,is_magnetic=False):
  global t,dt,M,M1,M2,M3,KE1,KE2,KE3,E,ME1,ME2,ME3
  global mdot,edot,jdot,phibh,vol,divb
  hst = np.loadtxt(file)
  n = 0;
  t = hst[:,n]; n+=1
  dt = hst[:,n]; n+=1
  M = hst[:,n]; n+=1
  M1 = hst[:,n]; n+=1
  M2 = hst[:,n]; n+=1
  M3 = hst[:,n]; n+=1
  KE1 = hst[:,n]; n+=1
  KE2 = hst[:,n]; n+=1
  KE3 = hst[:,n]; n+=1
  E   = hst[:,n]; n+=1
  if (is_magnetic==True):
    ME1 = hst[:,n]; n+=1
    ME3 = hst[:,n]; n+=1
    ME3 = hst[:,n]; n+=1
  mdot = hst[:,n]; n+=1
  jdot = hst[:,n]; n+=1
  edot = hst[:,n]; n+=1
  area = hst[:,n]; n+=1
  area = 4
  if (is_magnetic==True):
    phibh = hst[:,n]; n+=1
    #phibh = phibh/vol
    divb = hst[:,n]; n+=1
  # mdot = mdot/vol
  # jdot = jdot/vol
  # edot = edot/vol


def yt_extract_box(i_dump,box_radius = 0.5,mhd=True,gr=False,a=0.0,res=128):
  global region,x,y,z,rho,press,vel1,vel2,vel3,Bcc1,Bcc2,Bcc3,bsq,vsq,uu,bsq,bu,bd
  global nx,ny,nz
  yt_load(i_dump,gr=gr)
  resolution = complex("%dj" %res)
  region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):resolution,(-box_radius,'pc'):(box_radius,'pc'):resolution,
      (-box_radius,'pc'):(box_radius,'pc'):resolution]

  x = np.array(region['x'])
  y = np.array(region['y'])
  z = np.array(region['z'])

  rho = np.array(region['rho'])
  press = np.array(region['press'])
  vel1 = np.array(region['vel1'])
  vel2 = np.array(region['vel2'])
  vel3 = np.array(region['vel3'])

  vsq = vel1**2 + vel2**2 + vel3**2
  
  if (mhd==True):
    Bcc1 = np.array(region['Bcc1'])
    Bcc2 = np.array(region['Bcc2'])
    Bcc3 = np.array(region['Bcc3'])
    if (gr==False): bsq = Bcc1**2 + Bcc2**2 + Bcc3**2 


  # if (gr==True): region = ds2.r[(-box_radius,'pc'):(box_radius,'pc'):128j,(-box_radius,'pc'):(box_radius,'pc'):128j,
  #     (-box_radius,'pc'):(box_radius,'pc'):128j]

  if (gr==True):
      uu = [0,0,0,0]
      bu = [0,0,0,0]
      cks_metric(x,y,z,a)
      tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
      gamma = np.sqrt(1.0 + tmp);

      # Calculate 4-velocity
      cks_inverse_metric(x,y,z,a)
      alpha = np.sqrt(-1.0/gi[0,0]);
      uu[0] = gamma/alpha;
      uu[1] = vel1 - alpha * gamma * gi[0,1];
      uu[2] = vel2 - alpha * gamma * gi[0,2];
      uu[3] = vel3 - alpha * gamma * gi[0,3];

      uu = np.array(uu)
      if (mhd==True): 
          B_vec = np.zeros(uu.shape)
          B_vec[1] = Bcc1 
          B_vec[2] = Bcc2 
          B_vec[3] = Bcc3
          cks_metric(x,y,z,a)
          for i in range(1,4):
            for mu in range(0,4):
              bu[0] += g[i,mu]*uu[mu]*B_vec[i]
          bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
          bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
          bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
          bu = np.array(bu)
          bu_tmp = bu* 1.0

          bsq = 0
          for i in range(4):
            for j in range(4):
              bsq += g[i,j] * bu[i] * bu[j]

          bd = Lower(bu,g)

  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]

  global ke_ent,k_ent,ke_ent2,ke_ent3
  if (('athena_pp','r0') in ds.field_list ): 
    k_ent = np.array(region['r0'])
  if (('athena_pp','r1') in ds.field_list ): 
    ke_ent = np.array(region['r1'])
  if (('athena_pp','r2') in ds.field_list ): 
    ke_ent2 = np.array(region['r2'])
  if (('athena_pp','r3') in ds.field_list ): 
    ke_ent3 = np.array(region['r3'])

def yt_extract_box_chris(i_dump,box_radius = 0.5,mhd=True,gr=False,a=0.0):
  global region,x,y,z,rho,press,vel1,vel2,vel3,Bcc1,Bcc2,Bcc3,bsq,vsq,uu,bsq,bu,bd
  global nx,ny,nz
  yt_load(i_dump,gr=gr)
  region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):128j,(-box_radius,'pc'):(box_radius,'pc'):128j,
      (-box_radius,'pc'):(box_radius,'pc'):128j]

  x = np.array(region['x'])
  y = np.array(region['y'])
  z = np.array(region['z'])

  rho = np.array(region['rho'])
  press = np.array(region['press'])
  vel1 = np.array(region['vel1'])
  vel2 = np.array(region['vel2'])
  vel3 = np.array(region['vel3'])

  vsq = vel1**2 + vel2**2 + vel3**2
  
  if (mhd==True):
    Bcc1 = np.array(region['Bcc1'])
    Bcc2 = np.array(region['Bcc2'])
    Bcc3 = np.array(region['Bcc3'])
    if (gr==False): bsq = Bcc1**2 + Bcc2**2 + Bcc3**2 


  if (gr==True): region = ds2.r[(-box_radius,'pc'):(box_radius,'pc'):128j,(-box_radius,'pc'):(box_radius,'pc'):128j,
      (-box_radius,'pc'):(box_radius,'pc'):128j]

  if (gr==True):
    u0 = np.array(region['user_out_var1'])
    # u1 = np.array(region['user_out_var2'])
    # u2 = np.array(region['user_out_var3'])
    # u3 = np.array(region['user_out_var4'])
    # uu = np.array([u0,u1,u2,u3])
    if (mhd==True): 
      bsq = np.array(region['user_out_var2'])*2.0
      # B_vec = np.zeros(uu.shape)
      # bu = np.zeros(uu.shape)
      # B_vec[1] = Bcc1 
      # B_vec[2] = Bcc2 
      # B_vec[3] = Bcc3
      # for i in range(1,4):
      #   for mu in range(0,4):
      #     bu[0] += g[i,mu]*uu[mu]*B_vec[i]
      # bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
      # bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
      # bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
      # bd = Lower(bu,g)

  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]

def phibh():
  dx3 = np.diff(x3f)
  dx2 = np.diff(x2f)
  dOmega = (gdet * dx2[None,:,None]) * dx3[None,None,:]
  return 0.5*np.abs(Bcc1*dOmega).sum(-1).sum(-1)
def r_to_ir(r_input):
  dlog10r = np.diff(np.log10(r[-1,:]))[0]
  r_min = r[-1,0]
  r_out = r[-1,-1]
  #r = r_min * 10**(ir*dlog10r)
  return np.int(np.round(np.log10(r_input/r_min)/dlog10r))

def get_mdot(mhd=False,gr=False,a=0):
  global mdot, v,vr,bernoulli,mdot_in,mdot_out, Lx_dot,Ly_dot,Lz_dot
  global l_x,l_y,l_z
  global vth,vphi,Br,Bth,Bphi,Bz,r
  if (nz ==1):
    r = np.sqrt(x**2. + y**2.)
    v = np.sqrt(vel1**2. + vel2**2.)
    vr = vel1 * (x/r)[:,:,None] + vel2 * (y/r)[:,:,None]
    vphi = vel1* (-y/(r+1e-15))[:,:,None]  + vel2 * (x/(r+1e-15))[:,:,None]
    phi = np.arctan2(y,x)
    mdot = 2.*np.pi * r[:,:,None] * rho * vr
    if (mhd==True):
      Br = Bcc1 * (x/r)[:,:,None] + Bcc2 * (y/r)[:,:,None]
      Bphi = Bcc1* (-y/(r+1e-15))[:,:,None]  + Bcc2 * (x/(r+1e-15)) [:,:,None]
      Bz = Bcc3
  elif (gr==False):
    r = np.sqrt(x**2. + y**2. + z**2.)
    s = np.sqrt(x**2. + y**2.)
    vr = vel1 * (x/r) + vel2 * (y/r) + vel3 * (z/r)
    vth = vel1 * (x*z)/(r*s+1e-15)+ vel2 * (y*z)/(r*s+1e-15) + vel3 * (-s/(r+1e-15))
    vphi = vel1* (-y/(s+1e-15))  + vel2 * (x/(s+1e-15)) 
    if (mhd ==True):
      Br = Bcc1 * (x/r) + Bcc2 * (y/r) + Bcc3 * (z/r)
      Bth = Bcc1 * (x*z)/(r*s+1e-15)+ Bcc2 * (y*z)/(r*s+1e-15) + Bcc3 * (-s/(r+1e-15))
      Bphi = Bcc1* (-y/(s+1e-15))  + Bcc2 * (x/(s+1e-15))     
    mdot = 4.*np.pi * r**2. * rho * vr
    mdot_out = mdot * (mdot>0)
    mdot_in = mdot * (mdot<0)
    gam = 5./3.
    gm1 = gam-1.
    gm_ =  0.019264
    bernoulli = (vel1**2.+vel2**2.+vel3**2)/2. + gam*press/rho/gm1 - gm_/r
    l_x = y*vel3 - z*vel2
    l_y = z*vel1 - x*vel3
    l_z = x*vel2 - y*vel1
    Lx_dot = mdot * l_x
    Ly_dot = mdot * l_y 
    Lz_dot = mdot * l_z
  else: #GR
    R = np.sqrt(x**2+y**2+z**2)
    r = np.sqrt( R**2 -a**2 + np.sqrt( (R**2-a**2)**2 + 4.0*a**2*z**2 ) )/np.sqrt(2.0)
    global uu_ks,bu_ks,Bcc1_ks,Bcc2_ks,Bcc3_ks
    uu_ks = cks_vec_to_ks(uu,x,y,z,a=a)

    mdot = 4.0*np.pi/3.0 * (3.0*r**2+a**2) * rho * uu_ks[1]

    if (mhd==True):
      bu_ks = cks_vec_to_ks(bu,x,y,z,a=a)
      Bcc1_ks = (bu_ks[1] * uu_ks[0] - bu_ks[0] * uu_ks[1])
      Bcc2_ks = (bu_ks[2] * uu_ks[0] - bu_ks[0] * uu_ks[2])
      Bcc3_ks = (bu_ks[3] * uu_ks[0] - bu_ks[0] * uu_ks[3])

def cartesian_vector_to_spherical(vx,vy,vz):
    r = np.sqrt(x**2. + y**2. + z**2.)
    s = np.sqrt(x**2. + y**2.)
    vr = vx* (x/r) + vy * (y/r) + vz * (z/r)
    vth = vx * (x*z)/(r*s+1e-15)+ vy * (y*z)/(r*s+1e-15) + vz * (-s/(r+1e-15))
    vphi = vx* (-y/(s+1e-15))  + vy * (x/(s+1e-15))
    return (vr,vth,vphi)
class AthenaError(RuntimeError):
  """General exception class for Athena++ read functions."""
  pass

class AthenaWarning(RuntimeWarning):
  """General warning class for Athena++ read functions."""
  pass

def bhole(a = 0.0,facecolor='white'):
    rhor = ( 1.0 + np.sqrt(1.0-a**2) )
    ax = plt.gca()
    el = Ellipse((0,0), 2*rhor, 2*rhor, facecolor=facecolor, alpha=1)
    art=ax.add_artist(el)
    art.set_zorder(20)
    plt.draw()
def set_constants(n_levels = 8):
  global arc_secs,km_per_s,gm_,rho_to_n_cgs,mp_over_kev
  global v_kep_norm,l_norm, mue,mu_highT,keV_to_Kelvin
  global e_charge,me,cl,mp,pc,kyr,msun,Bunit
  D_BH = 8.3e3 #in parsecs
  #tan(theta) ~ theta ~ x_pc /D_BH
  arc_secs = 4.84814e-6 * D_BH
  km_per_s = 0.001022  # in parsec/k year
  gm_ = 0.019264
  rho_to_n_cgs = 40.46336
  mp_over_kev = 9.994827
  v_kep_norm = np.sqrt(gm_/arc_secs)
  l_norm = v_kep_norm * arc_secs
  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar

  muH_solar = 1./X_solar
  Z = 3. * Z_solar
  X = 0.
  mue = 2. /(1.+X)
  mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
  keV_to_Kelvin = 1.16045e7

  e_charge = 4.803e-10
  me = 9.109e-28
  cl = 2.997924e10
  mp = 1.6726e-24
  pc = 3.086e18
  kyr = 3.154e10
  msun = 1.989e33

  Bunit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 
  global k_green, silver,midnight_green,charcoal,dark_gray,light_blue
  k_green = '#4cbb17'
  silver = '#8D9093'
  midnight_green = '#004851'
  charcoal = '#54585A'
  dark_gray = '#A9A9A9'
  light_blue = "#ADD8E6"


def run_stellar_wind_test(suffix = ""):
  for i in range(1,5):
    N = 2.**i
    os.system("rm *.hst *.rst *.athdf*")
    os.system("sed -i 's_^star\_radius.*_star\_radius = %g_' athinput.star\_cartesian\_smr" %N)
    os.system("mpirun -n 16 ./athena -i athinput.star_cartesian_smr")
    os.system("mkdir N_%g%s -p" %(N,suffix))
    os.system("cp star_wind.out2.00500* *.hst N_%g%s" %(N,suffix))
    os.system("rm *.hst *.rst *.athdf*")

def run_bondi_test():
    for i in range(5,10):
        N = 2**i
        os.system("rm *.hst *.rst *.athdf*")
        os.system("sed -i 's_^nx1   .*_nx1      = %d_' athinput.bondi\_cyl" %(N))
        n_processors = np.amin(np.array([16,N/32]))
        os.system("mpirun -n %d ./athena -i athinput.bondi_cyl" %(n_processors))
        os.system("mkdir N_%g -p" %(N))
        dump_files = glob.glob("*.athdf*")
        dump_files.sort()
        last_dump = dump_files[-1]
        os.system("cp star_wind.out2.00501* *.hst N_%g" %(N))
        os.system("cp star_wind.out2.00000* *.hst N_%g" %(N))
        os.system("rm *.hst *.rst *.athdf*")

def plot_bondi_errors():
    global N_dirs,rho_err_arr,v_err_arr,press_err_arr
    N_dirs = [64,128,256,512]
    rho_err_arr = []
    v_err_arr = []
    press_err_arr = []
    for N in N_dirs:
        os.chdir("N_%g" %N)
        rdhdf5(0,block_level = 0, ndim=2)
        rho_init = rho
        v_init = vel1
        press_init = press
        rdhdf5(501,block_level = 0,ndim = 2)
        rho_err = np.sum(np.abs(rho-rho_init))/(1.*N)
        v_err =np.sum(np.abs(vel1-v_init))/(1.*N)
        press_err = np.sum(np.abs(press-press_init))/(1.*N)
        rho_err_arr.append(rho_err)
        v_err_arr.append(v_err)
        press_err_arr.append(press_err)
        os.chdir("..")



def plot_stellar_wind_test(suffix = ""):
  clf()
  for i in range(1,5):
    N = 2.**i
    os.chdir("N_%g%s" %(N,suffix))
    rdhdf5(500)
    ny = rho.shape[1]
    get_mdot()
    mdot_new = convert_to_polar(mdot).mean(-1)
    #plot(r[:,ny/2],mdot[:,ny/2,0],label = r'$N=%g$' %N)
    plot(r,mdot_new,label = r'$N=%g$' %N)
    os.chdir("..")
  plot(r,0.01*r/r,label = r'$\dot M_\ast$')
  plt.legend(loc = 'best')
  plt.xlabel(r'$r$',fontsize = 30)
  plt.ylabel(r'$\dot M$',fontsize = 30)
  plt.ylim(0.0099,0.0101)
  figure()
  for i in range(1,5):
    N = 2.**i
    os.chdir("N_%g%s" %(N,suffix))
    rdhdf5(500)
    ny = rho.shape[1]
    get_mdot()
    v_new = convert_to_polar(vr).mean(-1)
    #plot(r[:,ny/2],mdot[:,ny/2,0],label = r'$N=%g$' %N)
    plot(r,v_new,label = r'$N=%g$' %N)
    os.chdir("..")
  plot(r,1*r/r,label = r'$v_{wind}$')
  plt.legend(loc = 'best')
  plt.xlabel(r'$r$',fontsize = 30)
  plt.ylabel(r'$v$',fontsize = 30)
  plt.ylim(.95,1.05)


def submit_stellar_wind_test(suffix = ""):
  for i in range(1,5):
      N = 2.**i
      os.system("sed -i 's_^star\_radius.*_star\_radius = %g_' athinput.star\_cartesian\_smr" %N)
      os.system("mkdir N_%g%s -p" %(N,suffix))
      os.system("cp athena athinput.star_cartesian_smr qsub_mpi run_athena.py stellar_wind_test.txt N_%g%s" %(N,suffix))
      os.chdir("N_%g%s" %(N,suffix))
      os.system("sbatch qsub_mpi")
      os.chdir("..")

import scipy
from scipy import interpolate

def convert_to_polar(arr):
    global r,phi,xi,yi
    rmin = np.amin(np.sqrt(x**2.+y**2.))
    rmax = np.amax(np.sqrt(x**2.+y**2.))
    r = np.logspace(log10(rmin),log10(rmax),128)
    phi = np.linspace(0,2.*pi,128)
    xi = r[:,None]*cos(phi[None,:])
    yi = r[:,None]*sin(phi[None,:])
    result = scipy.interpolate.griddata((x.flatten(),y.flatten()),arr[:,:,0].flatten(),(xi,yi),method='nearest')
    return result

def get_conversion_array_2d():
    global r,phi,xi,yi
    global igrid_polar,jgrid_polar
    rmin = np.amin(np.sqrt(x**2.+y**2.))
    rmax = np.amax(np.sqrt(x**2.+y**2.))
    r = np.logspace(log10(rmin),log10(rmax),128)
    phi = np.linspace(0,2.*pi,128)
    xi = r[:,None]*cos(phi[None,:])
    yi = r[:,None]*sin(phi[None,:])
    jgrid,igrid = meshgrid(np.arange(0,ny),np.arange(0,nx))
    mgrid = igrid + jgrid*nx 
    mnew = scipy.interpolate.griddata((x.flatten(),y.flatten()),mgrid[:,:].flatten(),(xi,yi),method='nearest')

    igrid_polar= mod(mnew,nx)
    jgrid_polar = mnew//nx

def get_conversion_array_3d():
    global r,phi,theta,xi,yi,zi,igrid_spherical,jgrid_spherical,kgrid_spherical
    r_tmp = np.sqrt(x**2.+y**2.+z**2.)
    rmin =np.amin(r_tmp)
    rmax = np.amax(r_tmp)
    r = np.logspace(log10(rmin),log10(rmax),128)
    theta = np.linspace(0.,np.pi,128)
    phi = np.linspace(0,2.*np.pi,128)
    xi = r[:,None,None]*np.cos(phi[None,None,:])*np.sin(theta[None,:,None])
    yi = r[:,None,None]*np.sin(phi[None,None,:])*np.sin(theta[None,:,None])
    zi = r[:,None,None]*np.cos(theta[None,:,None]) * ((phi + 4.*np.pi)/(phi + 4.*np.pi))[None,None,:]
    kgrid,jgrid,igrid = meshgrid(np.arange(0,nz),np.arange(0,ny),np.arange(0,nx))
    mgrid = igrid + jgrid*nx  + kgrid*nx*ny
    mnew = scipy.interpolate.griddata((x.flatten(),y.flatten(),z.flatten()),mgrid[:,:,:].flatten(),(xi,yi,zi),method='nearest')

    igrid_spherical= mod(mod(mnew,ny*nx),nx)
    jgrid_spherical = mod(mnew,ny*nx)/nx
    kgrid_spherical = mnew/(ny*nx)
def convert_to_spherical(arr,th = 0,ph = 0):
    global r,phi,theta,xi,yi,zi
    z_hat = np.array([sin(th)*cos(ph),sin(th)*sin(ph),cos(th)])  #aligned with the angular momentum vector
    x_hat = np.array([cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)])  # theta direction at this angle
    y_hat = np.array([-sin(ph),cos(ph),0])    #phi direction at this angle

    x_hat_prime = [np.cos(th)*np.cos(ph),-np.sin(ph),np.sin(th)*np.cos(ph)]   #xhat vector of original coordinates in terms of rotated coords
    y_hat_prime = [np.cos(th)*np.sin(ph),np.cos(ph),np.sin(th)*np.sin(ph)]
    z_hat_prime = [-np.sin(th),0,np.cos(th)]
    r_tmp = np.sqrt(x**2.+y**2.+z**2.)
    rmin =np.amin(r_tmp)
    rmax = np.amax(r_tmp)
    r = np.logspace(log10(rmin),log10(rmax),128)
    theta = np.linspace(0.,np.pi,128)
    phi = np.linspace(0,2.*np.pi,128)
    r,theta,phi = np.meshgrid(r,theta,phi,indexing='ij')
    xi_prime = r*np.cos(phi)*np.sin(theta)
    yi_prime = r*np.sin(phi)*np.sin(theta)
    zi_prime = r*np.cos(theta)

    xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
    yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
    zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2]
    result = scipy.interpolate.griddata((x.flatten(),y.flatten(),z.flatten()),arr[:,:,:].flatten(),(xi,yi,zi),method='nearest')
    return result
def get_radial_profiles():
  global vr_avg,rho_avg,mdot_avg,p_avg,T_avg
  get_mdot()
  vr_avg = (convert_to_spherical(vr)*np.sin(theta)[None,:,None]).mean(-1).mean(-1)/(np.sin(theta)).mean()
  mdot_avg = (convert_to_spherical(mdot)*np.sin(theta)[None,:,None]).mean(-1).mean(-1)/(np.sin(theta)).mean()
  rho_avg = (convert_to_spherical(rho)*np.sin(theta)[None,:,None]).mean(-1).mean(-1)/(np.sin(theta)).mean()
  p_avg =(convert_to_spherical(press)*np.sin(theta)[None,:,None]).mean(-1).mean(-1)/(np.sin(theta)).mean()
  T_avg = p_avg/rho_avg

def get_radial_profile(num,denom=None):

  if (denom is None):
    return (convert_to_spherical(num)*np.sin(theta)).mean(-1).mean(-1)/(np.sin(theta)).mean(-1).mean(-1)

  else:
    return (convert_to_spherical(num)*np.sin(theta)).mean(-1).mean(-1)/(convert_to_spherical(denom)*np.sin(theta)).mean(-1).mean(-1)

def get_radial_profiles_2d(mhd=False):
  global vr_avg,vphi_avg,rho_avg,mdot_avg,p_avg,T_avg,Br_avg,Bphi_avg,bsq_avg
  get_mdot(mhd = mhd)
  # vr_avg = convert_to_polar(vr).mean(-1)
  # mdot_avg = convert_to_polar(mdot).mean(-1)
  # rho_avg = convert_to_polar(rho).mean(-1)
  # p_avg =convert_to_polar(press).mean(-1)

  vr_avg = vr[igrid_polar,jgrid_polar,0].mean(-1)
  vphi_avg = vphi[igrid_polar,jgrid_polar,0].mean(-1)
  mdot_avg = mdot[igrid_polar,jgrid_polar,0].mean(-1)
  rho_avg = rho[igrid_polar,jgrid_polar,0].mean(-1)
  p_avg = press[igrid_polar,jgrid_polar,0].mean(-1)
  T_avg = p_avg/rho_avg

  if (mhd==True):
    Br_avg = Br[igrid_polar,jgrid_polar,0].mean(-1)
    Bphi_avg = Bphi[igrid_polar,jgrid_polar,0].mean(-1)
    bsq_avg = bsq[igrid_polar,jgrid_polar,0].mean(-1)


from mpl_toolkits.mplot3d import Axes3D
def plot_orbits():
  hst_list = glob.glob("*.hst")
  hst_file = hst_list[0]
  hst = np.loadtxt(hst_file)
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  for i_star in range(30):
    x = hst[:,10+i_star]
    y = hst[:,11+i_star]
    z = hst[:,12+i_star]
    ax.plot(x,y,zs=z)

def mk1davg(blocklevel = 2,n_dim = 3):
    dump_files = glob.glob("*.athdf")
    dump_files.sort()
    n_dump = 0
    for dump_file in dump_files:
        rdhdf5(n_dump,block_level = blocklevel,ndim = n_dim)
        get_radial_profiles()
        dic = {"r": r, "vr_avg": vr_avg, "rho_avg": rho_avg,"p_avg": p_avg,"T_avg":T_avg,"mdot_avg":mdot_avg}
        np.savez("dump_%05d.npz" %n_dump,**dic)
        n_dump = n_dump + 1

def rd_1d_avg():
  fname = "1d_avg.npz"
  if (os.path.isfile(fname)): 
    rdnpz(fname)
    return
  dump_list = glob.glob("1d_dump_*")
  dump_list.sort()
  i_dump_max = len(dump_list)
  global r, mdot,Edot,Jdot,t,Phibh, EdotEM, Lx, Ly, Lz, Bx, By, Bz
  global A_jet_p, A_jet_m, x_jet_p,x_jet_m, y_jet_p,y_jet_m, z_jet_p,z_jet_m,rjet_max_p, rjet_max_m
  global gamma_jet_m, gamma_jet_p,vr_jet,rho,rho_fft,phi_fft,mdot_fft,Bphi
  global mdot_mid,Jdot_mid,BrBphi_mid
  r = []
  mdot = []
  Edot = []
  Jdot = []
  t = []
  Phibh = []
  EdotEM = []
  Lx = []
  Ly = []
  Lz = []
  Bx = []
  By = []
  Bz = []
  A_jet_p = []
  A_jet_m = []
  x_jet_p = []
  y_jet_p = []
  z_jet_p = []
  x_jet_m = []
  y_jet_m = []
  z_jet_m = []
  rjet_max_p = []
  rjet_max_m = []
  gamma_jet_p = []
  gamma_jet_m = []
  vr_jet = []
  rho = []
  rho_fft = []
  phi_fft = []
  mdot_fft = []
  Bphi = []
  mdot_mid = []
  Jdot_mid = []
  BrBphi_mid = []

  for dump in dump_list: ## in arange(i_dump_max):
    dic = np.load(dump) #"1d_dump_%04d.npz" %i)
    r.append(dic['r'])
    mdot.append(dic['mdot'])
    if ("Edot" in dic.keys()): Edot.append(dic['Edot'])
    Jdot.append(dic['Jdot'])
    t.append(dic['t'])
    if ("Phibh" in dic.keys()):Phibh.append(dic['Phibh'])
    if ("EdotEM" in dic.keys()): EdotEM.append(dic['EdotEM'])
    if ("Lx" in dic.keys()):
      Lx.append(dic['Lx'])
      Ly.append(dic['Ly'])
      Lz.append(dic['Lz'])
    if ("Bx" in dic.keys()):
      Bx.append(dic['Bx'])
      By.append(dic['By'])
      Bz.append(dic['Bz'])
    if ("A_jet_p" in dic.keys()): 
      A_jet_p.append(dic['A_jet_p'])
      rjet_max_p.append(dic['rjet_max_p'])
    if ("A_jet_m" in dic.keys()): 
      A_jet_m.append(dic['A_jet_m'])
      rjet_max_m.append(dic['rjet_max_m'])
    if ("x_jet_p" in dic.keys()): 
      x_jet_p.append(dic['x_jet_p'])
      y_jet_p.append(dic['y_jet_p'])
      z_jet_p.append(dic['z_jet_p'])
    if ("x_jet_m" in dic.keys()): 
      x_jet_m.append(dic['x_jet_m'])
      y_jet_m.append(dic['y_jet_m'])
      z_jet_m.append(dic['z_jet_m'])
    if ("gamma_jet_m" in dic.keys()):
      gamma_jet_m.append(dic['gamma_jet_m'] )
      gamma_jet_p.append(dic['gamma_jet_p'] )
      vr_jet.append(dic['vr_jet'] )
    if ("rho" in dic.keys()):
      rho.append(dic['rho'])
    if ("rho_fourier" in dic.keys()):
      rho_fft.append(dic['rho_fourier'])
    if ("phi_fourier" in dic.keys()):
      phi_fft.append(dic['phi_fourier'])
    if ("mdot_fourier" in dic.keys()):
      mdot_fft.append(dic['mdot_fourier'])
    if ("Bphi_r_5" in dic.keys()):
      Bphi.append(dic['Bphi_r_5'])  
    if ("mdot_mid" in dic.keys()):
      mdot_mid.append(dic['mdot_mid'])
    if ("Jdot_mid" in dic.keys()):
      Jdot_mid.append(dic['Jdot_mid'])
    if ("BphiBr_mid" in dic.keys()):
      BrBphi_mid.append(dic['Bphi_Br_mid'])


  r = np.array(r)
  mdot = np.array(mdot)
  Edot = np.array(Edot)
  Jdot = np.array(Jdot)
  t = np.array(t)
  Phibh = np.array(Phibh)
  EdotEM = np.array(EdotEM)
  Lx = np.array(Lx)
  Ly = np.array(Ly)
  Lz = np.array(Lz)
  Bx = np.array(Bx)
  By = np.array(By)
  Bz = np.array(Bz)
  A_jet_p = np.array(A_jet_p)
  A_jet_m = np.array(A_jet_m)
  x_jet_p = np.array(x_jet_p)
  y_jet_p = np.array(y_jet_p)
  z_jet_p = np.array(z_jet_p)
  x_jet_m = np.array(x_jet_m)
  y_jet_m = np.array(y_jet_m)
  z_jet_m = np.array(z_jet_m)

  rjet_max_p = np.array(rjet_max_p)
  rjet_max_m = np.array(rjet_max_m)

  gamma_jet_m = np.array(gamma_jet_m)
  gamma_jet_p = np.array(gamma_jet_p)
  vr_jet = np.array(vr_jet)
  rho = np.array(rho)
  rho_fft = np.array(rho_fft)
  mdot_fft = np.array(mdot_fft)
  phi_fft = np.array(phi_fft)
  Bphi = np.array(Bphi)
  mdot_mid = np.array(mdot_mid)
  Jdot_mid = np.array(Jdot_mid)
  BrBphi_mid = np.array(BrBphi_mid)


  dic = {"r":r,"mdot":mdot,"Edot":Edot,"Jdot":Jdot,"t":t,"Phibh":Phibh,"EdotEM":EdotEM,
  "Lx":Lx,"Ly":Ly,"Lz":Lz,"Bx":Bx,"By":By,"Bz":Bz,"A_jet_p": A_jet_p,"A_jet_m": A_jet_m, "x_jet_p": x_jet_p,
  "y_jet_p": y_jet_p,"z_jet_p": z_jet_p,"x_jet_m": x_jet_m,"y_jet_m": y_jet_m,"z_jet_m": z_jet_m,
  "rjet_max_p": rjet_max_p, "rjet_max_m": rjet_max_m,"gamma_jet_m": gamma_jet_m,"gamma_jet_p": gamma_jet_p,"vr_jet": vr_jet,
  "rho": rho,"rho_fft": rho_fft,"mdot_fft": mdot_fft, "phi_fft": phi_fft,"Bphi": Bphi,"BrBphi_mid": BrBphi_mid,"mdot_mid":mdot_mid,
  "Jdot_mid":Jdot_mid}
  np.savez(fname,**dic)


def rd_1d_torus_avg():
  fname = "1d_torus_avg.npz"
  if (os.path.isfile(fname)): 
    rdnpz(fname)
    return
  dump_list = glob.glob("1d_torus_dump_*")
  dump_list.sort()
  i_dump_max = len(dump_list)
  global r, rho,press,beta_inv,pmag,t
  r = []
  rho = []
  press = []
  beta_inv = []
  t = []
  pmag = []
  for dump in dump_list: ## in arange(i_dump_max):
    dic = np.load(dump) #"1d_dump_%04d.npz" %i)
    r.append(dic['r'])
    rho.append(dic['rho'])
    press.append(dic['press'])
    beta_inv.append(dic['beta_inv'])
    pmag.append(dic['pmag'])

    t.append(dic['t'])




  r = np.array(r)
  rho = np.array(rho)
  press = np.array(press)
  beta_inv = np.array(beta_inv)
  t = np.array(t)
  pmag = np.array(pmag)


  dic = {"r":r,"rho":rho,"press":press,"pmag":pmag,"t":t,"beta_inv":beta_inv}
  np.savez(fname,**dic)
if __name__ == "__main__":
    if len(sys.argv)>1:
        if sys.argv[1] == "mk1davg":
            mk1davg(blocklevel = np.int(sys.argv[2]),n_dim = np.int(sys.argv[3]))
        else:
            print( "Unknown command %s" % sys.argv[1] )



def run_scaling_test():
  n_cores = [1,2,4,16,28,56,112]

  for n in n_cores:
    os.system("mkdir n_%d" %n)
    os.system("sed -i 's_^\#SBATCH \-\-ntasks.*_\#SBATCH \-\-ntasks=%g_' qsub_mpi" %n)
    os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-r star\_wind\.04000\.rst time\/tlim=40.02_' qsub_mpi" %n)
    os.system("cp qsub_mpi athena rand_stars_test_2D.txt star_wind.04000.rst ./n_%d" %n)
    os.chdir("n_%d" %n)
    os.system("sbatch qsub_mpi")
    os.chdir("..")

def run_scaling_test_3D():
    import os
    n_cores = [1,2,4,8,16,23,46,92]
    n_cores = [113,226,452]
    n_cores =  [2,4,8,16,32,64,142,284,568,1136]
    n_cores = [64,142,284,568,1136]
    n_cores = [16,64,128,256,362,512,724,1024]
    n_cores = [16, 32, 64, 128, 208, 384, 624, 832, 1248]
    #1,2,4,8,16,32,64,128,256,512,1024,2048,4096

    for n in n_cores:
        os.system("mkdir n_%d" %n)
        os.system("sed -i 's_^\#SBATCH \-\-ntasks.*_\#SBATCH \-\-ntasks=%g_' qsub_strong_scaling" %n)
        #os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-r star\_wind\.00728\.rst time\/tlim=7.289_' qsub_mpi" %n)
        os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-i athinput\.strong\_scaling_' qsub_strong_scaling" %n)
        os.system("cp qsub_strong_scaling  athena orbits_spin_v3.in.txt athinput.strong_scaling ./n_%d" %n)
        os.chdir("n_%d" %n)
        os.system("sbatch qsub_strong_scaling")
        os.chdir("..")

def run_scaling_test_3D_stampede():
    import os
    n_cores = [136, 272, 340, 680, 1360, 2720, 5440]
    #n_cores = [142, 284, 568, 1136, 2272, 4544]
    N_nodes = np.array(n_cores)/48 +1
    #1,2,4,8,16,32,64,128,256,512,1024,2048,4096
#SBATCH -N 56
#SBATCH -n 3600#
    for i in range(len(n_cores)):
        n_core = n_cores[i]
        n_node = N_nodes[i]
        os.system("mkdir n_%d" %n_core)
        os.system("sed -i 's_^\#SBATCH \-N.*_\#SBATCH \-N %g_' qsub_strong_scaling" %n_node)
        os.system("sed -i 's_^\#SBATCH \-n.*_\#SBATCH \-n %g_' qsub_strong_scaling" %n_core)

        #os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-r star\_wind\.00728\.rst time\/tlim=7.289_' qsub_mpi" %n)
        os.system("cp qsub_strong_scaling  athena athinput.* ./n_%d" %n_core)

        os.chdir("n_%d" %n_core)
        os.system("sbatch qsub_strong_scaling")
        os.chdir("..")

def run_weak_scaling():
  n_cores =  [2,4,8,16,32,64,128,256,512,512*2]
  n_dims = [64,128,256,512,576,640,704,768]
  n_dims = [64,128,256,512,640,768]
  n_dims = np.array(n_dims)//2
  #n_dims = [640,704]
  for n_dim in n_dims:
    n_cores = (n_dim/32)**3
    os.system("mkdir n_%d" %n_cores)
    os.system("sed -i 's_^\#SBATCH \-\-ntasks.*_\#SBATCH \-\-ntasks=%g_' qsub_weak_scaling" %n_cores)
    #os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-r star\_wind\.00728\.rst time\/tlim=7.289_' qsub_mpi" %n)
    os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-i athinput\.weak\_scaling_' qsub_weak_scaling" %n_cores)
    os.system("sed -i 's_^nx1     .*_nx1           \= %d_' athinput.weak_scaling" %(n_dim))
    os.system("sed -i 's_^nx2     .*_nx2           \= %d_' athinput.weak_scaling" %(n_dim))
    os.system("sed -i 's_^nx3     .*_nx3           \= %d_' athinput.weak_scaling" %(n_dim))
    os.system("cp qsub_weak_scaling  athena orbits.in.txt athinput.weak_scaling ./n_%d" %n_cores)

    os.chdir("n_%d" %n_cores)
    os.system("sbatch qsub_weak_scaling")
    os.chdir("..")

  # n_dims = np.array(n_dims)/2
  # nx_block = 32
  # ny_block = 32
  # nz_block = 16
  # n_cores = [1,8,64,512,729, 1000,1331]
  # #n_dims = [640,704]
  # n_mults = [1,2,4,8,9,10,11]
  # for n_mult in n_mults:
  #   #n_cores = (n_dim/32)**3
  #   n_cores = n_mult**3
  #   os.system("mkdir n_%d" %n_cores)
  #   os.system("sed -i 's_^\#SBATCH \-\-ntasks.*_\#SBATCH \-\-ntasks=%g_' qsub_weak_scaling" %n_cores)
  #   #os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-r star\_wind\.00728\.rst time\/tlim=7.289_' qsub_mpi" %n)
  #   os.system("sed -i 's_^mpirun.*_mpirun \-np %g \.\/athena \-i athinput\.weak\_scaling_' qsub_weak_scaling" %n_cores)
  #   nx = nx_block*n_mult
  #   ny = ny_block*n_mult 
  #   nz = nz_block*n_mult
  #   os.system("sed -i 's_^nx1     .*_nx1           \= %d_' athinput.weak_scaling" %(nx))
  #   os.system("sed -i 's_^nx2     .*_nx2           \= %d_' athinput.weak_scaling" %(ny))
  #   os.system("sed -i 's_^nx3     .*_nx3           \= %d_' athinput.weak_scaling" %(nz))
  #   os.system("cp qsub_weak_scaling  athena orbits.in.txt athinput.weak_scaling ./n_%d" %n_cores)

  #   os.chdir("n_%d" %n_cores)
  #   os.system("sbatch qsub_weak_scaling")
  #   os.chdir("..")

import re

def get_scaling_results():
  global end_time_arr,cpu_time_arr, zone_cycles_per_second_arr
  global projected_time,n_vals
  ndirs = glob.glob('n_*')
  ndirs.sort()
  n_vals = []
  cpu_time_arr = []
  end_time_arr = []
  zone_cycles_per_second_arr = []
  for ndir in ndirs:
    n = int(ndir[2:])
    n_vals.append(n)
  n_vals.sort()
  n_vals = np.array(n_vals)
  for n in n_vals:
    os.chdir("n_%d" %n)
    if (glob.glob("slur*")==[]): 
      os.chdir("..")
      continue
    get_run_time()
    cpu_time_arr.append(cpu_time)
    end_time_arr.append(end_time)
    zone_cycles_per_second_arr.append(zone_cycles_per_second)
    os.chdir("..")
  cpu_time_arr = np.array(cpu_time_arr)
  zone_cycles_per_second_arr = np.array(zone_cycles_per_second_arr)
  end_time_arr = np.array(end_time_arr)
  projected_time = 2e4/(end_time_arr) * cpu_time_arr *n_vals
def print_scaling_results():
  n_cores = [1,2,4,16,28,56,112]
  n_cores = [1,2,4,8,16,23,46,92]

  cpu_time_arr = []
  for n in n_cores:
    os.chdir("n_%d" %n)
    print (n)
    slurm_file = glob.glob("slurm*")[0]
    with open(slurm_file) as f:
      for line in f:
        if "cpu time use" in line: #zone-cycles/cpu_second" in line:
          c1,c2 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          print (line)
      cpu_time = c1*10.**c2
      cpu_time_arr.append(cpu_time)

    os.chdir("..")


def get_gprof_output():
  gmon_list = glob.glob('gmon*')
  for file in gmon_list:
    os.system('gprof ./athena -p %s >> %s' % (file,'output'+file[8:]))
def get_gprof_results():
  global self_seconds_arr,percent_time_arr
  global self_seconds_riemann_arr
  self_seconds_arr = []
  percent_time_arr = []
  self_seconds_riemann_arr = []
  outputs = glob.glob("output*")
  for file in outputs:
    with open(file) as f:
      for line in f:
        if "cons_force" in line: 
          c1,c2,c3,c4,c5,c6 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          print (line)
    self_seconds = c3
    percent_time = c1
    self_seconds_arr.append(self_seconds)
    percent_time_arr.append(percent_time)
    with open(file) as f:
      for line in f:
        if "Hydro::RiemannSolver" in line: 
          c1,c2,c3,c4,c5,c6 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          print (line)
    self_seconds = c3
    self_seconds_riemann_arr.append(self_seconds)



def get_run_time():
  global cpu_time, zone_cycles_per_second
  global end_time
  slurm_dirs = glob.glob('slurm*')
  slurm_dirs.sort
  slurm_file = slurm_dirs[-1]
  with open(slurm_file) as f:
      for line in f:
        if "cpu time use" in line: #zone-cycles/cpu_second" in line:
          c1,c2 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          cpu_time = c1*10.**c2
        if "cpu_second" in line:
          c3,c4 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          zone_cycles_per_second = c3*10.**c4
        if ("time=" in line) and ("dt=" not in line):
          c5,c6,c7 = map(float,re.findall(r'[+-]?[0-9.]+', line))
          end_time = c5*10.**c6




def rdhdf5_new(file):
  import h5py
  f = h5py.File(file,'r')
  n_blocks = f.attrs['NumMeshBlocks']
  global x,y,z,rho,press,vel1,vel2,vel3
  x = []
  y = []
  z = []
  rho = []
  press = []
  vel1 = []
  vel2 = []
  vel3 = []
  for n in range(n_blocks):
    [X,Y,Z] = np.meshgrid(f['x1v'][n],f['x2v'][n],f['x3v'][n])
    x.append(X)
    y.append(Y)
    z.append(Z)
    rho.append(np.transpose(f['prim'][0][n]))
    press.append(np.transpose(f['prim'][1][n]))
    vel1.append(np.transpose(f['prim'][2][n]))
    vel2.append(np.transpose(f['prim'][3][n]))
    vel3.append(np.transpose(f['prim'][4][n]))

  x,y,z = np.array(x),np.array(y),np.array(z)
  rho = np.array(rho)
  press = np.array(rho)
  vel1 = np.array(vel1)
  vel2 = np.array(vel2)
  vel3 = np.array(vel3)


import yt
def yt_load(ifile,gr=False,a=0.0):
  global ds,ds2
  #T_0_kev = (press/rho * mu_highT*mp_over_kev)[0][0]
  #T_unit = mu_highT * mp_over_keV

  units_override = { "length_unit":(1.0,"pc") ,"time_unit":(1.0,"kyr"),"mass_unit":(1.0,"Msun") }
  if (gr==False): file_prefix = glob.glob("*.athdf")[0][:-11]
  else: file_prefix = glob.glob("*out2*.athdf")[0][:-11]
  ds = yt.load(file_prefix + "%05d.athdf" %ifile,units_override=units_override)
  if (gr==True and glob.glob("*out3*athdf") != []):
    file_prefix = glob.glob("*out3*.athdf")[0][:-11]
    fname = file_prefix +"%05d.athdf" %ifile
    if os.path.isfile(fname): ds2 = yt.load(fname,units_override=units_override)
  global gam 
  gam = ds.gamma

  a = ds.arr(a*1.0,"code_length")

  def _r(field,data):
    x = data['index','x']
    y = data['index','y']
    z = data['index','z']
    R = np.sqrt(x**2+y**2+z**2)
    if (gr==False): return R
    else: return np.sqrt(R**2-a**2 + np.sqrt((R**2-a**2)**2 + 4.0*a**2*z**2))/np.sqrt(2.0)
  def _theta(field,data):
    return np.arccos(data['index','z']/_r(field,data))
  def _phi(field,data):
    y = data['index','y']
    x = data['index','x']
    r = _r(field,data)
    if (gr==False): return np.arctan2(data['index','y'],data['index','x'])
    else: return np.arctan2(a*x-r*y,a*y+r*x)
  
  if (gr==False):
    def _vr(field,data):
      vx = data['gas',"velocity_x"]
      vy = data['gas',"velocity_y"]
      vz = data['gas',"velocity_z"]
      x = data['index','x']
      y = data['index','y']
      z = data['index','z']
      return (x*vx + y*vy + z*vz)/np.sqrt(x**2. + y**2. + z**2.)
    def _mdot(field,data):
      v_r = _vr(field,data)
      r = _r(field,data)
      rho = data['gas','density']

      return 4.*np.pi * rho * v_r * r**2.
    def _mdot_in(field,data):
      mdot = _mdot(field,data)
      return mdot * (mdot<0)
  def _temperature(field,data):
    mp_over_kev = ds.arr(9.994827,"code_time**2./code_length**2.")
    X = 0.7
    Z = 0.02
    muH = 1./X
    mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
    T_kev = data['gas','pressure']/data['gas','density'] * mu_highT*mp_over_kev
    from yt.units import keV
    return T_kev*keV
  def _kappa(field,data):
    return data['gas','pressure']/data['gas','density']**gam
  if (gr==False):
    def _kappa_dot(field,data):
      return _kappa(field,data) * _mdot(field,data)
    def _ldot_x(field,data):
      lx = data['gas','velocity_z'] * data['index','y'] - data['gas','velocity_y']*data['index','z']
      return lx * _mdot(field,data)
    def _ldot_y(field,data):
      ly = data['gas','velocity_x'] * data['index','z'] - data['gas','velocity_z']*data['index','x']
      return ly * _mdot(field,data)
    def _ldot_z(field,data):
      lz = data['gas','velocity_y'] * data['index','x'] - data['gas','velocity_x']*data['index','y']
      return lz * _mdot(field,data)
    def _ldot_x_in(field,data):
      ldot = _ldot_x(field,data)
      mdot = _mdot(field,data)
      return ldot * (mdot<0)
    def _ldot_y_in(field,data):
      ldot = _ldot_y(field,data)
      mdot = _mdot(field,data)
      return ldot * (mdot<0)
    def _ldot_z_in(field,data):
      ldot = _ldot_z(field,data)
      mdot = _mdot(field,data)
      return ldot * (mdot<0)
    def _bernoulli(field,data):
      vsq = data['gas','velocity_x']**2. + data['gas','velocity_y']**2. + data['gas','velocity_z']**2.
      csq = gam * data['gas','pressure']/data['gas','density']
      gm_ = ds.arr(0.019264,"code_length**3/code_time**2")
      return vsq/2. + csq/(gam-1.) - gm_/_r(field,data)
    def _Edot(field,data):
      return _bernoulli(field,data)*_mdot(field,data)


  if (gr==False): ds.add_field(('gas','mdot'),function = _mdot, units = "Msun/kyr",particle_type = False,sampling_type="cell")
  if (gr==False): ds.add_field(('gas','mdot_in'),function = _mdot_in, units = "Msun/kyr",particle_type = False,sampling_type="cell")
  #ds.add_field(('gas','temperature'),function = _temperature, units = "keV", force_override= True,particle_type = False,sampling_type="cell")
  if (gr==False): ds.add_field(('gas',"Ldot_x"),function = _ldot_x,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  if (gr==False): ds.add_field(('gas',"Ldot_y"),function = _ldot_y,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  if (gr==False): ds.add_field(('gas',"Ldot_z"),function = _ldot_z,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  if (gr==False): ds.add_field(('gas',"Ldot_x_in"),function = _ldot_x_in,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  if (gr==False): ds.add_field(('gas',"Ldot_y_in"),function = _ldot_y_in,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  if (gr==False): ds.add_field(('gas',"Ldot_z_in"),function = _ldot_z_in,units = "pc**2/kyr*Msun/kyr",particle_type = False,sampling_type = "cell")
  ##if (gr==False): ds.add_field(('gas','Edot'),function = _Edot, units = "pc**2/kyr**2*Msun/kyr",particle_type = False,sampling_type="cell")
  ##if (gr==False): ds.add_field(('gas',"kappa_dot"),function = _kappa_dot,units="code_length**4*code_pressure*code_velocity/code_mass**(2/3)",particle_type = False,sampling_type = "cell") #units = "(Msun/pc**3)**(5/3)"
  ds.add_field(('gas',"r"),function = _r,units="code_length",particle_type = False,sampling_type = "cell") #units = "(Msun/pc**3)**(5/3)"
  ds.add_field(('gas',"theta"),function = _theta,particle_type = False,sampling_type = "cell") #units = "(Msun/pc**3)**(5/3)"
  ds.add_field(('gas',"phi"),function = _phi,particle_type = False,sampling_type = "cell") #units = "(Msun/pc**3)**(5/3)"


def get_radial_profiles_yt():
  my_sphere = ds.sphere([0,0,0],(ds.domain_right_edge[0],"pc"))

  prof = my_sphere.profile("radius", ["density","pressure","mdot","Ldot_x","Ldot_y","Ldot_z","kappa_dot","Edot","Ldot_x_in","Ldot_y_in","Ldot_z_in","mdot_in"],weight_field = "cell_volume")

  global r 
  r = prof.x
  global rho_avg,p_avg,mdot_avg,Ldotx_avg,Ldoty_avg,Ldotz_avg,kappa_dot_avg
  global Edot_avg,Ldotx_in,Ldoty_in,Ldotz_in, mdot_in  
  rho_avg = prof['density']
  p_avg = prof['pressure']
  mdot_avg = prof['mdot']
  Ldotx_avg = prof['Ldot_x']
  Ldoty_avg = prof['Ldot_y']
  Ldotz_avg = prof['Ldot_z']
  Ldotx_in = prof['Ldot_x_in']
  Ldoty_in = prof['Ldot_y_in']
  Ldotz_in = prof['Ldot_z_in']
  kappa_dot_avg = prof['kappa_dot']
  Edot_avg = prof['Edot']
  mdot_in = prof['mdot_in']


def get_RM_map(file_prefix):

  e_charge = 4.803e-10
  me = 9.109e-28
  cl = 2.997924e10
  mp = 1.6726e-24
  pc = 3.086e18
  kyr = 3.154e10
  msun = 1.989e33

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

  ds.add_field(("gas","RM_integrand"),function = _RM_integrand,units="cm**-3",particle_type = False,sampling_type="cell",force_override=True)

  # prj = ds.proj('RM_integrand','z')
  # frb = prj.to_frb((0.5,'pc'),[1600,1600])
  global x_RM,y_RM,RM_map 

  # x_RM = np.array(frb['x'])
  # y_RM = np.array(frb['y'])
  # RM = np.array(frb['RM_integrand'].in_cgs())

  box_radius = 0.2
  region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,  #512
            (-box_radius,'pc'):(box_radius,'pc'):256j,   #512
            (-1,'pc'):(1,'pc'):1028j ]

  #box_radius = 1.0/2**5
  # region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,  #512
  #           (-box_radius,'pc'):(box_radius,'pc'):256j,   #512
  #           (-box_radius,'pc'):(box_radius,'pc'):256j]
  RM_map = np.array(region['RM_integrand'].mean(-1).in_cgs()) * 2 * pc



  c = matplotlib.pyplot.pcolormesh(region['x'].mean(-1),region['y'].mean(-1),log10(abs(RM_map)),cmap = 'cubehelix',vmin=-1.5,vmax=1.5)
  # c = matplotlib.pyplot.pcolormesh(region['x'].mean(-1),region['y'].mean(-1),log10((RM_map)),cmap = 'Reds',vmin=-1.5,vmax=1.5)
  # c = matplotlib.pyplot.pcolormesh(region['x'].mean(-1),region['y'].mean(-1),log10(-(RM_map)),cmap = 'Blues',vmin=-1.5,vmax=1.5)

  #c = matplotlib.pyplot.pcolormesh(region['x'].mean(-1),region['y'].mean(-1),log10(abs(RM_map)),cmap = 'cubehelix',vmin=-0,vmax=3)

  circ = matplotlib.patches.Circle((dalpha[0],ddelta[0]),radius = .01,fill=False,ls='--',lw=3,color='white')
  matplotlib.pyplot.gca().add_artist(circ)


  plt.xlabel(r'$x$ (pc)',fontsize = 20)
  plt.ylabel(r'$y$ (pc)',fontsize = 20)

  cb = plt.colorbar(c,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
  cb.set_label(r"$ \log_{10}|RM| $ ",fontsize=17)


  for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+cb.ax.get_yticklabels():
      label.set_fontsize(10)
  plt.tight_layout()



def get_Xray_Lum(file_prefix,r_out,type="cylinder",make_image = False):
    mp_over_kev = 9.994827
    Z_o_X_solar = 0.0177
    Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
    X_solar = 0.7491
    Z_solar = 1.0-X_solar - Y_solar

    muH_solar = 1./X_solar
    Z = 3.0 * Z_solar
    X = 0.
    mue = 2. /(1.+X)
    mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
    mp = 8.41175e-58
    def Lam_func(TK):
        f1 = file_prefix + "_H_only.dat"
        f2 = file_prefix + "_He_only.dat"
        f3 = file_prefix + "_Metals_only.dat"
        data = np.loadtxt(f1)
        T_tab = data[:,0] * 1.16045e7
        Lam_H = data[:,1]
        data = np.loadtxt(f2)
        Lam_He = data[:,1]
        data = np.loadtxt(f3)
        Lam_metals = data[:,1]
        # T_tab = 10.**data[:,0]
        T_min = np.amin(T_tab)
        T_max = np.amax(T_tab)
        # if isinstance(TK,)
        # TK[TK<T_min] = T_min
        # TK[TK>T_max] = T_max
        # Lam_tab = 10.**data[:,1]

        Lam_tab = (X/X_solar) * Lam_H + ( (1.-X-Z)/Y_solar) * Lam_He + (Z/Z_solar) * Lam_metals
        from scipy.interpolate import InterpolatedUnivariateSpline
        Lam = InterpolatedUnivariateSpline(T_tab,Lam_tab,k = 1,ext =3)  #ext = 3 returns boundary value if outside range of T
        return Lam(TK)
    def _Lam_chandra(field,data):
        T_kev = data['gas','pressure']/data['gas','density'] * mu_highT*mp_over_kev
        T_K = T_kev*1.16e7
        nH = data['gas','density']/mp/muH_solar
        ne = data['gas','density']/mp/mue
        return Lam_func(T_K) * ne*nH 

    ds.add_field(('gas','Lam_chandra'),function = _Lam_chandra,units="code_mass**2/code_length**6",particle_type = False,sampling_type="cell",force_override=True)
    Lz =(ds.domain_right_edge-ds.domain_left_edge)[2]
    
    if (type=="cylinder"):
        ad = ds.disk("c",[0,0,1],(r_out,"pc"),Lz/2.)  #all_data()
    else:
        ad = ds.sphere("c",(r_out,"pc"))
    average_value = ad.quantities.weighted_average_quantity("Lam_chandra","cell_volume")
    average_value = yt.YTQuantity(average_value,"erg/s*cm**3 /pc**6")
    from yt.units import pc 
    box_volume = (ds.domain_right_edge-ds.domain_left_edge)[0]*(ds.domain_right_edge-ds.domain_left_edge
    )[1]*(ds.domain_right_edge-ds.domain_left_edge)[2]

    if (type=="cylinder"):
        volume = np.pi * ad.radius**2. * Lz
    else:
        volume = 4./3.*np.pi * ad.radius**3.

    average_value = (average_value * volume).in_cgs()

    if (make_image==True): #      proj = ds.proj("Lam_chandra", "z", weight_field="cell_volume")
      global x_im,y_im,image
      box_radius = 1.0
      region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,
                    (-box_radius,'pc'):(box_radius,'pc'):256j,
                    (-Lz/2.0,'pc'):(Lz/2.0,'pc'):256j ]
      image = region['Lam_chandra'].mean(-1)*Lz
      max = np.amax(image)
      image = image  #/np.amax(image)
      x_im = region['x'].mean(-1)
      y_im = region['y'].mean(-1)


    #res = 300
    #length_im = r_out
#frb = proj.to_frb((length_im,"pc"),[res,res])
      #image = frb['Lam_chandra']*Lz
      #x_im = np.linspace(-length_im/2.,length_im/2.,res)
      #y_im = np.linspace(-length_im/2.,length_im/2.,res)


    return average_value



def angle_avg(arr):
  return (arr*sin(theta)).mean(-1).mean(-1)/(sin(theta).mean(-1).mean(-1))

def compute_alpha():
  global sh
  global alpha

  Sh = (angle_avg(Lxdot_avg/s/(4.*pi*r**2.)) - angle_avg(vr_avg)*angle_avg(Lz_avg/s) )   # units of rho v^2
  alpha = Sh/angle_avg(press_tavg)
def make_l_histogram(r_in,r_out):
  global hist,l_hist,mdot_hist
  get_mdot()
  dx = x[1,0,0]-x[0,0,0]
  dy = y[0,1,0] - y[0,0,0]
  dz = z[0,0,1] - z[0,0,0]
  dV = dx*dy*dz
  r = np.sqrt(x**2.+ y**2.+z**2.)
  th = np.arccos(z/r)
  phi = np.arctan2(y,x)
  l = np.sqrt(l_x**2. + l_y**2. + l_z**2.)

  index = (r>=r_in)*(r<=r_out) #*(mdot<=0)
  vol = np.sum(dV*(r/r)[index])

  l_phi = np.arctan2(l_y,l_z)
  l_th = np.arccos(l_z/l)

  hist = plt.hist(th[index].flatten()/pi*180.,weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')
  hist = plt.hist(phi[index].flatten()/pi*180.,weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')

  #plt.hist(log10(l)[index].flatten(),weights=((-mdot*dV/vol/l)[index]).flatten(),bins=100,log=True,histtype='step')

  hist = plt.hist(l_th[index].flatten()/pi*180.,weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')
  hist = plt.hist(l_phi[index].flatten()/pi*180.,weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')


  hist = plt.hist(log10(abs(l))[index].flatten(),weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')
  sign = (l_x>0)*1 - (l_x<0)*1
  hist_x = plt.hist(log10(abs(l_x))[index].flatten(),weights=((-mdot*dV/vol*sign)[index]).flatten(),bins=100,histtype='step')
  sign = (l_y>0)*1 - (l_y<0)*1
  hist_y = plt.hist(log10(abs(l_y))[index].flatten(),weights=((-mdot*dV/vol*sign)[index]).flatten(),bins=hist_x[1],histtype='step')
  sign = (l_z>0)*1 - (l_z<0)*1
  hist_z = plt.hist(log10(abs(l_z))[index].flatten(),weights=((-mdot*dV/vol*sign)[index]).flatten(),bins=hist_x[1],histtype='step')

  hist_z = plt.hist(log10(abs(l))[index].flatten(),weights=((-mdot*dV/vol)[index]).flatten(),bins=100,histtype='step')

  l_hist = 10.**hist[1][1:]
  mdot_hist = hist[0]



 # hist_z[1][1:] are bins
 # hist_z[0] are y axis
  # l_var = hist_x[1]
  # dmdot = hist_x[0]**2. + hist_x[1]





def t_to_it(t_input):
  if (t_input <=t[0]):
    return 0
  if (t_input >=t[-1]):
    return t.shape[0]-1
  for it in range(t.shape[0]-1):
    if ( (t[it] <t_input) and (t_input<=t[it+1]) ):
      if (t_input/t[it]-1. < 1. - t_input/t[it+1]):
        return it
      else:
        return it+1
def cuadra_plot(t_in=None,r_in = None,type = "angular_momentum"):
  simulation_start_time = -1100
  if ( (type =="angular_momentum") and (t_in is None) ):
    print ("time must be specified to plot angular momentum")
    return
  if ( (type=="accretion_rate") and (r_in is None) ):
    r_in = 0.05 * arc_secs
  if ((type=="ancular_momentum_evolution") and (r_in is None)):
    r_in = 0.3 * arc_secs
  if (type == "angular_momentum"):
    it = t_to_it(t_in)
    plt.semilogx(r[0,:]/arc_secs,(L_avg/rho_avg/l_norm)[it-3:it+3,:].mean(0), label = r'l')
    plt.semilogx(r[0,:]/arc_secs,(Lx_avg/rho_avg/l_norm)[it-3:it+3,:].mean(0), label = r'lx',ls="-")
    plt.semilogx(r[0,:]/arc_secs,(Ly_avg/rho_avg/l_norm)[it-3:it+3,:].mean(0), label = r'ly',ls=":")
    plt.semilogx(r[0,:]/arc_secs,(Lz_avg/rho_avg/l_norm)[it-3:it+3,:].mean(0), label = r'lz',ls = "--")
    plt.xlim(.1,2)
    plt.ylim(-1,1)
    plt.xlabel(r"R [arcsecs]",fontsize = 20)
    plt.ylabel(r"Normalised Ang Mom",fontsize = 20)
    plt.legend(loc='best',fontsize = 15)
  elif (type == "accretion_rate"):
    ir = r_to_ir(r_in)
    plt.plot(t*1e3+simulation_start_time,-mdot_avg[:,ir]*1e3,lw = 2)
    plt.ylim(0,8)
    plt.xlim(-1000,100)
    plt.xlabel(r"Time [yr]",fontsize = 20)
    plt.ylabel(r"Acc. Rate [$10^{-6} M_{\rm sun}$ yr$^{-1}$]",fontsize = 20)
  elif (type =="angular_momentum_evolution"):
    ir_max = r_to_ir(r_in)
    ir_min = r_to_ir(0.05*arc_secs)
    plt.plot(t*1e3+simulation_start_time,(L_avg/rho_avg/l_norm)[:,ir_min:ir_max].mean(1),label = r'l',ls="-")
    plt.plot(t*1e3+simulation_start_time,(Lx_avg/rho_avg/l_norm)[:,ir_min:ir_max].mean(1),label = r'lx',ls="-")
    plt.plot(t*1e3+simulation_start_time,(Ly_avg/rho_avg/l_norm)[:,ir_min:ir_max].mean(1),label = r'ly',ls=":")
    plt.plot(t*1e3+simulation_start_time,(Lz_avg/rho_avg/l_norm)[:,ir_min:ir_max].mean(1),label = r'lz',ls="--")
    plt.xlim(-1000,100)
    plt.ylim(-0.5,0.5)
    plt.xlabel(r"Time [yr]",fontsize = 20)
    plt.ylabel(r"Normalised Ang Mom",fontsize = 20)
    plt.legend(loc='best',fontsize = 12)

def plot_cooling_test():
  rd_hst('star_wind.hst')
  plt.clf()
  gam = 5./3.
  p = E/8.*(gam-1.) # divide by box volume
  d = M/8.
  M_sun = 1.989e33
  pc = 3.086e+18
  mp_over_kev = 9.994827
  mp = 1.6726219e-24
  k_b = 1.380658e-16
  X = 0.7
  Z = 0.02
  muH = 1./X
  mue = 2./(1. + X)
  CUADRA_COOL=0
  if (CUADRA_COOL ==0):
    kT_kev_tab = np.array([8.61733130e-06,   8.00000000e-04,   1.50000000e-03,2.50000000e-03,   7.50000000e-03,   
      2.00000000e-02,3.10000000e-02,   1.25000000e-01,   3.00000000e-01,2.26000000e+00,   1.00000000e+02])
    Lam_tab = np.array([1.24666909e-27,   3.99910139e-26,   1.47470970e-22, 1.09120314e-22,   4.92195285e-22,   
      5.38853593e-22, 2.32144473e-22,   1.38278507e-22,   3.66863203e-23,2.15641313e-23,   9.73848346e-23])
    exp_cc = np.array([0.76546122,  13.06493514,  -0.58959508,   1.37120661, 0.09233853,  -1.92144798,  -0.37157016,
      -1.51560627,-0.26314206,   0.39781441,   0.39781441])
  else:
    kT_kev_tab = np.array([8.61733130e-06,   8.00000000e-04,   1.50000000e-03,2.50000000e-03,   7.50000000e-03,   
      2.00000000e-02,3.10000000e-02,   1.25000000e-01,   3.00000000e-01,2.26000000e+00,   1.00000000e+02])
    Lam_tab = np.array([    1.89736611e-19,   7.95699530e-21,   5.12446122e-21,
      3.58388517e-21,   1.66099838e-21,   8.35970776e-22,
      6.15118667e-22,   2.31779533e-22,   1.25581948e-22,
      3.05517566e-23,   2.15234602e-24])
    exp_cc = np.array([    -0.7,  -0.7,  -0.7,   -0.7,
      -0.7,  -0.7,  -0.7,  -0.7,
      -0.7,   -0.7,  -0.7])
  mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
  T_kev = p/d * mu_highT*mp_over_kev
  T_K = T_kev * 1.16e7
  from scipy.interpolate import interp1d
  from scipy.interpolate import InterpolatedUnivariateSpline
  Lam = InterpolatedUnivariateSpline(kT_kev_tab,Lam_tab,k = 1)
  def Lam(T_kev):
    global k
    k = []

    for i in range(T_kev.size):
      for j in range(kT_kev_tab.size-1,-1,-1):
        if (T_kev[i] >= kT_kev_tab[j]):
          break
      k.append(j)
    
    return Lam_tab[k] * (T_kev/kT_kev_tab[k])** exp_cc[k]
  #Lam = s(x)
  #Lam = interp1d(kT_kev_tab,Lam_tab,fill_value = "extrapolate")
  rdhdf5(0)
  d_0 = rho[0][0] * M_sun/(pc**3.)
  T_0_kev = (press/rho * mu_highT*mp_over_kev)[0][0]
  T_0_K = T_0_kev * 1.16e7
  tcool = (muH * muH/mu_highT * mp *k_b * T_0_K / (gam-1.)/Lam(np.array([T_0_kev]))/d_0)[0][0]

  #(T) * (muH * muH) / ( gm1 * d * mu_highT * Lambda_T(T)/UnitLambda_times_mp_times_kev );
  kyr = 31557600000. 
  plt.loglog(t*kyr/tcool,T_K,lw=0,marker = 'o',label = 'small $\Delta t$')

  t_an = np.logspace(-4,2,1000)
  def dT_dt(T_dimensionless,t):
    T_kev = T_dimensionless * T_0_kev 
    return -Lam(T_kev)/Lam(np.array([T_0_kev]))[0][0]  
  from scipy.integrate import odeint

  T_an = odeint(dT_dt,1., t_an)*T_0_K
  plt.loglog(t_an,T_an,lw = 2,label = r'python odeint')
  plt.legend(loc='best')
  plt.xlabel(r'$t/t_{cool}$',fontsize = 25)
  plt.ylabel(r'$T$ (K)',fontsize = 25)
  plt.tight_layout()
  # alpha = -0.7
  # T_ref = 1e7/1.16e7
  # Lam_ref = 6e-23
  # Y_0 = 1./(1.-alpha) * (1. - (T_ref/T_0_kev)**(alpha-.1) )
  # def Y_inv(Y):
  #   return T_ref * (1-(1.-alpha)*Y)**(1./(1.-alpha))

  # T_an = Y_inv(Y_0 + T_0_kev/T_ref * Lam(np.array([T_ref]))[0] /Lam(np.array([T_0_kev]))[0]  * t_an)
  # T_an = (1.**(1.-alpha) - t_an * (1-alpha) )**(1./(1.-alpha))*T_0_K
class Star:
    eccentricity = 0.
    mean_angular_motion = 0.
    alpha = 0.
    beta = 0.
    tau = 0.
    gamma = 0.
    x10 = 0.
    x20 = 0.
    x30 = 0.
    v10 = 0.
    v20 = 0.
    v30 = 0.
    x1 = 0.
    x2 = 0.
    x3 = 0.
    v1 = 0.
    v2 = 0.
    v3 = 0.
    Mdot = 0.
    vwind = 0.
    radius = 0.
def rd_star_file(file):
    global gm_, simulation_start_time,Nstars
    global star_array
    global gd, mdot_star,vwind,ecc,x10,x20,x30,v10,v20,v30,alpha,beta,gamma,tau,mean_angular_motion
    global period, a,b,r_mean,nx1,nx2,nx3
    global mdot_tot
    star_array = []
    fin = open( file, "rb" )
    header = fin.readline().split()
    n = 0
    Nstars =int(header[n]); n+=1
    simulation_start_time =np.float64(header[n]); n+=1
    gm_ = np.float64(header[n]); n+=1

    if (n!=len(header)):
        print ('Extra or missing header variables')
        print (n, 'variables in script', len(header),  'variables in header')
    n=0
    gd = np.loadtxt( file,
                    dtype=np.float64,
                    skiprows=1,
                    unpack = True )

    tmp = gd[n]; n+=1
    mdot_star = gd[n]; n+=1
    mdot_tot = np.sum(mdot_star)
    vwind =gd[n]; n+=1
    x10 = gd[n]; n+=1
    x20 = gd[n]; n+=1
    x30 = gd[n]; n+=1
    v10 = gd[n]; n+=1
    v20 = gd[n]; n+=1
    v30 = gd[n]; n+=1
    alpha = gd[n]; n+=1
    beta = gd[n]; n+=1
    gamma = gd[n]; n+=1
    tau = gd[n]; n+=1
    mean_angular_motion = gd[n]; n+=1
    ecc = gd[n]; n+=1
    nx1 = gd[n]; n+=1
    nx2 = gd[n]; n+=1
    nx3 = gd[n]; n+=1
    period = 2.*np.pi/mean_angular_motion
    a = (gm_/mean_angular_motion**2.)**(1./3.)
    b = a * np.sqrt(1.- ecc**2.)
    r_mean = a * (1. + ecc**2./2.)  #mean radius of orbit in time
    n = 0

    gd=gd.transpose()
    for i in range(Nstars):
        star = Star()
        [tmp, star.Mdot, star.vwind, star.x10 ,star.x20 ,star.x30 ,star.v10 ,star.v20, star.v30 ,
         star.alpha, star.beta, star.gamma,star.tau ,star.mean_angular_motion, star.eccentricity,star.nx1,star.nx2,star.nx3] = gd[n]; n+=1
        star_array.append(star)

def resave_star_file(fname,VARY_MDOT=0,VARY_VWIND=0):
  rd_star_file(fname)
  if (VARY_MDOT==1):
    fname = fname + ".vary_mdot"
  if (VARY_VWIND==1):
    fname = fname + ".vary_vwind"
  else:
    print ("Nothing to do, quitting")
    return
  f = open(fname,"w")
  header = [str(Nstars), str(simulation_start_time), str(gm_)]
  f.write(" ".join(header) + "\n")
  for i in range(Nstars):
      star = star_array[i]
      if (VARY_MDOT==1):
          star.Mdot = star.Mdot/random.uniform(1.,6.)
      if (VARY_VWIND ==1):
          star.vwind = star.vwind/random.uniform(1,6.)
      star_list = [str(0), str(star.Mdot), str(star.vwind), str(star.x1) ,str(star.x2) ,str(star.x3) ,str(star.v1) ,str(star.v2), str(star.v3) ,
          str(star.alpha), str(star.beta), str(star.gamma) ,str(star.tau) ,str(star.mean_angular_motion), str(star.eccentricity)]
      f.write(" ".join(star_list) + "\n")

  f.close()
def get_R_orbit(t_vals):
  global R_orbit

  R_orbit = []

  for i_star in range(Nstars):
    def eqn(e_anomaly,m_anamoly):
      return m_anamoly - e_anomaly + ecc[i_star] * np.sin(e_anomaly)
    mean_anamoly = mean_angular_motion[i_star] * (t_vals -tau[i_star])
    eccentric_anomaly =  fsolve(eqn,mean_anamoly,args = (mean_anamoly,))

    R_orbit.append( a[i_star] * (1.-ecc[i_star]**2.)/ (1. + ecc[i_star]*np.cos(eccentric_anomaly)))
  R_orbit = np.array(R_orbit)

def plot_R_orbit(t_vals):
  plt.figure()
  for i in range(Nstars):
    plt.plot(t_vals*1e3 + simulation_start_time*1e3,R_orbit[i])


def rd_Lx():
  global Lx,t_arr
  file_list = glob.glob("Lx_*.npz")
  file_list.sort()
  t_arr = np.zeros(len(file_list))
  Lx = np.zeros((len(file_list),2))
  for ifile in range(len(file_list)):
    rdnpz(file_list[ifile])
    t_arr[ifile] = t
    Lx[ifile,0] = Lx_1_5
    Lx[ifile,1] = Lx_10

def rd_RM(moving=False):
  global t_arr,RM_sgra,RM_pulsar,z,RM_pulsar_rand,DM_pulsar
  if (moving == True): file_list = glob.glob("RM_dump_moving*.npz")
  else: file_list = glob.glob("RM_dump_*.npz")
  keys = np.load(file_list[0]).keys()
  file_list.sort()
  t_arr = np.zeros(len(file_list))
  RM_sgra = np.zeros(len(file_list))
  rdnpz(file_list[0])
  RM_pulsar = np.zeros((len(file_list),len(z_los)-1))
  RM_pulsar_rand = np.zeros((len(file_list),10))
  DM_pulsar = np.zeros((len(file_list),len(z_los)-1))

  for ifile in range(len(file_list)):
    rdnpz(file_list[ifile])
    t_arr[ifile] = t
    RM_sgra[ifile] = sgra_RM
    RM_pulsar[ifile] = pulsar_RM
    if "pulsar_RM_rand" in keys:
      RM_pulsar_rand[ifile] = pulsar_RM_rand
    if "pulsar_DM" in keys:
      DM_pulsar[ifile] = pulsar_DM
  z = z_los


def rd_column_density(x_pos,y_pos):
  global cd_arr, t_arr
  file_list = glob.glob("column_density_z_*.npz")
  file_list.sort()
  t_arr = np.zeros(len(file_list))
  cd_arr = np.zeros(len(file_list))
  n = 0
  rdnpz(file_list[0])
  def get_i(arg,arr):
    dx = np.diff(arr)[0]
    x_min = arr[0]
    return np.int(np.round((arg-x_min)/dx))

  ix = get_i(x_pos,x[:,0])
  iy = get_i(y_pos,y[0,:])



  for fname in file_list:
    rdnpz(fname)
    cd_arr[n] = column_density[ix,iy]
    t_arr[n] = t
    n = n + 1



# figure(1)
# clf()
# loglog(r,rho_avg,lw=2,ls = '-',color= 'red',label = r'$\rho$')
# loglog(x[:,0],rho[:,0,0],lw = 2,ls = '--',color = 'red')
# loglog(r,T_avg*2.,lw=2,ls ='-',color = 'blue',label = r'$T$')
# loglog(x[:,0],(press/rho*2.)[:,0,0],lw = 2,ls = '--',color = 'blue')
# ylim(1e-2,1e1)
# xlabel(r'$r$',fontsize = 25)
# plt.legend(loc='best',frameon=0,fontsize = 20)
# axvline(rbound,color = 'black',lw = 4)
# figure(2)
# clf()

# semilogx(r,vr_avg,lw=2,ls = '-',color= 'red',label = r'$v_r$')
# semilogx(x[:,0],vel1[:,0,0],lw = 2,ls = '--',color = 'red')
# xlabel(r'$r$',fontsize = 25)
# plt.legend(loc='best',frameon=0,fontsize = 20)
# axvline(rbound,color = 'black',lw = 4)

# ylim(-2,1)

# figure(1)
# clf()
# loglog(r[0,:],rho_avg[2000:4000,:].mean(0),lw=2,ls = '-',color= 'red',label = r'$\rho$')
# loglog(x[:,0],rho[:,0,0],lw = 2,ls = '--',color = 'red')
# loglog(r[0,:],(p_avg/rho_avg)[2000:4000,:].mean(0)*2.,lw=2,ls ='-',color = 'blue',label = r'$T$')
# loglog(x[:,0],(press/rho*2.)[:,0,0],lw = 2,ls = '--',color = 'blue')
# ylim(1e-2,1e2)
# xlabel(r'$r$',fontsize = 25)
# plt.legend(loc='best',frameon=0,fontsize = 20)
# axvline(rbound,color = 'black',lw = 4)
# figure(2)
# clf()

# semilogx(r[0,:],vr_avg[2000:4000,:].mean(0),lw=2,ls = '-',color= 'red',label = r'$v_r$')
# semilogx(x[:,0],vel1[:,0,0],lw = 2,ls = '--',color = 'red')
# xlabel(r'$r$',fontsize = 25)
# plt.legend(loc='best',frameon=0,fontsize = 20)
# axvline(rbound,color = 'black',lw = 4)

# ylim(-1.5,1)

#nspace = 4
#figure(1)
#clf()
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin_smaller_radii/star_wind.hst')
#loglog(r[0,:],rho_avg[1579,:],lw=2,ls = '-',color= 'red',label = r'$\rho$')
#loglog(x[:,0],rho[:,0,0],lw = 2,ls = '--',color = 'red')
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin/star_wind.hst')
#loglog(r[0,:],rho_avg[1579,:],lw=2,ls = '-.',color= 'red')
##rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_all_angles_uniform/star_wind.hst')
#rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_uniform_2000_stars/star_wind.hst')
#
#loglog(r[0,::nspace],rho_avg[2648,::nspace],marker = 'o',lw=0,color= 'red')
#
#
#
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin_smaller_radii/star_wind.hst')
#loglog(r[0,:],(p_avg/rho_avg)[1579,:]*2.,lw=2,ls ='-',color = 'blue',label = r'$T$')
#loglog(x[:,0],(press/rho*2.)[:,0,0],lw = 2,ls = '--',color = 'blue')
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin/star_wind.hst')
#loglog(r[0,:],(p_avg/rho_avg)[1579,:]*2.,lw=2,ls ='-.',color = 'blue')
##rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_all_angles_uniform/star_wind.hst')
#rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_uniform_2000_stars/star_wind.hst')
#
#loglog(r[0,::nspace],(p_avg/rho_avg)[2648,::nspace]*2.,lw=0,marker ='o',color = 'blue')
#
#
#ylim(1e-2,1e2)
#xlabel(r'$r$',fontsize = 25)
#plt.legend(loc='best',frameon=0,fontsize = 20)
##axvline(rbound,color = 'black',lw = 4)
#figure(2)
#clf()
#
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin_smaller_radii/star_wind.hst')
#semilogx(r[0,:],vr_avg[1579,:],lw=2,ls = '-',color= 'red',label = r'$v_r$')
#semilogx(x[:,0],vel1[:,0,0],lw = 2,ls = '--',color = 'red')
#rd_hst('/global/scratch/smressle/star_cluster/eliot_source/bin/star_wind.hst')
#semilogx(r[0,:],vr_avg[1579,:],lw=2,ls = '-.',color= 'red')
##rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_all_angles_uniform/star_wind.hst')
#rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_uniform_2000_stars/star_wind.hst')
#
#semilogx(r[0,::nspace/2],vr_avg[2648,::nspace/2],lw=0,marker = 'o',color= 'red')
#
#
#
#
#xlabel(r'$r$',fontsize = 25)
#plt.legend(loc='best',frameon=0,fontsize = 20)
##axvline(rbound,color = 'black',lw = 4)
#
#ylim(-1.5,1)



# set_constants()
# rbound = 10./128./2.**6.*2./arc_secs
# figure(1)
# clf()
# rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_smaller_radii/star_wind.hst')
# loglog(r[0,:]/arc_secs,rho_avg[650:750,:].mean(0)*rho_to_n_cgs,lw=2,ls = '-',color= 'red',label = r'3D Sim')
# loglog(x[:,0]/arc_secs,rho[:,0,0]*rho_to_n_cgs,lw = 2,ls = '--',color = 'red',label = r'1D Calc')


# rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_smaller_radii/star_wind.hst')
# loglog(r[0,:]/arc_secs,(p_avg[650:750,:].mean(0)/rho_avg[650:750,:].mean(0)*mp_over_kev)*2.,lw=2,ls ='-',color = 'blue')
# loglog(x[:,0]/arc_secs,(press/rho*2.*mp_over_kev)[:,0,0],lw = 2,ls = '--',color = 'blue')

# ylabel(r'$T$ (keV)     $n_e$ (cm$^{-3}$)',fontsize = 25)

# text(1,2,r'$T$',rotation=0,fontsize=20,color='blue')
# text(1,.6e3,r'$n_e$',rotation=0,fontsize=20,color='red')



# xlim(rbound*.9,2e2)

# ylim(5e-2,1e4)
# xlabel(r'$r$ (arcsecs)',fontsize = 25)
# plt.legend(loc='best',frameon=0,fontsize = 20)
# leg = plt.gca().get_legend()
# leg.legendHandles[0].set_color('black')
# leg.legendHandles[1].set_color('black')
# plt.draw()
# axvline(rbound,color = 'black',lw = 4)
# plt.setp(plt.gca().get_xticklabels(), fontsize=20)
# plt.setp(plt.gca().get_yticklabels(), fontsize=20)
# plt.tight_layout()
# plt.savefig("rho_T_rand_stars.pdf")
# figure(2)
# clf()

# rd_hst('/global/scratch/smressle/star_cluster/rand_stars_test/bin_smaller_radii/star_wind.hst')
# semilogx(r[0,:]/arc_secs,vr_avg[650:750,:].mean(0)/(1000.*km_per_s),lw=2,ls = '-',color= 'red',label = r'3D Sim')
# semilogx(x[:,0]/arc_secs,vel1[:,0,0]/(1000.*km_per_s),lw = 2,ls = '--',color = 'red',label = r'1D Calc')



# ylabel(r'$v_r/v_{wind}$',fontsize = 25)


# xlabel(r'$r$ (arcsecs)',fontsize = 25)
# axvline(rbound,color = 'black',lw = 4)
# plt.legend(loc='best',frameon=0,fontsize = 20)

# leg = plt.gca().get_legend()
# leg.legendHandles[0].set_color('black')
# leg.legendHandles[1].set_color('black')
# plt.setp(plt.gca().get_xticklabels(), fontsize=20)
# plt.setp(plt.gca().get_yticklabels(), fontsize=20)
# plt.draw()

# xlim(rbound*.9,2e2)

# ylim(-1.5,1)
# plt.tight_layout()
# plt.savefig("vr_rand_stars.pdf")



# figure(1)
# clf()
# loglog(r[0,:],rho_avg[110,:],lw=2,ls="-",label = r'$\rho$',color = 'black')
# loglog(r_high[0,:],rho_high[110,:],lw=2,ls="--",color = 'black')
# loglog(r[0,:],p_avg[110,:],lw=2,ls="-",label = r'$P$',color = 'blue')
# loglog(r_high[0,:],p_high[110,:],lw=2,ls="--",color = 'blue')
# loglog(r[0,:],abs(vr_avg[110,:]),lw=2,ls="-",label = r'$|v_r|$',color = 'red')
# loglog(r_high[0,:],abs(vr_high[110,:]),lw=2,ls="--",color = 'red')
# loglog(r[0,:],abs((mdot_avg)[110,:]),lw=2,ls="-",label = r'$\dot M$',color = 'green')
# loglog(r_high[0,:],abs(mdot_high[110,:]),lw=2,ls="--",color = 'green')
# xlabel(r'$r$ (pc)',fontsize = 25)
# plt.legend(loc='best',frameon=0)
# plt.tight_layout()

def compare_two_hsts(f1,f2,t1 = 1.1,t2 = 2.2,t0_1 = -1.1,t0_2 = -2.2,lab1=r"absorbing",lab2=r"none"):
  # f1 = "/global/scratch/smressle/star_cluster/cuadra_comp/min_ecc_run/star_wind.hst"
  # f2 = "/global/scratch/smressle/star_cluster/cuadra_comp/min_ecc_run_t0_2.2/star_wind.hst"
  figure(1)
  plt.clf()
  rd_hst(f1)
  it1 = t_to_it(t1)
  loglog(r[0,:],rho_avg[it1-3:it1+3,:].mean(0),lw=2,ls="-",label = r'$\rho$',color = 'black')
  loglog(r[0,:],np.sqrt(5./3.*p_avg[it1-3:it1+3,:].mean(0)/rho_avg[it1-3:it1+3,:].mean(0)),lw=2,ls="-",label = r'$c_s$',color = 'blue')
  loglog(r[0,:],abs((mdot_avg[it1-3:it1+3,:].mean(0)/rho_avg[it1-3:it1+3,:].mean(0)/(r**2.)[it1-3:it1+3,:].mean(0)/np.pi/4.)),lw=2,ls="-",label = r'$|v_r|$',color = 'red')
  loglog(r[0,:],abs((mdot_avg[it1-3:it1+3,:].mean(0))),lw=2,ls="-",label = r'$\dot M$',color = 'green')
  rd_hst(f2)
  it2 = t_to_it(t2)
  loglog(r[0,:],rho_avg[it2-3:it2+3,:].mean(0),lw=2,ls="--",color = 'black')
  loglog(r[0,:],np.sqrt(5./3.*p_avg[it2-3:it2+3,:].mean(0)/rho_avg[it2-3:it2+3,:].mean(0)),lw=2,ls="--",color = 'blue')
  loglog(r[0,:],abs((mdot_avg[it2-3:it2+3,:].mean(0)/rho_avg[it2-3:it2+3,:].mean(0)/(r**2.)[it2-3:it2+3,:].mean(0)/np.pi/4.)),lw=2,ls="--",color = 'red')
  loglog(r[0,:],abs((mdot_avg[it2-3:it2+3,:].mean(0))),lw=2,ls="--",color = 'green')

  xlabel(r'$r$ (pc)',fontsize = 25)
  plt.legend(loc='best',frameon=0)
  plt.tight_layout()


  figure(2)
  plt.clf()
  rd_hst(f1)
  l_kep = np.sqrt(gm_*r[0,:])
  plt.semilogx(r[0,:],abs(L_avg[it1-3:it1+3,:].mean(0)/rho_avg[it1-3:it1+3,:].mean(0))/l_kep,lw = 2, label = r'$l/l_{kep}$',color = 'red')
  # plt.semilogx(r[0,:],abs(Lx_avg/rho_avg)[it1-3:it1+3,:].mean(0),lw = 2, label = r'$l_x$',ls="-",color = 'blue')
  # plt.semilogx(r[0,:],abs(Ly_avg/rho_avg)[it1-3:it1+3,:].mean(0),lw=2, label = r'$l_y$',ls="-",color = 'green')
  # plt.semilogx(r[0,:],abs(Lz_avg/rho_avg)[it1-3:it1+3,:].mean(0),lw = 2, label = r'$l_z$',ls = "-",color = 'red')

  rd_hst(f2)
  l_kep = np.sqrt(gm_*r[0,:])
  plt.semilogx(r[0,:],abs(L_avg[it2-3:it2+3,:].mean(0)/rho_avg[it2-3:it2+3,:].mean(0))/l_kep,lw = 2,ls = "--",color = 'black')
  # plt.semilogx(r[0,:],abs(Lx_avg/rho_avg)[it2-3:it2+3,:].mean(0),lw = 2,ls="--",color = 'blue')
  # plt.semilogx(r[0,:],abs(Ly_avg/rho_avg)[it2-3:it2+3,:].mean(0),lw=2,ls="--",color = 'green')
  # plt.semilogx(r[0,:],abs(Lz_avg/rho_avg)[it2-3:it2+3,:].mean(0),lw = 2,ls = "--",color = 'red')
  plt.yscale('log')
  plt.xlabel(r'$r$ (pc)',fontsize = 25)
  plt.ylabel(r'$l/l_{kep}$',fontsize = 25)
  #plt.legend(loc='best',frameon=0)
  plt.tight_layout()


  figure(3)
  plt.clf()
  rd_hst(f1)
  ir1 = r_to_ir(0.05*arc_secs)
  plt.plot(t*1e3 + t0_1*1e3,-mdot_avg[:,ir1]*1e3,lw=2,ls = "-",color = 'red',label = lab1)
  rd_hst(f2)
  ir2 = r_to_ir(0.05*arc_secs)
  plt.plot(t*1e3 + t0_2*1e3,-mdot_avg[:,ir2]*1e3,lw =2,ls = "--",color = 'black',label = lab2)
  plt.ylim(0,8)
  plt.xlim(min(t0_2,t0_1)*1e3,100)
  plt.xlabel(r"Time [yr]",fontsize = 20)
  plt.ylabel(r"Acc. Rate [$10^{-6} M_{\rm sun}$ yr$^{-1}$]",fontsize = 20)
  plt.legend(loc = 'best',fontsize = 15,frameon=0)
  plt.tight_layout()

def plot_keplerian_torus(r_out,frac_kep):
    fig = plt.figure(1)
    fig.clf()

    q = 3./2.
    
    def v_phi(s):
        return frac_kep*np.sqrt(gm_/s)
    # def Omega(s):
    #     return np.sqrt(gm_/R_pmax**3.) * (s/R_pmax)**(-q)
    # def v_phi(s):
    #     return np.sqrt(gm_/R_pmax) * (s/R_pmax)**(-q+1.)

    global rho,C,RHS
    global rvals,xvals,zvals;
    rvals = np.logspace(np.log10(0.01*r_out),np.log10(2*r_out))
    thvals = np.linspace(0.0001,np.pi-0.0001)
    rvals,thvals = meshgrid(rvals,thvals)
    zvals = np.cos(thvals)*rvals
    xvals = np.sin(thvals)*rvals
    svals = rvals * np.sin(thvals)
    kappa = 1.

    C = -gm_/r_out + v_phi(r_out)**2./(2.*q-2.)
    #C = -gm_/rvals + l_kep**2./(2.*q-2.)/svals**(2.*q-2.)
        
        
    gam =5./3.
    gm1 = gam-1.

    #RHS = C + gm_/rvals - l_kep**2./(2.*q-2.)/svals**(2.*q-2.)
    RHS =C + gm_/rvals - v_phi(svals)**2./(2.*q-2.)
    rho = (RHS*gm1/gam/kappa)**(1./gm1)
    rho[isnan(rho)] = 0

    plt.contourf(xvals,zvals,log10(rho),isfilled=1,levels = np.linspace(log10(np.amax(rho))-5.,log10(np.amax(rho))+.1,200))
    plt.colorbar()


def plot_torus(r_out, R_pmax,q=2.):
    fig = plt.figure(1)
    fig.clf()
    l_kep = np.sqrt(gm_*R_pmax)
    
    # def v_phi(s):
    #     return frac_kep*np.sqrt(gm_/s)
    def Omega(s):
        return np.sqrt(gm_/R_pmax**3.) * (s/R_pmax)**(-q)
    def v_phi(s):
        return np.sqrt(gm_/R_pmax) * (s/R_pmax)**(-q+1.)

    global rho,C,RHS
    global rvals,xvals,zvals,thvals,svals;
    rvals = np.logspace(np.log10(0.01*r_out),np.log10(10*r_out),200)
    thvals = np.linspace(0.0001,np.pi-0.0001,200)
    rvals,thvals = meshgrid(rvals,thvals)
    zvals = np.cos(thvals)*rvals
    xvals = np.sin(thvals)*rvals
    svals = rvals * np.sin(thvals)
    kappa = 1.

    C = -gm_/r_out + v_phi(r_out)**2./(2.*q-2.)
    #C = -gm_/rvals + l_kep**2./(2.*q-2.)/svals**(2.*q-2.)
        
        
    gam =5./3.
    gm1 = gam-1.

    #RHS = C + gm_/rvals - l_kep**2./(2.*q-2.)/svals**(2.*q-2.)
    RHS =C + gm_/rvals - v_phi(svals)**2./(2.*q-2.)
    rho = (RHS*gm1/gam/kappa)**(1./gm1)
    rho[isnan(rho)] = 0

    RHS_max = C + gm_/R_pmax - v_phi(R_pmax)**2./(2.*q-2.)
    rho_max = (RHS_max*gm1/gam/kappa)**(1./gm1)

    plt.contourf(xvals,zvals,log10(rho),isfilled=1,levels = np.linspace(log10(np.amax(rho))-6.,log10(np.amax(rho))+.1,200))
    plt.colorbar()


def plot_rotating_star(r_star,rho_star=1.0):
    Omega = 0.5*np.sqrt(gm_/r_star**3)
    global rho,C,RHS
    global rvals,xvals,zvals,thvals,svals;
    rvals = np.logspace(np.log10(0.01*r_star),np.log10(10*r_star),200)
    thvals = np.linspace(0.0001,np.pi-0.0001,200)
    rvals,thvals = meshgrid(rvals,thvals)
    zvals = np.cos(thvals)*rvals
    xvals = np.sin(thvals)*rvals
    svals = rvals * np.sin(thvals)
    kappa = 1.

    gam = 5.0/3.0
    gm1 = gam-1.0
    ##C = gam/(gm1) * kappa * rho_star**(gam-1.0)
    C = -gm_/r_star

    RHS =C + gm_/rvals + Omega**2 * svals**2/2.0
    rho = (RHS*gm1/gam/kappa)**(1./gm1)
    plt.contourf(xvals,zvals,log10(rho),isfilled=1,levels = np.linspace(log10(np.nanmax(rho))-6.,log10(np.nanmax(rho))+.1,200))


def reshape_arrays():
  global r,th,ph,rho,press,vel1,vel2,vel3 
  r = r.repeat(4).reshape(nx,ny,4) *1.0
  th = th.repeat(4).reshape(nx,ny,4) *1.0
  ph =  ph.repeat(4).reshape(nx,ny,4) *1.0
  rho = rho.repeat(4).reshape(nx,ny,4) *1.0
  press = press.repeat(4).reshape(nx,ny,4) *1.0
  vel1 = vel1.repeat(4).reshape(nx,ny,4) * 1.0
  vel2 = vel2.repeat(4).reshape(nx,ny,4) * 1.0
  vel3 = vel3.repeat(4).reshape(nx,ny,4) * 1.0
def rescale_prim_for_inits(axisym=False):
    def fold_theta(arr,anti=False):
        fac = 1
        if (anti ==True): fac = -1
        return arr[:,::-1,:]/2.0 * fac + arr[:,:,:]/2.0
#        return arr[:,::-1,:]*(th_tavg<=pi/2.)*fac + arr[:,:,:]*(th_tavg>pi/2.)
    def axisymmetrize(arr):
        return arr.mean(-1).repeat(arr.shape[-1]).reshape(arr.shape)
    global r,th,ph,rho,press,vel1,vel2,vel3
    rg = 2.058e-7 #pc
    rin_9 = 2.*2./128./2**9.0
    #rg = rin_9
    r = r_tavg * (rg/rin_9)
    th = th_tavg
    ph = ph_tavg
    press = fold_theta(press_tavg * (rin_9/rg)**2.0)
    rho = fold_theta(rho_tavg * (rin_9/rg))
    vel1 = fold_theta(vr_avg * (rin_9/rg)**0.5)
    vel2 = fold_theta(vth_avg * (rin_9/rg)**0.5,anti=True)
    vel3 = fold_theta(vphi_avg * (rin_9/rg)**0.5)

    if (axisym==True):
      press = axisymmetrize(press)
      rho = axisymmetrize(rho)
      vel1 = axisymmetrize(vel1)
      vel2 = axisymmetrize(vel2)
      vel3 = axisymmetrize(vel3)
      #rand = np.random.uniform(0,1,size=rho.shape)
      #press = press * (1. + 4.0e-2*(rand-0.5))


def read_vector_potential(idump,low_res=False,spherical=True,th = 0, ph = 0,nr=356,nth=128,nphi=128):
  rd_yt_convert_to_spherical(idump,MHD=False,th=th,ph=ph,omega_phi = None,
    dump_name=None,low_res=low_res,nr=nr,nth=nth,nphi=nphi,double_precision=True)
  global A1,A2,A3, Ar, Ath,Aphi
  A1 = vel1
  A2 = vel2
  A3 = vel3 

  if (spherical==True):
    get_mdot(False)
    Ar = vr 
    Ath = vth
    Aphi = vphi


def make_athena_inits(fname,spherical=False,gr=False,mhd=False, vector_potential=False,electrons=False):
  if (spherical==False): data = [x.transpose(),y.transpose(),z.transpose(),rho.transpose(),vel1.transpose(),vel2.transpose(),vel3.transpose(),press.transpose()]
  elif (gr==False): data = [r.transpose(),th.transpose(),ph.transpose(),rho.transpose(),vel1.transpose(),vel2.transpose(),vel3.transpose(),press.transpose()]
  else: data = [r.transpose(),th.transpose(),ph.transpose(),rho.transpose(),vr.transpose(),vth.transpose(),vphi.transpose(),press.transpose()]

  if (mhd==True): 
    if (gr==False): data = data + ([Bcc1.transpose(),Bcc2.transpose(),Bcc3.transpose()])
    else: data = data + ([Br.transpose(),Bth.transpose(),Bphi.transpose()])

  if (vector_potential==True): 
    if (gr==False): data = data + ([np.array(A1.transpose()), np.array(A2.transpose()), np.array(A3.transpose())])
    else: data = data + ([np.array(Ar.transpose()), np.array(Ath.transpose()), np.array(Aphi.transpose())])

  n_electrons = 0
  if (electrons==True):
    n_electrons = 1
    data = data + ([np.array(ke_ent.transpose())])
    try:
      ke_ent2
    except NameError:
      print("ke_ent2 not defined")
    else:
      data = data + ([np.array(ke_ent2.transpose())])
      n_electrons += 1
    try:
      ke_ent3
    except NameError:
      print("ke_ent3 not defined")
    else: 
      data = data + ([np.array(ke_ent3.transpose())])
      n_electrons += 1
  data = np.array(data)
  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]
  header = [np.str(nx),np.str(ny),np.str(nz)]
  if (electrons==True): header = header + ([np.str(n_electrons)])

  fout = open(fname,"w")
  fout.write(" ".join(header) + "\n")
  #fout.flush()
  fout.close()
  fout = open(fname,"ab")
  data = data.transpose(1,2,3,0)
  data.tofile(fout)
  fout.close()


def make_mhd_restart_file(path_to_sim_data,path_to_vector_potential,idump_sim,idump_vector_potential,fname = "vector_potential_dump",electrons=False):
  global r,th,phi,A1,A2,A3,vel1,vel2,vel3,rho,press
  nr = 320
  nth=200
  nphi = 400
  os.chdir(path_to_vector_potential)
  read_vector_potential(idump_vector_potential,spherical=False,nr=nr,nth=nth,nphi=nphi,low_res=True)

  os.chdir(path_to_sim_data)
  rd_yt_convert_to_spherical(idump_sim,MHD=True,nr=nr,nth=nth,nphi=nphi,low_res=True,double_precision=True)
  make_athena_inits(fname,spherical=True,gr=False,mhd=True, vector_potential=True,electrons=electrons)

def make_grmhd_restart_file(path_to_sim_data,path_to_vector_potential,idump_sim,idump_vector_potential,fname = "gr_vector_potential_dump",
  low_res=False,th_tilt=0,ph_tilt = 0,electrons=False):
  global r,th,phi,A1,A2,A3,vel1,vel2,vel3,rho,press
  nr = 320
  nth=200
  nphi = 400

  # low_res = True
  # nr = 32
  # nth = 32
  # nphi = 32
  os.chdir(path_to_vector_potential)
  read_vector_potential(idump_vector_potential,spherical=False,nr=nr,nth=nth,nphi=nphi,low_res=True,th=th_tilt,ph=ph_tilt)

  os.chdir(path_to_sim_data)
  rd_yt_convert_to_spherical(idump_sim,MHD=True,nr=nr,nth=nth,nphi=nphi,low_res=True,th=th_tilt,ph=ph_tilt,double_precision=True)
  make_athena_inits(fname,spherical=True,gr=False,mhd=True, vector_potential=True,electrons=electrons)

def make_grmhd_restart_file_mks(path_to_sim_data,path_to_vector_potential,idump_sim,idump_vector_potential,fname = "gr_vector_potential_dump",
  low_res=False):
  global r,th,ph, phi,A1,A2,A3,vel1,vel2,vel3,rho,press
  nr = 320
  nth=200
  nphi = 400

  # low_res = True
  # nr = 32
  # nth = 32
  # nphi = 32
  os.chdir(path_to_vector_potential)
  read_vector_potential(idump_vector_potential,spherical=True,nr=nr,nth=nth,nphi=nphi,low_res=True,th =1.27542452,ph= -1.74037675)

  os.chdir(path_to_sim_data)
  rd_yt_convert_to_spherical(idump_sim,MHD=True,nr=nr,nth=nth,nphi=nphi,low_res=True,th =1.27542452,ph= -1.74037675)
  th = theta
  ph = phi

  get_mdot(mhd=True)
  make_athena_inits(fname,spherical=True,gr=True,mhd=True, vector_potential=True)


def rd_athena_inits(fname):
  global r, th, phi, rho, vr, vth, vphi, press, Ar, Ath,Aphi
  global Br,Bth,Bphi
  global header
  f = open(fname,"rb")
  header = f.readline().split()
  nx = int(header[0])
  ny = int(header[1])
  nz = int(header[2])
  body = np.fromfile(f,dtype=np.float64,count=-1)
  gd = body.view().reshape((-1,nx,ny,nz),order="F")

  gd = (gd.transpose(0,3,2,1))
  f.close()


  r = gd[0].transpose()
  th = gd[1].transpose()
  phi = gd[2].transpose()
  rho = gd[3].transpose()
  vr = gd[4].transpose()
  vth = gd[5].transpose()
  vphi = gd[6].transpose()
  press = gd[7].transpose()
  Br = gd[8].transpose()
  Bth = gd[9].transpose()
  Bphi = gd[10].transpose()
  Ar = gd[11].transpose()
  Ath = gd[12].transpose()
  Aphi = gd[13].transpose()

def make_athena_inits_wrapper():
  os.chdir("/global/scratch/smressle/star_cluster/poisson_from_restart/beta_1e2_t_120")
  read_vector_potential(123,spherical=True,th = 1.3,ph=-1.8)
  os.chdir("/global/scratch/smressle/star_cluster/test_new_code/without_S2_9_levels_beta_1e2_v3_orbits_comet") 
  rdnpz("dump_spher_120_th_1.3_phi_-1.8.npz")
  get_mdot(mhd=True)
  make_athena_inits("init_beta_1e2_120_Aphi.init",spherical = True,gr=True,mhd=True,vector_potential=True)

def rd_binary(fname,nx,ny,nz):
  global gd,header
  f = open(fname,"rb")
  header = f.readline()
  body = np.fromfile(f,dtype=np.float64,count=-1)
  gd = body.view().reshape((-1,nz,ny,nx),order="F")
  gd = (gd.transpose(0,3,2,1))
  f.close()
def rdnpz(file):
    dic = np.load(file)
    for key in dic.keys():
        exec("globals()['%s'] = dic['%s']" % (key,key))
def rd_yt_convert_to_spherical(idump,MHD=False,th=0,ph=0,omega_phi = None,dump_name=None,low_res =False,
  method='nearest',fill_value = 0.0,gr=False,a=0.0, ISOTHERMAL = False, nr = 356, nth = 128,nphi=128,
  double_precision=False,slice = False,rmin=None,rmax=None,midplane_slice = False):
  if (dump_name is None): 
    if (low_res == True): dump_name = "dump_spher_%d_th_%.2g_phi_%.2g_low_res.npz" %(idump,th,ph)
    elif (slice==True): dump_name = "dump_spher_%d_th_%.2g_phi_%.2g_slice.npz" %(idump,th,ph)
    elif (midplane_slice==True): dump_name = "dump_spher_%d_th_%.2g_phi_%.2g_midplane_slice.npz" %(idump,th,ph)
    else: dump_name = "dump_spher_%d_th_%.2g_phi_%.2g.npz" %(idump,th,ph)
  if (os.path.isfile(dump_name) or 0):
    rdnpz(dump_name)
  else:
    yt_load(idump,gr=gr,a=a)
    if (omega_phi is not None):
      ph = ph + omega_phi * ds.current_time/ds.time_unit
      dump_name = "dump_spher_%d_th_%.2g_phi_%.2g.npz" %(idump,th,ph)
    global r,phi,theta,xi,yi,zi
    global rho,press,vel1,vel2,vel3,x,y,z
    global Bcc1,Bcc2,Bcc3,t 
    global k_ent,ke_ent,ke_ent2,ke_ent3
    #unit vectors for the new coordinate system in the frame of the old coordinates
    z_hat = np.array([sin(th)*cos(ph),sin(th)*sin(ph),cos(th)])   #r
    x_hat = np.array([cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)])  #theta
    y_hat = np.array([-sin(ph),cos(ph),0])                        #phi

    #unit vectors for original coordinate system in the frame of the new coordinates
    #r_vec_old = x_old x_hat_prime + y_old y_hat_prime + z_old z_hat_prime 
    x_hat_prime = [np.cos(th)*np.cos(ph),-np.sin(ph),np.sin(th)*np.cos(ph)]   #in terms of [theta_hat, phi_hat, r_hat] 
    y_hat_prime = [np.cos(th)*np.sin(ph),np.cos(ph),np.sin(th)*np.sin(ph)]
    z_hat_prime = [-np.sin(th),0,np.cos(th)]

    index = np.arange(ds.r['density'].shape[0])
    r_extrema = ds.all_data().quantities.extrema("r")
    if (rmin is None): rmin =r_extrema[0]*2. #ds.domain_width[0]/(ds.domain_dimensions[0]*1.)/
    if (rmax is None): rmax = r_extrema[1]*.9

    #faces
    theta = np.linspace(0.,np.pi,nth+1)
    phi = np.linspace(0,2.*np.pi,nphi+1)

    if (slice==True): phi = np.array([0.0,np.pi])
    if (midplane_slice==True): theta = np.array([np.pi/2.0])

    #centers
    r = np.logspace(log10(rmin),log10(rmax),nr)

    if (midplane_slice==False):
      dth = np.diff(theta)[0]
      theta = (theta + dth/2.0)[:-1]
    if (slice==False):
      dphi = np.diff(phi)[0]
      phi = (phi + dphi/2.0)[:-1]
    r,theta,phi = np.meshgrid(r,theta,phi,indexing='ij')



    ##new x,y,z coords in terms of new r,th,phi coords
    if (gr==True):
      xi_prime = r*np.cos(phi)*np.sin(theta) + a*np.sin(phi)*np.sin(theta)
      yi_prime = r*np.sin(phi)*np.sin(theta) - a*np.cos(phi)*np.sin(theta)
      zi_prime = r*np.cos(theta)
    else:
      xi_prime = r*np.cos(phi)*np.sin(theta) 
      yi_prime = r*np.sin(phi)*np.sin(theta) 
      zi_prime = r*np.cos(theta)

    #r_vec = x_new x_hat_new + y_new y_hat_new + z_new z_hat_new  
    # x_old = r_vec dot x_hat_old 
    #original x,y,z coords in terms of new coords
    if (gr==False):
      xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
      yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
      zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2]
    else: #assume cooordinate system aligned with spin of black hole
      xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
      yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
      zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2] 
    new_index = scipy.interpolate.griddata((ds.r['x'],ds.r['y'],ds.r['z']),index,(xi,yi,zi),method = method,fill_value = fill_value)


    from yt.units import pc, msun,kyr
    rho = ds.r['density'][new_index] * pc**3/msun
    if (ISOTHERMAL == False): press = ds.r['press'][new_index] * pc**3/msun * kyr**2/pc**2
    vel1 = ds.r['vel1'][new_index] * kyr/pc
    vel2 = ds.r['vel2'][new_index] * kyr/pc
    vel3 = ds.r['vel3'][new_index] * kyr/pc

    global uu,bu,bsq
    uu = [0,0,0,0]
    bu = [0,0,0,0]

    if (gr==True):
        cks_metric(xi,yi,zi,a)
        tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
        gamma = np.sqrt(1.0 + tmp);

        # Calculate 4-velocity
        cks_inverse_metric(xi,yi,zi,a)
        alpha = np.sqrt(-1.0/gi[0,0]);
        uu[0] = gamma/alpha;
        uu[1] = vel1 - alpha * gamma * gi[0,1];
        uu[2] = vel2 - alpha * gamma * gi[0,2];
        uu[3] = vel3 - alpha * gamma * gi[0,3];
      # uu[0] = ds2.r['user_out_var1'][new_index]
      # uu[1] = ds2.r['user_out_var2'][new_index]
      # uu[2] = ds2.r['user_out_var3'][new_index]
      # uu[3] = ds2.r['user_out_var4'][new_index]
        uu = np.array(uu)

        uu_tmp = uu*1.0

    
    vx_tmp = vel1
    vy_tmp = vel2
    vz_tmp = vel3 


    #new coords in terms of new r,th,phi
    if (gr==True):
      x = r*np.cos(phi)*np.sin(theta) + a*np.sin(phi)*np.sin(theta)
      y = r*np.sin(phi)*np.sin(theta) - a*np.cos(phi)*np.sin(theta)
      z = r*np.cos(theta)
    else:
      x = r*np.cos(phi)*np.sin(theta) 
      y = r*np.sin(phi)*np.sin(theta) 
      z = r*np.cos(theta)

    if (MHD==True):
        B_unit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 
        Bcc1 = ds.r['Bcc1'][new_index]/B_unit
        Bcc2 = ds.r['Bcc2'][new_index]/B_unit
        Bcc3 = ds.r['Bcc3'][new_index]/B_unit

        if (gr==False):
          Bx_tmp = Bcc1
          By_tmp = Bcc2
          Bz_tmp = Bcc3
          Bcc1 = Bx_tmp*x_hat_prime[0] + By_tmp*y_hat_prime[0] + Bz_tmp*z_hat_prime[0]
          Bcc2 = Bx_tmp*x_hat_prime[1] + By_tmp*y_hat_prime[1] + Bz_tmp*z_hat_prime[1]   
          Bcc3 = Bx_tmp*x_hat_prime[2] + By_tmp*y_hat_prime[2] + Bz_tmp*z_hat_prime[2]

        if (gr==True):
          # bsq = ds2.r['user_out_var5'][new_index]*2.0
          B_vec = np.zeros(uu.shape)
          B_vec[1] = Bcc1 
          B_vec[2] = Bcc2 
          B_vec[3] = Bcc3
          cks_metric(xi,yi,zi,a)
          for i in range(1,4):
            for mu in range(0,4):
              bu[0] += g[i,mu]*uu[mu]*B_vec[i]
          bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
          bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
          bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
          bu = np.array(bu)
          bu_tmp = bu* 1.0

          bsq = 0
          for i in range(4):
            for j in range(4):
              bsq += g[i,j] * bu[i] * bu[j]


    if (gr==False):
        vel1 = vx_tmp*x_hat_prime[0] + vy_tmp*y_hat_prime[0] + vz_tmp*z_hat_prime[0]
        vel2 = vx_tmp*x_hat_prime[1] + vy_tmp*y_hat_prime[1] + vz_tmp*z_hat_prime[1]
        vel3 = vx_tmp*x_hat_prime[2] + vy_tmp*y_hat_prime[2] + vz_tmp*z_hat_prime[2]
    else:
        vel1 = vx_tmp*1.0
        vel2 = vy_tmp*1.0
        vel3 = vz_tmp*1.0
        # x_old = x \cos(\theta)\cos(\varphi) + y (-\sin(\varphi)) + z \sin(\theta)\cos(\varphi)\\
        # y_old = x \cos(\theta)\sin(\varphi) + y \cos(\varphi) + z \sin(\theta)\sin(\varphi) \\
        # z_old = x (-\sin(\theta)) + z \cos(\theta)
        # x_new = x_old * xhat_prime[0] + y_old * yhat_prime[0] + z_old * zhat_prime[0]
        # y_new = x_old * xhat_prime[1] + y_old * yhat_prime[1] + z_old * zhat_prime[1]
        # z_new = x_old * xhat_prime[2] + y_old * yhat_prime[2] + z_old * zhat_prime[2]
          # dxnew_dxold = np.cos(th) * np.cos(ph)
          # dxnew_dyold = - np.sin(ph)
          # dxnew_dzold = np.sin(th) * np.cos(ph)

          # dynew_dxold = np.cos(th) * np.sin(ph)
          # dynew_dyold = np.cos(ph)
          # dynew_dzold = np.sin(th)*np.sin(ph)

          # dznew_dxold = -np.sin(th)
          # dznew_dyold = 0.0
          # dznew_dzold = np.cos(th)
          # uu[1] = uu_tmp[1] * dxnew_dxold + uu_tmp[2] * dxnew_dyold + uu_tmp[3] * dxnew_dzold
          # uu[2] = uu_tmp[1] * dynew_dxold + uu_tmp[2] * dynew_dyold + uu_tmp[3] * dynew_dzold
          # uu[3] = uu_tmp[1] * dznew_dxold + uu_tmp[2] * dznew_dyold + uu_tmp[3] * dznew_dzold

        uu[1] = uu_tmp[1] * x_hat_prime[0] + uu_tmp[2] * y_hat_prime[0]  + uu_tmp[3] * z_hat_prime[0]
        uu[2] = uu_tmp[1] * x_hat_prime[1] + uu_tmp[2] * y_hat_prime[1]  + uu_tmp[3] * z_hat_prime[1]
        uu[3] = uu_tmp[1] * x_hat_prime[2] + uu_tmp[2] * y_hat_prime[2]  + uu_tmp[3] * z_hat_prime[2]

        if (MHD==True):
          bu[1] = bu_tmp[1] * x_hat_prime[0] + bu_tmp[2] * y_hat_prime[0]  + bu_tmp[3] * z_hat_prime[0]
          bu[2] = bu_tmp[1] * x_hat_prime[1] + bu_tmp[2] * y_hat_prime[1]  + bu_tmp[3] * z_hat_prime[1]
          bu[3] = bu_tmp[1] * x_hat_prime[2] + bu_tmp[2] * y_hat_prime[2]  + bu_tmp[3] * z_hat_prime[2]

          # bu[1] = bu_tmp[1] * dxnew_dxold + bu_tmp[2] * dxnew_dyold + bu_tmp[3] * dxnew_dzold
          # bu[2] = bu_tmp[1] * dynew_dxold + bu_tmp[2] * dynew_dyold + bu_tmp[3] * dynew_dzold
          # bu[3] = bu_tmp[1] * dznew_dxold + bu_tmp[2] * dznew_dyold + bu_tmp[3] * dznew_dzold

          Bcc1 = (bu[1] * uu[0] - bu[0] * uu[1])
          Bcc2 = (bu[2] * uu[0] - bu[0] * uu[2])
          Bcc3 = (bu[3] * uu[0] - bu[0] * uu[3])


    global nx,ny,nz 
    nx = x.shape[0]
    if (midplane_slice==False): ny = x.shape[1]
    else: ny = 1
    if (slice==False): nz = x.shape[2]
    else: nz = 2

    if (double_precision==False):
      rho = np.array(np.float32(rho))
      press = np.array(np.float32(press))
      vel1 = np.array(np.float32(vel1))
      vel2 = np.array(np.float32(vel2))
      vel3 = np.array(np.float32(vel3))
      r = np.array(np.float32(r))
      theta = np.array(np.float32(theta))
      x = np.array(np.float32(x))
      y = np.array(np.float32(y))
      z = np.array(np.float32(z))
      phi = np.array(np.float32(phi))

      if (MHD==True):
        Bcc1 = np.array(np.float32(Bcc1))
        Bcc2 = np.array(np.float32(Bcc2))
        Bcc3 = np.array(np.float32(Bcc3))
      if (gr==True):
        uu = np.array(np.float32(uu))
        if (MHD==True):
          bu = np.array(np.float32(bu))
          bsq = np.array(np.float32(bsq))

    if (ISOTHERMAL==False): dic = {"rho": rho, "press":press, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz,"th_tilt":th,"phi_tilt":ph, "t":ds.current_time, "r": r,"th": theta, "ph": phi  }
    else: dic = {"rho": rho, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz,"th_tilt":th,"phi_tilt":ph, "t":ds.current_time, "r": r,"th": theta, "ph": phi }
    if (gr==True):
      dic['uu'] = uu
    if (MHD==True):
        dic["Bcc1"] = Bcc1
        dic["Bcc2"] = Bcc2
        dic["Bcc3"] = Bcc3
        if (gr==True): dic["bu"] = bu
        if (gr==True): dic['bsq'] = bsq
    if (('athena_pp','r0') in ds.field_list ): 
      k_ent = ds.r['r0'][new_index]
    if (('athena_pp','r1') in ds.field_list ): 
      ke_ent = ds.r['r1'][new_index]
      dic["ke_ent"] = ke_ent
    if (('athena_pp','r2') in ds.field_list ): 
      ke_ent2 = ds.r['r2'][new_index]
      dic["ke_ent2"] = ke_ent2
    if (('athena_pp','r3') in ds.field_list ): 
      ke_ent3 = ds.r['r3'][new_index]
      dic["ke_ent3"] = ke_ent3




    t = ds.current_time

    np.savez(dump_name,**dic)

def rd_yt_convert_to_gammie(idump,MHD=False,dump_name=None,low_res =False,method='nearest',
  fill_value = 0.0,gr=False,a=0.0, ISOTHERMAL = False, nr = 356, nth = 128,nphi=128,double_precision=False,hslope=0.3):
  if (dump_name is None): 
    if (low_res == False): dump_name = "dump_gammie_%d.npz" %(idump)
    else: dump_name = "dump_gammie_%d_low_res.npz" %(idump)
  if (os.path.isfile(dump_name) or 0):
    rdnpz(dump_name)
  else:
    yt_load(idump,gr=gr,a=a)
    global r,phi,theta,xi,yi,zi
    global x1,x2,x3
    global rho,press,vel1,vel2,vel3,x,y,z
    global Bcc1,Bcc2,Bcc3,t 
    global k_ent,ke_ent,ke_ent2,ke_ent3
    #unit vectors for the new coordinate system in the frame of the old coordinates
    z_hat = np.array([0,0,1])   #r
    x_hat = np.array([1,0,0])  #theta
    y_hat = np.array([0,1,0])                        #phi

    #unit vectors for original coordinate system in the frame of the new coordinates
    #r_vec_old = x_old x_hat_prime + y_old y_hat_prime + z_old z_hat_prime 
    x_hat_prime = [1,0,0]   #in terms of [theta_hat, phi_hat, r_hat] 
    y_hat_prime = [0,1,0]
    z_hat_prime = [0,0,1]

    index = np.arange(ds.r['density'].shape[0])
    r_extrema = ds.all_data().quantities.extrema("r")
    rmin = 1.0 
    rmax = r_extrema[1]*.9

    #faces
    x2 = np.linspace(0.,1.0,nth+1)
    x3 = np.linspace(0,2.*np.pi,nphi+1)

    #centers
    x1 = np.linspace(np.log(rmin),np.log(rmax),nr+1)

    dx1 = np.diff(x1)[0]
    x1 = (x1 + dx1/2.0)[:-1]
    dx2 = np.diff(x2)[0]
    x2 = (x2 + dx2/2.0)[:-1]
    dx3 = np.diff(x3)[0] 
    x3 = (x3 + dx3/2.0)[:-1]
    x1,x2,x3 = np.meshgrid(x1,x2,x3,indexing='ij')


    ##new x,y,z coords in terms of new r,th,phi coords
    r = np.exp(x1)
    theta = np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
    phi = x3*1.0
    xi_prime = r*np.cos(phi)*np.sin(theta) + a*np.sin(phi)*np.sin(theta)
    yi_prime = r*np.sin(phi)*np.sin(theta) - a*np.cos(phi)*np.sin(theta)
    zi_prime = r*np.cos(theta)


    #r_vec = x_new x_hat_new + y_new y_hat_new + z_new z_hat_new  
    # x_old = r_vec dot x_hat_old 
    #original x,y,z coords in terms of new coords

    xi = xi_prime * x_hat[0] + yi_prime * y_hat[0] + zi_prime * z_hat[0]
    yi = xi_prime * x_hat[1] + yi_prime * y_hat[1] + zi_prime * z_hat[1]
    zi = xi_prime * x_hat[2] + yi_prime * y_hat[2] + zi_prime * z_hat[2] 
    new_index = scipy.interpolate.griddata((ds.r['x'],ds.r['y'],ds.r['z']),index,(xi,yi,zi),method = method,fill_value = fill_value)


    from yt.units import pc, msun,kyr
    rho = ds.r['density'][new_index] * pc**3/msun
    if (ISOTHERMAL == False): press = ds.r['press'][new_index] * pc**3/msun * kyr**2/pc**2
    vel1 = ds.r['vel1'][new_index] * kyr/pc
    vel2 = ds.r['vel2'][new_index] * kyr/pc
    vel3 = ds.r['vel3'][new_index] * kyr/pc

    global uu,bu,bsq
    uu = [0,0,0,0]
    bu = [0,0,0,0]

    if (gr==True):
        cks_metric(xi,yi,zi,a)
        tmp = g[1,1]*vel1*vel1 + 2.0*g[1,2]*vel1*vel2 + 2.0*g[1,3]*vel1*vel3+ g[2,2]*vel2*vel2 + 2.0*g[2,3]*vel2*vel3+ g[3,3]*vel3*vel3;
        gamma = np.sqrt(1.0 + tmp);

        # Calculate 4-velocity
        cks_inverse_metric(xi,yi,zi,a)
        alpha = np.sqrt(-1.0/gi[0,0]);
        uu[0] = gamma/alpha;
        uu[1] = vel1 - alpha * gamma * gi[0,1];
        uu[2] = vel2 - alpha * gamma * gi[0,2];
        uu[3] = vel3 - alpha * gamma * gi[0,3];

        uu = np.array(uu)

        uu_tmp = uu*1.0

    
    vx_tmp = vel1
    vy_tmp = vel2
    vz_tmp = vel3 


    #new coords in terms of new r,th,phi
    x = r*np.cos(phi)*np.sin(theta) + a*np.sin(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta) - a*np.cos(phi)*np.sin(theta)
    z = r*np.cos(theta)


    if (MHD==True):
        B_unit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 
        Bcc1 = ds.r['Bcc1'][new_index]/B_unit
        Bcc2 = ds.r['Bcc2'][new_index]/B_unit
        Bcc3 = ds.r['Bcc3'][new_index]/B_unit

        B_vec = np.zeros(uu.shape)
        B_vec[1] = Bcc1 
        B_vec[2] = Bcc2 
        B_vec[3] = Bcc3
        cks_metric(xi,yi,zi,a)
        for i in range(1,4):
          for mu in range(0,4):
            bu[0] += g[i,mu]*uu[mu]*B_vec[i]
        bu[1] = 1.0/uu[0] * (B_vec[1] + bu[0]*uu[1])
        bu[2] = 1.0/uu[0] * (B_vec[2] + bu[0]*uu[2])
        bu[3] = 1.0/uu[0] * (B_vec[3] + bu[0]*uu[3])
        bu = np.array(bu)
        bu_tmp = bu* 1.0

        bsq = 0
        for i in range(4):
          for j in range(4):
            bsq += g[i,j] * bu[i] * bu[j]


    #convert to gammie 4 vectors

    #first convert to ks:

    uu_ks = cks_vec_to_ks(uu,xi,yi,zi,a=a)
    bu_ks = cks_vec_to_ks(bu,xi,yi,zi,a=a)

    #then to gammie

    uu = ks_vec_to_gammie(uu_ks,x1,x2,x3,a=a,hslope=hslope)
    bu = ks_vec_to_gammie(bu_ks,x1,x2,x3,a=a,hslope=hslope)

    Bcc1 = (bu[1] * uu[0] - bu[0] * uu[1])
    Bcc2 = (bu[2] * uu[0] - bu[0] * uu[2])
    Bcc3 = (bu[3] * uu[0] - bu[0] * uu[3])



    global nx,ny,nz 
    nx = x.shape[0]
    ny = x.shape[1]
    nz = x.shape[2]

    if (double_precision==False):
      rho = np.float32(rho)
      press = np.float32(press)
      vel1 = np.float32(vel1)
      vel2 = np.float32(vel2)
      vel3 = np.float32(vel3)
      r = np.float32(r)
      theta = np.float32(theta)
      x = np.float32(x)
      y = np.float32(y)
      z = np.float32(z)
      phi = np.float32(phi)

      if (MHD==True):
        Bcc1 = np.float32(Bcc1)
        Bcc2 = np.float32(Bcc2)
        Bcc3 = np.float32(Bcc3)
      if (gr==True):
        uu = np.float32(uu)
        bu = np.float32(bu)
        bsq = np.float32(bsq)

    if (ISOTHERMAL==False): dic = {"rho": rho, "press":press, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz, "t":ds.current_time, "r": r,"th": theta, "ph": phi,"x1": x1,"x2":x2,"x3":x3 }
    else: dic = {"rho": rho, "vel1": vel1, "vel2": vel2,"vel3":vel3,"x":x,"y":y,"z":z,"nx":nx,"ny":ny,"nz":nz, "t":ds.current_time, "r": r,"th": theta, "ph": phi,"x1": x1,"x2":x2,"x3":x3 }
    if (gr==True):
      dic['uu'] = uu
    if (MHD==True):
        dic["Bcc1"] = Bcc1
        dic["Bcc2"] = Bcc2
        dic["Bcc3"] = Bcc3
        if (gr==True): dic["bu"] = bu
        if (gr==True): dic['bsq'] = bsq
    if (('athena_pp','r0') in ds.field_list ): 
      k_ent = ds.r['r0'][new_index]
    if (('athena_pp','r1') in ds.field_list ): 
      ke_ent = ds.r['r1'][new_index]
      dic["ke_ent"] = ke_ent
    if (('athena_pp','r2') in ds.field_list ): 
      ke_ent2 = ds.r['r2'][new_index]
      dic["ke_ent2"] = ke_ent2
    if (('athena_pp','r3') in ds.field_list ): 
      ke_ent3 = ds.r['r3'][new_index]
      dic["ke_ent3"] = ke_ent3




    t = ds.current_time

    np.savez(dump_name,**dic)

def make_movies_from_frames():
  for var in ["rho","T"]:
    for scale in ['outer','inner']:
      for direction in ['x','y','z']:
        os.system('ffmpeg -i frame_%s_%s_%s_%s.png -vcodec mpeg4 -qmax 5 movie_%s_%s_%s.mp4' %(var,scale,direction,"%d",var,scale,direction))
def make_lunch_talk_plots():
    r_in  = 2.*2./128./2.**8.
    plt.figure(1)
    plt.clf()
    f1 = "/global/scratch/smressle/star_cluster/restart/absorbing_8_levels_more_output/star_wind.hst"
    f2 = "/global/scratch/smressle/star_cluster/cuadra_comp/without_S2_new_outputs/star_wind.hst"

    f1 = "/global/scratch/smressle/star_cluster/cuadra_comp/test_new_output_8_levels/star_wind.hst"
    f2 = f1
    i_s = 0
    ie = 14
    rd_hst(f1)
    l_kep_in = sqrt(gm_*r_in)
    plt.loglog(r[0,:]/2.167e-7,-mdot_avg[i_s:ie,:].mean(0),lw = 2 ,ls='-',color='red')
    rd_hst(f2)
    plt.loglog(r[0,:]/2.167e-7,mdot_avg[i_s:ie,:].mean(0),lw = 2 ,ls='--',color = 'red')
    plt.xlabel(r"$r/r_g$",fontsize = 30)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+plt.gca().get_yticklabels():
        label.set_fontsize(20)
    plt.gca().axvspan(0.35*0.04/2.167e-7,2*0.04/2.167e-7, alpha=0.5, color='grey')
    plt.tight_layout()
    plt.ylim(1e-4,1.5e0)
    plt.savefig("mdot.png")
    plt.figure(2)
    plt.clf()
    rd_hst(f1)
    def Ldot_avg_func(istart,iend):
        return np.sqrt(Lxdot_avg[istart:iend,:].mean(0)**2. + Lydot_avg[istart:iend,:].mean(0)**2. + Lzdot_avg[istart:iend,:].mean(0)**2.)
    def L_avg_func(istart,iend):
        return np.sqrt(Lx_avg[istart:iend,:].mean(0)**2. + Ly_avg[istart:iend,:].mean(0)**2. + Lz_avg[istart:iend,:].mean(0)**2.)
    plt.loglog(r[0,:]/2.167e-7,abs(Ldot_avg_func(i_s,ie)/mdot_avg[i_s:ie,:].mean(0))/l_kep_in,lw = 2,label = r'$\langle l \rangle_\rho$')
    plt.loglog(r[0,:]/2.167e-7,abs(L_avg_func(i_s,ie)/rho_avg[i_s:ie,:].mean(0))/l_kep_in,lw=2,label = r'$\langle l \rangle_{\dot M}$')
#plt.ylim(5e-5,1e-2)
#plt.xlim(3e1,1e4)
    loglog(r[0,:]/2.167e-7,0.5*l_kep[0,:]/l_kep_in,lw = 2,ls = '--',label = r'$0.5 l_{\rm kep}$')
    plt.xlabel(r"$r/r_g$",fontsize = 30)
#plt.ylabel(r"$l/l_{\rm kep, in}$",fontsize = 30)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels()+plt.gca().get_yticklabels():
        label.set_fontsize(20)
    plt.tight_layout()
    plt.legend(loc = 'best',frameon = 0,fontsize = 20)

    plt.savefig("l_kep.png")

    plt.figure(3)
    plt.clf()

    rdnpz("dump_spher_avg_102_114.npz")

    plt.contourf(-x_tavg[:,:,0],z_tavg[:,:,0],log10(((-mdot_avg).mean(-1))),levels = np.linspace(-4,-2.5,200),extend='both',cmap = 'Blues')
    plt.contourf(-x_tavg[:,:,0],z_tavg[:,:,0],log10(((mdot_avg).mean(-1))),levels = np.linspace(-4,-2.5,200),extend='both',cmap = 'Reds')
    plt.contourf(x_tavg[:,:,0],z_tavg[:,:,0],log10(((-mdot_avg).mean(-1))),levels = np.linspace(-4,-2.5,200),extend='both',cmap = 'Blues')
    plt.contourf(x_tavg[:,:,0],z_tavg[:,:,0],log10(((mdot_avg).mean(-1))),levels = np.linspace(-4,-2.5,200),extend='both',cmap = 'Reds')
    plt.xlim(-0.003*3,.003*3)
    plt.ylim(-0.003*3,.003*3)

    plot_streamlines(box_radius = 0.003*3.)

    plt.axis('off')
    plt.savefig("streamlines.png")

def Eliot_plots():
  plt.figure(1)
  plt.clf()
  r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
  theta = np.arccos(z_tavg/r)
  gam = 5./3.
  gm_ = 0.0191744
  csq = gam * (press_tavg*np.sin(theta)).mean(-1).mean(-1)/(rho_tavg*np.sin(theta)).mean(-1).mean(-1)
  l = (Lz_avg*np.sin(theta)).mean(-1).mean(-1)/(rho_tavg*np.sin(theta)).mean(-1).mean(-1)
  a = gm_/r[:,0,0]
  plt.loglog(r[:,0,0],csq/(gam-1.)/a,label = r"$\frac{c_s^2}{\gamma-1} \frac{r}{GM}$",lw=2,ls = '-')
  plt.loglog(r[:,0,0],l**2./(a*r[:,0,0]**2.),label = r"$\frac{l^2}{l_{\rmkep}^2}$",lw=2,ls = "--")

  plt.xlabel(r'$r$(pc)',fontsize = 25)

  plt.ylim(.5e-2,2e0)
  plt.xlim(1e-4,1e-1)

  plt.legend(loc='best',fontsize = 15,frameon=0)
  plt.tight_layout()

  plt.figure(2)
  plt.clf()

  plt.contourf(x_tavg[:,:,0],z_tavg[:,:,0],log10(abs((Lz_avg).mean(-1))),levels = np.linspace(-1.5,0,200),extend = 'both')
  plt.xlim(0,.001*3)
  plt.ylim(-0.5e-3*3,0.5e-3*3)
  cb = plt.colorbar()

  cb.set_label(r'$\log_{10}\left(\langle L_z\rangle\right)$',fontsize=25)
  plt.xlabel(r'$R (pc)$',fontsize =25)
  plt.ylabel(r'$z (pc)$',fontsize = 25)
  plt.tight_layout()

  plt.figure(3)
  plt.clf()

  plt. contourf(x_tavg[:,:,0],z_tavg[:,:,0],log10(((-mdot_avg).mean(-1))),levels = np.linspace(-4.5,-3,200),extend='both')
  plt.xlim(0,.001*3)
  plt.ylim(-0.5e-3*3,0.5e-3*3)
  cb = plt.colorbar()

  cb.set_label(r'$\log_{10}\left(-\langle\dot M\rangle\right)$',fontsize=25)
  plt.xlabel(r'$R (pc)$',fontsize =25)
  plt.ylabel(r'$z (pc)$',fontsize = 25)
  plt.tight_layout()

  plt.figure(4)
  plt.clf()

  plt. contourf(x_tavg[:,:,0],z_tavg[:,:,0],log10(((rho_tavg).mean(-1))),levels = np.linspace(1.5,2.5,200),extend='both')
  plt.xlim(0,.001*3)
  plt.ylim(-0.5e-3*3,0.5e-3*3)
  cb = plt.colorbar()

  cb.set_label(r'$\log_{10}\left(\langle \rho\rangle\right)$',fontsize=25)
  plt.xlabel(r'$R (pc)$',fontsize =25)
  plt.ylabel(r'$z (pc)$',fontsize = 25)
  plt.tight_layout()


def plot_vectors(A_r,A_th,A_phi=None,box_radius = 0.003,spherical_coords=False):
  global x,y,vx,vz
  global vr_avg
  x,z = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
  s = np.sqrt(x_tavg**2. + y_tavg**2.)
  theta = np.arccos(z_tavg/r)

  #if (spherical_coords ==False ): vth_avg = v1 * (x_tavg*z_tavg)/(r*s+1e-15)+ v2 * (y_tavg*z_tavg)/(r*s+1e-15) + v3 * (-s/(r+1e-15))

  vxi = (A_r).mean(-1) * np.sin(theta[:,:,0]) + (A_th).mean(-1)  * np.cos(theta[:,:,0])
  vzi = (A_r).mean(-1) * np.cos(theta[:,:,0]) + (A_th).mean(-1)  * -np.sin(theta[:,:,0])

  vx = scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vxi.flatten(),(x,z),method = 'nearest')
  vz = scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vzi.flatten(),(x,z),method = 'nearest')

  plt.streamplot(x.transpose(),z.transpose(),vx.transpose(),vz.transpose(),color = 'black')
  plt.streamplot(-x.transpose(),z.transpose(),-vx.transpose(),vz.transpose(),color = 'black')



def plot_streamlines(box_radius = 0.003,spherical_coords=False,gr=False):
  global x,y,vx,vz
  global vr_avg
  x,z = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
  s = np.sqrt(x_tavg**2. + y_tavg**2.)
  theta = np.arccos(z_tavg/r)

  if (gr==True):
      vth_avg = (uu[2]/uu[0])*r
      vr_avg = (uu[1]/uu[0])
  elif (spherical_coords ==False ): vth_avg = vel1_tavg * (x_tavg*z_tavg)/(r*s+1e-15)+ vel2_tavg * (y_tavg*z_tavg)/(r*s+1e-15) + vel3_tavg * (-s/(r+1e-15))

  vxi = (rho_tavg*vr_avg).mean(-1)/rho_tavg.mean(-1) * np.sin(theta[:,:,0]) + (rho_tavg*vth_avg).mean(-1)/rho_tavg.mean(-1)  * np.cos(theta[:,:,0])
  vzi = (rho_tavg*vr_avg).mean(-1)/rho_tavg.mean(-1) * np.cos(theta[:,:,0]) + (rho_tavg*vth_avg).mean(-1)/rho_tavg.mean(-1) * -np.sin(theta[:,:,0])

  vx= scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vxi.flatten(),(x,z),method = 'nearest')
  vz = scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vzi.flatten(),(x,z),method = 'nearest')

  plt.streamplot(x.transpose(),z.transpose(),vx.transpose(),vz.transpose(),color = 'black')
  plt.streamplot(-x.transpose(),z.transpose(),-vx.transpose(),vz.transpose(),color = 'black')

def plot_fieldlines(box_radius = 0.003,spherical_coords=False):
  global x,y,vx,vz
  x,z = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
  s = np.sqrt(x_tavg**2. + y_tavg**2.)
  theta = np.arccos(z_tavg/r)

  if (spherical_coords ==False ): Bth_avg = Bcc1_tavg * (x_tavg*z_tavg)/(r*s+1e-15)+ Bcc2_tavg * (y_tavg*z_tavg)/(r*s+1e-15) + Bcc3_tavg * (-s/(r+1e-15))

  vxi = Br_avg.mean(-1) * np.sin(theta[:,:,0]) + Bth_avg.mean(-1) * np.cos(theta[:,:,0])
  vzi = Br_avg.mean(-1) * np.cos(theta[:,:,0]) + Bth_avg.mean(-1) * -np.sin(theta[:,:,0])

  vx= scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vxi.flatten(),(x,z),method = 'nearest')
  vz = scipy.interpolate.griddata((s[:,:,0].flatten(),z_tavg[:,:,0].flatten()),vzi.flatten(),(x,z),method = 'nearest')

  plt.streamplot(x.transpose(),z.transpose(),vx.transpose(),vz.transpose(),color = 'white',density=0.5)
  plt.streamplot(-x.transpose(),z.transpose(),-vx.transpose(),vz.transpose(),color = 'white',density = 0.5)

def plot_fieldlines_slice(box_radius = 0.003,xbox_radius=None,ybox_radius=None,spherical_coords=False,iphi=0,arrowstyle='->',lw=1,density=1,color='black'):
  global x,y,vx,vz
  global x_grid_r,x_grid_l,x_grid,z_grid
  if (xbox_radius is None and ybox_radius is None): 
    xbox_radius = box_radius
    ybox_radius = box_radius
  elif (ybox_radius is None): ybox_radius = xbox_radius
  elif (xbox_radius is None): xbox_radius = ybox_radius
  ym = -ybox_radius
  yp = ybox_radius
  xp = xbox_radius
  xm = -xbox_radius

  print(xm,xp,ym,yp)
  if (xm<0): x_grid,z_grid = np.meshgrid(np.linspace(0,xp,128),np.linspace(ym,yp ,128),indexing = 'ij')
  else: x_grid,z_grid = np.meshgrid(np.linspace(xm,xp,128),np.linspace(ym,yp ,128),indexing = 'ij')

  r = np.sqrt(x**2. + y**2. + z**2.)
  s = np.sqrt(x**2. + y**2.)
  theta = np.arccos(z/r)



  if (spherical_coords==False): Bth = Bcc1 * (x*z)/(r*s+1e-15)+ Bcc2 * (y*z)/(r*s+1e-15) + Bcc3 * (-s/(r+1e-15))
  else:
    Br = Bcc1
    Bth = Bcc2


  vxi = (Br[:,:,iphi] * np.sin(theta[:,:,0]) + Bth[:,:,iphi] * np.cos(theta[:,:,0]) ) 
  vzi = (Br[:,:,iphi] * np.cos(theta[:,:,0]) + Bth[:,:,iphi] * -np.sin(theta[:,:,0]) ) 

  vx = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  vxr = vx
  vzr = vz
  x_grid_r = x_grid
  z_grid_r = z_grid
  #plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = 'black')


  if (xp>0): x_grid,z_grid = np.meshgrid(np.linspace(xm,0,128),np.linspace(ym,yp ,128),indexing = 'ij')
  else: np.meshgrid(np.linspace(xm,xp,128),np.linspace(ym,yp ,128),indexing = 'ij')
  vxi = (Br[:,:,iphi+nz//2] * np.sin(theta[:,:,0]) + Bth[:,:,iphi+nz//2] * np.cos(theta[:,:,0])) * np.cos(ph[:,:,iphi+nz//2])
  vzi = Br[:,:,iphi+nz//2] * np.cos(theta[:,:,0]) + Bth[:,:,iphi+nz//2] * -np.sin(theta[:,:,0])

  vx= scipy.interpolate.griddata((x[:,:,iphi+nz//2].flatten(),z[:,:,iphi+nz//2].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,iphi+nz//2].flatten(),z[:,:,iphi+nz//2].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  vxl = vx
  vzl = vz

  x_grid_l = x_grid
  z_grid_l = z_grid

  vx = np.concatenate((vxl,vxr),axis=0)
  vz = np.concatenate((vzl,vzr),axis=0)
  x_grid = np.concatenate((x_grid_l,x_grid_r),axis=0)
  z_grid = np.concatenate((z_grid_l,z_grid_r),axis=0)
  plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)

def plot_fieldlines_right(xm,xp,ym,yp,spherical_coords=False,iphi=0,arrowstyle='->',lw=1,density=1,color='black'):
  global x,y,vx,vz
  global x_grid_r,x_grid_l,x_grid,z_grid

  x_grid,z_grid = np.meshgrid(np.linspace(xm,xp,128),np.linspace(ym,yp ,128),indexing = 'ij')

  th_grid = np.arccos(z_grid/sqrt(x_grid**2+z_grid**2))

  r = np.sqrt(x**2. + y**2. + z**2.)
  s = np.sqrt(x**2. + y**2.)
  theta = np.arccos(z/r)



  if (spherical_coords==False): Bth = Bcc1 * (x*z)/(r*s+1e-15)+ Bcc2 * (y*z)/(r*s+1e-15) + Bcc3 * (-s/(r+1e-15))
  else:
    Br = Bcc1
    Bth = Bcc2


  vxi = (Br[:,:,iphi] * np.sin(theta[:,:,0]) + Bth[:,:,iphi] * np.cos(theta[:,:,0]) ) 
  vzi = (Br[:,:,iphi] * np.cos(theta[:,:,0]) + Bth[:,:,iphi] * -np.sin(theta[:,:,0]) ) 

  vx = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  vx[th_grid<np.amin(theta)] = 0.0
  vx[th_grid>np.amax(theta)] = 0.0
  vz[th_grid<np.amin(theta)] = 0.0
  vz[th_grid>np.amax(theta)] = 0.0
  plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)

def plot_streamlines_phi_slice(box_radius = 0.003,iphi = 0):
  global x,y,vx,vz
  x_grid,z_grid = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  r = np.sqrt(x**2. + y**2. + z**2.)
  s = np.sqrt(x**2. + y**2.)
  theta = np.arccos(z/r)



  vth = vel1 * (x*z)/(r*s+1e-15)+ vel2 * (y*z)/(r*s+1e-15) + vel3 * (-s/(r+1e-15))

  vxi = (vr[:,:,iphi] * np.sin(theta[:,:,0]) + vth[:,:,iphi] * np.cos(theta[:,:,0]) ) 
  vzi = (vr[:,:,iphi] * np.cos(theta[:,:,0]) + vth[:,:,iphi] * -np.sin(theta[:,:,0]) ) 

  vx= scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = 'black')


  x_grid,z_grid = np.meshgrid(np.linspace(-box_radius,0,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')
  vxi = (vr[:,:,iphi+nz//2] * np.sin(theta[:,:,0]) + vth[:,:,iphi+nz//2] * np.cos(theta[:,:,0])) * np.cos(ph[:,:,iphi+nz//2])
  vzi = vr[:,:,iphi+nz//2] * np.cos(theta[:,:,0]) + vth[:,:,iphi+nz//2] * -np.sin(theta[:,:,0])

  vx= scipy.interpolate.griddata((x[:,:,iphi+nz//2].flatten(),z[:,:,iphi+nz//2].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,iphi+nz//2].flatten(),z[:,:,iphi+nz//2].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = 'black')



def plot_streamlines_midplane(box_radius = 0.003,density=1,is_tavg=False,color='black',lw=1):
  global x,y,vx,vz
  x_grid,y_grid = np.meshgrid(np.linspace(-box_radius,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  if (is_tavg==False):
    r = np.sqrt(x**2. + y**2. + z**2.)
    s = np.sqrt(x**2. + y**2.)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
  else:
    r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
    s = np.sqrt(x_tavg**2. + y_tavg**2.)
    theta = np.arccos(z_tavg/r)
    phi = np.arctan2(y_tavg,x_tavg)

  if (is_tavg==False):
    vxi = vr[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.cos(phi[:,ny//2,:]) + vphi[:,ny//2,:] * (-np.sin(phi[:,ny//2,:]))
    vyi = vr[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.sin(phi[:,ny//2,:]) + vphi[:,ny//2,:] * ( np.cos(phi[:,ny//2,:]))
    #vzi = vr[:,ny//2,:] * np.cos(theta[:,:,0]) + vth[:,ny//2,:] * -np.sin(theta[:,:,0])
    vx = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),vxi.flatten(),(x_grid,y_grid),method = 'nearest')
    vy = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),vyi.flatten(),(x_grid,y_grid),method = 'nearest')
  else:
    vxi = vr_avg[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.cos(phi[:,ny//2,:]) + vphi_avg[:,ny//2,:] * (-np.sin(phi[:,ny//2,:]))
    vyi = vr_avg[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.sin(phi[:,ny//2,:]) + vphi_avg[:,ny//2,:] * ( np.cos(phi[:,ny//2,:]))
    #vzi = vr[:,ny//2,:] * np.cos(theta[:,:,0]) + vth[:,ny//2,:] * -np.sin(theta[:,:,0])
    vx = scipy.interpolate.griddata((x_tavg[:,ny//2,:].flatten(),y_tavg[:,ny//2,:].flatten()),vxi.flatten(),(x_grid,y_grid),method = 'nearest')
    vy = scipy.interpolate.griddata((x_tavg[:,ny//2,:].flatten(),y_tavg[:,ny//2,:].flatten()),vyi.flatten(),(x_grid,y_grid),method = 'nearest')

  plt.streamplot(x_grid.transpose(),y_grid.transpose(),vx.transpose(),vy.transpose(),color = color,density = density,linewidth=lw)

def plot_fieldlines_midplane(box_radius = 0.003, is_tavg =False):
  global x,y,vx,vz
  x_grid,y_grid = np.meshgrid(np.linspace(-box_radius,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')

  if (is_tavg==False):
    r = np.sqrt(x**2. + y**2. + z**2.)
    s = np.sqrt(x**2. + y**2.)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
  else:
    r = np.sqrt(x_tavg**2. + y_tavg**2. + z_tavg**2.)
    s = np.sqrt(x_tavg**2. + y_tavg**2.)
    theta = np.arccos(z_tavg/r)
    phi = np.arctan2(y_tavg,x_tavg)


  if (is_tavg==False):
    Bxi = Br[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.cos(phi[:,ny//2,:])  + Bphi[:,ny//2,:] * (-np.sin(phi[:,ny//2,:]))
    Byi = Br[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.sin(phi[:,ny//2,:])  + Bphi[:,ny//2,:] * ( np.cos(phi[:,ny//2,:]))
    #vzi = vr[:,ny//2,:] * np.cos(theta[:,:,0]) + vth[:,ny//2,:] * -np.sin(theta[:,:,0])
    Bx = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),Bxi.flatten(),(x_grid,y_grid),method = 'nearest')
    By = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),Byi.flatten(),(x_grid,y_grid),method = 'nearest')
  else:
    Bxi = Br_avg[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.cos(phi[:,ny//2,:])  + Bphi_avg[:,ny//2,:] * (-np.sin(phi[:,ny//2,:]))
    Byi = Br_avg[:,ny//2,:] * np.sin(theta[:,ny//2,:]) * np.sin(phi[:,ny//2,:])  + Bphi_avg[:,ny//2,:] * ( np.cos(phi[:,ny//2,:]))
    #vzi = vr[:,ny//2,:] * np.cos(theta[:,:,0]) + vth[:,ny//2,:] * -np.sin(theta[:,:,0])
    Bx = scipy.interpolate.griddata((x_tavg[:,ny//2,:].flatten(),y_tavg[:,ny//2,:].flatten()),Bxi.flatten(),(x_grid,y_grid),method = 'nearest')
    By = scipy.interpolate.griddata((x_tavg[:,ny//2,:].flatten(),y_tavg[:,ny//2,:].flatten()),Byi.flatten(),(x_grid,y_grid),method = 'nearest')
  plt.streamplot(x_grid.transpose(),y_grid.transpose(),Bx.transpose(),By.transpose(),color = 'black')


  # vxi = vr[:,:,iphi+nz/2] * np.sin(theta[:,:,0]) + vth[:,:,iphi] * np.cos(theta[:,:,0])
  # vyi = vr[:,:,iphi+nz/2] * np.cos(theta[:,:,0]) + vth[:,:,iphi] * -np.sin(theta[:,:,0])

  # vx= scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  # vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  #plt.streamplot(-x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = 'black')
def plot_streamlines_spherical(box_radius = 0.003):
  global vx,vz 
  r = x 
  theta = y 
  x_grid,z_grid = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')


  vxi = vel1.mean(-1) * np.sin(theta) + vel2.mean(-1) * np.cos(theta)
  vzi = vel1.mean(-1) * np.cos(theta) + vel2.mean(-1) * -np.sin(theta)

  x_tmp = r*sin(theta) 
  z_tmp = r*cos(theta)
  vx= scipy.interpolate.griddata((x_tmp.flatten(),z_tmp.flatten()),vxi.flatten(),(x_grid,z_grid),method = 'nearest')
  vz = scipy.interpolate.griddata((x_tmp.flatten(),z_tmp.flatten()),vzi.flatten(),(x_grid,z_grid),method = 'nearest')

  plt.streamplot(x_grid.transpose(),z_grid.transpose(),vx.transpose(),vz.transpose(),color = 'black')
  plt.streamplot(-x_grid.transpose(),z_grid.transpose(),-vx.transpose(),vz.transpose(),color = 'black')

def rd_cooling_files(Z_solar = 1.0):
  global T,Lam_tot,Lam_metal,Lam_non_metal
  data = np.loadtxt('lambda.dat')
  T = data[:,0]
  Lam_non_metal = data[:,13] 
  Lam_metal = data[:,15]
  Lam_tot = Lam_non_metal + Lam_metal*Z_solar

def mk_lambda_file(fname):
  data = np.loadtxt('lambda.dat')
  T = data[:,0][::8]
  Lam_non_metal = data[:,13][::8]
  Lam_metal = data[:,15][::8]

  f = open(fname,"w")
  for i in range(T.shape[0]):
      array = [str(T[i]),str(Lam_non_metal[i]),str(Lam_metal[i])]
      f.write(" ".join(array) + "\n")
  f.close()


def spex_lam():
  global T_kev, L_arr
  
  isfine = 0
  limited_band = 0
  X_ray_image_band = 1
  T_kev = np.array([8.00000000e-04,   1.50000000e-03, 2.50000000e-03,   7.50000000e-03,
    2.00000000e-02, 3.10000000e-02,   1.25000000e-01,   3.00000000e-01,
     8.22000000e-01, 2.26000000e+00, 3.010000000e+00, 3.4700000000e+01, 1.00000000e+02 ])
  if (isfine==1):
      T_K = np.logspace(4,9,100)
      T_kev = T_K/1.16045e7
  if (limited_band ==1 or X_ray_image_band ==1):
    T_K = np.logspace(4,9,100)
    T_kev = T_K/1.16045e7
  L_arr = []
  n = 0
  W_to_erg_s = 1e7
  with_H = 1
  n_metals = 1.0 #Metallicity in Solar Units
  norm = 1e64 * 1e-6  #in cm^-3

  for i in range(3):
      if (i ==0):
        H_only = 1
        He_only = 0
        Metals_only = 0
      elif (i==1):
        H_only = 0
        He_only = 1
        Metals_only = 1
      else:
        H_only = 0
        He_only = 0
        Metals_only = 1
      L_arr = []
      n = 0

      #Lodders 2003
      Z_o_X_solar = 0.0177
      Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
      X_solar = 0.7491
      Z_solar = 1.0-X_solar - Y_solar

      if (with_H==0):
        Y = Y_solar/(1.-X)
        Z = Z_solar/(1.-X) * n_metals
        X = 0.
      for T in T_kev:
        f = open("spex.com",'w')
        f.write("spex << STOP\n")
        f.write("com cie\n")
        f.write("abundance solar\n")
        if (limited_band ==1):
          f.write("elim 2:10\n")
        elif (X_ray_image_band ==1):
          f.write("elim 2:8\n")
        else:  
          f.write("elim 0.0000001:1000000\n")
        f.write("par t val %g \n" %T)
        f.write("par it val %g \n" %T)
#f.write("par ed val 1.\n")
        if (H_only==1):
          f.write("par 02 val 0.0\n")
          for i_el in range(3,31):
                f.write("par %02d val %g\n" %(i_el,0.0))
        elif (He_only==1):
          f.write("par ref val 02\n")
          f.write("par 01 val 0.00001\n")
          for i_el in range(3,31):
                f.write("par %02d val %g\n" %(i_el,0.0))
        elif (Metals_only==1):
          f.write("par ref val 08\n")
          f.write("par 01 val 0.00001\n")
          f.write("par 02 val 0.0\n")
          for i_el in range(3,31):
                f.write("par %02d val %g\n" %(i_el,n_metals))
        f.write("calc\n")
        f.write("log out spex_%d o\n" %n)
        f.write("par show\n")
        f.write("quit\n")
        f.write("STOP\n")
        f.close()
        os.system("chmod +x spex.com")
        os.system("./spex.com")
        save_line = 0
        with open("spex_%d.out"%n) as f:
            for num,line in enumerate(f,0):
                if "(W)" in line:
                    i_line = num+1
                    break
        f = open("spex_%d.out"%n)
        lines = f.readlines()
        f.close()
        tmp = list(map(float,re.findall(r'[+-]?[0-9.]+', lines[i_line])) )
        Lum = tmp[-2] * 10.**(tmp[-1])
        L_arr.append(Lum)
        os.system("rm spex_%d.out"%n)

        n = n+1
      L_arr = np.array(L_arr)*W_to_erg_s / norm

      if (isfine==0 and limited_band==0):
        L_arr[-1] = L_arr[-2] * (T_kev[-1]/T_kev[-2])**(np.log(L_arr[-2]/L_arr[-3])/np.log(T_kev[-2]/T_kev[-3]))



      fname = "Lam_spex_Z_solar"

      if (limited_band==1):
        fname = fname + "_2_10_kev"

      if (X_ray_image_band==1):
        fname = fname + "_2_8_kev"

      if (isfine==1):
        file_suffix = "_fine.dat"
      else:
        file_suffix = ".dat"
      if (H_only==1):
        fname = fname + "_H_only" + file_suffix
      elif (He_only ==1):
        fname = fname + "_He_only" + file_suffix
      elif (Metals_only ==1):
        fname =fname + "_Metals_only" + file_suffix
      f = open(fname,"w")

      for i in range(T_kev.shape[0]):
          array = [str(T_kev[i]),str(L_arr[i])]
          f.write(" ".join(array) + "\n")
      f.close()

def rd_spex_lam(fname):
  global T_kev,T_K,Lam
  data = np.loadtxt(fname)
  T_kev = data[:,0]
  T_K = T_kev * 1.16045e7
  Lam = data[:,1]

def compute_total_lam(file_prefix,X=0,Z=3,isfine = 0):
  global T_K, T_kev,Lam,Lam_He,Lam_metals,Lam_H,muH_solar,mue,mu

  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar


  Z  = Z * Z_solar
  muH_solar = 1./X_solar
  mu = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
  mue = 2./(1+X)

  if (isfine==0):
    f1 = file_prefix + "_H_only.dat"
    f2 = file_prefix + "_He_only.dat"
    f3 = file_prefix + "_Metals_only.dat"
  else:
    f1 = file_prefix + "_H_only_fine.dat"
    f2 = file_prefix + "_He_only_fine.dat"
    f3 = file_prefix + "_Metals_only_fine.dat" 
  data = np.loadtxt(f1)
  T_kev = data[:,0]
  T_K = data[:,0] * 1.16045e7
  Lam_H = data[:,1]
  data = np.loadtxt(f2)
  Lam_He = data[:,1]
  data = np.loadtxt(f3)
  Lam_metals = data[:,1]

  Lam = Lam_H * X/X_solar + Lam_He * (1-X-Z)/Y_solar + Lam_metals * Z/Z_solar



def spex_ne_nH():
    global T_kev, nerat_arr

    T_kev = np.array([8.00000000e-04,   1.50000000e-03, 2.50000000e-03,   7.50000000e-03,
                  2.00000000e-02, 3.10000000e-02,   1.25000000e-01,   3.00000000e-01,
                  8.22000000e-01, 2.26000000e+00, 3.010000000e+00, 3.4700000000e+01,1.00000000e+02 ])
    T_K = np.logspace(4,9,100)
    T_kev = T_K/1.16045e7
    nerat_arr = []
    n = 0
    W_to_erg_s = 1e7
    with_H = 1
    n_metals = 1.0 #Metallicity in Solar Units
    norm = 1e64 * 1e-6  #in cm^-3


    #Lodders 2003
    Z_o_X_solar = 0.0177
    Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
    X_solar = 0.7491
    Z_solar = 1.0-X_solar - Y_solar
    
    X = 0
    Z = 3.*Z_solar
    Y = 1 - X - Z
    
    metal_frac = Z/Y / (Z_solar/Y_solar)

    for T in T_kev:
        f = open("spex.com",'w')
        f.write("spex << STOP\n")
        f.write("com cie\n")
        f.write("abundance solar\n")
        f.write("elim 0.00000001:100000\n")
        f.write("par t val %g \n" %T)
        f.write("par it val %g \n" %T)
        f.write("par ed val 1.\n")
        for i_el in range(3,31):
            f.write("par %02d val %g\n" %(i_el,metal_frac))
        f.write("par ref val 02\n")
        f.write("par 01 val 0.00001\n")
        f.write("log out spex_%d o\n" %n)
        f.write("ascdump terminal 1 1 plas\n")
        f.write("quit\n")
        f.write("STOP\n")
        f.close()
        os.system("chmod +x spex.com")
        os.system("./spex.com")
        save_line = 0
        with open("spex_%d.out"%n) as f:
            for num,line in enumerate(f,0):
                if "Electron/Hydrogen density" in line:
                    i_line = num
                    break
        f = open("spex_%d.out"%n)
        lines = f.readlines()
        f.close()
        tmp = map(float,re.findall(r'[+-]?[0-9.]+', lines[i_line]))
        if (np.array(tmp).shape[0]>1):
            ne_o_nh  = tmp[-2] * 10.**(tmp[-1])
        else:
            ne_o_nh =tmp[-1]
        nerat_arr.append(ne_o_nh)
        os.system("rm spex_%d.out"%n)

        n = n+1
    nerat_arr = np.array(nerat_arr)/ (1.-X_solar)

    fname = "ne_rat_fine.dat"
    f = open(fname,"w")

    for i in range(T_kev.shape[0]):
        array = [str(T_kev[i]),str(nerat_arr[i])]
        f.write(" ".join(array) + "\n")
    f.close()

def rd_ne_nh(fname):
  global T_kev,T_K,ne_o_nh
  data = np.loadtxt(fname)
  T_kev = data[:,0]
  T_K = T_kev * 1.16045e7
  ne_o_nh = data[:,1]


def cuadra_cool(T_K):
    Tbr1 = 3.0e4
    Tbr2 = 4.0e7
    case1 = -6.4e-23 * (Tbr2/1e7)**-0.7 * (T_K/Tbr2)**0.5
    case2 = -6.4e-23 * (T_K/1e7)**-0.7
    case3 = -6.4e-23 * (T_K/1e7)**-0.7 * (T_K/Tbr1)**2.0
    return (T_K>Tbr2)*case1 + (T_K>Tbr1) * (T_K<Tbr2) *case2 + (T_K<=Tbr1) * case3

def tavg(var,t_i,t_f):
  iti = t_to_it(t_i)
  itf = t_to_it(t_f)
  if (iti==itf): return var[itf,:]
  else: return var[iti:itf,:].mean(0)

def get_l_angles(t_i =1.0, t_f = 1.2,levels = 8):
  global th_l,phi_l,x_rat,y_rat,z_rat
  L_tot = np.sqrt( tavg(Lx_avg,t_i,t_f)**2. + tavg(Ly_avg,t_i,t_f)**2. + tavg(Lz_avg,t_i,t_f)**2.)
  r_in = 2.*2./2.**levels/128.

  x_rat = (tavg(Lx_avg,t_i,t_f)/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)
  y_rat = (tavg(Ly_avg,t_i,t_f)/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)
  z_rat = (tavg(Lz_avg,t_i,t_f)/L_tot)[r_to_ir(r_in*10):r_to_ir(0.8*arc_secs)].mean(-1)
  norm = np.sqrt(x_rat**2+y_rat**2+z_rat**2)
  x_rat = x_rat/norm
  y_rat = y_rat/norm
  z_rat = z_rat/norm
  th_l = np.arccos(z_rat)
  phi_l = np.arctan2(y_rat,x_rat)

def get_l_angles_grmhd():
  global th_l,phi_l
  L_tot = np.sqrt(Lx**2 + Ly**2 + Lz**2)
  i1 = 30
  i2 = 56
  it1 = 800
  it2 = -1

  nx_avg = (Lx/L_tot)[it1:it2,i1:i2].mean(-1).mean(-1)
  ny_avg = (Ly/L_tot)[it1:it2,i1:i2].mean(-1).mean(-1)
  nz_avg = (Lz/L_tot)[it1:it2,i1:i2].mean(-1).mean(-1)

  th_l = np.arccos(nz_avg)
  ph_l = np.arctan2(ny_avg,nx_avg)



def mdot_bondi(p,rho):
    qs = 1./4.
    cs = np.sqrt(5./3. * p/rho)
    return 4.*pi * qs * gm_**2. * rho / (cs**3.)

def get_bh_spin_vector(tilt_angle=0,th=0,ph=0):
  l_vector = [np.sin(th)*np.cos(ph),np.sin(th)*np.sin(ph),np.cos(th)]



def render_3d():
  import yt
  from yt.visualization.volume_rendering.api import Scene, VolumeSource 
  import numpy as np
  yt.enable_parallelism()
  sc  = Scene()
  vol = VolumeSource(ds, field="density")
  bounds = (1e-2, 10.**1.5)
  tf = yt.ColorTransferFunction(np.log10(bounds))
  tf.add_layers(8, colormap='bone')
  tf.grey_opacity = False
  vol.transfer_function = tf
  vol.tfh.tf = tf
  vol.tfh.bounds = bounds
  vol.tfh.plot('transfer_function.png', profile_field="cube_helix")
  cam = sc.add_camera(ds, lens_type='plane-parallel')
  cam.resolution = [512,512]
  # cam.position = ds.arr(np.array([282.*np.cos(theta)*np.sin(phi),282.*np.cos(theta)*np.cos(phi),282.*np.sin(theta)]), 'code_length')
  # cam.switch_orientation(normal_vector=normal_vector,
  #                        north_vector=north_vector)
  cam.set_width(ds.domain_width*0.25)

  cam.position = ds.arr(np.array([0,0,-0.5]), 'code_length')
  normal_vector = [0,0,-1]  #camera to focus
  north_vector = [0,1,0]  #up direction
  cam.switch_orientation(normal_vector=normal_vector,
                           north_vector=north_vector)
  sc.add_source(vol)
  sc.render()
  sc.save('tmp2.png',sigma_clip = 6.0)
  #sc.save('./RENDERING/rendering_temperature_2_'+str(i).zfill(3)+'.png', sigma_clip=6.0)

def yt_region(box_radius):
  global region,rho,press,vel1,vel2,vel3,x,y,z
  region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):256j,(-box_radius,'pc'):(box_radius,'pc'):256j,
      (-box_radius,'pc'):(box_radius,'pc'):256j]
  rho = region['rho']
  press = region['press']
  vel1 = region['vel1']
  vel2 = region['vel2']
  vel3 = region['vel3']
  x = region['x']
  y = region['y']
  z = region['z']


def Lambda_cool(TK,file_prefix,X=0,Z=3):
  global mue,mu_highT,muH_solar
  mp_over_kev = 9.994827
  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar

  muH_solar = 1./X_solar
  Z = Z * Z_solar
  mue = 2. /(1.+X)
  mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
  mp = 8.41175e-58
  f1 = file_prefix + "_H_only.dat"
  f2 = file_prefix + "_He_only.dat"
  f3 = file_prefix + "_Metals_only.dat"
  data = np.loadtxt(f1)
  T_tab = data[:,0] * 1.16045e7
  Lam_H = data[:,1]
  data = np.loadtxt(f2)
  Lam_He = data[:,1]
  data = np.loadtxt(f3)
  Lam_metals = data[:,1]
  # T_tab = 10.**data[:,0]
  T_min = np.amin(T_tab)
  T_max = np.amax(T_tab)
  # if isinstance(TK,)
  # TK[TK<T_min] = T_min
  # TK[TK>T_max] = T_max
  # Lam_tab = 10.**data[:,1]

  Lam_tab = (X/X_solar) * Lam_H + ( (1.-X-Z)/Y_solar) * Lam_He + (Z/Z_solar) * Lam_metals
  from scipy.interpolate import InterpolatedUnivariateSpline
  Lam = InterpolatedUnivariateSpline(T_tab,Lam_tab,k = 1,ext =3)  #ext = 3 returns boundary value if outside range of T
  return Lam(TK)

def t_cool_func(press,rho,file_prefix,X=0,Z=3):
  mp_over_kev = 9.994827
  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar

  muH_solar = 1./X_solar
  Z = Z * Z_solar
  mue = 2. /(1.+X)
  mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
  mp = 8.41175e-58

  T_kev = press/rho * mu_highT*mp_over_kev
  TK= T_kev*1.16e7
  Lam_cgs = Lambda_cool(TK,file_prefix,X,Z)
  gm1 = 5./3. -1.
  UnitLambda_times_mp_times_kev = 1.255436328493696e-21

  return (T_kev) * (mu_highT) / ( gm1 * rho *             Lam_cgs/UnitLambda_times_mp_times_kev )

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



def get_star_loc(t):
    for star in star_array:
      rotation_matrix = get_rotation_matrix(star.alpha,star.beta,star.gamma)
      inverse_rotation_matrix = transpose(rotation_matrix)
      X_orbit,V_orbit = get_orbit(star,t)
      x1_orbit,x2_orbit,x3_orbit = matrix_vec_mult(inverse_rotation_matrix,X_orbit)
      v1_orbit,v2_orbit,v3_orbit = matrix_vec_mult(inverse_rotation_matrix,V_orbit)
      star.x1 = np.float(x1_orbit)
      star.x2 = np.float(x2_orbit)
      star.x3 = np.float(x3_orbit)
      star.v1 = np.float(v1_orbit)
      star.v2 = np.float(v2_orbit)
      star.v3 = np.float(v3_orbit)
def set_star_size():
  for star in star_array:
    level_0 = 1.0
    level_1 = level_0/2.0
    level_2 = level_1/2.0
    level_3 = level_2/2.0
    level_4 = level_3/2.0
    dx = 2./128.
    star.radius = 2.* np.sqrt(3.0)*dx 

    if ( (fabs(star.x1)< level_1) and (fabs(star.x2) <level_1) and (fabs(star.x3)<level_1) ): star.radius= star.radius/2.0
    if ( (fabs(star.x1)< level_2) and (fabs(star.x2) <level_2) and (fabs(star.x3)<level_2) ): star.radius= star.radius/2.0
    if ( (fabs(star.x1)< level_3) and (fabs(star.x2) <level_3) and (fabs(star.x3)<level_3) ): star.radius= star.radius/2.0
    if ( (fabs(star.x1)< level_4) and (fabs(star.x2) <level_4) and (fabs(star.x3)<level_4) ): star.radius= star.radius/2.0
def mask_grid(x_arr,y_arr,z_arr):
  global mask_array
  mask_array = np.zeros(x_arr.shape) +1.0
  for star in star_array:
    print (star.Mdot)
    dr = np.sqrt( (np.array(x_arr)-star.x1)**2.0 + (np.array(y_arr)-star.x2)**2.0 + (np.array(z_arr)-star.x3)**2.0 )   
    mask_array = mask_array * (dr>star.radius)

def make_rendering():
  nx_disk = -0.12
  ny_disk = -0.79
  nz_disk = 0.6
  box_radius = 1.0 
  box_radius = 0.01
  box_radius = 1.0
  region = ds.r[(-box_radius,'pc'):(box_radius,'pc'):512j,
  (-box_radius,'pc'):(box_radius,'pc'):512j,
  (-box_radius,'pc'):(box_radius,'pc'):512j ]
  x,y,z = region['x'],region['y'],region['z']
  import numpy as np

  bbox = np.array([[-box_radius,box_radius],[-box_radius,box_radius],[-box_radius,box_radius]])
  rho = region['density']
  #rho = rho * mask_array
  r = np.sqrt(x**2. + y**2. + z**2.)
  rho_dot_r = rho*r
  press = region['press']
  set_constants()
  keV_to_Kelvin = 1.16045e7

  T  = press/rho *  mu_highT*mp_over_kev*keV_to_Kelvin
  data =  dict(density = (np.array(rho),"Msun/pc**3"),temperature = (np.array(T),"K"),rho_dot_r = (np.array(rho_dot_r),"Msun/pc**2"),x = (np.array(x),"pc"), y = (np.array(y),"pc"),z = (np.array(z),"pc"))
  ds = yt.load_uniform_grid(data,rho.shape,length_unit="pc",bbox=bbox)

  #phi = np.linspace(0,2*pi,100)
  #   for iphi in range(100):
  from yt.visualization.volume_rendering.api import Scene, VolumeSource
  sc  = Scene()
  vol = VolumeSource(ds, field="density")
  #vol.set_log(True)
  vol.set_log(True)
#  bound_min = ds.arr(1e5,"K").in_cgs()
#  bound_max = ds.arr(1e9,"K").in_cgs()

  bound_min = ds.arr(1e-2,"Msun/pc**3.").in_cgs()
  bound_max = ds.arr(10.**2.0,"Msun/pc**3.").in_cgs()
  tf_min = ds.arr(1e-2,"Msun/pc**3.").in_cgs()
  tf_max = ds.arr(10.**0.5,"Msun/pc**3.").in_cgs()

  # bound_min = ds.arr(0.04/1e-2,"Msun/pc**3.").in_cgs()
  # bound_max = ds.arr(0.24/1e-3,"Msun/pc**3.").in_cgs()
  # bound_min = ds.arr(0.04/2,"Msun/pc**2.") #.in_cgs()
  # bound_max = ds.arr(0.24*10,"Msun/pc**2.")#.in_cgs()
  # tf_min = bound_min #ds.arr(1e-2,"Msun/pc**3.").in_cgs()
  # tf_max = bound_max #ds.arr(10.**0.5,"Msun/pc**3.").in_cgs()

  bounds = (bound_min, bound_max)

  tf = yt.ColorTransferFunction(np.log10(bounds))
  def linramp(vals, minval, maxval):
    return (vals - vals.min())/(vals.max() - vals.min())
  #tf.map_to_colormap(np.log10(bounds[0]),np.log10(bounds[1]),colormap='ocean',scale_func=linramp)
  tf.add_layers(8, colormap='ocean',mi = np.log10(tf_min),ma = np.log10(tf_max),col_bounds=([np.log10(tf_min),np.log10(tf_max)])) #,w = 0.01,alpha = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])  #ds_highcontrast
  tf.add_step(np.log10(tf_max*2),np.log10(bound_max), [0.5,0.5,0.5,1.0])
  #tf.sample_colormap(5.,w=0.001,colormap='inferno')
#  tf.add_gaussian(np.log10(tf_max*2),width=0.01,height = [1.0,1.0,1.0,1.0])
#  tf.add_gaussian(np.log10(tf_max*4),width=0.01,height = [1.0,1.0,1.0,1.0])
#  tf.add_gaussian(np.log10(tf_max*8),width=0.01,height = [1.0,1.0,1.0,1.0])

  #tf.add_gaussian(np.log10(5e7),width = 0.01,height=[1.0, 0.0, 0, 0.9])
#  tf.add_step(np.log10(3e7),np.log10(1e9),[1.0,0.0,0.0,0.5])
#  tf.add_step(np.log10(1e6),np.log10(3e7),[0.5,0.0,0.5,0.2])
#  tf.add_step(np.log10(1e5),np.log10(1e6),[0.0,0.0,1.0,1.0])
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
  #cam.position = ds.arr(np.array([0.0,0,-0.01]), 'code_length')
  normal_vector = [0,0,-1]  #camera to focus  #CHANGE WITH PHI
  north_vector = [0,1,0]  #up direction 
  #north_vector = [nx_disk,ny_disk,nz_disk]  
  cam.switch_orientation(normal_vector=normal_vector,north_vector=north_vector)
  sc.add_source(vol)
  sc.render()
  # sc.save('tmp2.png',sigma_clip = 6.0)
  # sc = yt.create_scene(asc.ds,lens_type = 'perspective')
  # sc.camera.zoom(2.0)
  # sc[0].tfh.set_bounds([1e-4,1e2])
  fname = "test.png"
  sc.save(fname,sigma_clip = 6.0)
  for i in cam.iter_rotate(np.pi*2.0,50,rot_center=[0,0,0]):
    sc.render()
    sc.save("test_rotate_%04d.png" %i)

def fold_theta(arr):
  return arr[:,::-1,:]/2.0 + arr[:,:,:]/2.0

def fit_theta():
  rd_hst("star_wind.hst")
  rdnpz("dump_spher_avg_100_110.npz")
  plotting_radius = 0.1 * arc_secs
  theta = np.arccos(z_tavg/r_tavg)
  def get_ir(r_in):
    dlog10r = np.diff(np.log10(r_tavg[:,0,0]))[0]
    r_min = r_tavg[0,0,0]
    r_out = r_tavg[-1,0,0]
    #r = r_min * 10**(ir*dlog10r)
    return np.int(np.round(np.log10(r_in/r_min)/dlog10r))

  ir = get_ir(plotting_radius)

  os.system("export LD_PRELOAD=/global/software/sl-7.x86_64/modules/langs/intel/2016.4.072/mkl/lib/intel64_lin/libmkl_core.so:/global/software/sl-7.x86_64/modules/langs/intel/2016.4.072/mkl/lib/intel64_lin/libmkl_sequential.so")
  from scipy.optimize import curve_fit
  def rho_func(h,rho_0,rho_peak):
    return rho_0 + (rho_peak-rho_0) * np.sin(h)**3.0
  def press_func(h,rho_0,rho_peak):
    return rho_0 + (rho_peak-rho_0) * np.sin(h)**2.5
  def vr_func(h,vr_0,vr_peak):
    return vr_0 + (vr_peak-vr_0) * np.sin(h)**2.0
  def T_func(h,T0):
    return T0
  def vphi_func(h,vphi_peak):
    return vphi_peak * sin(h)

  popt,pcov = curve_fit(rho_func,theta[0,:,0],fold_theta(rho_tavg)[ir,:,:].mean(-1))
  rho_0,rho_peak = popt[0],popt[1]
  popt,pcov = curve_fit(vr_func,theta[0,:,0],fold_theta(vr_avg)[ir,:,:].mean(-1))
  vr_0, vr_peak = popt[0],popt[1]
  popt,pcov = curve_fit(vphi_func,theta[0,:,0],fold_theta(vphi_avg)[ir,:,:].mean(-1))
  vphi_peak = popt[0]
  popt,pcov = curve_fit(press_func,theta[0,:,0],fold_theta(press_tavg)[ir,:,:].mean(-1))
  p_0,p_peak = popt[0],popt[1]

  vr_pole = vr_0 * (r_tavg/plotting_radius)**-1.0
  vr_an = vr_pole + (vr_peak-vr_pole) * sin(theta)**2.0
  rho_an = rho_func(theta,rho_0,rho_peak) * (r_tavg/plotting_radius)**(-1.0)
  press_an = press_func(theta,p_0,p_peak) * (r_tavg/plotting_radius)**(-2.0)
  vphi_an = vphi_func(theta,vphi_peak) * (r_tavg/plotting_radius)**(-0.5)


def mk_spherical_frame(var,min = 4.5,max = 6,cmap="ocean",length_scale=40.0,magnetic = False,gr=False):
  plt.clf()
  rg = 2.058e-7
  if (gr==True): rg = 1;
  tm = 6.7161e-10
#  plt.figure(1)
#  plt.clf()
  plt.xlim(-length_scale,length_scale)
  plt.ylim(-length_scale,length_scale)
  plt.pcolormesh(-(r*sin(th))[:,:,0]/rg,(r*cos(th))[:,:,0]/rg,log10((var[:,:,nz//2]) ),vmin=min,vmax=max,cmap = cmap)
  plt.pcolormesh((r*sin(th))[:,:,0]/rg,(r*cos(th))[:,:,0]/rg,log10((var[:,:,0]) ),vmin=min,vmax=max,cmap = cmap)
  plt.colorbar()
  if magnetic:
    plt.contour((r*sin(th))[:,:,0]/rg,(r*cos(th))[:,:,0]/rg,(psicalc(B1=Bcc1)),0,colors='black')
    plt.contour((-r*sin(th))[:,:,0]/rg,(r*cos(th))[:,:,0]/rg,(psicalc(B1=Bcc1)),30,colors='black')

  


def psicalc(B1 = None,gr=False,xy=False):
    """
    Computes the field vector potential
    """
    if (B1 is None): B1 = Bcc1
    _dx2 = np.diff(x2f)
    daphi = -(r*np.sin(th)*B1).mean(-1)*_dx2[None,:]
    if (gr==True): daphi = -(gdet*B1).mean(-1)*_dx2[None,:]
    if (gr== False): aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]/np.sin(th)[:,:,0]
    else: aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta


    return(aphi)

def psicalc_slice(B1 = None,gr=False,xy=False,iphi = 0):
    """
    Computes the field vector potential
    """
    if (B1 is None): B1 = Bcc1
    if (xy==False):
      _dx2 = np.diff(x2f)
      daphi = -(r*np.sin(th)*B1)[:,:,iphi]*_dx2[None,:]
      if (gr==True): daphi = -(gdet*B1)[:,:,iphi]*_dx2[None,:]
      if (gr== False): aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]/np.sin(th)[:,:,0]
      else: aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
      aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    else: #Calculate Ay assuming By = 0 (i.e. projecting the magnetic field onto the plane)
      daphi = -B1[:,ny//2,:]
      aphi = daphi[:,::-1].cumsum(axis=1)[:,::-1]
      aphi-=0.5*daphi


    return(aphi)


#def compute_3D_vector_potential():
  

def psicalc_npz(B1 = None):
    """
    Computes the field vector potential
    """
    if (B1 is None): B1 = Br
    _dx2 = np.diff(th[0,:,0])[0]
    daphi = -(r*np.sin(th)*B1).mean(-1)*_dx2
    aphi  = daphi[:,::-1].cumsum(axis=1)[:,::-1]/(np.sin(th)[:,:,0]+1e-15)
    aphi2 = -daphi[:,:].cumsum(axis=1)[:,:]/(np.sin(th)[:,:,0]+1e-15)

    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    aphi2-=0.5*daphi

    aphi_avg = aphi * np.sin(th[:,:,0]/2.0)**2.0 + aphi2 * np.cos(th[:,:,0]/2.0)**2.0

    #aphi_avg = 0 
    return(aphi_avg)

def psicalc_npz_avg(B1 = None):
    """
    Computes the field vector potential
    """
    if (B1 is None): B1 = Br_avg
    _dx2 = np.diff(th_tavg[0,:,0])[0]
    daphi = -(r_tavg*np.sin(th_tavg)*B1).mean(-1)*_dx2
    aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]/(np.sin(th_tavg)[:,:,0]+1e-15)
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta


    return(aphi)

def psicalc_npz_gr(B1 = None,a = 0, gdet = None):
    """
    Computes the field vector potential

    """

  ##Bcc1 = (bu_ks[1] * uu_ks[0] - bu_ks[0] * uu_ks[1])

    if (B1 is None): B1 = (bu_ks[1] * uu_ks[0] - bu_ks[0] * uu_ks[1])
    _dx2 = np.diff(th[0,:,0])[0]
    daphi = -(r*np.sin(th)*B1).mean(-1)*_dx2
    if (gdet is None and "th" in globals() ): gdet =  (r**2 + a**2*np.cos(th)**2)*np.sin(th)
    elif (gdet is None): gdet =  (r**2 + a**2*np.cos(th)**2)*np.sin(theta)
    daphi = -(gdet*B1).mean(-1)*_dx2
    aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
    aphi-=0.5*daphi #correction for half-cell shift between face and center in theta


    return(aphi)

# def psicalc_npz_avg(B2 = None):
#     """
#     Computes the field vector potential
#     """
#     if (B2 is None): B2 = Bth_avg
#     _dx1 = np.diff(th_tavg[0,:,0])[0]
#     daphi = -(r_tavg*np.sin(th_tavg)*B1).mean(-1)*_dx2
#     aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]/(np.sin(th_tavg)[:,:,0]+1e-15)
#     aphi-=0.5*daphi #correction for half-cell shift between face and center in theta


#     return(aphi)

def psicalc_xy(B1=None,B2 = None,x1 = None,x2 = None,slice=False):
    """
    Computes the field vector potential
    """
    if (B2 is None): B2 = Bcc2
    if (B1 is None): B1 = Bcc1
    if (x1 is None): x1 = x1v
    if (x2 is None): x2 = x2v

    _dx1 = np.gradient(x1)
    _dx2 = np.gradient(x2)
    if (slice==False):
      daz1 = -(B2).mean(-1)*(_dx1[:,None])
      daz2 = (B1).mean(-1)*_dx2[None,:]
    else:
      daz1 = -(B2)[:,:,B2.shape[-1]//2]*_dx1[:,None]
      daz2 = (B1)[:,:,B1.shape[-1]//2]*_dx2[None,:]   
    az1=daz1[:,:].cumsum(axis=0)[:,:]
    az2=daz2[:,:].cumsum(axis=1)[:,:]
    az1-=0.5*daz1 #correction for half-cell shift between face and center in theta
    az2-=0.5*daz2

    az = az1 - np.gradient(az1,axis=1).cumsum(axis=1) + az2


    return(az)
def bl_metric(r,th,a=0):
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  gbl = np.zeros((4,4))
  Sigma = r**2 + a**2.*np.cos(th)**2;
  Delta = r**2 -2.0*r + a**2;
  A = (r**2 + a**2)**2 - a**2*Delta*np.sin(th)**2.;


  nx = r.shape[0]
  ny = r.shape[1]
  nz = r.shape[2]
  gbl  = np.zeros((4,4,nx,ny,nz))
  gbl[0][0] = - (1 - 2.0*r/Sigma);
  gbl[1][0] = 0;
  gbl[0][1] = 0;
  gbl[1][1] = Sigma/Delta;
  gbl[2][0] = 0;
  gbl[0][2] = 0;
  gbl[2][1] = 0;
  gbl[1][2] = 0;
  gbl[2][2] = Sigma;
  gbl[3][0] = -2*a*r*sin2/Sigma;
  gbl[0][3] = gbl[3][0];
  gbl[1][3] = 0;
  gbl[3][1] = 0;
  gbl[3][2] = 0;
  gbl[3][3] = A*sin2/Sigma;

  return gbl

def bl_vec_to_ks(A,a=0):
  Delta = r**2 -2.0*r + a**2;
  tmp = A*0
  tmp[0] = A[0] + A[1] * 2.*r/Delta
  tmp[1] = A[1] 
  tmp[2] = A[2]
  tmp[3] = A[3] + A[1] * a/Delta
  return tmp 
def ks_vec_to_bl(A,a=0):
  Delta = r**2 -2.0*r + a**2;
  tmp = A*0
  tmp[0] = A[0] - A[1] * 2.*r/Delta
  tmp[1] = A[1] 
  tmp[2] = A[2]
  tmp[3] = A[3] - A[1] * a/Delta
  return tmp


#uu_ks^\mu = uu_cks^\nu dx_ks^\mu_/dx_cks^\nu
def cks_vec_to_ks(A,x,y,z,a=0):
    R = np.sqrt(x**2+y**2+z**2)
    r = np.sqrt( R**2 -a**2 + np.sqrt( (R**2-a**2)**2 + 4.0*a**2*z**2 ) )/np.sqrt(2.0)
    A_ks = A*0
    #uu_ks^\mu = uu_cks^\nu dx_ks^\mu_/dx_cks^\nu
    SMALL = 1e-15
    sqrt_term = 2.0*r**2-R**2+a**2
    A_ks[0] = A[0] 
    A_ks[1] = A[1] * (x*r)/sqrt_term + \
               A[2] * (y*r)/sqrt_term + \
               A[3] * z/r * (r**2+a**2)/sqrt_term
    A_ks[2] = A[1] * (x*z)/(r * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL) + \
               A[2] * (y*z)/(r * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL) + \
               A[3] * ( (z*z)*(r**2+a**2)/(r**3 * sqrt_term * np.sqrt(1.0-z**2/r**2) + SMALL) - 1.0/(r*np.sqrt(1.0-z**2/r**2) + SMALL) )
    A_ks[3] = A[1] * (-y/(x**2+y**2+SMALL) + a*r*x/((r**2+a**2)*sqrt_term)) + \
               A[2] * (x/(x**2+y**2+SMALL) + a*r*y/((r**2+a**2)*sqrt_term)) + \
               A[3] * (a* z/r/sqrt_term) 

    return A_ks


## A_ks^mu_\nu = A_cks^_ dx_ks^\mu/dx_cks^\lam dx_cks^\beta/dx_ks^\nu
# def cks_udtensor_to_ks(A,x,y,z,a=0):

def cks_coord_to_ks(x,y,z,a=0):
    global r, th,ph
    def SQR(b):
      return b**2.0
    R = np.sqrt( SQR(x) + SQR(y) + SQR(z) );
    r = np.sqrt( SQR(R) - SQR(a) + np.sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/np.sqrt(2.0);

    th = np.arccos(z/r)

    ph = np.arctan2((r*y-a*x), (a*y+r*x) )

def bl_vec_to_cks(x,y,z,A,a=0):
    A_cks = np.array(A)*0
    def SQR(q):
      return q**2.0
    m = 1

    R = np.sqrt( SQR(x) + SQR(y) + SQR(z) );
    r = np.sqrt( SQR(R) - SQR(a) + np.sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/np.sqrt(2.0);
    delta = SQR(r) - 2.0*m*r + SQR(a);
    A_cks[0] = A[0] + 2.0*r/delta * A[1];
    A_cks[1] = A[1] * ( (r*x+a*y)/(SQR(r) + SQR(a)) - y*a/delta) + A[2] * x*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - A[3] * y
    A_cks[2] = A[1] * ( (r*y-a*x)/(SQR(r) + SQR(a)) + x*a/delta) + A[2] * y*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + A[3] * x
    A_cks[3] = A[1] * z/r - A[2] * r * np.sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)))

    return A_cks

def ks_metric(r,th,a):
  global g
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  g  = np.zeros((4,4,nx,ny,nz))
  g[0][0] = -(1.0 - 2.0*m*r/sigma);
  g[0][1] = 2.0*m*r/sigma;
  g[1][0] = g[0][1]
  g[0][3] = -2.0*m*a*r/sigma * sin2
  g[3][0] = g[0][3]
  g[1][1] = 1.0 + 2.0*m*r/sigma
  g[1][3] =  -(1.0 + 2.0*m*r/sigma) * a * sin2
  g[3][1] = g[1][3]
  g[2][2] = sigma
  g[3][3] = (r2 + a2 + 2.0*m*a2*r/sigma * sin2) * sin2

def ks_Gamma_ud(radius,theta,a,m=1):
  global gammaud
  gammaud = np.zeros((4,4,4,nx,ny))
  r= radius[:,:,0]
  th = theta[:,:,0]
  gammaud[0][0][0] = -2*(a**2*m**2*r*cos(th)**2 - m**2*r**3)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[0][0][1] = -(a**4*m*cos(th)**4 + 2*a**2*m**2*r*cos(th)**2 - 2*m**2*r**3 - m*r**4)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[0][0][2] = -2*a**2*m*r*cos(th)*sin(th)/(a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4) 
  gammaud[0][0][3] = 2*(a**3*m**2*r*cos(th)**2 - a*m**2*r**3)*sin(th)**2/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[0][1][1] = -2*(a**4*m*cos(th)**4 + a**2*m**2*r*cos(th)**2 - m**2*r**3 - m*r**4)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[0][1][2] = -2*a**2*m*r*cos(th)*sin(th)/(a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4) 
  gammaud[0][1][3] = (a**5*m*cos(th)**4 + 2*a**3*m**2*r*cos(th)**2 - 2*a*m**2*r**3 - a*m*r**4)*sin(th)**2/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[0][2][2] = -2*m*r**2/(a**2*cos(th)**2 + r**2) 
  gammaud[0][2][3] = 2*a**3*m*r*cos(th)*sin(th)**3/(a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4) 
  gammaud[0][3][3] = -2*((a**4*m**2*r*cos(th)**2 - a**2*m**2*r**3)*sin(th)**4 + (a**4*m*r**2*cos(th)**4 + 2*a**2*m*r**4*cos(th)**2 + m*r**6)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][0][0] = (a**2*m*r**2 - 2*m**2*r**3 + m*r**4 - (a**4*m - 2*a**2*m**2*r + a**2*m*r**2)*cos(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][0][1] = (2*a**2*m**2*r*cos(th)**2 - 2*m**2*r**3 - (a**4*m*cos(th)**2 - a**2*m*r**2)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][0][3] = -(a**3*m*r**2 - 2*a*m**2*r**3 + a*m*r**4 - (a**5*m - 2*a**3*m**2*r + a**3*m*r**2)*cos(th)**2)*sin(th)**2/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][1][1] = (2*a**4*m*cos(th)**4 + a**2*m*r**2 - 2*m**2*r**3 - m*r**4 - (a**4*m - 2*a**2*m**2*r + a**2*m*r**2)*cos(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][1][2] = -a**2*cos(th)*sin(th)/(a**2*cos(th)**2 + r**2) 
  gammaud[1][1][3] = ((a**5*m*cos(th)**2 - a**3*m*r**2)*sin(th)**4 + (a**5*r*cos(th)**4 + 2*a*m**2*r**3 + a*r**5 - 2*(a**3*m**2*r - a**3*r**3)*cos(th)**2)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[1][2][2] = -(a**2*r - 2*m*r**2 + r**3)/(a**2*cos(th)**2 + r**2) 
  gammaud[1][3][3] = ((a**4*m*r**2 - 2*a**2*m**2*r**3 + a**2*m*r**4 - (a**6*m - 2*a**4*m**2*r + a**4*m*r**2)*cos(th)**2)*sin(th)**4 - (a**2*r**5 - 2*m*r**6 + r**7 + (a**6*r - 2*a**4*m*r**2 + a**4*r**3)*cos(th)**4 + 2*(a**4*r**3 - 2*a**2*m*r**4 + a**2*r**5)*cos(th)**2)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][0][0] = -2*a**2*m*r*cos(th)*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][0][1] = -2*a**2*m*r*cos(th)*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][0][3] = 2*(a**3*m*r + a*m*r**3)*cos(th)*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][1][1] = -2*a**2*m*r*cos(th)*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][1][2] = r/(a**2*cos(th)**2 + r**2) 
  gammaud[2][1][3] = (a**5*cos(th)**5 + 2*a**3*r**2*cos(th)**3 + (2*a**3*m*r + 2*a*m*r**3 + a*r**4)*cos(th))*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[2][2][2] = -a**2*cos(th)*sin(th)/(a**2*cos(th)**2 + r**2) 
  gammaud[2][3][3] = -((a**6 - 2*a**4*m*r + a**4*r**2)*cos(th)**5 + 2*(a**4*r**2 - 2*a**2*m*r**3 + a**2*r**4)*cos(th)**3 + (2*a**4*m*r + 4*a**2*m*r**3 + a**2*r**4 + r**6)*cos(th))*sin(th)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][0][0] = -(a**3*m*cos(th)**2 - a*m*r**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][0][1] = -(a**3*m*cos(th)**2 - a*m*r**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][0][2] = -2*a*m*r*cos(th)/((a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4)*sin(th)) 
  gammaud[3][0][3] = (a**4*m*cos(th)**2 - a**2*m*r**2)*sin(th)**2/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][1][1] = -(a**3*m*cos(th)**2 - a*m*r**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][1][2] = -(a**3*cos(th)**3 + (2*a*m*r + a*r**2)*cos(th))/((a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4)*sin(th)) 
  gammaud[3][1][3] = (a**4*r*cos(th)**4 + 2*a**2*r**3*cos(th)**2 + r**5 + (a**4*m*cos(th)**2 - a**2*m*r**2)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6) 
  gammaud[3][2][2] = -a*r/(a**2*cos(th)**2 + r**2) 
  gammaud[3][2][3] = (a**4*cos(th)**5 - 2*(a**2*m*r - a**2*r**2)*cos(th)**3 + (2*a**2*m*r + r**4)*cos(th))/((a**4*cos(th)**4 + 2*a**2*r**2*cos(th)**2 + r**4)*sin(th)) 
  gammaud[3][3][3] = -((a**5*m*cos(th)**2 - a**3*m*r**2)*sin(th)**4 + (a**5*r*cos(th)**4 + 2*a**3*r**3*cos(th)**2 + a*r**5)*sin(th)**2)/(a**6*cos(th)**6 + 3*a**4*r**2*cos(th)**4 + 3*a**2*r**4*cos(th)**2 + r**6)

  for i in arange(4):
    for j in arange(1,4):
      for k in arange(0,j):
        gammaud[i][j][k] = gammaud[i][k][j]

def ks_inverse_metric(r,th,a):
  global gi
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  delta = r2 - 2.0*m*r + a2
  gi  = np.zeros((4,4,nx,ny,nz))
  gi[0][0] = -(1.0 + 2.0*m*r/sigma);
  gi[0][1] = 2.0*m*r/sigma;
  gi[1][0] = g[0][1]
  gi[0][3] = 0
  gi[3][0] = g[0][3]
  gi[1][1] = delta/sigma
  gi[1][3] = a/sigma
  gi[3][1] = g[1][3]
  gi[2][2] = 1.0/sigma
  gi[3][3] = 1.0 / (sigma * sin2)

def cks_metric(x,y,z,a):
  global g
  m = 1
  R = np.sqrt(x**2+y**2+z**2)
  r = np.sqrt(R**2-a**2 + np.sqrt( (R**2-a**2)**2 + 4*a**2*z**2 ))/np.sqrt(2.0)

  f = 2.0*r**3/(r**4 + a**2*z**2)
  l0 = 1.0
  l1 = (r*x+a*y)/(r**2+a**2)
  l2 = (r*y-a*x)/(r**2+a**2)
  l3 = z/r

  nx = x.shape[0]
  ny = x.shape[1]
  nz = x.shape[2]
  g  = np.zeros((4,4,nx,ny,nz))
  g[0][0] = -1.0 + f * l0*l0;
  g[0][1] = f*l0*l1;
  g[1][0] = g[0][1]
  g[0][2] = f*l0*l2
  g[2][0] = g[0][2]
  g[0][3] = f*l0*l3
  g[3][0] = g[0][3]
  g[1][1] = 1.0 + f*l1*l1
  g[1][3] =  f*l1*l3
  g[3][1] = g[1][3]
  g[2][2] = 1.0 + f*l2*l2
  g[2][3] = f*l2*l3 
  g[3][2] = g[2][3]
  g[1][2] = f*l1*l2
  g[2][1] = g[1][2]
  g[3][3] = 1.0 + f*l3*l3

  # def SQR(c):
  #   return c**2
  # def pow(c,d):
  #   return c**d

  # sqrt_term =  2.0*SQR(r)-SQR(R) + SQR(a)
  # rsq_p_asq = SQR(r) + SQR(a)

  # df_dx1 = SQR(f)*x/(2.0*pow(r,3)) * ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) )/ sqrt_term ;
  # df_dx2 = SQR(f)*y/(2.0*pow(r,3)) * ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) )/ sqrt_term ;
  # df_dx3 = SQR(f)*z/(2.0*pow(r,5)) * ( ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) * ( rsq_p_asq ) )/ sqrt_term - 2.0*SQR(a*r)) ;
  # dl1_dx1 = x*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
  # dl1_dx2 = y*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) )+ a/( rsq_p_asq );
  # dl1_dx3 = z/r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( (rsq_p_asq) * ( sqrt_term ) ) ;
  # dl2_dx1 = x*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) - a/( rsq_p_asq );
  # dl2_dx2 = y*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
  # dl2_dx3 = z/r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( (rsq_p_asq) * ( sqrt_term ) );
  # dl3_dx1 = - x*z/(r) /( sqrt_term );
  # dl3_dx2 = - y*z/(r) /( sqrt_term );
  # dl3_dx3 = - SQR(z)/(SQR(r)*r) * ( rsq_p_asq )/( sqrt_term ) + 1.0/r;

def cks_inverse_metric(x,y,z,a):
  global gi
  m = 1
  R = np.sqrt(x**2+y**2+z**2)
  r = np.sqrt(R**2-a**2 + np.sqrt( (R**2-a**2)**2 + 4*a**2*z**2 ))/np.sqrt(2.0)

  f = 2.0*r**3/(r**4 + a**2*z**2)
  l0 = - 1.0
  l1 = (r*x+a*y)/(r**2+a**2)
  l2 = (r*y-a*x)/(r**2+a**2)
  l3 = z/r

  nx = x.shape[0]
  ny = x.shape[1]
  nz = x.shape[2]
  gi  = np.zeros((4,4,nx,ny,nz))
  gi[0][0] = -1.0 - f * l0*l0;
  gi[0][1] = -f*l0*l1;
  gi[1][0] = gi[0][1]
  gi[0][2] = -f*l0*l2
  gi[2][0] = gi[0][2]
  gi[0][3] = -f*l0*l3
  gi[3][0] = gi[0][3]
  gi[1][1] = 1.0 - f*l1*l1
  gi[1][3] = -f*l1*l3
  gi[3][1] = gi[1][3]
  gi[2][2] = 1.0 - f*l2*l2
  gi[2][3] = -f*l2*l3 
  gi[3][2] = gi[2][3]
  gi[1][2] = -f*l1*l2
  gi[2][1] = gi[1][2]
  gi[3][3] = 1.0 - f*l3*l3

def Determinant_4b4(A):
  a11 = A[0][0];
  a12 = A[0][1];
  a13 = A[0][2];
  a14 = A[0][3];
  a21 = A[1][0];
  a22 = A[1][1];
  a23 = A[1][2];
  a24 = A[1][3];
  a31 = A[2][0];
  a32 = A[2][1];
  a33 = A[2][2];
  a34 = A[2][3];
  a41 = A[3][0];
  a42 = A[3][1];
  a43 = A[3][2];
  a44 = A[3][3];
  det = (a11 * Determinant_3b3(a22, a23, a24, a32, a33, a34, a42, a43, a44) 
           - a12 * Determinant_3b3(a21, a23, a24, a31, a33, a34, a41, a43, a44) 
           + a13 * Determinant_3b3(a21, a22, a24, a31, a32, a34, a41, a42, a44) 
           - a14 * Determinant_3b3(a21, a22, a23, a31, a32, a33, a41, a42, a43) )
  return det

def Determinant_3b3(a11, a12, a13, a21, a22, a23,a31, a32, a33):
  det = (a11 * Determinant_2b2(a22, a23, a32, a33) - 
              a12 * Determinant_2b2(a21, a23, a31, a33) + 
              a13 * Determinant_2b2(a21, a22, a31, a32) )
  return det

def Determinant_2b2(a11, a12, a21, a22):
  return a11 * a22 - a12 * a21

def get_bl_coords(x,y,z,a=0):
  global r,th

  R = np.sqrt(x**2+y**2+z**2)
  r = np.sqrt(R**2-a**2 + np.sqrt( (R**2-a**2)**2 + 4*a**2*z**2 ))/np.sqrt(2.0)

  th = np.arccos(z/r)

def cks_bl_jac(x,y,z,a=0):
  global jac_cks_bl
  jac_cks_bl = np.zeros((4,4,nx,ny,nz))

  ##dx^\mu_cks/dx^\nu_Bl
  m=1
  def SQR(c):
    return c**2
  def pow(c,d):
    return c**d
  R = np.sqrt(x**2+y**2+z**2)
  r = np.sqrt(R**2-a**2 + np.sqrt( (R**2-a**2)**2 + 4*a**2*z**2 ))/np.sqrt(2.0)
  delta = SQR(r) - 2.0*m*r + SQR(a);


    # *pa0 = a0_bl + 2.0*r/delta * a1_bl;
    # *pa1 = a1_bl * ( (r*x-a*y)/(SQR(r) + SQR(a)) - y*a/delta) + 
    #        a2_bl * x*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - 
    #        a3_bl * y; 
    # *pa2 = a1_bl * ( (r*y+a*x)/(SQR(r) + SQR(a)) + x*a/delta) + 
    #        a2_bl * y*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + 
    #        a3_bl * x;
    # *pa3 = a1_bl * z/r - 
    #        a2_bl * r * std::sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  jac_cks_bl[0][0] = 1.0;
  jac_cks_bl[0][1] = 2.0*r/delta
  jac_cks_bl[0][2] = 0.0
  jac_cks_bl[0][3] = 0.0
  jac_cks_bl[1][0] = 0.0
  jac_cks_bl[2][0] = 0.0
  jac_cks_bl[3][0] = 0.0
  jac_cks_bl[1][1] = ( (r*x+a*y)/(SQR(r) + SQR(a)) - y*a/delta)
  jac_cks_bl[1][2] = x*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y)))
  jac_cks_bl[1][3] = -y
  jac_cks_bl[2][1] = ( (r*y-a*x)/(SQR(r) + SQR(a)) + x*a/delta)

  jac_cks_bl[2][2] = y*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y)));
  jac_cks_bl[2][3] = +x
  jac_cks_bl[3][1] = z/r
  jac_cks_bl[3][2] = -r * np.sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  jac_cks_bl[3][3] = 0.0

def cks_ks_jac(x,y,z,a=0):
  global jac_cks_ks
  jac_cks_ks = np.zeros((4,4,nx,ny,nz))

  ##dx^\mu_cks/dx^\nu_Bl
  m=1
  def SQR(c):
    return c**2
  def pow(c,d):
    return c**d
  R = np.sqrt(x**2+y**2+z**2)
  r = np.sqrt(R**2-a**2 + np.sqrt( (R**2-a**2)**2 + 4*a**2*z**2 ))/np.sqrt(2.0)
  delta = SQR(r) - 2.0*m*r + SQR(a);


    # *pa0 = a0_bl + 2.0*r/delta * a1_bl;
    # *pa1 = a1_bl * ( (r*x-a*y)/(SQR(r) + SQR(a)) - y*a/delta) + 
    #        a2_bl * x*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - 
    #        a3_bl * y; 
    # *pa2 = a1_bl * ( (r*y+a*x)/(SQR(r) + SQR(a)) + x*a/delta) + 
    #        a2_bl * y*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + 
    #        a3_bl * x;
    # *pa3 = a1_bl * z/r - 
    #        a2_bl * r * std::sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  jac_cks_ks[0][0] = 1.0;
  jac_cks_ks[0][1] = 0.0
  jac_cks_ks[0][2] = 0.0
  jac_cks_ks[0][3] = 0.0
  jac_cks_ks[1][0] = 0.0
  jac_cks_ks[2][0] = 0.0
  jac_cks_ks[3][0] = 0.0
  jac_cks_ks[1][1] = ( (r*x+a*y)/(SQR(r) + SQR(a)) )
  jac_cks_ks[1][2] = x*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y)))
  jac_cks_ks[1][3] = -y
  jac_cks_ks[2][1] = ( (r*y-a*x)/(SQR(r) + SQR(a)))

  jac_cks_ks[2][2] = y*z/r * np.sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y)));
  jac_cks_ks[2][3] = +x
  jac_cks_ks[3][1] = z/r
  jac_cks_ks[3][2] = -r * np.sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  jac_cks_ks[3][3] = 0.0


def Tud_calc(uu,ud,bu,bd,is_magnetic = False,gam=5.0/3.0):
  global Tud,TudMA,TudEM
  w = rho+press * (gam)/(gam-1.0)
  Tud = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
  TudMA = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
  TudEM = np.zeros((4,4,nx,ny,nz),dtype=np.float32,order='F')
  for kapa in np.arange(4):
    for nu in np.arange(4):
      if(kapa==nu): delta = 1
      else: delta = 0
      TudMA[kapa,nu] = w*uu[kapa]*ud[nu]+press*delta
      if (is_magnetic==True): TudEM[kapa,nu] = bsq*uu[kapa]*ud[nu] + 0.5*bsq*delta - bu[kapa]*bd[nu]
      if (is_magnetic==True): Tud[kapa,nu] = TudEM[kapa,nu] + TudMA[kapa,nu]
      else: Tud[kapa,nu] = TudMA[kapa,nu]

def Tdd_calc(Tud,g):
  global Tdd
  Tdd = 0*Tud
  for i in range(4):
    for j in range(4):
      for k in range(4):
        Tdd[i][j] += g[k,i]*Tud[k,j]

def Tdd_cks_to_ks(Tdd,x,y,z,a=0):
  global Tdd_ks
  cks_ks_jac(x,y,z,a)
  Tdd_ks = 0*Tdd
  for i in range(4):
    for j in range(4):
      for k in range(4):
        for m in range(4):
          Tdd_ks[i][j] += Tdd[k][m] * jac_cks_ks[k][i] * jac_cks_ks[m][j]

def raise_Tdd_ks(Tdd,gi):
  global Tud_ks
  Tud_ks = 0 *Tdd
  for i in range(4):
    for j in range(4):
      for k in range(4):
        Tud_ks[i,j] += gi[i,k]*Tdd[k,j]

def convert_to_gr_units():
  rho_max = np.amax(rho)
  SMALL=1e-20
  cl = 306.4
  press = press/cl**2./rho_max
  rho = rho/rho_max
  vel1 = vel1/cl
  vel2 = vel2/cl/(r)
  vel3 = vel3/cl/(r*sin(th)+SMALL)


def Lower(uu,g):
  ud = 0
  for i in range(4):
    ud += g[i,:]*uu[i]
  return ud
# def gr_dot(uu1,uu2,g):
#   sum = 0
#   for i in range(4):
#     for j in range(4):
#       sum += g[i][j] * uu1[i] * uu2[j]
#   return sum

def angle_average_npz(arr,weight=None,gr=False,a=0,gdet = None):
  dx3 = 1
  if "th" in globals(): dx2 = np.diff(th[0,:,0])[0]
  else: dx2 = np.diff(theta[0,:,0])[0]
  if (gr==False): 
    if ("th" in globals()): dOmega = (np.sin(th)*dx2)*dx3
    else: dOmega = (np.sin(theta)*dx2)*dx3
  else: 
    if (gdet is None and "th" in globals() ): gdet =  (r**2 + a**2*np.cos(th)**2)*np.sin(th)
    elif (gdet is None): gdet =  (r**2 + a**2*np.cos(th)**2)*np.sin(theta)
    dOmega = gdet *dx2*dx3
  if weight is None: weight = 1.0
  return (arr * weight * dOmega).mean(-1).mean(-1)/(dOmega*weight).mean(-1).mean(-1)

def angle_average(arr,weight=None,gr=False):
  if ("x3f" in globals()): dx3 = np.diff(x3f)
  else: dx3 = np.diff(ph[0,0,:])
  if ("x2f" in globals()): dx2 = np.diff(x2f)
  else: dx2 = np.diff(th[0,:,0])
  dOmega = (np.sin(th)*dx2[None,:,None])*dx3[None,None,:]
  if weight is None: weight = 1.0
  if gr==True: dOmega = (gdet * dx2[None,:,None]) * dx3[None,None,:]
  return (arr * weight * dOmega).mean(-1).mean(-1)/(dOmega*weight).mean(-1).mean(-1)

def get_stress(mhd=True):
  global alpha_m, alpha_h,F_J,F_maxwell,F_reynolds,F_advection
  drhovr = rho*vel1 - angle_average(rho*vel1)[:,None,None]*(rho/rho)
  dvphi = vel3 - angle_average(vel3)[:,None,None]*(rho/rho)
  if (mhd==True): F_J = angle_average((rho*vel1*vel3-Bcc1*Bcc3)*sin(th)*r**3,gr=False)
  else: F_J = angle_average((rho*vel1*vel3)*sin(th)*r**3,gr=False)
  if (mhd==True): F_maxwell = angle_average(r**3*Bcc1*Bcc3*sin(th))*-1
  F_reynolds = angle_average(r**3*sin(th)*drhovr*dvphi)
  if (mhd==True): F_advection = F_J-F_maxwell-F_reynolds
  else: F_advection = F_J-F_reynolds
  Sh = angle_average(rho*vel1*vel3*sin(th)) - angle_average(rho*vel1)*angle_average(vel3*sin(th))
  if (mhd==True): Sm = angle_average(Bcc1*Bcc3*-1.0*sin(th))
  if (mhd==True): Ptot = angle_average(press + bsq/2.0) 
  else: Ptot = angle_average(press)
  if (mhd==True): alpha_m = Sm/Ptot
  alpha_h = Sh/Ptot

def get_stress_cart(mhd = True):
  global alpha_m, alpha_h,F_J,F_maxwell,F_reynolds,F_advection,S
  if (mhd==True):bsq = B1**2 + B2**2 + B3**2
  def angle_average(arr,weight=None):
    dx3 = np.diff(ph[0,0,:])[0]
    dx2 = np.diff(th[0,:,0])[0]
    dOmega = (np.sin(th)*dx2*dx3)
    if weight is None: weight = 1.0
    return (arr * weight * dOmega).mean(-1).mean(-1)/(dOmega*weight).mean(-1).mean(-1)
  global alpha_m, alpha_h,F_J,F_maxwell,F_reynolds,F_advection
  drhovr = rho*vr - angle_average(rho*vr)[:,None,None]*(rho/rho)
  dvphi = vphi - angle_average(vphi)[:,None,None]*(rho/rho)
  if (mhd==True): F_J = angle_average((rho*vr*vphi-Br*Bphi)*sin(th)*r**3)
  else: F_J = angle_average((rho*vr*vphi)*sin(th)*r**3)
  if (mhd==True): F_maxwell = angle_average(r**3*Br*Bphi*sin(th))*-1
  else: F_maxwell = 0
  F_reynolds = angle_average(r**3*sin(th)*drhovr*dvphi)
  F_advection = F_J-F_maxwell-F_reynolds
  Sh = angle_average(rho*vr*vphi*sin(th)) - angle_average(rho*vr)*angle_average(vphi*sin(th))
  if (mhd==True):Sm = angle_average(Br*Bphi*-1.0*sin(th))
  else: Sm = 0
  if (mhd==True): Ptot = angle_average(press + bsq/2.0)
  else: Ptot = angle_average(press)
  alpha_m = Sm/Ptot
  alpha_h = Sh/Ptot

def gr_dot(A,B):
  return A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + A[3]*B[3]

def get_tetrad(uu_avg,ud_avg):
  global omega_t,omega_r,omega_th,omega_ph
  C0 = uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3]
  C1 = uu_avg[1]*ud_avg[0]
  C2 = uu_avg[1]*ud_avg[3]
  l = ud_avg[3]/ud_avg[0]
  s = -C0/abs(C0)
  N1 = np.sqrt(- gbl[1,1] * (ud_avg[0]*uu_avg[0]+ud_avg[3]*uu_avg[3])*(1+ud_avg[2]*uu_avg[2])) ##g[1,1] * np.sqrt(g[0,0]*C1**2 + g[1,1]*C0**2 + g[3,3]*C2**2 + 2*g[0,3]*C1*C2)
  N2 = np.sqrt(gbl[2,2]*(1+uu_avg[2]*ud_avg[2]))
  N3 = np.sqrt(gbl[0,0]*l**2 -2*gbl[0,3]*l + gbl[3,3])
  omega_t = uu_avg
  omega_r = uu_avg*0
  omega_r[0] = s/N1 * ud_avg[1]*uu_avg[0]
  omega_r[1] = s/N1 * - (uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3])
  omega_r[2] = 0
  omega_r[3] = s/N1 * ud_avg[1]*uu_avg[3]
  omega_th = uu_avg*0
  omega_th[0] = 1./N2 * ud_avg[2]*uu_avg[0]
  omega_th[1] = 1./N2 * ud_avg[2]*uu_avg[1]
  omega_th[2] = 1./N2 * (1.0 + ud_avg[2]*uu_avg[2])
  omega_th[3] = 1./N2 * ud_avg[2]*uu_avg[3]
  omega_ph = uu_avg*0
  omega_ph[0] = 1./N3 * (-l)
  omega_ph[3] = 1./N3


# def get_LNRF_ks(a=0):
#   global econ_t,econ_r,econ_th,econ_phi
#   r2 = r**2.
#   a2 = a**2.
#   sin2 = np.sin(th)**2.
#   gbl = np.zeros((4,4))
#   Sigma = r**2 + a**2.*np.cos(th)**2;
#   Delta = r**2 -2.0*r + a**2;
#   A = (r**2 + a**2)**2 - a**2*Delta*np.sin(th)**2.;


#   alpha = 1.0/np.sqrt(1.0 + 2.0*r/Sigma)
#   betar = 2.0*r/Sigma /(1.0 + 2.0*r/Sigma)
#   gammarr = 1.0 + 2.0*r/(Sigma)
#   gammathth = Sigma
#   gammaphiphi = A * sin2/Sigma
#   gammarphi = - a * sin2*(1.0+2.0*r/Sigma)

#   ecov_t = [alpha,0,0,0]
#   ecov_r = [betar/np.sqrt(gammarr),1.0/sqrt(gammarr),0,0]
#   ecov_th  = [0,0,np.sqrt(gammathth),0]
#   ecov_phi = [betar*gammarphi/np.sqrt(gammaphiphi),gammarphi/np.sqrt(gammaphiphi),0,np.sqrt(gammaphiphi)]

#   econ_t = [1.0/alpha,- betar/alpha,0,0]
#   econ_r = [0,1.0/np.sqrt(gammarr),0,-gammarphi/np.sqrt(gammarr)/gammaphiphi]
#   econ_th = [0,0,1.0/np.sqrt(gammathth),0]
#   econ_phi = [0,0,0,1.0/np.sqrt(gammaphiphi)]


def get_LNRF_bl(a=0):
  global econ_t,econ_r,econ_th,econ_phi
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  gbl = np.zeros((4,4))
  Sigma = r**2 + a**2.*np.cos(th)**2;
  Delta = r**2 -2.0*r + a**2;
  A = (r**2 + a**2)**2 - a**2*Delta*np.sin(th)**2.;
  rhosq = r2 + a2*np.cos(th)**2.0
  omega = 2.0*a*r/A 


  alpha = np.sqrt(Delta*rhosq/A)
  betacon_phi = -omega 
  gammacov_rr = rhosq/Delta
  gammacov_thth = rhosq 
  gammacov_phiphi = A*sin2/rhosq
  gammacon_rr = Delta/rhosq 
  gammacon_thth = 1.0/rhosq 
  gammacon_phiphi = rhosq/(A*sin2)


  ecov_t = [alpha,0,0,0]
  ecov_r = [0,np.sqrt(gammacov_rr),0,0]
  ecov_th  = [0,0,np.sqrt(gammacov_thth),0]
  ecov_phi = [betacon_phi*np.sqrt(gammacov_phiphi),0,0,np.sqrt(gammacov_phiphi)]

  zero = 0.0*r 
  econ_t = np.array([1.0/alpha,zero,zero,-betacon_phi/alpha])
  econ_r = np.array([zero,np.sqrt(gammacon_rr),zero,zero])
  econ_th = np.array([zero,zero,np.sqrt(gammacon_thth),zero])
  econ_phi = np.array([zero,zero,zero,np.sqrt(gammacon_phiphi)])



def ks_metric(r,th,a):
  global g
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  g  = np.zeros((4,4,nx,ny,nz))
  g[0][0] = -(1.0 - 2.0*m*r/sigma);
  g[0][1] = 2.0*m*r/sigma;
  g[1][0] = g[0][1]
  g[0][3] = -2.0*m*a*r/sigma * sin2
  g[3][0] = g[0][3]
  g[1][1] = 1.0 + 2.0*m*r/sigma
  g[1][3] =  -(1.0 + 2.0*m*r/sigma) * a * sin2
  g[3][1] = g[1][3]
  g[2][2] = sigma
  g[3][3] = (r2 + a2 + 2.0*m*a2*r/sigma * sin2) * sin2

def get_mri_q():
  global bphi,bth,dphi,dth,Qphi,Qth,gbl
  gbl =  bl_metric(r,th)
  uu_bl = ks_vec_to_bl(uu)
  ud_bl = Lower(uu_bl,gbl)
  get_tetrad(uu_bl,ud_bl)
  omega_t = bl_vec_to_ks(omega_t)
  omega_r = bl_vec_to_ks(omega_r)
  omega_th = bl_vec_to_ks(omega_th)
  omega_ph = bl_vec_to_ks(omega_ph)

  bphi = gr_dot(bd,omega_ph)
  bth = gr_dot(bd,omega_th)

  dx3 = np.diff(x3f)
  dx2 = np.diff(x2f)
  dx1 = np.diff(x1f)

  dx3 = (gdet/gdet) * dx3[None,None,:]
  dx2 = (gdet/gdet) * dx2[None,:,None]
  dx1 = (gdet/gdet) * dx1[:,None,None]
  
  dx_mu = uu*0 
  dx_mu[0] = 1.87213890018754e-03
  dx_mu[1] = dx1 
  dx_mu[2] = dx2 
  dx_mu[3] = dx3 

  dphi = gr_dot(Lower(dx_mu,g),omega_ph)
  dth = gr_dot(Lower(dx_mu,g),omega_th)

  Omega = uu[3]/uu[0]
  Qphi = 2.*np.pi / (Omega * dphi) * bphi/np.sqrt(rho)
  Qth = 2.*np.pi / (Omega * dth) * bth/np.sqrt(rho)

def get_mri_q_cartesian(x,y,z,a=0,xmax = 50.0,refinement_levels = 4):
  global bphi,bth,dphi,dth,Qphi,Qth,gbl,Qr
  nx = x.shape[0]
  ny = x.shape[1]
  nz = x.shape[2]
  cks_metric(x,y,z,a=a)
  bd = Lower(bu,g)

  uu_ks = cks_vec_to_ks(uu,x,y,z,a)


  gbl = bl_metric(r,th,a) 

  global econ_t,econ_r,econ_th,econ_phi
  get_LNRF_bl(a=a)

  econ_t = bl_vec_to_cks(x,y,z,econ_t,a=a)
  econ_r = bl_vec_to_cks(x,y,z,econ_r,a=a)
  econ_th = bl_vec_to_cks(x,y,z,econ_th,a=a)
  econ_phi = bl_vec_to_cks(x,y,z,econ_phi,a=a)

  bphi = gr_dot(bd,econ_phi)
  bth = gr_dot(bd,econ_th)
  br = gr_dot(bd,econ_r)


  DX = xmax*2.0/(nx*1.0)
  dx_list = []
  r_lim_list = []
  for n in range(refinement_levels):
    dx_list.append(DX/2.0**n)
    r_lim_list.append(xmax/2.0**n)

  dx_arr = rho*0 + DX

  for i in range(len(dx_list)-1):
    r_lim = r_lim_list[i]
    ind = (x<r_lim)*(x>-r_lim) * (y<r_lim)*(y>-r_lim) * (z<r_lim)*(z>-r_lim)
    dx_arr[ind] = dx_list[i]
  
  dx_mu = uu*0 
  dx_mu[0] = 1.46484375000429e-02
  dx_mu[1] = dx_arr 
  dx_mu[2] = dx_arr
  dx_mu[3] = dx_arr

  dphi = gr_dot(Lower(dx_mu,g),econ_phi)
  dth = gr_dot(Lower(dx_mu,g),econ_th)
  dr = gr_dot(Lower(dx_mu,g),econ_r)

  Omega = uu_ks[3]/uu_ks[0]
  Qphi = 2.*np.pi / (Omega * dphi) * bphi/np.sqrt(rho)
  Qth = 2.*np.pi / (Omega * dth) * bth/np.sqrt(rho)
  Qr = 2.*np.pi/(Omega * dr) * br/np.sqrt(rho)
def get_mri_q_newt():
  global Qphi,Qth,Qr
  
  dphi = np.gradient(ph,axis=2)
  dth = np.gradient(th,axis=1)
  dr = np.gradient(r,axis=0)


  Omega = vel3/r
  Qphi = 2.*np.pi / (Omega * dphi*r*sin(th)) * Bcc3/np.sqrt(rho)
  Qr = 2.*np.pi / (Omega * dr) * Bcc1/np.sqrt(rho)
  Qth = 2.*np.pi / (Omega * dth*r) * Bcc2/np.sqrt(rho)

def get_time_step_limit():
  dphi = np.gradient(ph,axis=2)
  dth = np.gradient(th,axis=1)
  dr = np.gradient(r,axis=0)

  dx1_min = np.amin(abs(dr/vel1),axis=2)
  dx2_min = np.amin(abs(dth*r/vel2),axis=2)
  dx3_min = np.amin(abs(dphi*r*sin(th)/vel3),axis=2)

  va = sqrt(bsq/rho)
  dx1_min_B = np.amin(dr/va,axis=2)
  dx2_min_B = np.amin(dth*r/va,axis=2)
  dx3_min_B = np.amin(dphi*r*sin(th)/va,axis=2)


def get_gr_stress(a=0):
  global omega_t,omega_r,omega_th,omega_ph
  global l,C0,C1,C2,N1,s,N2,N3
  gbl =  bl_metric(r,th)  
  uu_avg = uu*0

  vr_avg = angle_average(uu[1]/uu[0],weight=rho,gr=True)[:,None,None]*(rho/rho)
  vth_avg = angle_average(uu[2]/uu[0],weight=rho,gr=True)[:,None,None]*(rho/rho)
  vph_avg = angle_average(uu[3]/uu[0],weight=rho,gr=True)[:,None,None]*(rho/rho)
  vu = uu*0
  vu[0] = (rho/rho)
  vu[1] = vr_avg
  vu[2] = vth_avg
  vu[3] = vph_avg

  tmp = 0
  for mu in range(4):
    for nu in range(4):
      tmp += g[mu,nu]*vu[mu]*vu[nu]
  norm = np.sqrt(-1./tmp)

  uu_avg = vu * norm

  #for i in range(4): uu_avg[i] = angle_average(uu[i],gr=True)[:,None,None]*(rho/rho)

  #uu_avg = uu 
  #ud_avg = ud

  uu_avg = ks_vec_to_bl(uu_avg)
  ud_avg = Lower(uu_avg,gbl)
  C0 = uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3]
  C1 = uu_avg[1]*ud_avg[0]
  C2 = uu_avg[1]*ud_avg[3]
  l = ud_avg[3]/ud_avg[0]
  s = -C0/abs(C0)
  N1 = np.sqrt(- gbl[1,1] * (ud_avg[0]*uu_avg[0]+ud_avg[3]*uu_avg[3])*(1+ud_avg[2]*uu_avg[2])) ##g[1,1] * np.sqrt(g[0,0]*C1**2 + g[1,1]*C0**2 + g[3,3]*C2**2 + 2*g[0,3]*C1*C2)
  N2 = np.sqrt(gbl[2,2]*(1+uu_avg[2]*ud_avg[2]))
  N3 = np.sqrt(gbl[0,0]*l**2 -2*gbl[0,3]*l + gbl[3,3])
  omega_t = uu_avg
  omega_r = uu_avg*0
  omega_r[0] = s/N1 * ud_avg[1]*uu_avg[0]
  omega_r[1] = s/N1 * - (uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3])
  omega_r[2] = 0
  omega_r[3] = s/N1 * ud_avg[1]*uu_avg[3]
  omega_th = uu_avg*0
  omega_th[0] = 1./N2 * ud_avg[2]*uu_avg[0]
  omega_th[1] = 1./N2 * ud_avg[2]*uu_avg[1]
  omega_th[2] = 1./N2 * (1.0 + ud_avg[2]*uu_avg[2])
  omega_th[3] = 1./N2 * ud_avg[2]*uu_avg[3]
  omega_ph = uu_avg*0
  omega_ph[0] = 1./N3 * (-l)
  omega_ph[3] = 1./N3

  omega_t = bl_vec_to_ks(omega_t)
  omega_r = bl_vec_to_ks(omega_r)
  omega_th = bl_vec_to_ks(omega_th)
  omega_ph = bl_vec_to_ks(omega_ph)
  global reynolds_stress,maxwell_stress,total_stress,alpha,alpha_m,alpha_r

  total_stress=0
  maxwell_stress=0
  reynolds_stress=0
  Tud_calc(is_magnetic=True)
  for mu in range(4):
    for nu in range(4):
      total_stress += Tud[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
      maxwell_stress += TudEM[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
      reynolds_stress += TudMA[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
  alpha_m = angle_average(maxwell_stress,gr=True)/angle_average(press + bsq/2.0,gr=True)
  alpha_r = angle_average(reynolds_stress,gr=True)/angle_average(press+bsq/2.0,gr=True)
  alpha = alpha_m + alpha_r

def get_total_stress(a=0):
  global omega_t,omega_r,omega_th,omega_ph
  gbl =  bl_metric(r,th)  
  uu_avg = uu*0
  vu = uu*0
  vu[0] = (rho/rho)
  vu[1] = 0
  vu[2] = 0
  vu[3] = 0

  tmp = 0
  for mu in range(4):
    for nu in range(4):
      tmp += g[mu,nu]*vu[mu]*vu[nu]
  norm = np.sqrt(-1./tmp)

  uu_avg = vu * norm


  uu_avg = ks_vec_to_bl(uu_avg)
  ud_avg = Lower(uu_avg,gbl)
  C0 = uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3]
  C1 = uu_avg[1]*ud_avg[0]
  C2 = uu_avg[1]*ud_avg[3]
  l = ud_avg[3]/ud_avg[0]
  s = -C0/abs(C0)
  N1 = np.sqrt(- gbl[1,1] * (ud_avg[0]*uu_avg[0]+ud_avg[3]*uu_avg[3])*(1+ud_avg[2]*uu_avg[2])) ##g[1,1] * np.sqrt(g[0,0]*C1**2 + g[1,1]*C0**2 + g[3,3]*C2**2 + 2*g[0,3]*C1*C2)
  N2 = np.sqrt(gbl[2,2]*(1+uu_avg[2]*ud_avg[2]))
  N3 = np.sqrt(gbl[0,0]*l**2 -2*gbl[0,3]*l + gbl[3,3])
  omega_t = uu_avg
  omega_r = uu_avg*0
  omega_r[0] = s/N1 * ud_avg[1]*uu_avg[0]
  omega_r[1] = s/N1 * - (uu_avg[0]*ud_avg[0] + uu_avg[3]*ud_avg[3])
  omega_r[2] = 0
  omega_r[3] = s/N1 * ud_avg[1]*uu_avg[3]
  omega_th = uu_avg*0
  omega_th[0] = 1./N2 * ud_avg[2]*uu_avg[0]
  omega_th[1] = 1./N2 * ud_avg[2]*uu_avg[1]
  omega_th[2] = 1./N2 * (1.0 + ud_avg[2]*uu_avg[2])
  omega_th[3] = 1./N2 * ud_avg[2]*uu_avg[3]
  omega_ph = uu_avg*0
  omega_ph[0] = 1./N3 * (-l)
  omega_ph[3] = 1./N3

  omega_t = bl_vec_to_ks(omega_t)
  omega_r = bl_vec_to_ks(omega_r)
  omega_th = bl_vec_to_ks(omega_th)
  omega_ph = bl_vec_to_ks(omega_ph)
  global reynolds_stress,maxwell_stress,total_stress,alpha,alpha_m,alpha_r

  total_stress=0
  maxwell_stress=0
  reynolds_stress=0
  Tud_calc(is_magnetic=True)
  for mu in range(4):
    for nu in range(4):
      total_stress += Tud[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
      maxwell_stress += TudEM[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
      reynolds_stress += TudMA[mu][nu]*Lower(omega_r,g)[mu]*omega_ph[nu]
  alpha_m = angle_average(maxwell_stress,gr=True)/angle_average(press + bsq/2.0,gr=True)
  alpha_r = angle_average(reynolds_stress,gr=True)/angle_average(press+bsq/2.0,gr=True)
  alpha = alpha_m + alpha_r
def risco(a):
  Z1 = 1.0 + (1.0-a*a)**(1.0/3.0) * ( (1.0+a)**(1.0/3.0) + (1.0-a)**(1.0/3.0) ) 
  Z2 = sqrt(3.0*a*a + Z1*Z1)
  sgn = 1
  if (a>0): sgn = -1
  return 3.0 + Z2 + sgn*sqrt((3.0-Z1) * (3.0+Z1 + 2.0*Z2))


def ks_vec_to_gammie(uu_ks,x1,x2,x3,a=0.0,hslope = 0.3):
  r = np.exp(x1)

  dx1_dr = 1.0/r 
  dr_dx1 = 1./dx1_dr 
  dtheta_dx2 = np.pi + np.pi * (1.0-hslope) * np.cos(2*np.pi*x2)
  dx2_dtheta = 1.0/dtheta_dx2

  uu_gammie = uu_ks* 1.0

  #u^\mu_gammie = u^\nu_athena dx^\mu_gammie/dx^\nu_athena
  #u_\mu_gammie = u_\nu_athena dx^\nu_athena/dx^\mu_gammie

  uu_gammie[1] *= dx1_dr
  uu_gammie[2] *= dx2_dtheta

  return uu_gammie




def convert_to_gammie(a=0):
  global hslope, uu_gammie,ud_gammie,bu_gammie,bd_gammie
  hslope = 0.3
  #x1gammie = log(r)  r = exp(x1) 
  # dx1/dr = 1/r -> dx1/dr = r 
  #theta = pi*x2 + 0.5*(1-h)*sin(2*pi*x2)
  # dtheta/dx2 = pi + pi* (1-h) * cos(2*pi*x2)
  def thfunc(x2):
    return np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
  def x2func(th):
    def fun(x2):
      return thfunc(x2)-th
    return fsolve(fun,.5)

  x2 = np.zeros(ny)
  for j in range(ny):
    if (th[0,j,0] ==0):
      x2[j] = 0.0
      continue
    if (th[0,j,0]==np.pi):
      x2[j] = 1.0
      continue
    x2[j] = x2func(th[0,j,0])
  x2 = x2[None,:,None] * (r/r)



  dx1_dr = 1.0/r 
  dr_dx1 = 1./dx1_dr 
  dtheta_dx2 = np.pi + np.pi * (1.0-hslope) * np.cos(2*np.pi*x2)
  dx2_dtheta = 1.0/dtheta_dx2

  uu_gammie = uu 
  ud_gammie = ud 
  bu_gammie = bu 
  bd_gammie = bd 

  #u^\mu_gammie = u^\nu_athena dx^\mu_gammie/dx^\nu_athena
  #u_\mu_gammie = u_\nu_athena dx^\nu_athena/dx^\mu_gammie

  uu_gammie[1] *= dx1_dr
  uu_gammie[2] *= dx2_dtheta
  bu_gammie[1] *= dx1_dr
  bu_gammie[2] *= dx2_dtheta

  ud_gammie[1] *= dr_dx1 
  ud_gammie[2] *= dtheta_dx2
  bd_gammie[1] *= dr_dx1
  bd_gammie[2] *= dtheta_dx2



  # gprime_mu nu = dx^alpha/dx^mu dx^sigma/dx^nu g_alpha sigma
  # prime^mu^nu = dx^mu/dx^alpha dx^nu/dx^sigma g^alpha sigma

  # so gi_gammie ^mu nu = dx_gammie^mu/dx^sig dx_gammie^nu/dx^alph g^sig alph
  # only nonzero are dx1/dr and dx2/dth

  # gi_gammie00 is unchanged
  # gi_gammie10 = dx1/dr 
  #   g_inv(I00,i) = -(1.0 + 2.0*m*r/sigma);
  #   g_inv(I01,i) = 2.0*m*r/sigma;
  #   g_inv(I11,i) = delta/sigma;
  #   g_inv(I13,i) = a/sigma;
  #   g_inv(I22,i) = 1.0/sigma;
  #   g_inv(I33,i) = 1.0 / (sigma * sin2);
  sigma = r**2.0 + a**2.0 * np.cos(th)**2.0
  m = 1
  gi00 = -(1.0 + 2.0*m*r/sigma)
  gi01 = 2.0*m*r/sigma *dx1_dr
  gi02 = 0
  gi03 = 0
  global v1,v2,v3,B1,B2,B3,gdet_gammie
  v1 = uu_gammie[1] - gi01/gi00 * uu_gammie[0]
  v2 = uu_gammie[2] - gi02/gi00 * uu_gammie[0]
  v3 = uu_gammie[3] - gi03/gi00 * uu_gammie[0]

  B1 = bu_gammie[1]*uu_gammie[0] - bu_gammie[0]*uu_gammie[1]
  B2 = bu_gammie[2]*uu_gammie[0] - bu_gammie[0]*uu_gammie[2]
  B3 = bu_gammie[3]*uu_gammie[0] - bu_gammie[0]*uu_gammie[3]

  gdet_gammie = gdet * dr_dx1 * dtheta_dx2




def gammie_metric(r,th,a=0,hslope = 0.3):
  global gg

  def thfunc(x2):
    return np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
  def x2func(th):
    def fun(x2):
      return thfunc(x2)-th
    return fsolve(fun,.5)

  x2 = np.zeros(ny)
  for j in range(ny):
    if (th[0,j,0] ==0):
      x2[j] = 0.0
      continue
    if (th[0,j,0]==np.pi):
      x2[j] = 1.0
      continue
    x2[j] = x2func(th[0,j,0])
  x2 = x2[None,:,None] * (r/r)
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  rfac = r*1.0;
  hfac = np.pi + (1. - hslope) * np.pi * np.cos(2. * np.pi * x2);
  gg  = np.zeros((4,4,nx,ny,nz))
  gg[0][0] = -(1.0 - 2.0*m*r/sigma);
  gg[0][1] = 2.0*m*r/sigma * rfac
  gg[1][0] = gg[0][1] 
  gg[0][3] = -2.0*m*a*r/sigma * sin2
  gg[3][0] = gg[0][3]
  gg[1][1] = (1.0 + 2.0*m*r/sigma) * rfac * rfac
  gg[1][3] =  -(1.0 + 2.0*m*r/sigma) * a * sin2 * rfac
  gg[3][1] = gg[1][3] 
  gg[2][2] = sigma * hfac * hfac
  gg[3][3] = (r2 + a2 + 2.0*m*a2*r/sigma * sin2) * sin2

def gammie_gcon(r,th,a=0,hslope=0.3):
  global ggcon

  def thfunc(x2):
    return np.pi*x2 + 0.5*(1.-hslope)*np.sin(2.*np.pi*x2)
  def x2func(th):
    def fun(x2):
      return thfunc(x2)-th
    return fsolve(fun,.5)

  x2 = np.zeros(ny)
  for j in range(ny):
    if (th[0,j,0] ==0):
      x2[j] = 0.0
      continue
    if (th[0,j,0]==np.pi):
      x2[j] = 1.0
      continue
    x2[j] = x2func(th[0,j,0])
  x2 = x2[None,:,None] * (r/r)
  m = 1
  r2 = r**2.
  a2 = a**2.
  sin2 = np.sin(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2;
  delta = r**2 - 2.0*m*r + a**2
  rfac = r*1.0;
  hfac = np.pi + (1. - hslope) * np.pi * np.cos(2. * np.pi * x2);
  ggcon  = np.zeros((4,4,nx,ny,nz))
  ggcon[0][0] = -(1.0 + 2.0*m*r/sigma);
  ggcon[0][1] =  2.0*m*r/sigma /(rfac)
  ggcon[1][0] = ggcon[0][1] 
  ggcon[1][1] = delta/sigma /( rfac * rfac)
  ggcon[1][3] =  a/sigma / rfac
  ggcon[3][1] = ggcon[1][3] 
  ggcon[2][2] = 1.0/sigma /( hfac * hfac )
  ggcon[3][3] = 1.0 / (sigma * sin2)




def gammie_grid():
  global ri,thi,phii,x1_grid,x2_grid,x3_grid
  global igrid_new,jgrid_new,kgrid_new
  global igrid,jgrid,kgrid
  global x1,x2,x3
  # dx3grid = 2.0*pi/(nz*1.)
  
  # dx1grid = log(np.amax(r) /np.amin(r))/(nx*1.0)
  # dx2grid = 1.0/(ny*1.0)
  x1_grid_faces = np.linspace(log(np.amin(r)),log(np.amax(r)),nx+1)  ##faces
  x2_grid_faces = np.linspace(0,1,ny+1)       ##faces
  x3_grid_faces = np.linspace(0,2.0*pi,nz+1)  ##faces
  x1_grid = ( (x1_grid_faces) + 0.5*np.diff(x1_grid_faces)[0] )[:-1]
  x2_grid = ( x2_grid_faces + 0.5*np.diff(x2_grid_faces)[0] ) [:-1]
  if (nz==1): x3_grid = x3_grid_faces[0] + np.pi
  else: x3_grid =( x3_grid_faces + 0.5*np.diff(x3_grid_faces)[0] )[:-1]

  # x1_grid = np.linspace(log(np.amin(r)) + 0.5*dx1grid,log(np.amax(r))-0.5*dx1grid,nx)
  # x2_grid = np.linspace(0+0.5*dx2grid,1-0.5*dx2grid,ny)
  # x3_grid = np.linspace(0+0.5*dx3grid,2.0*pi-0.5*dx3grid,nz)
  #x1_grid = x1_grid + 0.5*np.diff(x1_grid)[0]
  #x2_grid = x2_grid + 0.5*np.diff(x2_grid)[0]
  #if (nz==1): x3_grid = x3_grid + np.pi
  #else: x3_grid = x3_grid + 0.5*np.diff(x3_grid)[0]
  ri = np.exp(x1_grid)
  thi = np.pi*x2_grid + 0.5*(1.0-hslope)*np.sin(2.0*pi*x2_grid)
  phii = x3_grid

  ri,thi,phii = np.meshgrid(ri,thi,phii,indexing='ij')


  kgrid,jgrid,igrid = meshgrid(np.arange(0,nz),np.arange(0,ny),np.arange(0,nx),indexing='ij')
  igrid,jgrid,kgrid = meshgrid(np.arange(0,nx),np.arange(0,ny),np.arange(0,nz),indexing='ij')
  mgrid = igrid + jgrid*nx  + kgrid*nx*ny

  mnew = scipy.interpolate.griddata((r.flatten(),th.flatten(),ph.flatten()),mgrid[:,:,:].flatten(),(ri,thi,phii),method='nearest')


  # index = np.arange(nx*ny*nz)

  # new_index  = scipy.interpolate.griddata((r.flatten(),th.flatten(),ph.flatten()),index,(ri,thi,phii),method='nearest')

  igrid_new= mod(mod(mnew,ny*nx),nx)
  jgrid_new = mod(mnew,ny*nx)//nx
  kgrid_new = mnew//(ny*nx)


def cross_product(a,b):
    return [ a[1]*b[2] - b[1]*a[2], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0] ]
def dot_product(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def curl(a):
  x_tmp = x[:,0,0]
  y_tmp = y[0,:,0]
  z_tmp = z[0,0,:]
  return [ gradient(a[2],y_tmp,axis=1) - gradient(a[1],z_tmp,axis=2), gradient(a[0],z_tmp,axis=2) - gradient(a[2],x_tmp,axis=0), gradient(a[1],x_tmp,axis=0) - gradient(a[0],y_tmp,axis=1)]

def divergence(a):
  x_tmp = x[:,0,0]
  y_tmp = y[0,:,0]
  z_tmp = z[0,0,:]
  return gradient(a[0],x_tmp,axis=0) + gradient(a[1],y_tmp,axis=1) + gradient(a[2],z_tmp,axis=2)

def curl_spherical(a):
  iphi = 2
  ith = 1
  ir = 0
  small = 1e-15
  r_tmp = r[:,0,0]
  th_tmp = th[0,:,0]
  phi_tmp = ph[0,0,:]
  curl_r   = 1.0/(r*np.sin(th) + small) * (gradient(a[iphi]*np.sin(th),th_tmp,axis=ith)             - gradient(a[ith]   , phi_tmp , axis=iphi) )
  curl_th  = 1.0/r                      * ( 1.0/(sin(th)+small) * gradient(a[ir],phi_tmp,axis=iphi) - gradient(r*a[iphi], r_tmp   , axis=ir)   )
  curl_phi = 1.0/r                      * ( gradient(r*a[ith],r_tmp,axis=ir)                        - gradient(a[ir]    , th_tmp  , axis=ith)  )
  return [ curl_r,curl_th,curl_phi]


def curl_gr(a):
  iphi = 2
  ith = 1
  ir = 0
  small = 1e-15
  r_tmp = r[:,0,0]
  th_tmp = th[0,:,0]
  phi_tmp = ph[0,0,:]
  curl_r   = 1.0/(r**2*np.sin(th) + small) * (gradient(a[iphi],th_tmp,axis=ith)             - gradient(a[ith]   , phi_tmp , axis=iphi) )
  curl_th  = 1.0/(r**2*np.sin(th) + small) * (gradient(a[ir],phi_tmp,axis=iphi)             - gradient(a[iphi]  , r_tmp   , axis=ir)   )
  curl_phi = 1.0/(r**2*np.sin(th) + small) * (gradient(a[ith],r_tmp,axis=ir)                - gradient(a[ir]    , th_tmp  , axis=ith)  )
  return [ curl_r,curl_th,curl_phi]

def div_spherical(a):
  iphi = 2
  ith = 1
  ir = 0
  small = 1e-15
  r_tmp = r[:,0,0]
  th_tmp = th[0,:,0]
  phi_tmp = ph[0,0,:]
  return 1.0/r**2.0 * gradient(r**2.0*a[ir],r_tmp,axis=ir) + 1.0/(r*np.sin(th) + small) * gradient(a[ith]*np.sin(th),th_tmp,axis=ith) + 1.0/(r*np.sin(th)+small) * gradient(a[iphi],phi_tmp,axis=iphi)

def advection_derivative_spherical(a,b):
  iphi = 2 ;ith = 1; ir = 0
  small = 1e-15
  r_tmp = r[:,0,0]; th_tmp = th[0,:,0]; phi_tmp = ph[0,0,:]

  def ddr(c):
    return gradient(c,r_tmp,axis=ir)
  def ddth(c):
    return gradient(c,th_tmp,axis=ith)
  def ddphi(c):
    return gradient(c,phi_tmp,axis=iphi)

  ar = a[ir]; ath = a[ith]; aphi = a[iphi]
  br = b[ir]; bth = b[ith]; bphi = b[iphi]

  r_term = ar * ddr(br) + ath/r*ddth(br) + aphi/(r*np.sin(th)+small)*ddphi(br) - (ath*bth+aphi*bphi)/r
  th_term = ar * ddr(bth) + ath/r * ddth(bth) + aphi/(r*np.sin(th)+small)*ddphi(bth) + ath*br/r - aphi*bphi*np.cos(th)/(r*np.sin(th)+small)
  phi_term = ar * ddr(bphi) + ath/r * ddth(bphi) + aphi/(r*np.sin(th)+small)*ddphi(bphi) + aphi*br/r + aphi*bth*np.cos(th)/(r*np.sin(th)+small)
  return [r_term,th_term,phi_term]

def curl_2d(a):
  x_tmp = x[:,0]
  y_tmp = y[0,:]
  return [ gradient(a[2],y_tmp,axis=1), - gradient(a[2],x_tmp,axis=0), gradient(a[1],x_tmp,axis=0) - gradient(a[0],y_tmp,axis=1)]

def make_grmonty_dump(fname,a=0,gam=5./3.):
  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]
  convert_to_gammie(a = a)
  gammie_grid()
  gammie_metric(ri,thi,a=a)
  gammie_gcon(ri,thi,a=a)
  dx1 = np.diff(x1_grid)[0]
  dx2 = np.diff(x2_grid)[0]
  if (nz==1): dx3 = 2.*np.pi
  else: dx3 = np.diff(x3_grid)[0]


  Nprim = 8 
  header = [str(np.array(t)), str(nx), str(ny), str(nz), str(np.amin(x1_grid)-0.5*dx1),str(np.amin(x2_grid)-0.5*dx2),str(np.amin(x3_grid)-0.5*dx3), str(dx1),str(dx2),str(dx3),str(a),str(gam),str(np.amin(ri)),str(hslope),str(Nprim)]

  rhoi =rho[igrid_new,jgrid_new,kgrid_new]
  #uui = uu_gammie[:,igrid_new,jgrid_new,kgrid_new]

  v1i = v1[igrid_new,jgrid_new,kgrid_new]
  v2i = v2[igrid_new,jgrid_new,kgrid_new]
  v3i = v3[igrid_new,jgrid_new,kgrid_new]

  qsq = v1i*v1i*gg[1,1] + v2i*v2i*gg[2,2] + v3i*v3i*gg[3,3] + 2.*v1i*v2i*gg[1,2] + 2.*v1i*v3i*gg[1,3] + 2.*v2i*v3i*gg[2,3]
  alpha = 1./np.sqrt(-ggcon[0,0]) 
  beta = 0*uu_gammie
  for l in range(1,4): beta[l] = ggcon[0][l]*alpha*alpha ;

  qsq[qsq<0] = 1e-10
  
  gamma = np.sqrt(1.0 + qsq)
  uui = 0*uu_gammie
  uui[0] = gamma/alpha 
  uui[1] = v1i - gamma*beta[1]/alpha
  uui[2] = v2i - gamma*beta[2]/alpha
  uui[3] = v3i - gamma*beta[3]/alpha

  udi = Lower(uui,gg)



  B1i = B1[igrid_new,jgrid_new,kgrid_new]
  B2i = B2[igrid_new,jgrid_new,kgrid_new]
  B3i = B3[igrid_new,jgrid_new,kgrid_new]

  bui = bu_gammie*0

  bui[0] = B1i*udi[1] + B2i*udi[2] + B3i*udi[3]
  bui[1] = (B1i + bui[0]*uui[1])/uui[0]
  bui[2] = (B2i + bui[0]*uui[2])/uui[0]
  bui[3] = (B3i + bui[0]*uui[3])/uui[0]

  bdi = Lower(bui,gg)



  #udi = ud_gammie[:,igrid_new,jgrid_new,kgrid_new]
  #bui = bu_gammie[:,igrid_new,jgrid_new,kgrid_new]
  #bdi = bd_gammie[:,igrid_new,jgrid_new,kgrid_new]
  pressi = press[igrid_new,jgrid_new,kgrid_new]
  gdeti = gdet_gammie[igrid_new,jgrid_new,kgrid_new]

  tmp = rhoi*0
  x1_grid,x2_grid,x3_grid = meshgrid(x1_grid,x2_grid,x3_grid,indexing='ij')
  data = [igrid,jgrid,kgrid,x1_grid.astype(float32),x2_grid.astype(float32),x3_grid.astype(float32),ri.astype(float32),thi.astype(float32),phii.astype(float32),
          rhoi.astype(float32),(pressi/(5./3.-1.)).astype(float32),v1i.astype(float32),v2i.astype(float32),v3i.astype(float32),B1i.astype(float32),B2i.astype(float32),
          B3i.astype(float32),(pressi/rhoi**(5./3.)).astype(float32),
          uui[0].astype(float32),uui[1].astype(float32),uui[2].astype(float32),uui[3].astype(float32),udi[0].astype(float32),udi[1].astype(float32),udi[2].astype(float32),
          udi[3].astype(float32), bui[0].astype(float32),bui[1].astype(float32),bui[2].astype(float32),bui[3].astype(float32),bdi[0].astype(float32),bdi[1].astype(float32),
          bdi[2].astype(float32),bdi[3].astype(float32),gdeti.astype(float32)]
  data = np.array(data).astype(float32)
  fout = open(fname,"w")
  fout.write(" ".join(header) + "\n")
  #fout.flush()
  fout.close()
  fout = open(fname,"ab")
  data = data.transpose(1,2,3,0)
  data.tofile(fout)
  fout.close()

def make_grmonty_dump_cartesian(fname,idump,a=0,gam=5./3.,hslope = 0.3,high_res = False):

  if (high_res == True):rd_yt_convert_to_gammie(idump,MHD=True,gr=True,a=a,hslope = hslope,low_res=True,nr=356,nth=200,nphi=400)
  else: rd_yt_convert_to_gammie(idump,MHD=True,gr=True,a=a,hslope = hslope)
  nx = rho.shape[0]
  ny = rho.shape[1]
  nz = rho.shape[2]


  dx1_dr = 1.0/r 
  dr_dx1 = 1./dx1_dr 
  dtheta_dx2 = np.pi + np.pi * (1.0-hslope) * np.cos(2*np.pi*x2)
  dx2_dtheta = 1.0/dtheta_dx2
  gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 ) * abs(dr_dx1 * dtheta_dx2)

  gammie_metric(r,th,a=a,hslope=hslope)
  gammie_gcon(r,th,a=a,hslope=hslope)
  dx1 = np.diff(x1[:,0,0])[0]
  dx2 = np.diff(x2[0,:,0])[0]
  if (nz==1): dx3 = 2.*np.pi
  else: dx3 = np.diff(x3[0,0,:])[0]


  Nprim = 8 
  # electron_data = []
  # if ("ke_ent" in globals()):  
  #   Nprim += 1
  #   electron_data.append(ke_ent.astype(float32))
  # if ("ke_ent2" in globals()): 
  #   Nprim += 1
  #   electron_data.append(ke_ent2.astype(float32))
  # if ("ke_ent3" in globals()): 
  #   Nprim += 1
  #   electron_data.append(ke_ent3.astype(float32))
  # if ("ke_ent4" in globals()):
  #   Nprim +=1 
  #   electron_data.append(ke_ent4.astype(float32))


  header = [str(np.array(t)), str(nx), str(ny), str(nz), str(np.amin(x1)-0.5*dx1),str(np.amin(x2)-0.5*dx2),str(np.amin(x3)-0.5*dx3), str(dx1),str(dx2),str(dx3),str(a),str(gam),str(np.amin(r)),str(hslope),str(Nprim)]

  ud = Lower(uu,gg)
  bd = Lower(bu,gg)

  igrid,jgrid,kgrid = meshgrid(np.arange(0,nx),np.arange(0,ny),np.arange(0,nz),indexing='ij')
  data = [igrid,jgrid,kgrid,x1.astype(float32),x2.astype(float32),x3.astype(float32),r.astype(float32),th.astype(float32),ph.astype(float32),
          rho.astype(float32),(press/(5./3.-1.)).astype(float32),vel1.astype(float32),vel2.astype(float32),vel3.astype(float32),Bcc1.astype(float32),Bcc2.astype(float32),
          Bcc3.astype(float32),(press/rho**(5./3.)).astype(float32),
          uu[0].astype(float32),uu[1].astype(float32),uu[2].astype(float32),uu[3].astype(float32),ud[0].astype(float32),ud[1].astype(float32),ud[2].astype(float32),
          ud[3].astype(float32), bu[0].astype(float32),bu[1].astype(float32),bu[2].astype(float32),bu[3].astype(float32),bd[0].astype(float32),bd[1].astype(float32),
          bd[2].astype(float32),bd[3].astype(float32),gdet.astype(float32)]
  # data = data + electron_data
  data = np.array(data).astype(float32)
  fout = open(fname,"w")
  fout.write(" ".join(header) + "\n")
  #fout.flush()
  fout.close()
  fout = open(fname,"ab")
  data = data.transpose(1,2,3,0)
  data.tofile(fout)
  fout.close()
def box_limits(a):
  plt.xlim(-a,a)
  plt.ylim(-a,a)

def subplot_axis_label(fig,x_label,y_label,fontsize=15,bottom=0.15,left=0.15):
  fig.subplots_adjust(bottom=bottom)
  fig.subplots_adjust(left=left)
  fig.text(0.5, 0.04, x_label, fontsize=fontsize,ha='center')
  fig.text(0.02, 0.5,y_label, fontsize=fontsize, va='center', rotation='vertical')

def subplot_draw_cb(fig,c,r_lim=0.83,b_lim = 0.15,height=0.77,width = 0.03):
    global cb
    fig.subplots_adjust(right=r_lim)
    cbar_ax = fig.add_axes([r_lim+0.02, b_lim, width, height])
    cb = plt.colorbar(c, cax=cbar_ax)
  


def r_to_ir_npz(r_input,r_arr):
  dlog10r = np.diff(np.log10(r_arr[:,0,0]))[0]
  r_min = r_arr[0,0,0]
  r_out = r_arr[-1,0,0]
  #r = r_min * 10**(ir*dlog10r)
  return np.int(np.round(np.log10(r_input/r_min)/dlog10r))

def single_star_accretion_diagram():
  clf()
  plt.style.use('dark_background')  
  from matplotlib.patches import Ellipse
  from matplotlib.patches import FancyArrowPatch
  l_array = np.linspace(0,2,20) #np.array([,0.1,0.3,0.,1])
  th_array = np.linspace(0,np.pi,20)
  cmap = plt.get_cmap('RdBu_r')
  indices = np.linspace(0,cmap.N,len(l_array))
  my_colors = [cmap(int(i)) for i in indices]
  i = 0
  for l in l_array:
    lw = 2 #fabs(l)*2
    #l = 1+cos(th)
    if l<1: color = 'blue'
    else: color = 'red'
    if (l<=np.sqrt(2)):
      ecc = 1-l**2
      # r = a (1 - ecc*cos(E) ) .... cos(E) = -1
      a = 1/(1+ecc)
      b = a*sqrt(1-ecc**2)
      x_cent = 1-a
      r_circ = sqrt(l)
      e1 = Ellipse((x_cent,0),2*a,2*b,fill=False,lw=lw,color = color) #my_colors[i]) 
      e1 = Ellipse((0,0),r_circ*2,r_circ*2,fill=False,lw=lw,color = color) #my_colors[i]) 

      plt.gca().add_patch(e1)
    else:
      continue
      ecc =l**2-1
      a = 1/(ecc-1) #1/(ecc-1) #-1*l/(1-ecc**2)
      # a = 1/(2 - l**2)
      # ecc = 1-1/a
      b = a*sqrt(ecc**2-1)
      x_cent = 1+a 
      x_arr = np.linspace(-10,1,2000)
      y_arr = b*np.sqrt((x_arr-x_cent)**2/a**2 -1)
      plt.plot(x_arr,y_arr,lw=lw,color = color )#my_colors[i])
    i = i + 1
  plt.xlim(-2.25,1.25)
  plt.ylim(-1.5,2)
  plt.xlim(-1.5,1.5)
  plt.ylim(-1.5,1.5)
  plt.plot(1,0,marker='*',ms=20,color = 'gold')
  arrow = FancyArrowPatch((1.15,-.1),(1.15,.4),arrowstyle='simple,head_width=8,head_length=8',color = 'gold', \
    lw=2)#,connectionstyle="arc3,rad=0.4")
  plt.gca().add_patch(arrow)
  bh = Ellipse((0,0),.01,.01,color = 'white',lw=2) 
  star_orbit = Ellipse((0,0),2,2,color = 'gold',lw=2,ls = '--',fill=False)
  plt.gca().add_patch(star_orbit)
  plt.gca().add_patch(bh)
  plt.gca().get_xaxis().set_visible(False)
  plt.gca().get_yaxis().set_visible(False)
  plt.gca().set_aspect('equal')
  plt.savefig('single_star_circ.pdf')


def run_dependency(qsub_file,i_dep = 3,orig_id = None):
  import subprocess
  if (orig_id is None):
    out = subprocess.check_output(['sbatch',qsub_file])
  else:
    out = subprocess.check_output(['sbatch','--dependency=afterany:%d' %orig_id,qsub_file])
  for i in range(i_dep):
     prev_job = [int(s) for s in out.split() if s.isdigit()][0]
     out = subprocess.check_output(['sbatch','--dependency=afterany:%d' %prev_job,qsub_file])
     print (out)
     print ("with dependency: %d" %prev_job)


def get_RM(x=0,y=0,cum=False):
  from scipy.integrate import cumtrapz
  global z_los,ray
  e_charge = 4.803e-10
  me = 9.109e-28
  cl = 2.997924e10
  mp = 1.6726e-24
  pc = 3.086e18
  kyr = 3.154e10
  msun = 1.989e33


  Z_o_X_solar = 0.0177
  Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
  X_solar = 0.7491
  Z_solar = 1.0-X_solar - Y_solar

  muH_solar = 1./X_solar
  Z = 3. * Z_solar
  X = 0.
  mue = 2. /(1.+X)
  mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)

  Bunit = pc/kyr * np.sqrt(4. * np.pi * msun/pc**3 ) 

  ray = ds.r[(x,"pc"),(y,"pc"),:]

  ne = np.array(ray['density'].in_cgs())/mue/mp
  z = np.array(ray['z'].in_cgs())

  integrand_RM = e_charge**3/(2.0*np.pi * me**2 * cl**4) * ne *np.array(-ray['Bcc3'].in_cgs())
  integrand_DM = ne
  z_los = z
  if (cum==False):
    RM = np.trapz(integrand_RM[z<0],z[z<0])
    DM = np.trapz(integrand_DM[z<0],z[z<0])
    return RM,DM
  else:
    RM = cumtrapz(integrand_RM,z)
    DM = cumtrapz(integrand_DM,z)
    return RM,DM


def get_Xray_Lum_los(file_prefix,x=0,y=0):
    mp_over_kev = yt.YTQuantity(9.994827,"kyr**2/pc**2")
    Z_o_X_solar = 0.0177
    Y_solar = 0.2246 + 0.7409 * (Z_o_X_solar)
    X_solar = 0.7491
    Z_solar = 1.0-X_solar - Y_solar

    muH_solar = 1./X_solar
    Z = 3.0 * Z_solar
    X = 0.
    mue = 2. /(1.+X)
    mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.)
    mp = yt.YTQuantity(8.41175e-58,"Msun")
    kb = yt.YTQuantity(1.380658e-16,"erg/K")

    def Lam_func(TK):
        f1 = file_prefix + "_H_only.dat"
        f2 = file_prefix + "_He_only.dat"
        f3 = file_prefix + "_Metals_only.dat"
        data = np.loadtxt(f1)
        T_tab = data[:,0] * 1.16045e7
        Lam_H = data[:,1]
        data = np.loadtxt(f2)
        Lam_He = data[:,1]
        data = np.loadtxt(f3)
        Lam_metals = data[:,1]
        # T_tab = 10.**data[:,0]
        T_min = np.amin(T_tab)
        T_max = np.amax(T_tab)
        # if isinstance(TK,)
        # TK[TK<T_min] = T_min
        # TK[TK>T_max] = T_max
        # Lam_tab = 10.**data[:,1]

        Lam_tab = (X/X_solar) * Lam_H + ( (1.-X-Z)/Y_solar) * Lam_He + (Z/Z_solar) * Lam_metals
        from scipy.interpolate import InterpolatedUnivariateSpline
        Lam = InterpolatedUnivariateSpline(T_tab,Lam_tab,k = 1,ext =3)  #ext = 3 returns boundary value if outside range of T
        return Lam(TK)* yt.YTQuantity(1,"g * cm**5/s**3")
    def _Lam_chandra(field,data):
        T_K = (data['gas','pressure']/data['gas','density'] * mu_highT*mp/kb).in_units("K")
        #T_K = T_kev*1.16e7
        nH = data['gas','density']/mp/muH_solar
        ne = data['gas','density']/mp/mue
        return Lam_func(T_K).in_units("Msun *pc**5/kyr**3 ") * (ne*nH).in_units("pc**-6")

    ds.add_field(('gas','Lam_chandra'),function = _Lam_chandra,units="Msun/kyr**3/pc",particle_type = False,sampling_type="cell",force_override=True)
    
    ray = ds.r[(x,"pc"),(y,"pc"),:]

    return np.trapz(np.array(ray['Lam_chandra'].in_cgs()),np.array(ray['z'].in_cgs()))



    #res = 300
    #length_im = r_out
#frb = proj.to_frb((length_im,"pc"),[res,res])
      #image = frb['Lam_chandra']*Lz
      #x_im = np.linspace(-length_im/2.,length_im/2.,res)
      #y_im = np.linspace(-length_im/2.,length_im/2.,res)


    return average_value
def PSR_pos():
  global t_yr,ddelta,dalpha,YEAR,MONTH,DAY,valpha,vdelta
  arc_secs = 0.040239562
  dat = np.loadtxt('PSR.dat',usecols=(0,13,16))
  t_mjd = dat[:,0]

  #correct for Errata Bower+ 2016
  dalpha = (dat[:,1] + 18.36) * 1e-3 * arc_secs
  ddelta = (dat[:,2] + 32.00) * 1e-3 * arc_secs

  JD = t_mjd + 2400000.5 
  L= JD+68569
  N= 4*L//146097
  L= L-(146097*N+3)//4
  I= 4000*(L+1)//1461001
  L= L-1461*I//4+31
  J= 80*L//2447
  K= L-2447*J//80
  L= J//11
  J= J+2-12*L
  I= 100*(N-49)+I+L
  YEAR= I
  MONTH= J
  DAY= K

  t_yr = YEAR + MONTH/12.0 + DAY/365.25

  valpha = 2.45 * arc_secs
  vdelta = 5.89 * arc_secs

  #t_yr = (t_mjd + 2400000.5)/365.25 - 4713

def set_Be_units(r_star = 7.0,m_star = 9.2):
  global rho_unit, p_unit,B_unit,L_unit,T_unit,mass_unit,t_unit,v_unit
  global mp,kb
  rho_unit = 1e-11 #Okazaki 2001
  R_sun = 6.96e10
  M_sun = 1.99e33
  L_unit =  r_star * R_sun
  G_newt = 6.67259e-8
  kb = 1.380649e-16
  mp = 1.67e-24
  #GM/L_unit = v_unit
  #M = r_star/G
  mass_unit = m_star * M_sun
  t_unit = L_unit/np.sqrt(G_newt*mass_unit/L_unit)
  v_unit = L_unit/t_unit
  T_unit = mp * v_unit**2.0 / (kb)


  #2 kb T/mp = press/rho

def set_units():
  global UnitB,Unitlength
  UnitDensity = 6.767991e-23; 
  UnitEnergyDensity = 6.479592e-7; 
  UnitTime = 3.154e10;  
  Unitlength = 3.086e+18; 
  UnitB = Unitlength/UnitTime * np.sqrt(4. * np.pi* UnitDensity);



def compare_old_wind_new_wind():
  old_dir = '/global/scratch/smressle/star_cluster/stellar_wind_test_mhd/no_refine_cos_theta'
  new_dir = '/global/scratch/smressle/star_cluster/stellar_wind_test_mhd/3D_B_source_sinr_norm'

  os.chdir(old_dir)
  rdhdf5(100,ndim=3,coord="xy",user_x2=False,gr=False,a=0)

  plt.close()
  plt.figure(figsize=(10,5))
  plt.clf()
  plt.subplot(121)
  pcolormesh(x[:,0,:],z[:,0,:],log10(bsq/np.amax(bsq))[:,ny//2,:],cmap = 'ocean',vmin=-3,vmax=0)

  plt.xlabel(r'$x$',fontsize = 20)
  plt.ylabel(r'$z$ ',fontsize = 20)

  plt.subplot(122)
  pcolormesh(x[:,:,0],y[:,:,0],log10(bsq/np.amax(bsq))[:,:,nz//2],cmap = 'ocean',vmin=-3,vmax=0)

  plt.xlabel(r'$x$',fontsize = 20)
  plt.ylabel(r'$y$ ',fontsize = 20)
  cb = plt.colorbar()
  cb.set_label(r'$\log_{10}\left(b^2\right)$',fontsize=25)

  plt.suptitle(r'Old (Bad)',fontsize=20)


  os.chdir(new_dir)
  rdhdf5(100,ndim=3,coord="xy",user_x2=False,gr=False,a=0)

  plt.figure(figsize=(10,5))
  plt.clf()
  plt.subplot(121)
  pcolormesh(x[:,0,:],z[:,0,:],log10(bsq/np.amax(bsq))[:,ny//2,:],cmap = 'ocean',vmin=-3,vmax=0)

  plt.xlabel(r'$x$',fontsize = 20)
  plt.ylabel(r'$z$ ',fontsize = 20)

  plt.subplot(122)
  pcolormesh(x[:,:,0],y[:,:,0],log10(bsq/np.amax(bsq))[:,:,nz//2],cmap = 'ocean',vmin=-3,vmax=0)

  plt.xlabel(r'$x$',fontsize = 20)
  plt.ylabel(r'$y$ ',fontsize = 20)
  cb = plt.colorbar()
  cb.set_label(r'$\log_{10}\left(b^2\right)$',fontsize=25)

  plt.suptitle(r'New (Good)',fontsize=20)

  os.chdir(old_dir)

  rd_yt_convert_to_spherical(100,MHD=True)
  Bcc1 = B1
  Bcc2 = B2
  Bcc3 = B3
  get_mdot(mhd=True)
  
  set_units()
  B_a = 1.0/UnitB
  r_a = 6.957e10/Unitlength
  plt.figure(1)
  plt.clf()
  ir = 200
  B_sol = (B_a * r_a/r * np.sin(theta) ).mean(-1)[ir,ny//2]
  plt.plot(theta[200,:,0],Bphi[ir,:,:].mean(-1)/B_sol,lw=2,label = r'Old')

  os.chdir(new_dir)
  rd_yt_convert_to_spherical(100,MHD=True)
  Bcc1 = B1
  Bcc2 = B2
  Bcc3 = B3
  get_mdot(mhd=True)
  
  plt.figure(1)
  B_a = 30/UnitB
  B_sol = (B_a * r_a/r * np.sin(theta) ).mean(-1)[ir,ny//2]
  plt.plot(theta[200,:,0],Bphi[ir,:,:].mean(-1)/B_sol,lw=2,label = r'New')

  plt.plot(theta[200,:,0],sin(theta[200,:,0]),lw=2,ls='--',label = r'$\sin(\theta))$')

  plt.xlabel(r'$\theta$',fontsize = 20)
  plt.ylabel(r'$B_\varphi$ (Norm.) ',fontsize = 20)
  plt.legend(loc = 'best',frameon=False,fontsize=15)
  plt.setp(plt.gca().get_xticklabels(), fontsize=15)
  plt.setp(plt.gca().get_yticklabels(), fontsize=15)
  plt.tight_layout()


def rd_rst(fname):
  fin = open(fname,'rb')
  while b'par_end' not in fin.readline():
    continue
  dtype = np.float32
  body = np.fromfile(fin,dtype=dtype,count=-1)


def brackett_gamma():
  TK = press/rho * mu_highT*mp_over_kev * keV_to_Kelvin
  TK[TK<1e4] = 1e4
  ne = rho/mue * rho_to_n_cgs
  npro = rho/mu_highT * rho_to_n_cgs
  j = 3.44e-27 * ne*npro *(1e4/TK)**1.09 * ne * npro 

  pc = 3.086e18
  surface_brightness = j.mean(-1) * (amax(z)-amin(z) ) * pc 



def plot_SMR_grid(levels =9):
  plt.clf()
  plt.ylim(-1,1)
  plt.xlim(-1,1)
  for level in np.arange(levels+1):
    xmax = 1.0/2.0**level
    for x in np.linspace(-xmax,xmax,17):
      plt.plot([-xmax,xmax],[x,x],lw=1,color='k')
      plt.plot([x,x],[-xmax,xmax],lw=1,color='k')


def plot_fieldlines_gr(box_radius = 0.003,xbox_radius = None, ybox_radius = None,a=0,density=1,color='black',npz=False,phi_avg = False,arrowstyle=None,lw=1):
  global x_stream,z_stream,Bx,Bz
  if (xbox_radius is None and ybox_radius is None): 
    xbox_radius = box_radius
    ybox_radius = box_radius
  elif (ybox_radius is None): ybox_radius = xbox_radius
  elif (xbox_radius is None): xbox_radius = ybox_radius

  x_stream,z_stream = np.meshgrid(np.linspace(0,xbox_radius,128),np.linspace(-ybox_radius,ybox_radius ,128),indexing = 'ij')

  dx_dr = np.sin(th)
  dz_dr = np.cos(th)

  dx_dtheta = r * np.cos(th)
  dz_dtheta = -r * np.sin(th)

  dx_dphi = a *np.cos(ph)
  dz_dphi = 0.0

  if (npz==True):
    Bx = Bcc1 
    Bz = Bcc3
  else:
    Bx  = dx_dr * Bcc1 + dx_dtheta * Bcc2 + dx_dphi * Bcc3
    Bz  = dz_dr * Bcc1 + dz_dtheta * Bcc2 + dz_dphi * Bcc3

  if (phi_avg==True):
    Bx = Bx.mean(-1)[:,:,None] * (r/r)
    Bz = Bz.mean(-1)[:,:,None] * (r/r)


  vx= scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),Bx[:,:,0].flatten(),(x_stream,z_stream),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),Bz[:,:,0].flatten(),(x_stream,z_stream),method = 'nearest')

  plt.streamplot(x_stream.transpose(),z_stream.transpose(),vx.transpose(),vz.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)
  vx = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),Bx[:,:,nz//2].flatten(),(x_stream,z_stream),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),Bz[:,:,nz//2].flatten(),(x_stream,z_stream),method = 'nearest')

  if (npz==False): plt.streamplot(-x_stream.transpose(),z_stream.transpose(),-vx.transpose(),vz.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)
  else: plt.streamplot(-x_stream.transpose(),z_stream.transpose(),vx.transpose(),vz.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)
  
def plot_fieldlines_gr_midplane(box_radius = 0.003,a = 0,color ='black',npz=False,density = 1,arrowstyle=None,lw=1):
  global x_stream,y_stream,Bx,By
  x_stream,y_stream = np.meshgrid(np.linspace(-box_radius,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')


  dx_dr = np.cos(ph)
  dy_dr = np.sin(ph)
  dx_dphi = -r * np.sin(ph) + a * np.cos(ph)
  dy_dphi =  r * np.cos(ph) + a * np.sin(ph)

  if (npz==True):
    Bx = Bcc1
    By = Bcc2
  else:
    Bx  = dx_dr * Bcc1 + dx_dphi * Bcc3 
    By  = dy_dr * Bcc1 + dy_dphi * Bcc3


  vx = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),Bx[:,ny//2,:].flatten(),(x_stream,y_stream),method = 'nearest')
  vy = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),By[:,ny//2,:].flatten(),(x_stream,y_stream),method = 'nearest')

  plt.streamplot(x_stream.transpose(),y_stream.transpose(),vx.transpose(),vy.transpose(),color = color,density=density,arrowstyle=arrowstyle,linewidth=lw)

def plot_streamlines_gr_midplane(box_radius = 0.003,a = 0,color ='black'):
  global x_stream,y_stream,Bx,By
  x_stream,y_stream = np.meshgrid(np.linspace(-box_radius,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')


  dx_dr = np.cos(ph)
  dy_dr = np.sin(ph)
  dx_dphi = -r * np.sin(ph) + a * np.cos(ph)
  dy_dphi =  r * np.cos(ph) + a * np.sin(ph)

  Bx  = dx_dr * uu[1]/uu[0] + dx_dphi * uu[3]/uu[0] 
  By  = dy_dr * uu[1]/uu[0] + dy_dphi * uu[3]/uu[0]


  vx = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),Bx[:,ny//2,:].flatten(),(x_stream,y_stream),method = 'nearest')
  vy = scipy.interpolate.griddata((x[:,ny//2,:].flatten(),y[:,ny//2,:].flatten()),By[:,ny//2,:].flatten(),(x_stream,y_stream),method = 'nearest')

  plt.streamplot(x_stream.transpose(),y_stream.transpose(),vx.transpose(),vy.transpose(),color = color)

def plot_streamlines_gr(box_radius = 0.003,npz=False):
  global x_stream,z_stream,Bx,Bz
  x_stream,z_stream = np.meshgrid(np.linspace(0,box_radius,128),np.linspace(-box_radius,box_radius ,128),indexing = 'ij')


  dx_dr = np.sin(th)
  dz_dr = np.cos(th)
  dx_dtheta = r * np.cos(th)
  dz_dtheta = -r * np.sin(th)

  if (npz==True):
    vvx = uu[1]/uu[0]
    vvz = uu[3]/uu[0]
  else: 
    vvx =    dx_dr * uu[1]/uu[0] + dx_dtheta * uu[2]/uu[0]
    vvz =    dz_dr * uu[1]/uu[0] + dz_dtheta * uu[2]/uu[0]


  vx= scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vvx[:,:,0].flatten(),(x_stream,z_stream),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vvz[:,:,0].flatten(),(x_stream,z_stream),method = 'nearest')

  plt.streamplot(x_stream.transpose(),z_stream.transpose(),vx.transpose(),vz.transpose(),color = 'black')
  vx = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vvx[:,:,nz//2].flatten(),(x_stream,z_stream),method = 'nearest')
  vz = scipy.interpolate.griddata((x[:,:,0].flatten(),z[:,:,0].flatten()),vvz[:,:,nz//2].flatten(),(x_stream,z_stream),method = 'nearest')

  if (npz==False): plt.streamplot(-x_stream.transpose(),z_stream.transpose(),-vx.transpose(),vz.transpose(),color = 'black')
  else: plt.streamplot(-x_stream.transpose(),z_stream.transpose(),vx.transpose(),vz.transpose(),color = 'black')
  


def rd_inflow_file(fname,a=0,create_vector_potential = False):
  data = np.loadtxt(fname)
  global r, rho, ur,uphi, Br,Bphi,atheta
  r = data[:,0]
  rho = data[:,1]
  ur = data[:,2]
  uphi = data[:,3]
  Br = data[:,4]
  Bphi = data[:,5]


  th = np.pi/2.0
  gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )

  #atheta = -(gdet*Br) * 2.0 * np.pi 

  if (create_vector_potential == True):

    dr = np.diff(log(r))[0]*r

    atheta2 = (gdet*Bphi*dr).cumsum()

    data_new = np.array([r,rho,ur,uphi,Br,Bphi,atheta2]).transpose()

    np.savetxt(fname[:-4] + "_vector_potential.dat",data_new)


def convert_inflow_to_cartesian(fname, a=0):

    rd_inflow_file(fname)
    nx = r.shape[0]
    ny = 1
    nz = 1

    g_bl = bl_metric(r[:,None,None],th = pi/2,a=0)
    g_bl = g_bl[:,:,:,0,0]*1.0

    ut = np.sqrt((-ur*ur*g_bl[1][1] - uphi*uphi*g_bl[3][3] -1) / g_bl[0][0] )

    bt = g_bl[0,3]*ut*Bphi + g_bl[1][1]*ur*Br + g_bl[3][3]*uphi*Bphi 
    br = 1.0/ut * (Br + bt * ur);
    bphi = 1.0/ut * (Bphi + bt * uphi)

    phi_vals = np.linspace(0,2*pi,200)

    r_grid,phi_grid = np.meshgrid(r,phi_vals,indexing='ij')

    x_grid = r_grid * np.cos(phi_grid) + a * np.sin(phi_grid)
    y_grid = r_grid * np.sin(phi_grid) - a * np.cos(phi_grid)
    z_grid = 0.0*x_grid

    ut = ut[:,None]*(r_grid/r_grid)
    uphi = uphi[:,None]*(r_grid/r_grid)
    ur = ur[:,None]*(r_grid/r_grid)

    bt = bt[:,None]*(r_grid/r_grid)
    bphi = bphi[:,None]*(r_grid/r_grid)
    br = br[:,None]*(r_grid/r_grid)

    uu_bl = np.array([ut,ur,0*ur,uphi])
    bu_bl = np.array([bt,br,0*br,bphi])


    g_bl = bl_metric(r_grid[:,:,None]*1.0,pi/2*(r_grid/r_grid)[:,:,None],a=0)
    g_bl = g_bl[:,:,:,:,0]*1.0
    ud_bl = Lower(uu_bl,g_bl)
    bd_bl = Lower(bu_bl,g_bl)

    bsq_bl = gr_dot(bu_bl,bd_bl)

    uu_cks = bl_vec_to_cks(x_grid,y_grid,z_grid,np.array([ut,ur,ur*0,uphi]),a=a)
    bu_cks = bl_vec_to_cks(x_grid,y_grid,z_grid,np.array([bt,br,br*0,bphi]),a=a)

    cks_metric(x_grid[:,:,None]*1.0,y_grid[:,:,None]*1.0,z_grid[:,:,None]*1.0,a=a)
    g = g[:,:,:,:,0]

    ud_cks = Lower(uu_cks,g)
    bd_cks = Lower(bu_cks,g)

    bsq_cks = gr_dot(bu_cks,bd_cks)

    Bx = bu_cks[1]*uu_cks[0] - bu_cks[0]*uu_cks[1]
    By = bu_cks[2]*uu_cks[0] - bu_cks[0]*uu_cks[2]

    ## d Az/dy = Bx 
    ## d Az/dx = -By 



def psicalc_slice(B1 = None,gr=False,xy=False,iphi = 0):
    """
    Computes the field vector potential
    """
    if (B1 is None): B1 = Bcc1
    if (xy==False):
      _dx2 = np.diff(x2f)
      daphi = -(r*np.sin(th)*B1)[:,:,iphi]*_dx2[None,:]
      if (gr==True): daphi = -(gdet*B1)[:,:,iphi]*_dx2[None,:]
      if (gr== False): aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]/np.sin(th)[:,:,0]
      else: aphi=daphi[:,::-1].cumsum(axis=1)[:,::-1]
      aphi-=0.5*daphi #correction for half-cell shift between face and center in theta
    else: #Calculate Ay assuming By = 0 (i.e. projecting the magnetic field onto the plane)
      daphi = -B1[:,ny//2,:]
      aphi = daphi[:,::-1].cumsum(axis=1)[:,::-1]
      aphi-=0.5*daphi


    return(aphi)


def test_cks_functions():
  a =0.95
  nx = 15
  ny = 15
  nz = 15
  x = np.linspace(-5,5,nx)
  y = np.linspace(-5,5,ny)
  z = np.linspace(-5,5,nz)

  x,y,z = np.meshgrid(x,y,z,indexing='ij')


  rand_vec = rand(4,nx,ny,nz)


  cks_metric(x,y,z,a=a)

  rand_vec_lower = Lower(rand_vec,g)

  rand_vec_sq_cks = gr_dot(rand_vec,rand_vec_lower)

  rand_vec_ks = cks_vec_to_ks(rand_vec,x,y,z,a=a)

  cks_coord_to_ks(x,y,z,a=a)

  ks_metric(r,th,a=a)

  rand_vec_ks_lower = Lower(rand_vec_ks,g)

  rand_vec_sq_ks = gr_dot(rand_vec_ks,rand_vec_ks_lower)



  g = bl_metric(r,th,a=a)
  rand_vec_bl_lower = Lower(rand_vec,g)
  rand_vec_sq_bl = gr_dot(rand_vec_bl_lower,rand_vec)

  rand_vec_cks = bl_vec_to_cks(x,y,z,rand_vec,a=a)

  cks_metric(x,y,z,a=a)
  rand_vec_cks_lower = Lower(rand_vec_cks,g)
  rand_vec_sq_cks = gr_dot(rand_vec_cks,rand_vec_cks_lower)


def get_xyz_angular_momentum():
  global lz,ly,lx,Lx,Ly,Lz,Ltot
  uu_bl = ks_vec_to_bl(uu,0)  
  uu_cks = bl_vec_to_cks(x,y,z,uu_bl,0)

  cks_metric(x,y,z,0)
  ud_cks = Lower(uu_cks,g)

  ##ud_phi_ks = u_mu dx^mu/dphi_ks = u_x dx/dphi + u_y dy/dphi + ..
  ##ud_phi_ks =  u_x (-y) + u_y (x) + ..


  lz = ud_cks[1] * (-y) + ud_cks[2] * x 
  ly = ud_cks[1] * z - ud_cks[3] * x
  lx = ud_cks[3] * (y) - ud_cks[2] * z       

  Lx = angle_average(rho*lx,gr=True)
  Ly = angle_average(rho*ly,gr=True)
  Lz = angle_average(rho*lz,gr=True)

  Ltot = np.sqrt(Lx**2 + Ly**2 + Lz**2)


def test_restart_file():

  yr = 31556926.0
  pc = 3.09e18;    
  cl = 2.99792458e10 * (1e3 * yr)/pc 
  gm_ = 0.019
  L_unit = gm_/cl**2;
  r_in = 2.0*2.0/128.0/(2.0**9.0)/L_unit
  rs = 2.0
  rho_unit = 1960.53
  B_unit = cl * np.sqrt(rho_unit);
  
  rdhdf5(00,ndim=3,coord='spherical',gr=True,a=0,user_x2=True)

  r_orig=r 

  vr = angle_average(uu[1]/uu[0],weight=rho,gr=True)
  vth = angle_average(uu[2]/uu[0] * r,weight=rho,gr=True)
  vphi = angle_average(uu[3]/uu[0] * r * np.sin(th),weight=rho,gr=True)

  vsq_restart = angle_average((uu[1]/uu[0])**2 + (uu[2]/uu[0])**2*r**2 + (uu[3]/uu[0])**2*r**2*sin(th)**2,weight=rho,gr=True)

  Br = angle_average(Bcc1,gr=True)
  Bphi = angle_average(Bcc3 * r * np.sin(th),gr=True)
  Bth = angle_average(Bcc2 * r ,gr=True)

  bsq = angle_average(bsq,gr=True)

  clf()
  # loglog(r[:,0,0],-vr,ls='-',color='blue')
  # loglog(r[:,0,0],vphi,ls='-',color='red')

  loglog(r[:,0,0],Br,ls='-',color='blue')
  loglog(r[:,0,0],Bphi,ls='-',color='red')
  loglog(r[:,0,0],Bth,ls='-',color='green')


  # loglog(r[:,0,0],bsq,ls='-',color='red')

  rdnpz("/global/scratch/smressle/star_cluster/test_new_code/without_S2_9_levels_beta_1e2_v3_orbits_comet/dump_spher_120_th_1.3_phi_-1.8.npz")

  get_mdot(True)

  vsq_orig = angle_average_npz(vr**2 + vphi**2 + vth**2,weight=rho)
  bsq = angle_average_npz(Br**2 + Bphi**2 + Bth**2)

  vr = angle_average_npz(vr,weight=rho)
  vphi = angle_average_npz(vphi,weight=rho)

  Br = angle_average_npz(Br)
  Bphi = angle_average_npz(Bphi)
  Bth = angle_average_npz(Bth)



  # loglog(r[:,0,0]/L_unit * (rs/r_in),-vr/cl * sqrt(r_in/rs),ls='--',color='blue')
  # loglog(r[:,0,0]/L_unit * (rs/r_in),vphi/cl * sqrt(r_in/rs),ls='--',color='red')

  loglog(r[:,0,0]/L_unit * (rs/r_in),Br/B_unit * (r_in/rs),ls='--',color='blue')
  loglog(r[:,0,0]/L_unit * (rs/r_in),Bphi/B_unit * (r_in/rs),ls='--',color='red')
  loglog(r[:,0,0]/L_unit * (rs/r_in),Bth/B_unit * (r_in/rs),ls='--',color='green')

def cks_metric_code(x1,x2,x3,a):
  global g
  x = x1;
  y =x2;
  z = x3;
  def SQR(q):
    return q**2.0
  R = np.sqrt(SQR(x) + SQR(y) + SQR(z));
  r = SQR(R) - SQR(a) + np.sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
  r = np.sqrt(r/2.0);


  eta = [0,0,0,0]
  l_lower = [0,0,0,0]
  l_upper = [0,0,0,0]

  f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  l_upper[0] = -1.0;
  l_upper[1] = (r*x + a*y)/( SQR(r) + SQR(a) );
  l_upper[2] = (r*y - a*x)/( SQR(r) + SQR(a) );
  l_upper[3] = z/r;

  l_lower[0] = 1.0;
  l_lower[1] = l_upper[1];
  l_lower[2] = l_upper[2];
  l_lower[3] = l_upper[3];

  eta[0] = -1.0;
  eta[1] = 1.0;
  eta[2] = 1.0;
  eta[3] = 1.0;


  g = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
  g[0][0] = eta[0] + f * l_lower[0]*l_lower[0];
  g[0][1] = f * l_lower[0]*l_lower[1];
  g[0][2] = f * l_lower[0]*l_lower[2];
  g[0][3] = f * l_lower[0]*l_lower[3];
  g[1][0] = g[0][1]
  g[2][0] = g[0][2]
  g[3][0] = g[0][3]
  g[1][1] = eta[1] + f * l_lower[1]*l_lower[1];
  g[1][2] = f * l_lower[1]*l_lower[2];
  g[2][1] = g[1][2]
  g[1][3] = f * l_lower[1]*l_lower[3];
  g[3][1] = g[1][3]
  g[2][2] = eta[2] + f * l_lower[2]*l_lower[2];
  g[2][3] = f * l_lower[2]*l_lower[3];
  g[3][2] = g[2][3]
  g[3][3] = eta[3] + f * l_lower[3]*l_lower[3];
  ##loglog(r[:,0,0]/L_unit * (rs/r_in),bsq/B_unit**2 * (r_in/rs)**2,ls='--',color='red')


def plot_mass_temp_hist():

  plt.clf()
  low_res = "/global/scratch/smressle/star_cluster/binary_wind_shock/symmetric_128_tcool_off_axis"
  high_res = "/global/scratch/smressle/star_cluster/binary_wind_shock/symmetric_512_tcool_off_axis"

  hist_avg_high_res = 0.0
  hist_avg_low_res = 0.0
  set_constants()
  dump_range = arange(8,14)
  for i in dump_range:
    for res_dir in [low_res,high_res]:
      os.chdir(res_dir)
      rdhdf5(i,ndim=3,coord='xy')
      box_length = 2.0*1.018e-3
      TK = press/rho * mu_highT*mp_over_kev*keV_to_Kelvin
      if (res_dir==low_res): res = 128.0
      else: res = 512.0
      dx = box_length/res
      cell_vol = dx**3.0 
      cell_mass = rho * cell_vol
      if (res_dir==low_res): nbins = 30
      else: 200
      T_bins = np.logspace(3.7,7.2,nbins)

      index_array = (y<(-x*1.75+.0006)) * (y>(-x*1.75-.0006))
      T_array = TK[index_array].flatten()
      cell_mass_array = cell_mass[index_array].flatten()
      if (res_dir==low_res): line_style = '--'
      else: line_style = '-'
      hist,bins = np.histogram(T_array,bins = T_bins, weights = cell_mass_array,density=True)

      if (res_dir==high_res): bins_high_res = bins 
      else: bins_low_res = bins

      if (res_dir==high_res): hist_avg_high_res += hist/(len(dump_range))
      else: hist_avg_low_res  += hist/(len(dump_range))

  plt.loglog(bins_low_res[1:],hist_avg_low_res,lw=2,label = r"$N=%d$" %(np.int(128)),ls="--")
  plt.loglog(bins_high_res[1:],hist_avg_high_res,lw=2,label = r"$N=%d$" %(np.int(512)),ls="--")

  plt.ylabel(r'$dM/dT$',fontsize = 20)
  plt.xlabel(r'$T$ (K)',fontsize=20)

  plt.legend(loc='best',frameon=0,fontsize=15)

  for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
      label.set_fontsize(10)
  plt.tight_layout()
  plt.savefig("dM_dT_binary_wind.png")

  os.chdir(high_res)
  rdhdf5(13,ndim=3,coord='xy')
  plt.figure(2)
  plt.clf()

  c1 = pcolormesh(x[:,:,0]*206265,y[:,:,0]*206265,log10(rho[:,:,nz//2]),vmin=3,vmax=5)
  cb = plt.colorbar(c1,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
  cb.set_label(r"$\log_{10}(\int \rho dz)$ ($T > 9 \times 10^4 $K) ",fontsize=17)
  plt.xlabel(r'$x$ (Au)',fontsize = 20)
  plt.ylabel(r'$y$ (Au)',fontsize = 20)
  plt.tight_layout()
  plt.savefig("binary_wind_512.png")


  os.chdir(low_res)
  rdhdf5(13,ndim=3,coord='xy')
  plt.figure(3)
  plt.clf()

  c1 = pcolormesh(x[:,:,0]*206265,y[:,:,0]*206265,log10(rho[:,:,nz//2]),vmin=3,vmax=5)
  cb = plt.colorbar(c1,ax = plt.gca()) #,pad = 0.001) #,location='top', orientation = 'horizontal') #cax = cax,orientation = 'horizontal') #orientation= 'horizontal') #,ax =plt.gca()) ,orientation = 'horizontal') #cax=cb2axes)
  cb.set_label(r"$\log_{10}( \rho)$ ",fontsize=17)
  plt.xlabel(r'$x$ (Au)',fontsize = 20)
  plt.ylabel(r'$y$ (Au)',fontsize = 20)
  plt.tight_layout()
  plt.savefig("binary_wind_128.png")



def plot_1d_grmhd():
  rd_1d_avg()
  plt.figure(1)
  plt.clf()
  ir = 10
  plt.plot(t,(((EdotEM)/abs(mdot)))[:,ir]*100,lw=2,ls='-',label = r'$\frac{P_{\rm jet}}{\dot M c^2} \times 100 $')
  plt.plot(t,(Phibh/sqrt(-mdot))[:,ir],lw=2,ls = '--',label = r'$\phi_{\rm BH}$')
  plt.xlabel(r'$t$ (M)',fontsize = 20)
  plt.legend(loc='best',frameon=False,fontsize=15)
  plt.ylim(0,150)
  plt.tight_layout()
  plt.figure(2)
  plt.clf()
  ir = 10
  p0 = 5.9880242e-03
  rho0 = 1
  gm_ = 1

  plt.plot(t,((-mdot))[:,ir]/mdot_bondi(p0,rho0),lw=2,ls='-')
  plt.xlabel(r'$t$ (M)',fontsize = 20)
  plt.ylabel(r'$|\dot M/\dot M_{\rm Bondi}|$',fontsize = 20)

  plt.ylim(0,0.4)
  plt.tight_layout()

  def plot_restart_mhd():
    os.chdir("/global/scratch/smressle/star_cluster/restart_mhd/beta_1e2_cooling_125_comet")
    rd_hst("star_wind.hst",True)
    plt.figure(1)
    plt.clf()
    t_ff = np.sqrt(1e-3**1.5/gm_/2.0)
    PhiBH = 2.0*pi*r**2*Br_abs_avg*sqrt(4*pi)
    phibh = PhiBH/sqrt(abs(mdot_avg*v_kep*r**2))

    plt.plot(t/t_ff,-mdot_avg[:,r_to_ir(2e-6)]*1e5,lw=2,label = r'$\dot M$')

    plt.xlabel(r'$t/[t_{ff}(r= 1  $ mpc$)]$',fontsize = 20)
    plt.ylabel(r'$\dot M$ ($10^{-8} M_\odot/$yr)',fontsize = 20)
    plt.setp(plt.gca().get_xticklabels(), fontsize=20)
    plt.setp(plt.gca().get_yticklabels(), fontsize=20)

    #plt.legend(loc='best',fontsize = 15,frameon=0)
    plt.ylim(0,3)
    plt.tight_layout()

    plt.savefig("mdot_t_restart_mhd.png")
    plt.figure(2)
    plt.clf()
    t_ff = np.sqrt(1e-3**1.5/gm_/2.0)
    PhiBH = 2.0*pi*r**2*Br_abs_avg*sqrt(4*pi)
    phibh = PhiBH/sqrt(abs(mdot_avg*v_kep*r**2))

    plt.plot(t/t_ff,phibh[:,r_to_ir(2e-6)]*1e5,lw=2,label = r'$\dot M$')

    plt.xlabel(r'$t/[t_{ff}(r= 1  $ mpc$)]$',fontsize = 20)
    plt.ylabel(r'$\phi_{BH}$',fontsize = 20)
    plt.setp(plt.gca().get_xticklabels(), fontsize=20)
    plt.setp(plt.gca().get_yticklabels(), fontsize=20)

    #plt.legend(loc='best',fontsize = 15,frameon=0)
    plt.ylim(0,10)
    plt.tight_layout()
    plt.savefig("phibh_restart_mhd.png")

    plt.savefig("mdot_t_restart_mhd.png")
    plt.figure(2)
    plt.clf()
    t_ff = np.sqrt(1e-3**1.5/gm_/2.0)
    PhiBH = 2.0*pi*r**2*Br_abs_avg*sqrt(4*pi)
    phibh = PhiBH/sqrt(abs(mdot_avg*v_kep*r**2))

    plt.plot(t/t_ff,phibh[:,r_to_ir(2e-6)]*1e5,lw=2,label = r'$\dot M$')

    plt.xlabel(r'$t/[t_{ff}(r= 1  $ mpc$)]$',fontsize = 20)
    plt.ylabel(r'$\phi_{BH}$',fontsize = 20)
    plt.setp(plt.gca().get_xticklabels(), fontsize=20)
    plt.setp(plt.gca().get_yticklabels(), fontsize=20)

    #plt.legend(loc='best',fontsize = 15,frameon=0)
    plt.ylim(0,10)
    plt.tight_layout()
    plt.savefig("phibh_restart_mhd.png")

    plt.figure(3)
    plt.clf()
    t_ff = np.sqrt(1e-3**1.5/gm_/2.0)
    PhiBH = 2.0*pi*r**2*Br_abs_avg*sqrt(4*pi)
    phibh = PhiBH/sqrt(abs(mdot_avg*v_kep*r**2))

    plt.plot(t/t_ff,phibh[:,r_to_ir(2e-6)]*1e5,lw=2,label = r'$\dot M$')

    plt.xlabel(r'$t/[t_{ff}(r= 1  $ mpc$)]$',fontsize = 20)
    plt.ylabel(r'$\phi_{BH}$',fontsize = 20)
    plt.setp(plt.gca().get_xticklabels(), fontsize=20)
    plt.setp(plt.gca().get_yticklabels(), fontsize=20)

    #plt.legend(loc='best',fontsize = 15,frameon=0)
    plt.ylim(0,10)
    plt.tight_layout()
    plt.savefig("phibh_restart_mhd.png")

def get_Te(Rhigh,Rlow=1):
  global Te
  mue = 2
  mu = 1.351
  mui = mue * mu / (mue - mu)
  beta = press/bsq*2.0
  Ti_o_Te = (Rhigh*beta**2 + Rlow)/(1.0 + beta**2.0)

  T_tot = mu * press/(rho)

  Te = T_tot/ (mu/mue + mu/mui *Ti_o_Te)


def EHT_comp(a=0.9375):

  thmin = pi/3.0
  thmax = 2.0*pi/3.0


  get_mdot(mhd=True,gr=True,a=a)
  plt.figure(1)
  plt.clf()
  plt.subplot(231)

  loglog(r[:,0,0],angle_average_npz(rho,weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))
  plt.xlim(1,50)
  plt.ylim(1e-2,1e0)

  plt.subplot(232)
  loglog(r[:,0,0],angle_average_npz(press,weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))

  plt.xlim(1,50)
  plt.ylim(1e-6,1e-2)

  plt.subplot(233)

  loglog(r[:,0,0],angle_average_npz(np.sqrt(bsq),weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))

  plt.xlim(1,50)
  plt.ylim(1e-4,1e-1)

  plt.subplot(234)

  loglog(r[:,0,0],angle_average_npz(uu_ks[3],weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))

  plt.xlim(1,50)
  plt.ylim(1e-3,1e1)

  plt.subplot(235)

  loglog(r[:,0,0],angle_average_npz(press+bsq/2.0,weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))

  plt.xlim(1,50)
  plt.ylim(1e-6,1e-2)

  plt.subplot(236)

  loglog(r[:,0,0],angle_average_npz(bsq/2.0/press,weight = (th>thmin)*(th<thmax),gr=True,a=0.9375))

  plt.xlim(1,50)
  plt.ylim(1e-2,1e0)


  Edotmean = (Edot/abs(mdot))[:,50][500:].mean()
  plt.figure(2)

  ir = 50

  plt.subplot()


def Chris_EHT_comp():



  rd_1d_avg()
  dic = np.load("code_comparison_192.npz")
  a = 0.9375
  rh = ( 1.0 + np.sqrt(1.0-a**2) )
  ir = r_to_ir(rh)
  plt.figure(1)
  plt.clf()
  plt.subplot(411)



  plt.plot(t,-mdot[:,ir],lw=2,ls='--',label='Cartesian')
  plt.plot(dic['t'],dic['Mdot_horizon'],lw=2,ls='-',label='Spherical')
  plt.ylabel(r'$\dot M$')
  plt.ylim(0,1)
  plt.xlim(0,1e4)
  plt.legend(loc='best',frameon=False)


  plt.subplot(412)

  plt.plot(t,abs(Jdot/mdot)[:,ir],lw=2,ls='--')
  plt.plot(dic['t'],abs(dic['Jdot_horizon']/dic['Mdot_horizon']),lw=2,ls='-')
  plt.ylabel(r'$\dot J/ \dot M$')
  plt.ylim(1.5,2.5)
  plt.xlim(0,1e4)


  plt.subplot(413)

  plt.plot(t,abs(Edot/mdot)[:,ir],lw=2,ls='--')
  plt.plot(dic['t'],abs((dic['Edot_horizon']-dic['Mdot_horizon'])/dic['Mdot_horizon']),lw=2,ls='-')
  plt.ylabel(r'$|\dot E - \dot M|/ \dot M$')

  plt.ylim(0,.2)
  plt.xlim(0,1e4)

  plt.subplot(414)

  plt.plot(t,abs(Phibh/sqrt(-mdot)/sqrt(4*pi))[:,ir],lw=2,ls='--')
  plt.plot(dic['t'],abs((dic['Phi_horizon'])/sqrt(dic['Mdot_horizon'])/sqrt(4*pi)),lw=2,ls='-')
  plt.ylabel(r'$\phi_{\rm BH}$')

  plt.ylim(0,2)
  plt.xlim(0,1e4)

  plt.xlabel(r'$t$ $[M]$')

  plt.tight_layout()


  rd_1d_torus_avg()

  plt.figure(2)
  plt.clf()

  plt.subplot(221)
  plt.loglog(r[-1,:],rho[500:,:].mean(0),lw =2,ls='--',label='Cartesiann')
  plt.loglog(dic['r'],dic['rho_ave'],lw=2,ls='-',label = 'Spherical')
  plt.legend(loc='best',frameon=False)
  plt.ylabel(r'$\rho$')
  plt.ylim(1e-2,1e0)
  plt.xlim(1,50)

  subplot(222)
  plt.loglog(r[-1,:],press[500:,:].mean(0),lw=2,ls='--')
  plt.loglog(dic['r'],dic['pgas_ave'],lw=2,ls='-')
  plt.ylabel(r'$P_{\rm g}$')
  plt.ylim(1e-6,1e-2)
  plt.xlim(1,50)


  subplot(223)
  plt.loglog(r[-1,:],beta_inv[500:,:].mean(0),lw=2,ls='--')
  plt.loglog(dic['r'],1/dic['beta_inv_ave'],lw=2,ls='-')
  plt.xlabel(r'$r$ $[r_{\rm g}]$')

  plt.ylabel(r'$\beta^{-1}$')
  plt.ylim(1e-2,1e0)
  plt.xlim(1,50)

  subplot(224)
  plt.loglog(r[-1,:],pmag[500:,:].mean(0),lw=2,ls='--')
  plt.loglog(dic['r'],dic['pmag_ave'],lw=2,ls='-')

  plt.ylabel(r'$P_{B}$')
  #plt.ylim()
  plt.xlim(1,50)
  plt.xlabel(r'$r$ $[r_{\rm g}]$')

  plt.tight_layout()



  Edotmean = (Edot/abs(mdot))[:,ir][500:].mean()
  Edotstd = (Edot/abs(mdot))[:,ir][500:].std()

  phibhmean = abs(Phibh/sqrt(-mdot)/sqrt(4*pi))[:,ir][500:].mean()
  phibhstd = abs(Phibh/sqrt(-mdot)/sqrt(4*pi))[:,ir][500:].std()

  print("Edot/Mdot: ", Edotmean,"+/- ",Edotstd)
  print("phibh: ", phibhmean,"+/- ",phibhstd)





def Harrison_plots():

  c = 2.99792458e10
  gg_msun = 1.3271244e26
  kb = 1.380649e-16
  kpc = 3.085677581491368e21
  distance_kpc = 8.178
  nu = 230.0e9
  dic = np.load("star_wind_1e6_select.npz")


  I1 = dic['iquv_nu_select'][0,0] * c**2 / (2.0 * nu**2 * kb)
  I2 = dic['iquv_nu_select'][1,0] * c**2 / (2.0 * nu**2 * kb)
  I3 = dic['iquv_nu_select'][2,0] * c**2 / (2.0 * nu**2 * kb)
  I4 = dic['iquv_nu_select'][3,0] * c**2 / (2.0 * nu**2 * kb)

  Imax = 1.2
  Imin = 0.0

  n =1
  for I in [I1,I2,I3,I4]:
    plt.figure(figsize=(8,8))
    plt.clf()
    contourf(I/1e10,levels=np.linspace(Imin,Imax,400),cmap = "inferno")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("I%d.png" %n)
    n = n + 1

  plt.figure(figsize=(8,8))
  plt.clf()
  plt.subplot(221)
  contourf(I1/1e10,levels=np.linspace(Imin,Imax,400),cmap = "inferno")
  plt.axis('off')
  plt.subplot(222)
  contourf(I2/1e10,levels=np.linspace(Imin,Imax,400),cmap = "inferno")
  plt.axis('off')
  plt.subplot(223)
  contourf(I4/1e10,levels=np.linspace(Imin,Imax,400),cmap = "inferno")
  plt.axis('off')
  plt.subplot(224)
  contourf(I3/1e10,levels=np.linspace(Imin,Imax,400),cmap = "inferno")
  plt.axis('off')

  plt.tight_layout()
  plt.savefig("I_panel.png")

def gamma_rel(theta):
  return (10.0 + 20.0*theta)/(6.0 + 15.0*theta)
def ue_to_kappae(ue,rho,mue=2.0):
  mp_over_me = 1836.15267507
  cl = 306.4

  rhoe = rho/mp_over_me/mue 
  urat = ue/(rhoe *cl**2);
  theta_e = 1.0/30.0 * (-6.0 + 5.0 * urat + np.sqrt(36.0 + 180.0*urat +25.0*(urat**2)) ) ;
  return theta_e**(3.0/2.0) *  (theta_e + 2.0/5.0)**(3.0/2.0) / rhoe;

def get_Te_Tg(kappa,rho,gr=False):
  global Te, Tg
  mp_over_kev = 9.994827
  mp_over_me = 1836.15267507
  mue = 2.0;
  set_constants()
  cl = 306.4
  rhoe = rho/mp_over_me/mue;
  theta_e = 1.0/5.0 * (np.sqrt(1.0 + 25.0*(rhoe*kappa)**(2.0/3.0)) -1.0 )
  Te = theta_e * cl**2 /mp_over_me * mp_over_kev 
  Tg = press/rho * mp_over_kev * mu_highT
  if (gr==True):
    Te = theta_e/mp_over_me 
    Tg = press/rho * mu_highT
def kappa_to_ue(kappa,rho,gr=False,mue=2.0):
  global theta_e
  mp_over_me = 1836.15267507
  cl = 306.4
  if (gr==True): cl = 1.0

  rhoe = rho/mp_over_me/mue;
  theta_e = 1.0/5.0 * (np.sqrt(1.0 + 25.0*(rhoe*kappa)**(2.0/3.0)) -1.0 )
  pe_ = rhoe * theta_e * (cl**2.0);
  return pe_ / (gamma_rel(theta_e) - 1.0); 

  
def gravity_term_gr(r,th,a,m=1):
  global aterm,pressterm,EMterm, advection_term,massterm,centrifugal_term
  global pressterm_EM,tensionterm_EM
  get_mdot(mhd=True,gr=True,a=a)
  ks_metric(r,th,a)
  ks_Gamma_ud(r,th,a)

  ud_ks = nan_to_num(Lower(uu_ks,g))
  bd_ks = nan_to_num(Lower(bu_ks,g))
  Tud_calc(uu_ks,ud_ks,bu_ks,bd_ks,is_magnetic = True,gam=5.0/3.0)
  gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )

  TudKinetic = 0 * Tud
  gam = 5./3.
  Tudpress = 0*Tud
  Tudpress_EM = 0*Tud
  Tudtension_EM = 0 * Tud
  for kapa in np.arange(4):
    for nu in np.arange(4):
      TudKinetic[kapa,nu] = (rho+press*(gam/(gam-1.0) ))*uu_ks[kapa]*ud_ks[nu]
      Tudpress[kapa,nu] = press*(kapa==nu)
      Tudpress_EM[kapa,nu]  = bsq/2.0 * (kapa==nu)
      Tudtension_EM[kapa,nu]  = - bu_ks[kapa]*bd_ks[nu]
  aterm = 0
  pressterm = 0
  pressterm_EM = 0
  advection_term= 0
  EMterm = 0
  centrifugal_term = 0
  massterm = 0
  tensionterm_EM = 0
  for i in arange(4):
    for j in arange(4):
      if(i==0 and j==0): aterm+= (gdet * TudKinetic[j][i]) * gammaud[i][1][j][:,:,None]
      else: advection_term += (gdet * TudKinetic[j][i]) * gammaud[i][1][j][:,:,None]
      EMterm += (gdet * TudEM[j][i]) * gammaud[i][1][j][:,:,None]
      pressterm += (gdet * Tudpress[j][i] * gammaud[i][1][j][:,:,None])
      pressterm_EM += (gdet * Tudpress_EM[j][i] * gammaud[i][1][j][:,:,None])
      tensionterm_EM += (gdet * Tudtension_EM[j][i] * gammaud[i][1][j][:,:,None])
      if ((i==3 and j==3) or (i==2 and j==2)): centrifugal_term += (gdet * TudKinetic[j][i]) * gammaud[i][1][j][:,:,None]

      massterm += (gdet * Tud[j][i]) * gammaud[i][1][j][:,:,None]
  #centrifugal_term += (gdet * TudKinetic[3][3]) * gammaud[3][1][3][:,:,None]
  EMterm -= np.gradient(gdet * TudEM[1][1],r[:,0,0],axis=0) + np.gradient(gdet * TudEM[2][1],th[0,:,0],axis=1)
  EMterm -= np.gradient(gdet * TudEM[3][1],ph[0,0,:],axis=2)
  tensionterm_EM -= np.gradient(gdet * Tudtension_EM[1][1],r[:,0,0],axis=0) + np.gradient(gdet * Tudtension_EM[2][1],th[0,:,0],axis=1)
  tensionterm_EM -= np.gradient(gdet * Tudtension_EM[3][1],ph[0,0,:],axis=2)
  pressterm -= np.gradient(press*gdet,r[:,0,0],axis=0)
  pressterm_EM -= np.gradient(bsq/2.0*gdet,r[:,0,0],axis=0)
  advection_term -= np.gradient(gdet * TudKinetic[1][1],r[:,0,0],axis=0) + np.gradient(gdet * TudKinetic[2][1],th[0,:,0],axis=1)
  advection_term -= np.gradient(gdet * TudKinetic[3][1],ph[0,0,:],axis=2)
  massterm -= np.gradient(gdet * Tud[1][1],r[:,0,0],axis=0) + np.gradient(gdet * Tud[2][1],th[0,:,0],axis=1)
  massterm -= np.gradient(gdet * Tud[3][1],ph[0,0,:],axis=2)
def Edot_terms_gr(r,th,a,m=1):
  global Edot,EdotEM,EdotMA
  get_mdot(mhd=True,gr=True,a=a)
  ks_metric(r,th,a)

  ud_ks = nan_to_num(Lower(uu_ks,g))
  bd_ks = nan_to_num(Lower(bu_ks,g))
  Tud_calc(uu_ks,ud_ks,bu_ks,bd_ks,is_magnetic = True,gam=5.0/3.0)
  mdot =rho*uu_ks[1]*4.0 * np.pi/3.0 * (3.0*r**2 + a**2)
  Edot = - (Tud[1][0]*4.0 * np.pi/3.0 * (3.0*r**2 + a**2) +mdot )
  EdotEM = -(TudEM[1][0]*4.0 * np.pi/3.0 * (3.0*r**2 + a**2))
  EdotMA = Edot - EdotEM

def force_terms_gr(r,th,a):
  global EMforce, press_force
  get_mdot(mhd=True,gr=True,a=a)
  ks_metric(r,th,a)
  ud_ks = nan_to_num(Lower(uu_ks,g))
  bd_ks = nan_to_num(Lower(bu_ks,g))
  Tud_calc(uu_ks,ud_ks,bu_ks,bd_ks,is_magnetic = True,gam=5.0/3.0)
  gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )
  EMforce = np.gradient(gdet * Tud[1][1],r[:,0,0],axis=1)

def get_uu_ff(r,th,a):
  global uu_ff_ks, ud_ff_ks

  uu_ff_bl = 0*uu
  uu_ff_bl[0] = (r**2+a**2+2.0*a**2/r)/(r**2+a**2-2.0*r)
  uu_ff_bl[1] = - sqrt(2.0*(1.0/r)+a**2/r**3) 
  uu_ff_bl[3] = (2*a/r)/(r**2+a**2-2.0*r)


  ut_m1_bl =2.0/r * (a**2+r**2)/(a**2+r**2-2.0*r) 
  Delta = r**2 -2.0*r + a**2
  ut_m1_ks = ut_m1_bl + uu_ff_bl[1] * 2.*r/Delta


  uu_ff_ks = bl_vec_to_ks(uu_ff_bl,a=a)
  ks_metric(r,th,a=a)
  ud_ff_ks = Lower(uu_ff_ks,g)



def udm1_approx(r,th,a):
  global ud_t_p1
  ks_inverse_metric(r,th,a=a)
  v1 = uu_ks[1] - gi[0,1]/gi[0,0] * uu_ks[0]
  v2 = uu_ks[2] - gi[0,2]/gi[0,0] * uu_ks[0]
  v3 = uu_ks[3] - gi[0,3]/gi[0,0] * uu_ks[0]


  ks_metric(r,th,a)
  tmp = g[1,1]*v1*v1 + 2.0*g[1,2]*v1*v2 + 2.0*g[1,3]*v1*v3+ g[2,2]*v2*v2 + 2.0*g[2,3]*v2*v3+ g[3,3]*v3*v3
  gamma = np.sqrt(1.0 + tmp);

  sigma = r**2 + a**2.*np.cos(th)**2
  gi_tt = -(1.0 + 2.0*r/sigma);
  g_tt = -(1.0 - 2.0*r/sigma);

  alpha = np.sqrt(-1.0/gi[0,0])

  A = 2.0*r/sigma
  alphai_approx = 1.0 + A/2.0 - A**2.0 /8.0 
  gamma_approx =  1.0 + tmp/2.0 - tmp**2.0/8.0 

  term_1 = A/2.0 - A**2.0 /8.0 
  term_2 = tmp/2.0 - tmp**2.0/8.0 
  gamma_over_alpha_approx = 1.0  + term_1 + term_2 +A*tmp/4.0

  term_3 = - 2.0*r/sigma
  ud_term_1_approx = -(1.0 +term_3 + term_1 + term_2 + A*tmp/4.0 + term_3*A/2.0 + term_3*tmp/2.0 )



  ud_t_p1  = -(term_3 + term_1 + term_2 + A*tmp/4.0 + term_3*A/2.0 + term_3*tmp/2.0 )

  for i in arange(1,4):
    ud_t_p1 += g[i,0]*uu_ks[i]


def get_conv_flux():


  Bx_dir = "/global/scratch/smressle/star_cluster/gr_magnetically_frustrated/Bx_192_stampede"
  Bz_dir  = "/global/scratch/smressle/star_cluster/gr_magnetically_frustrated/Bz_192_stampede"
  sixty_degree_dir = "/global/scratch/smressle/star_cluster/gr_magnetically_frustrated/60_degree_192_stampede"
  thirty_degree_dir = '/global/scratch/smressle/star_cluster/gr_magnetically_frustrated/30_degree_192_stampede'

  dir_list = [Bz_dir,thirty_degree_dir, sixty_degree_dir, Bx_dir ]
  for i in arange(1,5):
    plt.figure(i)
    plt.clf()
    os.chdir(dir_list[i-1])
    rdnpz('dump_spher_600_th_0_phi_0.npz')
    a = 0.9375
    get_mdot(mhd=True,gr=True,a=0.9375)
    ks_metric(r,th,a=0.9375)
    ud_ks = Lower(uu_ks,g)
    bd_ks = Lower(bu_ks,g)
    gam = 5.0/3.0
    w = 1.0 + press/rho * gam/(gam-1.0)
    Be = -ud_ks[0]*w  - 1.0

    Tud_calc(uu_ks,ud_ks,bu_ks,bd_ks,is_magnetic=True,gam =gam)

    FMA = -(TudMA[1][0] + rho * uu_ks[1]) * 4.0*np.pi/3.0 * (3.0*r**2+a**2)
    FEM = -(TudEM[1][0] ) * 4.0*np.pi/3.0 * (3.0*r**2+a**2)

    angle_limit = abs(th-pi/2) < 30.0/180.0*pi
    scale_height = angle_average_npz(abs(th-pi/2),weight=rho,gr=True,a=0.9375)
    scale_height_3D = scale_height[:,None,None] * (rho/rho)
    angle_limit = abs(th-pi/2) < scale_height_3D
    Edot_ma = angle_average_npz(FMA,weight=angle_limit,gr=True,a=a)
    mdot = angle_average_npz(rho * uu_ks[1] * 4.0*np.pi/3.0 * (3.0*r**2+a**2),weight=angle_limit,gr=True,a=a)
    Be_avg = angle_average_npz(Be,weight=angle_limit,gr=True,a=a)
    Be_avg_rho = angle_average_npz(Be*angle_limit,weight=rho,gr=True,a=a)

    Edot_EM = angle_average_npz(FEM,weight=angle_limit,gr=True,a=a)

    Edot_conv = Edot_ma - Be_avg* mdot 

    plt.loglog(r[:,0,0],-Edot_ma,lw=2,ls='-',label = r'$-F_{\rm M}$')
    plt.loglog(r[:,0,0],Edot_conv,lw=2,ls='--',label = r'$F_{\rm conv}$')
    plt.loglog(r[:,0,0],-Be_avg*mdot,lw=2,ls=':',label = r'$-F_{\rm adv}$')
    plt.loglog(r[:,0,0],Edot_EM,lw=2,ls='-.',label = r'$F_{\rm EM}$')
    plt.xlim(1e0,1e3)
    plt.ylim(1e0,1e4)

    plt.legend(loc='best',frameon=0,fontsize=12)


def get_gr_bondi_sol(r_bondi):
  #r_b = 2 GM/ a_inf^2
  global mdotgr_bondi,rho,T,cs,ur,r,rc,ut,grr,gtt
  gam = 5.0/3.0
  n = 1.0/(gam-1.0)
  a_inf = sqrt(2/r_bondi)
  T_inf = a_inf**2.0/gam

  def eqn_(rc):
    Tc = - n/(n**2 -2.0*(n+1.0)*rc +4.0*n+3.0)
    uc = sqrt(1.0/(2.0*rc))
    C1 = uc * Tc**n * rc**2
    C2 = (1.0 + (1.0+n)*Tc)**2.0 * (1.0-2.0/rc + C1**2/(rc**4*Tc**(2.0*n)))
    C2_prime = (1.0+(1.0+n)*T_inf)**2.0
    return C2-C2_prime

  rc = scipy.optimize.fsolve(eqn_,5.0)

  uc = sqrt(1.0/(2.0*rc))
  Tc = - n/(n**2 -2.0*(n+1.0)*rc +4.0*n+3.0)
  C1 = uc * Tc**n * rc**2
  C2 = (1.0 + (1.0+n)*Tc)**2.0 * (1.0-2.0/rc + C1**2/(rc**4*Tc**(2.0*n)))
  r = np.logspace(np.log10(2.0),4,1000)

  def eqn_(T,r):
      return (((1 + (1+n)*T)**2*(1 -2.0/r + C1**2/r**4/T**(2*n)) - C2))[0]
  T_guess = 2*T_inf
  T = []
  for r_val in r[::-1]:
      T_new = scipy.optimize.fsolve(eqn_,T_guess,args=(r_val))[0]
      T.append(T_new)
      T_guess = T_new
  T = np.array(T)[::-1]
  ur = C1/(T**n*r**2)
  grr = 1./(1.0-2.0/r)
  gtt = -(1.0-2.0/r)
  ut = np.sqrt((-1 - ur**2.0*grr)/gtt)
  gam = 5.0/3.0

  rho = (T/T_inf)**n
  cs = sqrt(gam*T*rho/(gam/(gam-1.0)*rho*T + rho))
  mdotgr_bondi = rho * ur * r**2.0 * 4.0*pi
  v_loc = sqrt(grr/abs(gtt))*ur/ut



def pcolormesh_corner(r,th,myvar,coords = 'xz',flip_x = False,**kwargs):
  r_face = np.logspace(log10(np.amin(r)),log10(np.amax(r)),nx+1)
  th_face = np.linspace(0,pi,ny+1) 
  ph_face = np.linspace(0,2.0*pi,nz+1)
  if (coords =='xz'):
    x_corner = r_face[:,None] * np.sin(th_face[None,:])
    y_corner = r_face[:,None] * np.cos(th_face[None,:])
  else:
    x_corner = r_face[:,None] * np.cos(ph_face[None,:])
    y_corner = r_face[:,None] * np.sin(ph_face[None,:])
  if (flip_x==True): x_corner = x_corner * -1.0

  return plt.pcolormesh(x_corner,y_corner,myvar,**kwargs)

def gravity_term_bondi(rb,a,m=1):
  global aterm,pressterm, advection_term,massterm,centrifugal_term
  get_gr_bondi_sol(rb)
  global r,th,ph  
  th = np.linspace(0,pi,200)
  ph = np.linspace(0,2*pi,200)
  r,th,ph = np.meshgrid(r,th,ph,indexing='ij')
  global nx,ny,nz
  nx = r.shape[0]
  ny = r.shape[1]
  nz = r.shape[2]
  ks_metric(r,th,a)
  ks_Gamma_ud(r,th,a)

  global rho,T,ut,ur, press

  press = (rho*T)[:,None,None]*(r/r)
  rho = rho[:,None,None]*(r/r)

  global uu_bl,uu_ks,ud_ks,bd_ks
  uu_bl = np.array([ut[:,None,None]*(r/r),-ur[:,None,None]*(r/r),0*r,0*r])
  uu_ks = bl_vec_to_ks(uu_bl,a)

  ud_ks = nan_to_num(Lower(uu_ks,g))
  bd_ks = nan_to_num(Lower(uu_ks,g))
  global Tud,Tudpress,Tudkinetic

  Tud_calc(uu_ks,ud_ks,uu_ks,bd_ks,is_magnetic = False,gam=5.0/3.0)
  gdet = np.sqrt( np.sin(th)**2.0 * ( r**2.0 + a**2.0*np.cos(th)**2.0)**2.0 )

  TudKinetic = 0 * Tud
  gam = 5./3.
  Tudpress = 0*Tud
  for kapa in np.arange(4):
    for nu in np.arange(4):
      TudKinetic[kapa,nu] = (rho+press*(gam/(gam-1.0)))*uu_ks[kapa]*ud_ks[nu]
      Tudpress[kapa,nu] = press*(kapa==nu)
  aterm = 0
  pressterm = 0
  advection_term= 0
  centrifugal_term = 0
  massterm = 0
  for i in arange(4):
    for j in arange(4):
      if(i==0 and j==0): aterm+= (gdet * TudKinetic[j][i]) * gammaud[i][1][j][:,:,None]
      else: advection_term += (gdet * TudKinetic[j][i]) * gammaud[i][1][j][:,:,None]
      pressterm += (gdet * Tudpress[j][i] * gammaud[i][1][j][:,:,None])
      if ((i==3 and j==3) or (i==2 and j==2)): centrifugal_term += (gdet * TudKinetic[i][j]) * gammaud[i][1][j][:,:,None]
      massterm += (gdet * Tud[j][i]) * gammaud[i][1][j][:,:,None]
  #centrifugal_term += (gdet * TudKinetic[3][3]) * gammaud[3][1][3][:,:,None]


  pressterm -= np.gradient(press*gdet,r[:,0,0],axis=0)
  advection_term -= np.gradient(gdet * TudKinetic[1][1],r[:,0,0],axis=0) + np.gradient(gdet * TudKinetic[2][1],th[0,:,0],axis=1)
  advection_term -= np.gradient(gdet * TudKinetic[3][1],ph[0,0,:],axis=2)
  massterm -= np.gradient(gdet * Tud[1][1],r[:,0,0],axis=0) + np.gradient(gdet * Tud[2][1],th[0,:,0],axis=1)
  massterm -= np.gradient(gdet * Tud[3][1],ph[0,0,:],axis=2)


def make_Veronica_npz():

  f1 = "/global/scratch/smressle/star_cluster/cuadra_comp/without_S2_9_levels_new_orbits_Z_3_no_H_solar_intel/star_wind.hst"
  t_i = -0.1
  t_f = 0.00

  k_green = '#4cbb17'
  silver = '#8D9093'
  midnight_green = '#004851'
  charcoal = '#54585A'
  dark_gray = '#A9A9A9'
  light_blue = "#ADD8E6"

  arc_secs = 0.03878512
  r_in = 2.*2./128./2.**8.
  r_in = 2.*2./128./2.**9.
  #r_in = 2.*2./128./2.**10.

  simulation_start_time =  - 1.1

  t_i = t_i - simulation_start_time
  t_f = t_f - simulation_start_time

  rd_hst(f1)
  rho_unit = 6.768e-23 #g/cm^3  
  v_unit = 9.785e7
  L_unit = 3.086e18
  t_unit = 3.154e10
  iti1,itf1 = t_to_it(t_i),t_to_it(t_f)
  def tavg(var):
    return var[iti1:itf1,:].mean(0)
  rho_1d = rho_avg[iti1:itf1,:-3].mean(0) * rho_unit
  cs_1d = np.sqrt(5./3.*p_avg[iti1:itf1,:-3].mean(0)/rho_avg[iti1:itf1,:-3].mean(0)) * v_unit
  vr_abs_1d = abs((mdot_avg[iti1:itf1,:-3].mean(0)/rho_avg[iti1:itf1,:-3].mean(0)/(r**2.)[iti1:itf1,:-3].mean(0)))/np.pi/4. * v_unit
  mdot_1d = abs((mdot_avg[iti1:itf1,:-3].mean(0))) * rho_unit*L_unit**3.0 /t_unit
  press_1d = p_avg[iti1:itf1,:-3].mean(0) * rho_unit * v_unit**2.0
  L_1d = np.sqrt( tavg(Lx_avg)**2. + tavg(Ly_avg)**2. + tavg(Lz_avg)**2.)[:-3] * rho_unit * v_unit * L_unit
  Ldot_1d = np.sqrt( tavg(Lxdot_avg)**2. + tavg(Lydot_avg)**2. + tavg(Lzdot_avg)**2.)[:-3] * rho_unit * v_unit * L_unit/t_unit * L_unit**3.0
  Edot_1d = Edot_avg[iti1:itf1,:-3].mean(0) * rho_unit * v_unit**2/t_unit * L_unit**3.0


  dic = {"r":r[iti1:itf1,:-3].mean(0)*L_unit,
          "rho":rho_1d,
          "cs":cs_1d,
          "vr_abs":vr_abs_1d,
          "mdot":-mdot_1d,
          "press":press_1d,
          "L":L_1d,
          "Jdot":Ldot_1d,
          "Edot":Edot_1d }

  np.savez("hydro_wind_1d.npz",**dic)



def two_fes():
  global fe_tot, fe_ion
  set_constants()
  mrat = 1836.152672
  beta = press/bsq*2.0
  ue = kappa_to_ue(ke_ent,rho,gr=True)
  Te = theta_e/mrat
  Ttot = press/rho * mu_highT
  fe_tot = fe_howes(beta,Te,Ttot)
  mue = 2.0
  mui = 1.0/(1.0/mu_highT - 1.0/mue)

  pe = theta_e/mrat *rho /mue;
  pi = press - pe
  pi[pi<0] = press[pi<0] * 0.01
  Ti = pi/rho * mui
  betai = 2.0 * pi/(bsq + 1e-15)
  fe_ion = fe_howes(betai,Te,Ti)

def fe_howes(beta,Te, Ttot):
  mrat = 1836.152672;
  Te[Te<1e-15] = 1e-15
  Trat = abs(Ttot/Te)
  c1 = .92 
  
  beta[isnan(beta)] = 1e20
  beta[isinf(beta)] = 1e20 
  beta[beta>1e20] = 1e20
  mbeta = 2.-.2*log10(Trat)

  c2 = 1.6 / Trat * (Trat<=1) + 1.2/ Trat *(Trat>1)
  c3 = (18. + 5.*log10(Trat)) * (Trat<=1) + (18.0) * (Trat>1)
  
  c22 = c2**2.0
  c32 = c3**2.0
  
  Qp_over_Qe = c1 * (c22+beta**mbeta)/(c32 + beta**mbeta) * exp(-1./beta)*(mrat*Trat)**0.5 ;
  

  
  return 1./(1.+Qp_over_Qe);



def ud_zamo(a):
  ks_inverse_metric(r,th,a)
  return np.array([-1.0/sqrt(-gi[0][0]),0.0*r,0.0*r,0.0*r])

def wald_solution():
  r2 = r**2.
  a2 = a**2.
  s2 = np.sin(th)**2.
  c2 = np.cos(th)**2.
  sigma = r**2 + a**2.*np.cos(th)**2
  Delta = r**2 -2.0*r + a**2
  Ex = -a * r *s2/sigma - a*(r2-a2*c2)*(1+c2)/sigma**2 
  Ey = -sqrt(Delta)*a*sin(th)*cos(th)/sigma
  Ez = -0*r
  Bx = -cos(th)/sigma * (r2+a2-2*r*a2*(1+c2)/sigma) 
  By = sqrt(Delta)*r*sin(th)/sigma
  Bz = 0*r

  bsq = Bx**2+By**2+Bz**2 - Ex**2 - Ey**2 -Ez**2


def energy_spectrum():
  gr = False
  yt_extract_box(125,box_radius = 1,gr=gr)
  ##yt_extract_box(1900,box_radius = 40,gr=gr)
  Lx = np.amax(x)-np.amin(x)
  Ly = np.amax(y)-np.amin(y)
  Lz = np.amax(z)-np.amin(z)
  dx = np.diff(x[:,0,0])[0]
  dy = dx*1.0
  dz = dx*1.0
  if (gr==False):
    vel1_tilde = np.fft.fftn(sqrt(rho)*vel1) * dx**3.0
    vel2_tilde = np.fft.fftn(sqrt(rho)*vel2) * dx**3.0
    vel3_tilde = np.fft.fftn(sqrt(rho)*vel3) * dx**3.0
  else:
    def SQR(var):
      return var**2.0
    R = np.sqrt(SQR(x) + SQR(y) + SQR(z));
    r = SQR(R) - SQR(a) + np.sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
    r = np.sqrt(r/2.0);
    th = np.arccos(z/r)
    vel1_tilde = np.fft.fftn(sqrt(rho)*uu[1]/uu[0]) * dx**3.0
    vel2_tilde = np.fft.fftn(sqrt(rho)*uu[2]/uu[0] * r) * dx**3.0
    vel3_tilde = np.fft.fftn(sqrt(rho)*uu[3]/uu[0] * r * sin(th)) * dx**3.0
  # vel1_tilde = np.fft.fftn(vel1)
  # vel2_tilde = np.fft.fftn(vel2)
  # vel3_tilde = np.fft.fftn(vel3)
  E_tilde_vec = 0.5*(vel1_tilde*np.conj(vel1_tilde) + vel2_tilde*np.conj(vel2_tilde)+ vel3_tilde*np.conj(vel3_tilde))

  k1 = np.fft.fftfreq(nx,d=dx) #np.linspace(0,kmax,nx)#/Lx
  k2 = np.fft.fftfreq(ny,d=dy) #np.linspace(0,kmax,ny)#/Ly
  k3 = np.fft.fftfreq(nz,d=dz) #np.linspace(0,kmax,nz)#/Lz
  dk = np.diff(k1)[0]
  k1,k2,k3 = np.meshgrid(k1,k2,k3,indexing='ij')
  ksq = k1**2 + k2**2 + k3**2
  kr = np.sqrt(ksq)
  kph = np.arctan2(k2,k1)
  th_k = np.arccos(k3/sqrt(ksq))

  kmax = np.amax(sqrt(ksq))
  kr_new = np.linspace(k1[1,0,0],kmax,nx*4+1)[:-1] + np.diff(np.linspace(k1[1,0,0],kmax,nx*4+1))[0]
  kth_new = np.linspace(0,pi,ny+1)[:-1] + np.diff(np.linspace(0,pi,ny+1))[0]
  if(nz==1): kth_new = np.array([np.pi/2])
  if (nz>1): kph_new = np.linspace(0,pi*2.0,nz+1)[:-1] + np.diff(np.linspace(0,pi*2.0,nz+1))[0]
  else: kph_new = np.linspace(0,pi*2.0,ny+1)[:-1] + np.diff(np.linspace(0,pi*2.0,ny+1))[0]


  kr_new,kth_new,kph_new = np.meshgrid(kr_new,kth_new,kph_new,indexing='ij')

  k1_new = kr_new * np.sin(kth_new) * np.cos(kph_new)
  k2_new = kr_new * np.sin(kth_new) * np.sin(kph_new)
  k3_new = kr_new * np.cos(kth_new)

  E_tilde_vec_spher = scipy.interpolate.griddata((k1.flatten(),k2.flatten(),k3.flatten()),E_tilde_vec.flatten(),
    (k1_new,k2_new,k3_new), method='nearest',fill_value = 0.0)


  dr_k = np.diff(kr_new[:,0,0])[0]
  if (nz>1): dth_k = np.diff(kth_new[0,:,0])[0]
  else: dth_k = 1.0
  dph_k = np.diff(kph_new[0,0,:])[0]

  E_tilde = (E_tilde_vec_spher * kr_new**2 * sin(kth_new) * dth_k * dph_k).sum(-1).sum(-1) 


def Okazaki(alpha=0.2,a=0.1):
  #R_crit = 
  #vphi_crit = np.sqrt(1.0/R_crit - 5.0/2.0*a**2) 
  vr_crit = a
  vr_0_guess = 8.7e-3*a #a/(R_crit - (1-vphi_crit)/(alpha*a))
  vphi_0 = 1.0




  R_crit_max = 2.0/5.0 * 1/a**2.0 
  R_crit_min = 15.0


  def get_sol(R_crit,full_range=False):
    global sol
    vphi_crit = np.sqrt(1.0/R_crit - 5.0/2.0*a**2) 
    l_crit = vphi_crit * R_crit 

    v_r0 = - ( (1.0 - l_crit)/(alpha * a**2) - R_crit/a )**-1
    if (full_range==False): r_vec = np.logspace(0,log10(R_crit),200)
    else: r_vec = np.logspace(0,log10(500),200)
    def dvr_dr(v_r,v_phi,r):
      return (-1.0/r**2 + v_phi**2/r + 5.0/2.0*a**2/r)/(v_r - a**2.0/v_r)
    def func(r,v):
      v_r = v
      #v_phi = v[1]
      #v_phi = vphi_crit*R_crit/r + alpha * a**2/r  * (R_crit/a - r/v_r)
      v_phi = ( 1.0/r + alpha * a**2/r * (1.0/v_r0 - r/v_r) ) * (r<=R_crit) + ( 1.0/R_crit + alpha * a**2/R_crit * (1.0/v_r0 - R_crit/a) )*R_crit/r * (r>R_crit)

      #dvr_dr = (-1.0/r**2 + v_phi**2/r + 5.0/2.0*a**2/r)/(v_r - a**2.0/v_r)
      dvrdr = -1/2*(5*a**2*r + 2*r*v_phi**2 - 2)*v_r/(a**2*r**2 - r**2*v_r**2)
      #dvphi_dr = -1/2*(7*a**4*alpha*r + 2*a**2*alpha*r*v_phi**2 - 2*a**2*alpha*r*v_r**2 + 2*a**2*r*v_phi*v_r - 2*r*v_phi*v_r**3 - 2*a**2*alpha)/(a**2*r**2*v_r - r**2*v_r**3)
      return dvrdr #dvr_dr(v_r,v_phi,r)
    y_0 = [v_r0]

    from scipy.integrate import solve_ivp

    def dfunc(r,v):
      v_r = v
      return 2*(a**2*alpha*(r/v_r - 1/v_r0)/r - 1/r)*a**2*alpha*r/((a**2*r**2 - r**2*v_r**2)*v_r) - (2*(a**2*alpha*(r/v_r - 1/v_r0)/r - 1/r)**2*r + 5*a**2*r - 2)*r**2*v_r**2/(a**2*r**2 - r**2*v_r**2)**2 - 1/2*(2*(a**2*alpha*(r/v_r - 1/v_r0)/r - 1/r)**2*r + 5*a**2*r - 2)/(a**2*r**2 - r**2*v_r**2)


    sol = solve_ivp(func,(r_vec[0],r_vec[-1]),y_0,t_eval=r_vec) #Dfun=dfunc# ,Dfun=dfunc)
    #vphi_sol = vphi_crit*R_crit/r_vec + alpha * a**2/r_vec  * (R_crit/a - r_vec/sol[0])
    vr_sol = sol['y'][0,:]
    vphi_sol = (1.0/sol['t'] + alpha * a**2/sol['t'] * (1.0/v_r0 - sol['t']/sol['y'][0,:]) ) * (sol['t']<=R_crit) + ( 1.0/R_crit + alpha * a**2/R_crit * (1.0/v_r0 - R_crit/a) )*R_crit/sol['t'] * (sol['t']>R_crit)
    return [sol['t'],vr_sol,vphi_sol]

  def objective(R_crit):
    r,vr_sol,vphi_sol = get_sol(R_crit)
    return vr_sol[-1]/a - 1

  import scipy

  root = []
  r_crit_arr = np.logspace(log10(R_crit_min),log10(R_crit_max*0.99),100)
  for r_crit in r_crit_arr:
    root.append(objective(r_crit))
  root = np.array(root)
  for i in arange(len(root)):
    if (root[i] >0 ): break
  R_crit = scipy.optimize.bisect(objective, R_crit_min, r_crit_arr[i])

  r,vr,vphi = get_sol(R_crit,full_range=True)
  Sigma = 1.0/(vr * r)
  Sigma = Sigma/Sigma[0]
  vk = sqrt(1.0/r)
  H = r * (a/vk)
  rho = Sigma/H
  rho = rho/rho[0]

  return [r,vr,vphi,rho]


# import os
# import numpy as np

# dir_list = ["/scratch/03496/smressle/chris_archived/jet_resolution/jet_0",
#             "/scratch/03496/smressle/chris_archived/jet_resolution/jet_1",
#             "/scratch/03496/smressle/chris_archived/jet_resolution/jet_2",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_plm_0",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_plm_1",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_plm_2",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_plm_3",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_ppm_0",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_ppm_1",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_ppm_2",
#             "/scratch/03496/smressle/chris_archived/mad_resolution/mad_ppm_3",
#             "/scratch/03496/smressle/chris_archived/steady_state/sane_base",
#             "/scratch/03496/smressle/chris_archived/steady_state/sane_loop",
#             "/scratch/03496/smressle/chris_archived/steady_state/sane_loop_old",
#             "/scratch/03496/smressle/chris_archived/steady_state/sane_gamma",
#             "/scratch/03496/smressle/chris_archived/tilted_disks/s90_t00_hight"]

# chris_dir_list = ["/scratch/04939/cjwhite/archived/jet_resolution/jet_0",
#             "/scratch/04939/cjwhite/archived/jet_resolution/jet_1",
#             "/scratch/04939/cjwhite/archived/jet_resolution/jet_2",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_plm_0",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_plm_1",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_plm_2",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_plm_3",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_ppm_0",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_ppm_1",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_ppm_2",
#             "/scratch/04939/cjwhite/archived/mad_resolution/mad_ppm_3",
#             "/scratch/04939/cjwhite/archived/steady_state/sane_base",
#             "/scratch/04939/cjwhite/archived/steady_state/sane_loop",
#             "/scratch/04939/cjwhite/archived/steady_state/sane_loop_old",
#             "/scratch/04939/cjwhite/archived/steady_state/sane_gamma",
#             "/scratch/04939/cjwhite/archived/tilted_disks/s90_t00_hight"]

# for idir in np.arange(len(dir_list)):
#   dir_ = dir_list[idir]
#   chris_dir = chris_dir_list[idir]

#   os.chdir(dir_)
#   os.system("ls %s/*.tar | xargs -i tar -xf {}" %chris_dir)




# DX = 1600.0*2./192.
# dx_list = np.array([DX,DX/2,DX/2**2,DX/2**3,DX/2**4,DX/2**5,DX/2**6,DX/2**7,DX/2**8 ,DX/2**9])#,DX/2**10])
# r_lim_list = np.array([1.0,1.0/2.0,1.0/2.0**2,1.0/2.0**3,1.0/2.0**4,1.0/2.0**5,1.0/2.0**6,1.0/2.0**7,1.0/2.0**8,1.0/2.0**9]) #,1.0/2.0**10]
# r_lim_list = r_lim_list * 1600.

# dx_arr = rho*0 + DX

# for i in range(len(dx_list)):
#     r_lim = r_lim_list[i]
#     ind = (x<r_lim)*(x>-r_lim) * (y<r_lim)*(y>-r_lim) * (z<r_lim)*(z>-r_lim)
#     dx_arr[ind] = dx_list[i]

# plt.figure(figsize=(8,8*4))
# ir = r_to_ir(5.0)
# subplot(411)
# plot(t,-mdot[:,ir])
# ylim(0,60)
# ylabel(r'$|\dot M|$',fontsize=20)

# subplot(412)
# plot(t,(Phibh/sqrt(-mdot*4*pi))[:,ir])
# ylim(0,20)
# ylabel(r'$\phi$',fontsize=20)

# subplot(413)

# plot(t,((Edot+mdot)/(-mdot))[:,ir])
# ylim(-1,2)
# ylabel(r'$\dot E/|\dot M|$',fontsize=20)

# subplot(414)
# plot(t,(Jdot/(-mdot))[:,ir])
# ylabel(r'$\dot J/|\dot M|$',fontsize=20)
# ylim(-2,15)

# xlabel(r'$t$',fontsize=20)


# yt_load(120)
# from yt.visualization.volume_rendering.api import Scene, VolumeSource 
# import numpy as np
# sc  = Scene()
# vol = VolumeSource(ds, field="density")
# bounds = (10**-1, 10.**2)
# tf = yt.ColorTransferFunction(np.log10(bounds))
# def linramp(vals, minval, maxval):
#     return (vals - vals.min())/(vals.max() - vals.min())
# #tf.add_layers(8, colormap='ocean')
# tf.map_to_colormap(np.log10(bounds[0]),np.log10(bounds[1]),colormap='Blues_r',scale_func=linramp)
# #tf.add_layers(8, colormap='ocean')
# tf.grey_opacity = False
# vol.transfer_function = tf
# vol.tfh.tf = tf
# vol.tfh.bounds = bounds
# vol.tfh.plot('transfer_function.png', profile_field="density")
# cam = sc.add_camera(ds, lens_type='plane-parallel')
# cam.resolution = [512,512]
# # cam.position = ds.arr(np.array([282.*np.cos(theta)*np.sin(phi),282.*np.cos(theta)*np.cos(phi),282.*np.sin(theta)]), 'code_length')
# # cam.switch_orientation(normal_vector=normal_vector,
# #                        north_vector=north_vector)
# cam.set_width(ds.domain_width*0.2)

# cam.position = ds.arr(np.array([0,0,-0.5]), 'code_length')
# normal_vector = [0,0,-1]  #camera to focus
# north_vector = [0,1,0]  #up direction
# cam.switch_orientation(normal_vector=normal_vector,north_vector=north_vector)
# sc.add_source(vol)
# sc.render()
# # sc.save('tmp2.png',sigma_clip = 6.0)
# # sc = yt.create_scene(ds,lens_type = 'perspective')
# # sc.camera.zoom(2.0)
# # sc[0].tfh.set_bounds([1e-4,1e2])
# os.system("mkdir -p frames")
# sc.save(fname,sigma_clip = 4.0)
# plt.imshow(imread(fname))