#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

#include <iostream>


#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"

/* Problem Constants */
// Units
Real cm_to_pc = 3.24e-19;
Real g_to_msun = 5.03e-34;
Real s_to_kyr = 3.168e-11;
Real erg_to_code = g_to_msun*SQR(cm_to_pc/s_to_kyr);
Real icm3_to_code = 2.94e55;
// Problem 
Real pfloor, rhofloor;
Real vmax = 3/std::sqrt(3);        // ~1700 km/s
Real gas_gamma; 
Real kb = 1.38e-16*erg_to_code;    // per Kelvin
Real mh2 = 3.32e-24*g_to_msun;
Real GM;
// Torus parameters (calculared using Python)
Real R_in = 1.4; 
Real R_k = 1.9;
Real q = 1.56; 
Real T_tar = 1e3;
Real n_tar = 1.e5; //target number density in 1/cm3
Real rho_tar = n_tar*icm3_to_code*mh2;
// More parameters
Real C, K, P_amb, A; 
// Inner boundary
Real r_inner_boundary=0.0;
int n_mb = 0;
double SMALL = 1e-20;
bool amr_increase_resolution;
LogicalLocation *loc_list;
int max_refinement_level = 0; 

/* Disk pressure functions*/
// Gravitational potential
Real potential(Real r){
  return -GM/r;
}

// Centrifugal term
Real centrifugal(Real R){
  return SQR(C) * std::pow(R, -2*q + 2) / (2*q - 2);
}

// Integration constant 
Real integration_cons(){
  Real t1 = centrifugal(R_in);
  Real t2 = gas_gamma * P_amb*std::pow(K, 1/gas_gamma) / ((gas_gamma - 1) * std::pow(P_amb, 1/gas_gamma));
  Real t3 = potential(R_in);
  return -(t1 + t2 + t3);
}



/* Functions to deal with the inner boundary*/

// Floors out variables at the center of the simulation
void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
// Source function wrapper around 'apply_inner_boundary_condition'
static void integrate_cool(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim)
{
  apply_inner_boundary_condition(pmb,prim);
}

// Initialize variables
void Mesh::InitUserMeshData(ParameterInput *pin){
  pfloor = pin->GetReal("hydro", "pfloor");
  rhofloor = pin->GetReal("hydro", "dfloor");
  GM = pin->GetReal("problem", "GM");
  gas_gamma = pin->GetReal("hydro", "gamma");
  //EnrollUserExplicitSourceFunction(gravity);
  //EnrollUserExplicitSourceFunction(SourceFunc);
  EnrollUserRadSourceFunction(integrate_cool);

  C = std::sqrt(GM) * std::pow(R_k, q-1.5);
  K = (rho_tar * kb * T_tar) / (mh2 * std::pow(rho_tar, gas_gamma));
  P_amb = pfloor;
  A = integration_cons();
}


/* Functions to calculate the disk pressure */ 
// Used the values calculated from python script
// Torus constants
//Real C = 0.14325243622329745;//std::sqrt(GM) * std::pow(R_k, q-1.5);
//Real K = 1.4515744576165966e-07;//(rho_tar * kb * T_tar) / (mh2 * std::pow(rho_tar, gas_gamma));
//Real P_amb = 1.e-10;//pfloor;

//Real A = 0.0010017317369239685;//integration_cons();
//fprintf(stdout, " Integration constant is: " , A);

// Disk pressure
Real disk_press(Real r_vec[3]){
  // r_vec must be in cartesian coords i.e. <x,y,z>
  Real r = std::sqrt(SQR(r_vec[0]) + SQR(r_vec[1]) + SQR(r_vec[2]));
  Real R = std::sqrt(SQR(r_vec[0]) + SQR(r_vec[1]));
  Real factor = (gas_gamma - 1) / (gas_gamma * std::pow(K, 1/gas_gamma));
  Real terms  = -(A + potential(r) + centrifugal(R));
  // To avoid nans:
  if (terms < 0){
    terms = 0;
  }

  Real P = std::pow(factor * terms, gas_gamma / (gas_gamma-1));
  return std::max(P, P_amb);
}

// Source funtion to apply spped limits 
void SourceFunc(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

/* Useful functions*/
 
void get_uniform_box_spacing(const RegionSize box_size, Real *DX, Real *DY, Real *DZ){
  if (COORDINATE_SYSTEM == "cartesian"){
    *DX = (box_size.x1max-box_size.x1min)/(1. * box_size.nx1);
    *DY = (box_size.x2max-box_size.x2min)/(1. * box_size.nx2);
    *DZ = (box_size.x3max-box_size.x3min)/(1. * box_size.nx3);
  }
  else if (COORDINATE_SYSTEM == "cylindrical"){
    *DX = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DY = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DZ = (box_size.x3max-box_size.x3min)/(1. * box_size.nx3);
  }
  else if (COORDINATE_SYSTEM == "spherical_polar"){
    *DX = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DY = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
    *DZ = (box_size.x1max-box_size.x1min) *2./(1. * box_size.nx1);
  }
}

void get_minimum_cell_lengths(const Coordinates * pcoord, Real *dx_min, Real *dy_min, Real *dz_min){
  RegionSize block_size;
  enum BoundaryFlag block_bcs[6];

  *dx_min = 1e15;
  *dy_min = 1e15;
  *dz_min = 1e15;
  Real DX,DY,DZ;
  if (amr_increase_resolution){
    get_uniform_box_spacing(pcoord->pmy_block->pmy_mesh->mesh_size,&DX,&DY,&DZ);
    *dx_min = DX/std::pow(2.,max_refinement_level);
    *dy_min = DY/std::pow(2.,max_refinement_level);
    *dz_min = DZ/std::pow(2.,max_refinement_level);
    return;
   }
   block_size = pcoord->pmy_block->block_size;
   /* Loop over all mesh blocks by reconstructing the block sizes to find the block that the
   *          star is located in */
for (int j=0; j<n_mb; j++){
  pcoord->pmy_block->pmy_mesh->SetBlockSizeAndBoundaries(loc_list[j], block_size, block_bcs);
  get_uniform_box_spacing(block_size,&DX,&DY,&DZ);

  if (DX < *dx_min) *dx_min = DX;
  if (DY < *dy_min) *dy_min = DY;
  if (DZ < *dz_min) *dz_min = DZ;
  }
}

void get_cartesian_coords(const Real x1, const Real x2, const Real x3, Real *x, Real *y, Real *z){
  if (COORDINATE_SYSTEM == "cartesian"){
      *x = x1;
      *y = x2;
      *z = x3;
    }
    else if (COORDINATE_SYSTEM == "cylindrical"){
      *x = x1*std::cos(x2);
      *y = x1*std::sin(x2);
      *z = x3;
  }
    else if (COORDINATE_SYSTEM == "spherical_polar"){
      *x = x1*std::sin(x2)*std::cos(x3);
      *y = x1*std::sin(x2)*std::sin(x3);
      *z = x1*std::cos(x2);
    }
}

void get_cylindrical_coords(const Real x1, const Real x2, const Real x3, Real *r, Real *phi, Real *z){  if (COORDINATE_SYSTEM == "cartesian"){
    *r = std::sqrt(SQR(x1) + SQR(x2));
    *phi = std::atan2(x2,x1);
    *z = x3;
  }
  else if (COORDINATE_SYSTEM == "cylindrical"){
    *r = x1;
    *phi = x2;
    *z = x3;
  }
  else if (COORDINATE_SYSTEM == "spherical_polar"){
    *r = std::sqrt(SQR(x1*std::sin(x2)*std::cos(x3)) + SQR(x1*std::sin(x2)*std::sin(x3)));
    *phi = x3;
    *z = x1*std::cos(x2);
  }
}

void get_spherical_coords(const Real x1, const Real x2, const Real x3, Real *r, Real *theta, Real *phi){
  if (COORDINATE_SYSTEM == "cartesian"){
    *r = std::sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    *theta = std::acos(x3 / std::sqrt(SQR(x1) + SQR(x2) + SQR(x3)));
    *phi = std::atan2(x2, x1);
  }
  else if (COORDINATE_SYSTEM == "cylindrical"){
    *r = std::sqrt(SQR(x1) + SQR(x3));
    *theta = std::acos(x3 / std::sqrt(SQR(x1) + SQR(x3)));
    *phi = x2;
  }
  else if (COORDINATE_SYSTEM == "spherical_polar"){
    *r = x1;
    *theta = x2;
    *phi = x3;
  }
}


void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim){
  Real v_ff = std::sqrt(2.*GM/(r_inner_boundary+SMALL));
  Real va_max; /* Maximum Alfven speed allowed */
  Real bsq,bsq_rho_ceiling;
  Real dx,dy,dz,dx_min,dy_min,dz_min;
  get_uniform_box_spacing(pmb->block_size,&dx,&dy,&dz);  /* spacing of this block */
  get_minimum_cell_lengths(pmb->pcoord, &dx_min, &dy_min, &dz_min); /* spacing of the smallest block */
 /* Allow for larger Alfven speed if grid is coarser */
  va_max = v_ff * std::sqrt(dx/dx_min);
  Real r,x,y,z;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
          get_cartesian_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &x, &y, &z);
          Real r1,r2,new_x,new_y,new_z, r_hat_x,r_hat_y,r_hat_z;
          Real dE_dr, drho_dr,dM1_dr,dM2_dr,dM3_dr;
          Real dU_dr;
          int is_3D,is_2D;
          int i1,j1,k1,i2,j2,k2,ir;
          Real m_r;
          is_3D = (pmb->block_size.nx3>1);
          is_2D = (pmb->block_size.nx2>1);
          r = sqrt( SQR(x) + SQR(y)*is_2D + SQR(z)*is_3D);
          if (MAGNETIC_FIELDS_ENABLED){
            bsq = SQR(pmb->pfield->bcc(IB1,k,j,i)) + SQR(pmb->pfield->bcc(IB2,k,j,i)) + SQR(pmb->pfield->bcc(IB3,k,j,i));
            bsq_rho_ceiling = SQR(va_max);

            if (prim(IDN,k,j,i) < bsq/bsq_rho_ceiling){
              // pmb->user_out_var(16,k,j,i) += bsq/bsq_rho_ceiling - prim(IDN,k,j,i);
              prim(IDN,k,j,i) = bsq/bsq_rho_ceiling;
             }
           }
       if (r < r_inner_boundary){
          Real rho_flr = rhofloor;
          Real p_floor = pfloor;
          if (MAGNETIC_FIELDS_ENABLED){
            bsq_rho_ceiling = SQR(va_max);
            Real new_rho = bsq/bsq_rho_ceiling;
            if (new_rho>rho_flr) rho_flr = new_rho;
          }
          prim(IDN,k,j,i) = rho_flr;
          prim(IVX,k,j,i) = 0.;
          prim(IVY,k,j,i) = 0.;
          prim(IVZ,k,j,i) = 0.;
          prim(IPR,k,j,i) = p_floor;
          Real drho = prim(IDN,k,j,i) - rho_flr;
          /* Prevent outflow from inner boundary */
          if (prim(IVX,k,j,i)*x/r >0 ) prim(IVX,k,j,i) = 0.;
          if (prim(IVY,k,j,i)*y/r >0 ) prim(IVY,k,j,i) = 0.;
          if (prim(IVZ,k,j,i)*z/r >0 ) prim(IVZ,k,j,i) = 0.;
        }

      /* Add in the kinematic velocity limits */
        if (std::abs(prim(IVX,k,j,i)) > vmax){
          if (prim(IVX,k,j,i) > 0) prim(IVX,k,j,i) = vmax;
          else prim(IVX,k,j,i) = -vmax; 
        }        
        if (std::abs(prim(IVY,k,j,i)) > vmax){
          if (prim(IVY,k,j,i) > 0) prim(IVY,k,j,i) = vmax;
          else prim(IVY,k,j,i) = -vmax; 
        }
        if (std::abs(prim(IVZ,k,j,i)) > vmax){
          if (prim(IVZ,k,j,i) > 0) prim(IVZ,k,j,i) = vmax;
          else prim(IVZ,k,j,i) = -vmax; 
        }
      /* Add the pressure check */
        if (prim(IPR,k,j,i) < pfloor){
          prim(IPR,k,j,i) = pfloor;
        }
      /* Add the temperature check */
        Real v_therm = std::sqrt(prim(IPR,k,j,i)/prim(IDN,k,j,i));        
        if (v_therm > vmax || std::isnan(prim(IPR,k,j,i))){
          prim(IPR,k,j,i) = SQR(vmax)*prim(IDN,k,j,i);
        }
      /* Add in the density check */
        if (prim(IDN,k,j,i) < rhofloor){
          prim(IDN,k,j,i) = rhofloor;
        }
      }
    }
  }
}
void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
  n_mb = pmy_mesh->nbtotal;
  loc_list = pmy_mesh->loclist;
  amr_increase_resolution = pin->GetOrAddBoolean("problem","increase_resolution",false);
  r_inner_boundary = 0.;
  Real dx_min,dy_min,dz_min;
  get_minimum_cell_lengths(pcoord, &dx_min, &dy_min, &dz_min);
  max_refinement_level = pin->GetOrAddReal("mesh","numlevel",0);
  int N_cells_per_boundary_radius = pin->GetOrAddInteger("problem", "boundary_radius", 2);
  if (block_size.nx3>1){
    r_inner_boundary = N_cells_per_boundary_radius * std::max(std::max(dx_min,dy_min),dz_min); 
  }
  else if (block_size.nx2>1){
    r_inner_boundary = N_cells_per_boundary_radius * std::max(dx_min,dy_min); 
  }
  else{
    r_inner_boundary = N_cells_per_boundary_radius * dx_min;
  }
}


void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  int i=0,j=0,k=0;
  Real gm1 = peos->GetGamma() - 1.0;
  Real x,y,z;
  Real R, phi;
  std::cout<< "GM is " << GM << "\n";
  std::cout<< "pfloor is " << pfloor << "\n";
  std::cout << "Integration constant is : " << A << "\n";
  std::cout << "Omega constant is: " << C << "\n";
  for (k=ks; k<=ke; k++){
    for (j=js; j<=je; j++){
      for (i=is; i<=ie; i++){
        get_cartesian_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &x, &y, &z);
        Real r_vec[3] = {x,y,z};
        R = std::sqrt(SQR(x) + SQR(y));
        Real press = disk_press(r_vec);
        Real torus_rho = std::pow(press/K, 1/gas_gamma);
        //if (R < 1.5 && R > 1.4){
        //  std::cout << "Disk pressure is: " << press << i<< j<< k<<"\n"; 
        // std::cout << "Disk density  is: " << torus_rho << i<< j<< k<<"\n"; 
        //}
        Real int_energy = press / (gas_gamma - 1);
        Real torus_omega = C * std::pow(R, -q);
        Real torus_v = torus_omega * R;
        phi = std::atan2(y,x);

        
        if (disk_press(r_vec) > pfloor){
          torus_rho = std::pow(press/K, 1/gas_gamma);
        }
        // When pressure = pfloor, v = 0 & rho = rhofloor
        else if (disk_press(r_vec) <= pfloor){ 
           torus_rho = rhofloor;
           torus_v = 0;
        }
        
        // Outer regions
        if (R >= 0.1){
          phydro->u(IDN,k,j,i) = torus_rho;
          phydro->u(IM1,k,j,i) = -torus_rho * torus_v * std::sin(phi);
          phydro->u(IM2,k,j,i) = torus_rho * torus_v * std::cos(phi);
          phydro->u(IM3,k,j,i) = 0;
          phydro->u(IEN,k,j,i) = (0.5 * torus_rho) * SQR(torus_v) + int_energy;
        }
        // Assign floors and 0 velocity to the inner region
        // pressure is already at floor in this region... 
        else if (R <0.1){
          phydro->u(IDN,k,j,i) = rhofloor;
          phydro->u(IM1,k,j,i) = 0;
          phydro->u(IM2,k,j,i) = 0;
          phydro->u(IM3,k,j,i) = 0;
          phydro->u(IEN,k,j,i) = int_energy;
        }
      }
    }
  }
}
