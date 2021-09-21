#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
Real erg_to_code = g_to_msun*SQR(cm_to_pc)/SQR(s_to_kyr);
Real icm3_to_code = 2.94e55;
// Gas variables
Real mh2 = 3.32e-24*g_to_msun;
Real kb = 1.38e-16*erg_to_code;
Real gas_gamma; 
// Problem 
Real pfloor, rhofloor;
Real vmax = 10; //3.e8*cm_to_pc/s_to_kyr;
Real GM;
// For the inner boundary 
Real r_inner_boundary=0.0;
int n_mb = 0;
double SMALL = 1e-20;
bool amr_increase_resolution;
LogicalLocation *loc_list;
int max_refinement_level = 0; 

/* Functions to deal with the inner boundary*/
// Floors out variables at the center of the simulation
void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);

// Source function wrapper around 'apply_inner_boundary_condition'
static void integrate_cool(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim)
{
  apply_inner_boundary_condition(pmb,prim);
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

void get_cylindrical_coords(const Real x1, const Real x2, const Real x3, Real *r, Real *phi, Real *z){
  if (COORDINATE_SYSTEM == "cartesian"){
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
              prim(IDN,k,j,i) = bsq/bsq_rho_ceiling;
             }
           }
       /* Floor primitives in the inner boundary*/
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

void Mesh::InitUserMeshData(ParameterInput *pin){
  pfloor = pin->GetReal("hydro", "pfloor");
  rhofloor = pin->GetReal("hydro", "dfloor");
  gas_gamma = pin->GetReal("hydro", "gamma");
  GM = pin->GetReal("problem", "GM");
  EnrollUserRadSourceFunction(integrate_cool);
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
  Real r, phi;
  Real press = 1.e-6;
  for (k=ks-NGHOST; k<=ke+NGHOST; k++){
    for (j=js-NGHOST; j<=je+NGHOST; j++){
      for (i=is-NGHOST; i<=ie+NGHOST; i++){
        get_cartesian_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &x, &y, &z);
        // Infalling blob
        //if (SQR(x-1.5) + SQR(y) + SQR(z) <= SQR(0.2)){
        //  phydro->u(IDN,k,j,i) = 1;
        //  phydro->u(IM1,k,j,i) = 0;
        //  phydro->u(IM2,k,j,i) = 0;
        //  phydro->u(IM3,k,j,i) = 0;
        //   phydro->u(IEN,k,j,i) = press/gm1; 
        //}
        // Keplerian disk
         
        get_cylindrical_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &r, &phi, &z); 
        if (r >= 1 && r <= 1.1){
          phydro->u(IDN,k,j,i) = 1;
          phydro->u(IM1,k,j,i) = std::sqrt(GM/r)*std::cos(phi);
          phydro->u(IM2,k,j,i) = std::sqrt(GM/r)*std::sin(phi);
          phydro->u(IM3,k,j,i) = 0;
          phydro->u(IEN,k,j,i) = press/gm1;
        }
        
        // Assign ambient parameters everywhere else
        else{
          phydro->u(IDN,k,j,i) = rhofloor;
          phydro->u(IM1,k,j,i) = 0;
          phydro->u(IM2,k,j,i) = 0;
          phydro->u(IM3,k,j,i) = 0;
          phydro->u(IEN,k,j,i) = press/gm1;
        }
      }
    }
  }
}
