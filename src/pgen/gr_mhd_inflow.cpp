//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_mhd_inflow.cpp
//  \brief Problem generator for magnetized equatorial inflow around Kerr black hole.

// C++ headers
#include <algorithm>  // max(), max_element(), min(), min_element()

#include <cmath>      // cos, sin, sqrt
#include <fstream>    // ifstream
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // string, c_str()
#include <cfloat>    // FLT_MIN
#include <stdio.h>


// Athena++ headers
#include "../mesh/mesh.hpp"
#include "../athena.hpp"                   // macros, enums, FaceField
#include "../athena_arrays.hpp"            // AthenaArray
#include "../parameter_input.hpp"          // ParameterInput
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro

// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

#define USE_VECTOR_POTENTIAL 0

// Declarations
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);
static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi);
static void BoyerLindquistToCartesian(Real r, Real theta, Real phi, Real *x,
                                         Real *y, Real *z);
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
static void CalculateFromTable(Real r, Real theta, Real *prho, Real *put, Real *pur,
                               Real *puphi, Real *pbt, Real *pbr, Real *pbphi);
void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
static void inner_boundary(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim );
void Cartesian_GR(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);


// Global variables
static Real m;                           // mass M of black hole
static Real a;                           // spin of black hole (0 <= a < M)
static Real temperature;                 // temperature pgas/rho
static AthenaArray<Real> interp_values;  // table for analytic solution
static int num_lines;                    // number of lines in table

Real rh;                /* Horizon radius */
Real dfloor,pfloor;


//----------------------------------------------------------------------------------------
// Function for initializing global mesh properties
// Inputs:
//   pin: input parameters
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read temperature
  temperature = pin->GetReal("problem", "temperature");

  // Allocate array for interpolation points
  num_lines = pin->GetInteger("problem", "num_data_lines");

  if (COORDINATE_SYSTEM=="gr_user" && USE_VECTOR_POTENTIAL)  interp_values.NewAthenaArray(7, num_lines);
  else interp_values.NewAthenaArray(6, num_lines);

  // Read interpolation data from file
  std::string filename = pin->GetString("problem", "data_file");
  std::ifstream file(filename.c_str());
  if (not file.is_open()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in Problem Generator\n"
        << "file " << filename << " cannot be opened" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  for (int n = 0; n < num_lines; ++n) {
    Real r, rho, ur, uphi, bbr, bbphi,atheta_2;
    if (COORDINATE_SYSTEM == "gr_user" && USE_VECTOR_POTENTIAL) file >> r >> rho >> ur >> uphi >> bbr >> bbphi >> atheta_2;
    else file >> r >> rho >> ur >> uphi >> bbr >> bbphi;
    interp_values(0,n) = r;
    interp_values(1,n) = rho;
    interp_values(2,n) = ur;
    interp_values(3,n) = uphi;
    interp_values(4,n) = bbr;
    interp_values(5,n) = bbphi;

#if USE_VECTOR_POTENTIAL
    if (COORDINATE_SYSTEM == "gr_user") interp_values(6,n) = atheta_2;
#endif
  }



  // Enroll fixed outer boundary function
  EnrollUserBoundaryFunction(OUTER_X1, FixedBoundary);

    if (COORDINATE_SYSTEM == "gr_user"){
    // Enroll boundary functions
    EnrollUserBoundaryFunction(INNER_X1, FixedBoundary);
    EnrollUserBoundaryFunction(INNER_X2, FixedBoundary);
    EnrollUserBoundaryFunction(OUTER_X2, FixedBoundary);
    EnrollUserBoundaryFunction(INNER_X3, FixedBoundary);
    EnrollUserBoundaryFunction(OUTER_X3, FixedBoundary);
  }

  if(COORDINATE_SYSTEM=="gr_user") EnrollUserMetric(Cartesian_GR);

  EnrollUserRadSourceFunction(inner_boundary);
  return;
}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {

  int N_user_vars = 6;
  if (MAGNETIC_FIELDS_ENABLED) {
    AllocateUserOutputVariables(N_user_vars);
  } else {
    AllocateUserOutputVariables(N_user_vars);
  }
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie+1);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie+1);

  a = pcoord->GetSpin();
  m = pcoord->GetMass();
  rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));

  r_inner_boundary = rh*1.1;
  return;
}


/* Apply inner "absorbing" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim){


  Real r,th,ph;
  AthenaArray<Real> g, gi;
  g.InitWithShallowCopy(pmb->ruser_meshblock_data[0]);
  gi.InitWithShallowCopy(pmb->ruser_meshblock_data[1]);



   for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
      pmb->pcoord->CellMetric(k, j, pmb->is, pmb->ie, g, gi);
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {


         GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r, &th, &ph);

          if (r < pmb->r_inner_boundary){




            Real x1 = pmb->pcoord->x1v(i);
            Real x2 = pmb->pcoord->x2v(j);
            Real x3 = pmb->pcoord->x3v(k);
            Real rho,ut,ur,uphi,bt,br,bphi,u0,u1,u2,u3,b0,b1,b2,b3;
            CalculateFromTable(r, th, &rho, &ut, &ur, &uphi, &bt, &br, &bphi);
            TransformVector(ut, ur, 0.0, uphi, x1,x2,x3, &u0, &u1, &u2, &u3);

              

              Real uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
              Real uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
              Real uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;

              
              prim(IDN,k,j,i) = dfloor;
              prim(IVX,k,j,i) = uu1;
              prim(IVY,k,j,i) = uu2;
              prim(IVZ,k,j,i) = uu3;
              prim(IPR,k,j,i) = pfloor;




              x1 = pmb->pcoord->x1f(i);
              x2 = pmb->pcoord->x2v(j);
              x3 = pmb->pcoord->x3v(k);

              GetBoyerLindquistCoordinates(x1, x2, x3, &r, &th, &ph);

              CalculateFromTable(pmb->r_inner_boundary, th, &rho, &ut, &ur, &uphi, &bt, &br, &bphi);
              BoyerLindquistToCartesian(pmb->r_inner_boundary,th,ph,&x1,&x2,&x3);
              TransformVector(ut, ur, 0.0, uphi, x1,x2,x3, &u0, &u1, &u2, &u3);
              TransformVector(bt, br, 0.0, bphi, x1,x2,x3, &b0, &b1, &b2, &b3);
              pmb->pfield->b.x1f(k,j,i) = b1*u0 - b0*u1;





              x1 = pmb->pcoord->x1v(i);
              x2 = pmb->pcoord->x2f(j);
              x3 = pmb->pcoord->x3v(k);

              GetBoyerLindquistCoordinates(x1, x2, x3, &r, &th, &ph);

              CalculateFromTable(pmb->r_inner_boundary, th, &rho, &ut, &ur, &uphi, &bt, &br, &bphi);
              BoyerLindquistToCartesian(pmb->r_inner_boundary,th,ph,&x1,&x2,&x3);
              TransformVector(ut, ur, 0.0, uphi, x1,x2,x3, &u0, &u1, &u2, &u3);
              TransformVector(bt, br, 0.0, bphi, x1,x2,x3, &b0, &b1, &b2, &b3);
              pmb->pfield->b.x2f(k,j,i) = b2*u0 - b0*u2;


              pmb->pfield->b.x3f(k,j,i) = 0.0;
            
              
              
          }



}}}

g.DeleteAthenaArray();
gi.DeleteAthenaArray();



}
static void inner_boundary(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim )
{
  int i, j, k, kprime;
  int is, ie, js, je, ks, ke;


  apply_inner_boundary_condition(pmb,prim);

  return;
}

//----------------------------------------------------------------------------------------
// Function responsible for storing useful quantities for output
// Inputs: (none)
// Outputs: (none)
// Notes:
//   writes to user_out_var array the following quantities:
//     0: gamma (normal-frame Lorentz factor)
//     1: p_mag (magnetic pressure)
void MeshBlock::UserWorkInLoop(void)
{
  // Create aliases for metric
  AthenaArray<Real> g, gi;
  g.InitWithShallowCopy(ruser_meshblock_data[0]);
  gi.InitWithShallowCopy(ruser_meshblock_data[1]);


  // Go through all cells
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pcoord->CellMetric(k, j, is, ie, g, gi);
      for (int i = is; i <= ie; ++i) {

        // Calculate normal frame Lorentz factor
        Real uu1 = phydro->w(IM1,k,j,i);
        Real uu2 = phydro->w(IM2,k,j,i);
        Real uu3 = phydro->w(IM3,k,j,i);
        Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
                 + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
                 + g(I33,i)*uu3*uu3;
        Real gamma = std::sqrt(1.0 + tmp);
        user_out_var(0,k,j,i) = gamma;

        // Calculate 4-velocity
        Real alpha = std::sqrt(-1.0/gi(I00,i));
        Real u0 = gamma/alpha;
        Real u1 = uu1 - alpha * gamma * gi(I01,i);
        Real u2 = uu2 - alpha * gamma * gi(I02,i);
        Real u3 = uu3 - alpha * gamma * gi(I03,i);
        Real u_0, u_1, u_2, u_3;

        user_out_var(1,k,j,i) = u0;
        user_out_var(2,k,j,i) = u1;
        user_out_var(3,k,j,i) = u2;
        user_out_var(4,k,j,i) = u3;
        if (not MAGNETIC_FIELDS_ENABLED) {
          continue;
        }

        pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

        // Calculate 4-magnetic field
        Real bb1 = pfield->bcc(IB1,k,j,i);
        Real bb2 = pfield->bcc(IB2,k,j,i);
        Real bb3 = pfield->bcc(IB3,k,j,i);
        Real b0 = g(I01,i)*u0*bb1 + g(I02,i)*u0*bb2 + g(I03,i)*u0*bb3
                + g(I11,i)*u1*bb1 + g(I12,i)*u1*bb2 + g(I13,i)*u1*bb3
                + g(I12,i)*u2*bb1 + g(I22,i)*u2*bb2 + g(I23,i)*u2*bb3
                + g(I13,i)*u3*bb1 + g(I23,i)*u3*bb2 + g(I33,i)*u3*bb3;
        Real b1 = (bb1 + b0 * u1) / u0;
        Real b2 = (bb2 + b0 * u2) / u0;
        Real b3 = (bb3 + b0 * u3) / u0;
        Real b_0, b_1, b_2, b_3;
        pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);

        // Calculate magnetic pressure
        Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
        user_out_var(5,k,j,i) = b_sq/2.0;
      }
    }
  }
  return;
}
//----------------------------------------------------------------------------------------
// Function for cleaning up global mesh properties
// Inputs:
//   pin: parameters (unused)
// Outputs: (none)

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  // Free interpolation table
  interp_values.DeleteAthenaArray();
  return;
}

//----------------------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: pointer to runtime inputs
// Outputs: (none)
// Notes:
//   initializes equatorial inflow
//   see Gammie 1999, ApJ 522 L57
//       Gammie, McKinney, & Toth 2003, ApJ 589 444

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Get mass and spin of black hole
  m = pcoord->GetMass();
  a = pcoord->GetSpin();

  // Prepare variables to hold results from multiple-return functions
  Real r, theta, phi;   // Boyer-Lindquist (BL) coordinates
  Real rho;             // density
  Real ut, ur, uphi;    // BL u^\mu
  Real bt, br, bphi;    // BL b^\mu
  Real u0, u1, u2, u3;  // preferred coordinates u^\mu
  Real b0, b1, b2, b3;  // preferred coordinates b^\mu

  Real dath1_dphi,ath_2,tmp;
  Real phip,phim;
  Real ath_1p,ath_2p,ath_1m,ath_2m;
  Real dath1_dphip,dath1_dphim;
  Real az_1p,az_2p,az_1m,az_2m;
  Real rp,rm,thetap,thetam;

  // Initialize magnetic field
  if (MAGNETIC_FIELDS_ENABLED) {

    // Initialize radial field components
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is-NGHOST; i <= ie+NGHOST+1; ++i) {
          Real x1 = pcoord->x1f(i);
          Real x2 = pcoord->x2v(j);
          Real x3 = pcoord->x3v(k);
          GetBoyerLindquistCoordinates(x1, x2, x3, &r, &theta, &phi);

          CalculateFromTable(r, theta, &rho, &ut, &ur, &uphi, &bt, &br, &bphi);
          TransformVector(ut, ur, 0.0, uphi, x1,x2,x3, &u0, &u1, &u2, &u3);
          TransformVector(bt, br, 0.0, bphi, x1,x2,x3, &b0, &b1, &b2, &b3);

          if (r <= r_inner_boundary && COORDINATE_SYSTEM=="gr_user"){
            CalculateFromTable(r_inner_boundary, theta, &rho, &ut, &ur, &uphi, &bt, &br, &bphi);
            BoyerLindquistToCartesian(r_inner_boundary,theta,phi,&x1,&x2,&x3);
            TransformVector(ut, ur, 0.0, uphi, x1,x2,x3, &u0, &u1, &u2, &u3);
            TransformVector(bt, br, 0.0, bphi, x1,x2,x3, &b0, &b1, &b2, &b3);
          }
          pfield->b.x1f(k,j,i) = b1*u0 - b0*u1;

          // if (COORDINATE_SYSTEM == "gr_user"){
          //   Real x2p = pcoord->x2f(j+1);
          //   Real x2m = pcoord->x2f(j);
          //   GetBoyerLindquistCoordinates(x1, x2p, x3, &rp, &thetap, &phip);
          //   CalculateFromTable(rp, thetap, phip, &rho, &ut, &ur, &uphi, &bt, &br, &bphi, &dath1_dphip, &ath_2p);


          //   GetBoyerLindquistCoordinates(x1, x2m, x3, &rm, &thetam, &phim);
          //   CalculateFromTable(rm, thetam, phim, &rho, &ut, &ur, &uphi, &bt, &br, &bphi, &dath1_dphim, &ath_2m);

          //   if ( fabs(phip-phim)> PI ){ //Account for branch cut at phi = pi
          //     if (phip<phim) phip += 2.0*PI;
          //     else phim += 2.0*PI;
          //   }
          //   az_1p = -dath1_dphip * phip/rp;
          //   az_2p = -ath_2p/rp;
          //   az_1m = -dath1_dphim * phim/rm;
          //   az_2m = -ath_2m/rm;

          //   //ath = - az *r since theta = pi/2

          //   //pfield->b.x1f(k,j,i) =  (az_2p - az_2m) /pcoord->dx2f(j);
          //   pfield->b.x1f(k,j,i) = 0;
          //   pfield->b.x1f(k,j,i) += (az_1p - az_1m) /pcoord->dx2f(j); 
          //   }
          }
        }
      }
    

    // Initialize poloidal field components
    for (int k = ks; k <= ke; ++k) {
      for (int j = js; j <= je+1; ++j) {
        for (int i = is-NGHOST; i <= ie+NGHOST; ++i) {
          Real x1 = pcoord->x1v(i);
          Real x2 = pcoord->x2f(j);
          Real x3 = pcoord->x3v(k);

          GetBoyerLindquistCoordinates(x1, x2, x3, &r, &theta, &phi);
          CalculateFromTable(r, theta, &rho, &ut, &ur, &uphi, &bt, &br, &bphi);
          TransformVector(ut, ur, 0.0, uphi, x1,x2,x3, &u0, &u1, &u2, &u3);
          TransformVector(bt, br, 0.0, bphi, x1,x2,x3, &b0, &b1, &b2, &b3);

          if (r <= r_inner_boundary && COORDINATE_SYSTEM=="gr_user"){
            CalculateFromTable(r_inner_boundary, theta, &rho, &ut, &ur, &uphi, &bt, &br, &bphi);
            BoyerLindquistToCartesian(r_inner_boundary,theta,phi,&x1,&x2,&x3);
            TransformVector(ut, ur, 0.0, uphi, x1,x2,x3, &u0, &u1, &u2, &u3);
            TransformVector(bt, br, 0.0, bphi, x1,x2,x3, &b0, &b1, &b2, &b3);
          }
          pfield->b.x2f(k,j,i) = b2*u0 - b0*u2;
          // if (COORDINATE_SYSTEM == "gr_user"){
          //   Real x1p = pcoord->x1f(i+1);
          //   Real x1m = pcoord->x1f(i);
          //   GetBoyerLindquistCoordinates(x1p, x2, x3, &rp, &thetap, &phip);
          //   CalculateFromTable(rp, thetap, phip, &rho, &ut, &ur, &uphi, &bt, &br, &bphi, &dath1_dphip, &ath_2p);

          //   GetBoyerLindquistCoordinates(x1m, x2, x3, &rm, &thetam, &phim);
          //   CalculateFromTable(rm, thetam, phim, &rho, &ut, &ur, &uphi, &bt, &br, &bphi, &dath1_dphim, &ath_2m);
            

          //   if ( fabs(phip-phim)> PI ){ //Account for branch cut at phi = pi
          //     if (phip<phim) phip += 2.0*PI;
          //     else phim += 2.0*PI;
          //   }
          //   az_1p = -dath1_dphip * phip/rp;
          //   az_2p = -ath_2p/rp;
          //   az_1m = -dath1_dphim * phim/rm;
          //   az_2m = -ath_2m/rm;

          //   //pfield->b.x2f(k,j,i) =  -(az_2p - az_2m) /pcoord->dx1f(i);
          //   pfield->b.x2f(k,j,i) = 0;
          //   pfield->b.x2f(k,j,i) += -(az_1p - az_1m) /pcoord->dx1f(i); 
          // }        
        }
      }
    }

    // Initialize azimuthal field components
    for (int k = ks; k <= ke+1; ++k) {
      for (int j = js; j <= je; ++j) {
        for (int i = is-NGHOST; i <= ie+NGHOST; ++i) {
          Real x1 = pcoord->x1v(i);
          Real x2 = pcoord->x2v(j);
          Real x3 = pcoord->x3f(k);
          GetBoyerLindquistCoordinates(x1, x2, x3, &r, &theta, &phi);
          CalculateFromTable(r, theta, &rho, &ut, &ur, &uphi, &bt, &br, &bphi);
          TransformVector(ut, ur, 0.0, uphi, x1,x2,x3, &u0, &u1, &u2, &u3);
          TransformVector(bt, br, 0.0, bphi, x1,x2,x3, &b0, &b1, &b2, &b3);
          pfield->b.x3f(k,j,i) = b3*u0 - b0*u3;
          if (COORDINATE_SYSTEM == "gr_user") pfield->b.x3f(k,j,i)=0;
        }
      }
    }
  }

  // Calculate cell-centered magnetic field
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ke+1, je+1, ie+NGHOST+1);
  if (MAGNETIC_FIELDS_ENABLED) {
    pfield->CalculateCellCenteredField(pfield->b, bb, pcoord, is-NGHOST, ie+NGHOST, js,
        je, ks, ke);
  }

  // Initialize primitive values
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC,ie+NGHOST+1);
  gi.NewAthenaArray(NMETRIC,ie+NGHOST+1);
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      pcoord->CellMetric(k, j, is-NGHOST, ie+NGHOST, g, gi);
      for (int i = is-NGHOST; i <= ie+NGHOST; ++i) {
        Real x1 = pcoord->x1v(i);
        Real x2 = pcoord->x2v(j);
        Real x3 = pcoord->x3v(k);
        GetBoyerLindquistCoordinates(x1, x2, x3, &r, &theta, &phi);
        CalculateFromTable(r, theta, &rho, &ut, &ur, &uphi, &bt, &br, &bphi);
        TransformVector(ut, ur, 0.0, uphi, x1,x2,x3, &u0, &u1, &u2, &u3);
        TransformVector(bt, br, 0.0, bphi, x1,x2,x3, &b0, &b1, &b2, &b3);
        Real pgas = temperature * rho;
        Real uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
        Real uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
        Real uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;
        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas;
        phydro->w(IVX,k,j,i) = phydro->w1(IVX,k,j,i) = uu1;
        phydro->w(IVY,k,j,i) = phydro->w1(IVY,k,j,i) = uu2;
        phydro->w(IVZ,k,j,i) = phydro->w1(IVZ,k,j,i) = uu3;
      }
    }
  }
  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();

  // Initialize conserved values
  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is-NGHOST, ie+NGHOST, js,
      je, ks, ke);
  bb.DeleteAthenaArray();

  UserWorkInLoop();

  return;
}

//----------------------------------------------------------------------------------------
// Fixed boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   time,dt: current time and timestep of simulation
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones
// Notes:
//   does nothing

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh) {
  return;
}


//----------------------------------------------------------------------------------------
// Function for calculating quantities based on table
// Inputs:
//   r,theta: Boyer-Lindquist radial and polar coordinates
// Outputs:
//   prho: value set to interpolated density
//   put,pur,puphi: values set to interpolated u^\mu in Boyer-Lindquist coordinates
//   pbt,pbr,pbphi: values set to interpolated b^\mu in Boyer-Lindquist coordinates

static void CalculateFromTable(Real r, Real theta, Real *prho, Real *put, Real *pur,
                               Real *puphi, Real *pbt, Real *pbr, Real *pbphi) {
  // Find location in interpolation table
  int n;
  Real fraction = 0.0;
  if (r < interp_values(0,0)) {
    n = 0;
    fraction = 0.0;
  } else if (r >= interp_values(0,num_lines-1)) {
    n = num_lines - 1;
    fraction = 1.0;
  } else {
    for (n = 0; n < num_lines-1; ++n) {
      if (r < interp_values(0,n+1)) {
        fraction = (r-interp_values(0,n)) / (interp_values(0,n+1)-interp_values(0,n));
        break;
      }
    }
  }

  // Interpolate to location based on table
  *prho = (1.0-fraction)*interp_values(1,n) + fraction*interp_values(1,n+1);
  Real ur = (1.0-fraction)*interp_values(2,n) + fraction*interp_values(2,n+1);
  Real uphi = (1.0-fraction)*interp_values(3,n) + fraction*interp_values(3,n+1);
  Real bbr = (1.0-fraction)*interp_values(4,n) + fraction*interp_values(4,n+1);
  Real bbphi = (1.0-fraction)*interp_values(5,n) + fraction*interp_values(5,n+1);


  // Calculate velocity
  Real sin = std::sin(theta);
  Real cos = std::cos(theta);
  Real delta = SQR(r) - 2.0*m*r + SQR(a);
  Real sigma = SQR(r) + SQR(a) * SQR(cos);
  Real g_tt = -(1.0 - 2.0*m*r/sigma);
  Real g_tphi = -2.0*m*a*r/sigma * SQR(sin);
  Real g_rr = sigma/delta;
  Real g_phiphi = (SQR(r) + SQR(a) + 2.0*m*SQR(a)*r/sigma * SQR(sin)) * SQR(sin);
  Real var_a = g_tt;
  Real var_b = 2.0*g_tphi*uphi;
  Real var_c = g_rr*SQR(ur) + g_phiphi*SQR(uphi) + 1.0;
  Real ut;
  if (var_a == 0.0) {
    ut = -var_c/var_b;
  } else {
    Real a1 = var_b/var_a;
    Real a0 = var_c/var_a;
    Real s2 = SQR(a1) - 4.0*a0;
    Real s = (s2 < 0.0) ? 0.0 : std::sqrt(s2);
    ut = (s2 >= 0.0 and a1 >= 0.0) ? -2.0*a0/(a1+s) : (-a1+s)/2.0;
  }
  *put = ut;
  *pur = ur;
  *puphi = uphi;

  // Calculate covariant magnetic field
  Real bt = g_tphi*ut*bbphi + g_rr*ur*bbr + g_phiphi*uphi*bbphi;
  *pbt = bt;
  *pbr = 1.0/ut * (bbr + bt * ur);
  *pbphi = 1.0/ut * (bbphi + bt * uphi);
  return;
}

static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi) {


    if (COORDINATE_SYSTEM == "gr_user"){
      Real x = x1;
      Real y = x2;
      Real z = x3;
      Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
      Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);

      *pr = r;
      *ptheta = std::acos(z/r);
      *pphi = std::atan2( (r*y-a*x )/(SQR(r)+SQR(a) ), (a*y+r*x)/(SQR(r) + SQR(a) )  );
  }
  else if (COORDINATE_SYSTEM == "kerr-schild" || COORDINATE_SYSTEM == "schwarzschild"){
     *pr = x1;
     *ptheta = x2;
     *pphi = x3;
  }
  return;
}

static void BoyerLindquistToCartesian(Real r, Real theta, Real phi, Real *x,
                                         Real *y, Real *z) {

  *x = (r * std::cos(phi) + a * std::sin(phi) ) * std::sin(theta);
  *y = (r * std::sin(phi) - a * std::cos(phi) ) * std::sin(theta);
  *z = r * std::cos(theta);


  return;
}



//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   r,theta,phi: Boyer-Lindquist coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0

static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real x1,
                     Real x2, Real x3, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {

  if (COORDINATE_SYSTEM == "schwarzschild") {
    *pa0 = a0_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl;
  } else if (COORDINATE_SYSTEM == "kerr-schild") {
    Real r = x1;
    Real delta = SQR(r) - 2.0*m*r + SQR(a);
    *pa0 = a0_bl + 2.0*m*r/delta * a1_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl + a/delta * a1_bl;
  }
    else if (COORDINATE_SYSTEM == "gr_user"){
    Real x = x1;
    Real y = x2;
    Real z = x3;

    Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);
    Real delta = SQR(r) - 2.0*m*r + SQR(a);
    *pa0 = a0_bl + 2.0*r/delta * a1_bl;
    *pa1 = a1_bl * ( (r*x+a*y)/(SQR(r) + SQR(a)) - y*a/delta) + 
           a2_bl * x*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) - 
           a3_bl * y; 
    *pa2 = a1_bl * ( (r*y-a*x)/(SQR(r) + SQR(a)) + x*a/delta) + 
           a2_bl * y*z/r * std::sqrt((SQR(r) + SQR(a))/(SQR(x) + SQR(y))) + 
           a3_bl * x;
    *pa3 = a1_bl * z/r - 
           a2_bl * r * std::sqrt((SQR(x) + SQR(y))/(SQR(r) + SQR(a)));
  }
  return;
}


void Cartesian_GR(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3)
{
  // Extract inputs
  Real x = x1;
  Real y = x2;
  Real z = x3;

  Real a_spin = a; //-a;

  Real SMALL = 1e-5;
  if (std::fabs(z)<SMALL) z= SMALL;
  Real R = std::sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = SQR(R) - SQR(a) + std::sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
  r = std::sqrt(r/2.0);


  //if (r<0.01) r = 0.01;


  Real eta[4],l_lower[4],l_upper[4];

  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
  l_upper[0] = -1.0;
  l_upper[1] = (r*x + a_spin*y)/( SQR(r) + SQR(a) );
  l_upper[2] = (r*y - a_spin*x)/( SQR(r) + SQR(a) );
  l_upper[3] = z/r;

  l_lower[0] = 1.0;
  l_lower[1] = l_upper[1];
  l_lower[2] = l_upper[2];
  l_lower[3] = l_upper[3];

  eta[0] = -1.0;
  eta[1] = 1.0;
  eta[2] = 1.0;
  eta[3] = 1.0;





  // Set covariant components
  g(I00) = eta[0] + f * l_lower[0]*l_lower[0];
  g(I01) = f * l_lower[0]*l_lower[1];
  g(I02) = f * l_lower[0]*l_lower[2];
  g(I03) = f * l_lower[0]*l_lower[3];
  g(I11) = eta[1] + f * l_lower[1]*l_lower[1];
  g(I12) = f * l_lower[1]*l_lower[2];
  g(I13) = f * l_lower[1]*l_lower[3];
  g(I22) = eta[2] + f * l_lower[2]*l_lower[2];
  g(I23) = f * l_lower[2]*l_lower[3];
  g(I33) = eta[3] + f * l_lower[3]*l_lower[3];




  // // Set contravariant components
  g_inv(I00) = eta[0] - f * l_upper[0]*l_upper[0];
  g_inv(I01) = -f * l_upper[0]*l_upper[1];
  g_inv(I02) = -f * l_upper[0]*l_upper[2];
  g_inv(I03) = -f * l_upper[0]*l_upper[3];
  g_inv(I11) = eta[1] - f * l_upper[1]*l_upper[1];
  g_inv(I12) = -f * l_upper[1]*l_upper[2];
  g_inv(I13) = -f * l_upper[1]*l_upper[3];
  g_inv(I22) = eta[2] + -f * l_upper[2]*l_upper[2];
  g_inv(I23) = -f * l_upper[2]*l_upper[3];
  g_inv(I33) = eta[3] - f * l_upper[3]*l_upper[3];


  Real sqrt_term =  2.0*SQR(r)-SQR(R) + SQR(a);
  Real rsq_p_asq = SQR(r) + SQR(a);

  Real df_dx1 = SQR(f)*x/(2.0*std::pow(r,3)) * ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) )/ sqrt_term ;
  //4 x/r^2 1/(2r^3) * -r^4/r^2 = 2 x / r^3
  Real df_dx2 = SQR(f)*y/(2.0*std::pow(r,3)) * ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) )/ sqrt_term ;
  Real df_dx3 = SQR(f)*z/(2.0*std::pow(r,5)) * ( ( ( 3.0*SQR(a*z)-SQR(r)*SQR(r) ) * ( rsq_p_asq ) )/ sqrt_term - 2.0*SQR(a*r)) ;
  //4 z/r^2 * 1/2r^5 * -r^4*r^2 / r^2 = -2 z/r^3
  Real dl1_dx1 = x*r * ( SQR(a)*x - 2.0*a_spin*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
  // x r *(-r^2 x)/(r^6) + 1/r = -x^2/r^3 + 1/r
  Real dl1_dx2 = y*r * ( SQR(a)*x - 2.0*a_spin*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) )+ a_spin/( rsq_p_asq );
  Real dl1_dx3 = z/r * ( SQR(a)*x - 2.0*a_spin*r*y - SQR(r)*x )/( (rsq_p_asq) * ( sqrt_term ) ) ;
  Real dl2_dx1 = x*r * ( SQR(a)*y + 2.0*a_spin*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) - a_spin/( rsq_p_asq );
  Real dl2_dx2 = y*r * ( SQR(a)*y + 2.0*a_spin*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
  Real dl2_dx3 = z/r * ( SQR(a)*y + 2.0*a_spin*r*x - SQR(r)*y )/( (rsq_p_asq) * ( sqrt_term ) );
  Real dl3_dx1 = - x*z/(r) /( sqrt_term );
  Real dl3_dx2 = - y*z/(r) /( sqrt_term );
  Real dl3_dx3 = - SQR(z)/(SQR(r)*r) * ( rsq_p_asq )/( sqrt_term ) + 1.0/r;

  Real dl0_dx1 = 0.0;
  Real dl0_dx2 = 0.0;
  Real dl0_dx3 = 0.0;

  if (std::isnan(f) || std::isnan(r) || std::isnan(sqrt_term) || std::isnan (df_dx1) || std::isnan(df_dx2)){
    fprintf(stderr,"ISNAN in metric\n x y y: %g %g %g r: %g \n",x,y,z,r);
    exit(0);
  }



  //expressioons for a = 0

  // f = 2.0/R;
  // l_lower[1] = x/R;
  // l_lower[2] = y/R;
  // l_lower[3] = z/R;
  // df_dx1 = -2.0 * x/SQR(R)/R;
  // df_dx2 = -2.0 * y/SQR(R)/R;
  // df_dx3 = -2.0 * z/SQR(R)/R;

  // dl1_dx1 = -SQR(x)/SQR(R)/R + 1.0/R;
  // dl1_dx2 = -x*y/SQR(R)/R; 
  // dl1_dx3 = -x*z/SQR(R)/R;

  // dl2_dx1 = -x*y/SQR(R)/R;
  // dl2_dx2 = -SQR(y)/SQR(R)/R + 1.0/R;
  // dl2_dx3 = -y*z/SQR(R)/R;

  // dl3_dx1 = -x*z/SQR(R)/R;
  // dl3_dx2 = -y*z/SQR(R)/R;
  // dl3_dx3 = -SQR(z)/SQR(R)/R;


  // // Set x-derivatives of covariant components
  dg_dx1(I00) = df_dx1*l_lower[0]*l_lower[0] + f * dl0_dx1 * l_lower[0] + f * l_lower[0] * dl0_dx1;
  dg_dx1(I01) = df_dx1*l_lower[0]*l_lower[1] + f * dl0_dx1 * l_lower[1] + f * l_lower[0] * dl1_dx1;
  dg_dx1(I02) = df_dx1*l_lower[0]*l_lower[2] + f * dl0_dx1 * l_lower[2] + f * l_lower[0] * dl2_dx1;
  dg_dx1(I03) = df_dx1*l_lower[0]*l_lower[3] + f * dl0_dx1 * l_lower[3] + f * l_lower[0] * dl3_dx1;
  dg_dx1(I11) = df_dx1*l_lower[1]*l_lower[1] + f * dl1_dx1 * l_lower[1] + f * l_lower[1] * dl1_dx1;
  dg_dx1(I12) = df_dx1*l_lower[1]*l_lower[2] + f * dl1_dx1 * l_lower[2] + f * l_lower[1] * dl2_dx1;
  dg_dx1(I13) = df_dx1*l_lower[1]*l_lower[3] + f * dl1_dx1 * l_lower[3] + f * l_lower[1] * dl3_dx1;
  dg_dx1(I22) = df_dx1*l_lower[2]*l_lower[2] + f * dl2_dx1 * l_lower[2] + f * l_lower[2] * dl2_dx1;
  dg_dx1(I23) = df_dx1*l_lower[2]*l_lower[3] + f * dl2_dx1 * l_lower[3] + f * l_lower[2] * dl3_dx1;
  dg_dx1(I33) = df_dx1*l_lower[3]*l_lower[3] + f * dl3_dx1 * l_lower[3] + f * l_lower[3] * dl3_dx1;

  // Set y-derivatives of covariant components
  dg_dx2(I00) = df_dx2*l_lower[0]*l_lower[0] + f * dl0_dx2 * l_lower[0] + f * l_lower[0] * dl0_dx2;
  dg_dx2(I01) = df_dx2*l_lower[0]*l_lower[1] + f * dl0_dx2 * l_lower[1] + f * l_lower[0] * dl1_dx2;
  dg_dx2(I02) = df_dx2*l_lower[0]*l_lower[2] + f * dl0_dx2 * l_lower[2] + f * l_lower[0] * dl2_dx2;
  dg_dx2(I03) = df_dx2*l_lower[0]*l_lower[3] + f * dl0_dx2 * l_lower[3] + f * l_lower[0] * dl3_dx2;
  dg_dx2(I11) = df_dx2*l_lower[1]*l_lower[1] + f * dl1_dx2 * l_lower[1] + f * l_lower[1] * dl1_dx2;
  dg_dx2(I12) = df_dx2*l_lower[1]*l_lower[2] + f * dl1_dx2 * l_lower[2] + f * l_lower[1] * dl2_dx2;
  dg_dx2(I13) = df_dx2*l_lower[1]*l_lower[3] + f * dl1_dx2 * l_lower[3] + f * l_lower[1] * dl3_dx2;
  dg_dx2(I22) = df_dx2*l_lower[2]*l_lower[2] + f * dl2_dx2 * l_lower[2] + f * l_lower[2] * dl2_dx2;
  dg_dx2(I23) = df_dx2*l_lower[2]*l_lower[3] + f * dl2_dx2 * l_lower[3] + f * l_lower[2] * dl3_dx2;
  dg_dx2(I33) = df_dx2*l_lower[3]*l_lower[3] + f * dl3_dx2 * l_lower[3] + f * l_lower[3] * dl3_dx2;

  // Set phi-derivatives of covariant components
  dg_dx3(I00) = df_dx3*l_lower[0]*l_lower[0] + f * dl0_dx3 * l_lower[0] + f * l_lower[0] * dl0_dx3;
  dg_dx3(I01) = df_dx3*l_lower[0]*l_lower[1] + f * dl0_dx3 * l_lower[1] + f * l_lower[0] * dl1_dx3;
  dg_dx3(I02) = df_dx3*l_lower[0]*l_lower[2] + f * dl0_dx3 * l_lower[2] + f * l_lower[0] * dl2_dx3;
  dg_dx3(I03) = df_dx3*l_lower[0]*l_lower[3] + f * dl0_dx3 * l_lower[3] + f * l_lower[0] * dl3_dx3;
  dg_dx3(I11) = df_dx3*l_lower[1]*l_lower[1] + f * dl1_dx3 * l_lower[1] + f * l_lower[1] * dl1_dx3;
  dg_dx3(I12) = df_dx3*l_lower[1]*l_lower[2] + f * dl1_dx3 * l_lower[2] + f * l_lower[1] * dl2_dx3;
  dg_dx3(I13) = df_dx3*l_lower[1]*l_lower[3] + f * dl1_dx3 * l_lower[3] + f * l_lower[1] * dl3_dx3;
  dg_dx3(I22) = df_dx3*l_lower[2]*l_lower[2] + f * dl2_dx3 * l_lower[2] + f * l_lower[2] * dl2_dx3;
  dg_dx3(I23) = df_dx3*l_lower[2]*l_lower[3] + f * dl2_dx3 * l_lower[3] + f * l_lower[2] * dl3_dx3;
  dg_dx3(I33) = df_dx3*l_lower[3]*l_lower[3] + f * dl3_dx3 * l_lower[3] + f * l_lower[3] * dl3_dx3;
  return;
}
