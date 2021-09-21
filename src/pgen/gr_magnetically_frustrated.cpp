//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_bondi.cpp
//  \brief Problem generator for spherically symmetric black hole accretion.

// C++ headers
#include <cmath>  // abs(), NAN, pow(), sqrt()

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Athena++ headers
#include "../mesh/mesh.hpp"
#include "../athena.hpp"                   // macros, enums, FaceField
#include "../athena_arrays.hpp"            // AthenaArray
#include "../parameter_input.hpp"          // ParameterInput
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../utils/utils.hpp" //ran2()


// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif


#define UNIFORM 0
#define MULTI_LOOP 1
#define FIELD_TYPE UNIFORM

// Declarations
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);
static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi);
static void GetCKSCoordinates(Real r, Real th, Real phi, Real *x, Real *y, Real *z);
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
static void TransformCKSLowerVector(Real a0_cks, Real a1_cks, Real a2_cks, Real a3_cks, Real r,
                     Real theta, Real phi, Real x , Real y, Real z,Real *pa0, Real *pa1, Real *pa2, Real *pa3);
void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);


Real CompressedX2(Real x, RegionSize rs);
Real ExponetialX1(Real x, RegionSize rs);


// Global variables
static Real m, a;          // black hole mass and spin
static Real rh;
static Real r_cut;        // initial condition cut off
static Real bondi_radius;  // b^2/rho at inner radius
static Real field_norm;     
static Real magnetic_field_inclination;   

Real x1_harm_max;  //maximum x1 coordinate for hyper-exponentiation   
Real cpow2,npow2;  //coordinate parameters for hyper-exponentiation
Real rbr; //break radius for hyper-exp.
//----------------------------------------------------------------------------------------
// Function for initializing global mesh properties
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read problem parameters

  bondi_radius  = pin->GetReal("problem", "bondi_radius");

  magnetic_field_inclination = pin->GetOrAddReal("problem","field_inclination",0.0);


  r_cut = pin->GetReal("problem", "r_cut");
  field_norm =  pin->GetReal("problem", "field_norm");

  EnrollUserBoundaryFunction(INNER_X1, InflowBoundary);
  EnrollUserBoundaryFunction(OUTER_X1, FixedBoundary);


  EnrollUserMeshGenerator(X2DIR, CompressedX2);
  //EnrollUserMeshGenerator(X1DIR, ExponetialX1);



  return;
}

//----------------------------------------------------------------------------------------
// Function for preparing MeshBlock
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)
// Notes:
//   user arrays are metric and its inverse

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


  Real x1_harm_max0 = 1;
  x1_harm_max = 2;
  npow2 = 4.0;
  cpow2 = 1.0;
  rbr = 100.0;
  Real RELACC = 1e-14;
  const int ITERMAX = 50;
  int iter;
  Real Rout = pmy_mesh->mesh_size.x1max;
  Real x1br = std::log(rbr);
  Real R0 = 0.0;
      
  if (Rout>rbr){
      //find the x1_harm_max via iterations
      for( iter = 0; iter < ITERMAX; iter++ ) {
        if( fabs((x1_harm_max - x1_harm_max0)/x1_harm_max) < RELACC ) {
          break;
        }
        x1_harm_max0 = x1_harm_max;
        Real dxmax= (pow( (log(Rout-R0) - x1_harm_max0)/cpow2, 1./npow2 ) + x1br) - x1_harm_max0;
        
        // need a slight damping factor
        Real dampingfactor=0.5;
        x1_harm_max = x1_harm_max0 + dampingfactor*dxmax;
        if (x1_harm_max> log(Rout-R0)){x1_harm_max = log(Rout-R0);}
      }

      if( iter == ITERMAX ) {
          printf( "Error: iteration procedure for finding x1_harm_max has not converged: x1_harm_max = %g, dx1_harm_max/x1_harm_max = %g, iter = %d\n",
                 x1_harm_max, (x1_harm_max-x1_harm_max0)/x1_harm_max, iter );
          printf( "Error: iteration procedure for finding x1_harm_max has not converged: rbr= %g, x1br = %g, log(Rout-R0) = %g\n",
                 rbr, x1br, log(Rout-R0) );
        exit(1);
      }
      else {
        printf( "x1_harm_max = %g (dx1_harm_max/x1_harm_max = %g, itno = %d)\n", x1_harm_max, (x1_harm_max-x1_harm_max0)/x1_harm_max, iter );
      }

  }
  else{
    x1_harm_max = std::log(Rout);
  }

  return;
}


    static Real exp_cut_off(Real r){

      if (r<=rh) return 0.0;
      else if (r<= r_cut) return std::exp(5 * (r-r_cut)/r);
      else return 1.0;
    }

    static Real Ax_func(Real x,Real y, Real z){

      return 0.0 * field_norm;
    }
    static Real Ay_func(Real x, Real y, Real z){
      return  (-z * std::sin(magnetic_field_inclination) + x * std::cos(magnetic_field_inclination) ) * field_norm;   //x 
    }
    static Real Az_func(Real x, Real y, Real z){
      return 0.0 * field_norm;
    }

    static Real Aphi_func(Real r, Real th, Real ph){
 
      //Real r_loop = 20.0 ;  //Typical radius of loop
      Real pot_r_min = 10.0;
      Real pot_theta_min = PI/6.0 ;

      Real theta_length = std::cos(pot_theta_min) - std::cos(PI - pot_theta_min) ;

      Real n_loops_in_theta = 4.0 ; 

      Real pot_r_cent = (pot_r_min)/(1.0-theta_length/n_loops_in_theta/2.0);
      Real pot_r_max = 2.0*pot_r_cent - pot_r_min;
 
      Real n_loops_in_r = 1.0 ; 

      Real pot_r_fac = n_loops_in_r;
      Real pot_th_fac = n_loops_in_theta;

      Real arg_r = PI * pot_r_fac * (r-pot_r_min)/(pot_r_max-pot_r_min);
      Real arg_th = PI * pot_th_fac * (th- pot_theta_min)/(PI-2*pot_theta_min);


      if ( (th<pot_theta_min) || (th > (PI-pot_theta_min) ) ) return 0.0;
      else if (r<pot_r_max) return SQR(r) * std::sin(th) * std::fabs( std::sin(arg_r) * std::sin(arg_th)) * field_norm;


      Real r_max = 1e4;
      while ( pot_r_max < r_max){
        pot_r_min = pot_r_max;
        pot_r_cent = (pot_r_min)/(1.0-theta_length/n_loops_in_theta/2.0);
        pot_r_max = 2.0*pot_r_cent - pot_r_min;
        if (pot_r_max > r_max) return 0.0; 
        arg_r = PI * pot_r_fac * (r-pot_r_min)/(pot_r_max-pot_r_min);

        if (r<pot_r_max) return SQR(r) * std::sin(th) * std::fabs( std::sin(arg_r) * std::sin(arg_th)) * field_norm;

      }


      return 0.0;

     // return r * std::sin(th) * SQR( std::sin(z/r_loop ) * std::sin(s/r_loop)  ) * field_norm;
    }
//----------------------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)
// Notes:
//   sets primitive and conserved variables according to input primitives
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Parameters
  Real cs_0 = 1.0/std::sqrt(bondi_radius);
  Real rho_0 = 1.0;
  Real gam = peos->GetGamma();
  Real P_0 = SQR(cs_0)*rho_0/gam;

  // Prepare index bounds
  int il = is - NGHOST;
  int iu = ie + NGHOST;
  int jl = js;
  int ju = je;
  if (block_size.nx2 > 1) {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = ks;
  int ku = ke;
  if (block_size.nx3 > 1) {
    kl -= (NGHOST);
    ku += (NGHOST);
  }

  // Get mass of black hole
  m = pcoord->GetMass();
  a = pcoord->GetSpin();

  rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );



  // Prepare scratch arrays
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);


  int64_t iseed = -1 - gid;

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i = il; i <= iu; ++i) {
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k),
            &r, &theta, &phi);
        Real u0 = std::sqrt(-1.0/g(I00,i));
        Real uu1 = 0.0 - gi(I01,i)/gi(I00,i) * u0;
        Real uu2 = 0.0 - gi(I02,i)/gi(I00,i) * u0;
        Real uu3 = 0.0 - gi(I03,i)/gi(I00,i) * u0;

        Real amp = 0.00;
        if (std::fabs(a)<1e-1) amp = 0.05;
        Real rval = amp*(ran2(&iseed) - 0.5);
    

        if (r<r_cut){
          phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = 0.0;
          phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = 0.0;
          phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = 0.0;
          phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = 0.0;
          phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = 0.0;
        }
        else{ 
          phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho_0 * (1.0 + 2.0*rval);
          phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = P_0;
          phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
          phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
          phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;
      }
      }
    }
  }

  // Initialize magnetic field
  if (MAGNETIC_FIELDS_ENABLED) {


    Real delta =0.0; // 1e-1;  //perturbation in B-field amplitude
    Real pert = 0.0;


    int ncells1 = block_size.nx1 + 2*(NGHOST);
    int ncells2 = 1, ncells3 = 1;
    if (block_size.nx2 > 1) ncells2 = block_size.nx2 + 2*(NGHOST);
    if (block_size.nx3 > 1) ncells3 = block_size.nx3 + 2*(NGHOST);

    AthenaArray<Real> A3,A1,A2;

    A1.NewAthenaArray( ncells3  +1,ncells2 +1, ncells1+2   );
    A2.NewAthenaArray( ncells3  +1,ncells2 +1, ncells1+2   );
    A3.NewAthenaArray( ncells3  +1,ncells2 +1, ncells1+2   );

      // Set B^1
      for (int k = kl; k <= ku+1; ++k) {
        for (int j = jl; j <= ju+1; ++j) {
          for (int i = il; i <= iu+1; ++i) {

            //A1 defined at cell center in x1 but face in x2 x3, 
            //A2 defined at cell center in x2 but face in x1 x3,
            //A3 defined at cell center in x3 but face in x1 x2

            Real r,theta,phi;
            Real x_coord;
            if (i<= iu) x_coord = pcoord->x1v(i);
            else x_coord = pcoord->x1v(iu) + pcoord->dx1v(iu);
            GetBoyerLindquistCoordinates(x_coord,pcoord->x2f(j),pcoord->x3f(k), &r, &theta,&phi);

            pert = delta * std::cos(phi);
            Real x,y,z;
            GetCKSCoordinates(r,theta,phi,&x,&y,&z);
            Real Ax = Ax_func(x,y,z) * (1 + pert);
            Real Ay = Ay_func(x,y,z) * (1 + pert);
            Real Az = Az_func(x,y,z) * (1 + pert);

            Real Ar,Ath,Aphi,A0;;

            TransformCKSLowerVector(0.0,Ax,Ay,Az,r,theta,phi,x,y,z,&A0,&Ar,&Ath,&Aphi);

            if (FIELD_TYPE == MULTI_LOOP) Ar = 0;

            A1(k,j,i) = Ar * exp_cut_off(r);

            Real y_coord;
            if (j<= ju) y_coord = pcoord->x2v(j);
            else y_coord = pcoord->x2v(ju) + pcoord->dx2v(ju);
            GetBoyerLindquistCoordinates(pcoord->x1f(i),y_coord,pcoord->x3f(k), &r, &theta,&phi);
            pert = delta * std::cos(phi);
            GetCKSCoordinates(r,theta,phi,&x,&y,&z);
            Ax = Ax_func(x,y,z) * (1 + pert);
            Ay = Ay_func(x,y,z) * (1 + pert);
            Az = Az_func(x,y,z) * (1 + pert);
            TransformCKSLowerVector(0.0,Ax,Ay,Az,r,theta,phi,x,y,z,&A0,&Ar,&Ath,&Aphi);

            if (FIELD_TYPE == MULTI_LOOP) Ath = 0;

            A2(k,j,i) = Ath * exp_cut_off(r);

            Real z_coord;
            if (k<= ku) z_coord = pcoord->x3v(k);
            else z_coord = pcoord->x3v(ku) + pcoord->dx3v(ku);
            GetBoyerLindquistCoordinates(pcoord->x1f(i),pcoord->x2f(j),z_coord, &r, &theta,&phi);
            pert = delta * std::cos(phi);
            GetCKSCoordinates(r,theta,phi,&x,&y,&z);
            Ax = Ax_func(x,y,z) * (1 + pert);
            Ay = Ay_func(x,y,z) * (1 + pert);
            Az = Az_func(x,y,z) * (1 + pert);
            TransformCKSLowerVector(0.0,Ax,Ay,Az,r,theta,phi,x,y,z,&A0,&Ar,&Ath,&Aphi);

            if (FIELD_TYPE == MULTI_LOOP) Aphi = Aphi_func(r,theta,phi);

            A3(k,j,i) = Aphi * exp_cut_off(r);



            }
          }
        }


      // Initialize interface fields
    AthenaArray<Real> area;
    area.NewAthenaArray(ncells1+1);

    // for 1,2,3-D
    for (int k=kl; k<=ku; ++k) {
      // reset loop limits for polar boundary
      int jl=js; int ju=je+1;
      if (pcoord->pmy_block->pbval->block_bcs[INNER_X2] == 5) jl=js+1;
      if (pcoord->pmy_block->pbval->block_bcs[OUTER_X2] == 5) ju=je;
      for (int j=jl; j<=ju; ++j) {
        pcoord->Face2Area(k,j,il,iu,area);
        for (int i=il; i<=iu; ++i) {
          pfield->b.x2f(k,j,i) = -1.0*(pcoord->dx3f(k)*A3(k,j,i+1) - pcoord->dx3f(k)*A3(k,j,i))/area(i);
          if (area(i)==0.0) pfield->b.x2f(k,j,i) = 0;
          //if (j==ju) fprintf(stderr,"B: %g area: %g theta: %g j: %d A3: %g %g \n",pfield->b.x2f(k,j,i), area(i),pcoord->x2f(j),j, 
           // A3(k,j,i+1), A3(k,j,i));

          if (std::isnan((pfield->b.x2f(k,j,i)))) fprintf(stderr,"isnan in bx2!\n");
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=js; j<=je; ++j) {
        pcoord->Face3Area(k,j,il,iu,area);
        for (int i=il; i<=iu; ++i) {
          pfield->b.x3f(k,j,i) = (pcoord->dx2f(j)*A2(k,j,i+1) - pcoord->dx2f(j)*A2(k,j,i))/area(i);
          //if (area(i)==0) pfield->b.x3f(k,j,i) = 0.0;

          if (std::isnan((pfield->b.x3f(k,j,i)))){

           fprintf(stderr,"isnan in bx3!\n A2: %g %g \n area: %g dx2f: %g \n", A2(k,j,i+1),A2(k,j,i),area(i),pcoord->dx2f(j));
           exit(0);
         }
        }
      }
    }

    // for 2D and 3D
    if (block_size.nx2 > 1) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je; ++j) {
          pcoord->Face1Area(k,j,il,iu+1,area);
          for (int i=il; i<=iu+1; ++i) {
            pfield->b.x1f(k,j,i) = (pcoord->dx3f(k)*A3(k,j+1,i) - pcoord->dx3f(k)*A3(k,j,i))/area(i);
            //if (area(i)==0) pfield->b.x1f(k,j,i) = 0.0;
            if (std::isnan((pfield->b.x1f(k,j,i)))) fprintf(stderr,"isnan in bx1!\n");
          }
        }
      }
      for (int k=kl; k<=ku+1; ++k) {
        for (int j=js; j<=je; ++j) {
          pcoord->Face3Area(k,j,il,iu,area);
          for (int i=il; i<=iu; ++i) {
            pfield->b.x3f(k,j,i) -= (pcoord->dx1f(i)*A1(k,j+1,i) - pcoord->dx1f(i)*A1(k,j,i))/area(i);
            //if (area(i)==0) pfield->b.x3f(k,j,i) = 0.0;
            if (std::isnan((pfield->b.x3f(k,j,i)))) {
              fprintf(stderr,"isnan in bx3!\n A1: %g %g \n area: %g dx1f: %g \n", A1(k,j+1,i),A1(k,j,i),area(i),pcoord->dx1f(i));
              exit(0);
            }
          }
        }
      }
    }
    // for 3D only
    if (block_size.nx3 > 1) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je; ++j) {
          pcoord->Face1Area(k,j,il,iu+1,area);
          for (int i=il; i<=iu+1; ++i) {
            pfield->b.x1f(k,j,i) -= (pcoord->dx2f(j)*A2(k+1,j,i) - pcoord->dx2f(j)*A2(k,j,i))/area(i);
            //if (area(i)==0) pfield->b.x1f(k,j,i) = 0.0;
            if (std::isnan((pfield->b.x1f(k,j,i)))) fprintf(stderr,"isnan in bx1!\n");
          }
        }
      }
      for (int k=kl; k<=ku; ++k) {
        // reset loop limits for polar boundary
        int jl=js; int ju=je+1;
        if (pcoord->pmy_block->pbval->block_bcs[INNER_X2] == 5) jl=js+1;
        if (pcoord->pmy_block->pbval->block_bcs[OUTER_X2] == 5) ju=je;
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face2Area(k,j,il,iu,area);
          for (int i=il; i<=iu; ++i) {
            pfield->b.x2f(k,j,i) += (pcoord->dx1f(i)*A1(k+1,j,i) - pcoord->dx1f(i)*A1(k,j,i))/area(i);
            if (area(i)==0.0) pfield->b.x2f(k,j,i) = 0;
            if (std::isnan((pfield->b.x2f(k,j,i)))) fprintf(stderr,"isnan in bx2!\n");
            //if ( ju==je && j==je) fprintf(stderr,"B: %g area: %g theta: %g j: %d A1: %g %g \n",pfield->b.x2f(k,j,i), area(i),pcoord->x2f(j),j, 
            //A1_bound(k+1,j,i), A1_bound(k,j,i));
          }
        }
      }
    }

    area.DeleteAthenaArray();
    A1.DeleteAthenaArray();
    A2.DeleteAthenaArray();
    A3.DeleteAthenaArray();

    // Calculate cell-centered magnetic field
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
        ku);
  }

  // Initialize conserved variables
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
      kl, ku);

  // Free scratch arrays
  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();

    // Call user work function to set output variables
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
// Inflow boundary condition
// Inputs:
//   pmb: pointer to MeshBlock
//   pcoord: pointer to Coordinates
//   is,ie,js,je,ks,ke: indices demarkating active region
// Outputs:
//   prim: primitives set in ghost zones
//   bb: face-centered magnetic field set in ghost zones

void InflowBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                    FaceField &bb, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // Set hydro variables
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        prim(IDN,k,j,i) = prim(IDN,k,j,is);
        prim(IEN,k,j,i) = prim(IEN,k,j,is);
        prim(IM1,k,j,i) = std::min(prim(IM1,k,j,is), static_cast<Real>(0.0));
        prim(IM2,k,j,i) = prim(IM2,k,j,is);
        prim(IM3,k,j,i) = prim(IM3,k,j,is);
      }
    }
  }
  if (not MAGNETIC_FIELDS_ENABLED) {
    return;
  }

  // Set radial magnetic field
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        bb.x1f(k,j,i) = bb.x1f(k,j,is);
      }
    }
  }

  // Set polar magnetic field
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je+1; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        bb.x2f(k,j,i) = bb.x2f(k,j,is);
      }
    }
  }

  // Set azimuthal magnetic field
  for (int k = ks; k <= ke+1; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is-ngh; i <= is-1; ++i) {
        bb.x3f(k,j,i) = bb.x3f(k,j,is);
      }
    }
  }
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

  int ncycle = pmy_mesh->ncycle;
  int64_t iseed = -1 - gid;

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



        // if (ncycle ==1428246){ ///perturb pressure at one particular time

        //     Real amp = 0.05;
        //     Real rval = amp*(ran2(&iseed) - 0.5);
        //     phydro->w(IPR,k,j,i) *= (1.0 + 2.0*rval);


        // }
      }
    }
  }

    // if (ncycle== 1428246) {
    //   peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, is, ie, js, je,
    //   ks, ke);
    //   fprintf(stderr, "Adding perturbations\n");
    // }
  return;
}
//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// Inputs:
//   x1,x2,x3: global coordinates to be converted
// Outputs:
//   pr,ptheta,pphi: variables pointed to set to Boyer-Lindquist coordinates
// Notes:
//   conversion is trivial in all currently implemented coordinate systems

static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi) {
  if (COORDINATE_SYSTEM == "schwarzschild" or COORDINATE_SYSTEM == "kerr-schild") {
    *pr = x1;
    *ptheta = x2;
    *pphi = x3;
  }
  return;
}
static void GetCKSCoordinates(Real r, Real th, Real phi, Real *x, Real *y, Real *z){
  *x = std::sin(th) * (r * std::cos(phi) + a * std::sin(phi) );
  *y = std::sin(th) * (r * std::sin(phi) - a * std::cos(phi) );
  *z = std::cos(th) * r; 

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

static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real *pa0, Real *pa1, Real *pa2, Real *pa3) {
  if (COORDINATE_SYSTEM == "schwarzschild") {
    *pa0 = a0_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl;
  } else if (COORDINATE_SYSTEM == "kerr-schild") {
    Real delta = SQR(r) - 2.0*m*r + SQR(a);
    *pa0 = a0_bl + 2.0*m*r/delta * a1_bl;
    *pa1 = a1_bl;
    *pa2 = a2_bl;
    *pa3 = a3_bl + a/delta * a1_bl;
  }
  return;
}

static void TransformCKSLowerVector(Real a0_cks, Real a1_cks, Real a2_cks, Real a3_cks, Real r,
                     Real theta, Real phi, Real x , Real y, Real z,Real *pa0, Real *pa1, Real *pa2, Real *pa3) {

    *pa0 = a0_cks ; 
    *pa1 =         (std::sin(theta)*std::cos(phi)) * a1_cks 
                 + (std::sin(theta)*std::sin(phi)) * a2_cks 
                 + (std::cos(theta)              ) * a3_cks ;

    *pa2 =         (std::cos(theta) * (r*std::cos(phi) + a*std::sin(phi) ) ) * a1_cks
                 + (std::cos(theta) * (r*std::sin(phi) - a*std::cos(phi) ) ) * a2_cks
                 + (-r*std::sin(theta)                                     ) * a3_cks;

    *pa3 =          -y * a1_cks 
                + (x) * a2_cks;


  return;
}

/* Mesh Constructor For Compressed Theta Resolution */

Real CompressedX2(Real x, RegionSize rs)
{

    Real h = 0.3;
    return PI*x + 0.5*(1.0-h) * std::sin(2.0*PI*x);
    //fprintf(stderr,"is this ever called?\n");
    // Real t=2.0*x-1.0;
    // Real w=0.25*(t*(t*t+1.0))+0.5;

    // //fprintf(stderr,"x: %g w: %g result: %g\n", x,w,w*rs.x2max+(1.0-w)*rs.x2min );
    // return w*rs.x2max+(1.0-w)*rs.x2min;
}
/* Mesh Constructor For exponentiated x1 */

Real ExponetialX1(Real x, RegionSize rs)
{

    Real logrmin = std::log(rs.x1min) ;
    Real xbr = std::log(rbr);

    Real x_scaled = logrmin + (x1_harm_max - logrmin) * x; 

    if (x_scaled < xbr) return std::exp(x_scaled);
    else return std::exp(x_scaled + std::pow(cpow2*(x_scaled-xbr),npow2));

}


