//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file lw_implode.cpp
//  \brief Problem generator for square implosion problem
//
// REFERENCE: R. Liska & B. Wendroff, SIAM J. Sci. Comput., 25, 995 (2003)
//========================================================================================


// C++ headers
#include <cmath>  // abs(), NAN, pow(), sqrt()

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../parameter_input.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../utils/utils.hpp" //ran2()





Real gam, gm1,Omega_over_Omega_crit,r_star,gm_,density_floor_,pressure_floor_,cs_ratio,beta_kep,B0;
Real field_norm;

void Oscillating_ix1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);
Real CompressedX2(Real x, RegionSize rs);

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
static void integrate_cool(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim );


Real time_step(MeshBlock *pmb);

Real time_step(MeshBlock *pmb){

  if (pmb->pmy_mesh->ncycle==0) return 1e-20;
  else return 1e10;

}
void GetSphericalCoordinates(Real x1, Real x2, Real x3, Real *pr, Real *ptheta, Real *pphi){

    *pr = std::sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
    *pphi = std::atan2(x2,x1);
    *ptheta = std::acos(x3/(*pr));


  return;
}


Real get_stellar_solution(Real r, Real th, Real *rho, Real* press, Real *vphi){
    // Real Omega_crit = 2.0* std::sqrt(6.0)/9.0 * std::sqrt(gm_/std::pow(r_star,3));
    // Real Omega = Omega_over_Omega_crit * Omega_crit;
    // Real C = -gm_/r_star;
     Real s = r * std::sin(th);

     Real vkep = std::sqrt(gm_/s);
     Real cs = cs_ratio * vkep;



    // Real RHS =C + gm_/r + SQR(Omega * s)/2.0;

    // //fprintf(stderr,"r: %g th: %g RHS: %g Omega: %g\n gm_: %g Omega_rat: %g r_star: %g\n",r,th,RHS,Omega,gm_,Omega_over_Omega_crit, r_star);
    // if (RHS >0 && r < 1.5 * r_star) *rho = std::pow( (RHS*gm1/gam/kappa),(1./gm1) );
    // else *rho = density_floor_;
    // *press = kappa * std::pow(*rho,gam);
    // *vphi = Omega * s;

  *rho = 1.0;
  *press = SQR(cs)*(*rho) *gm1/gam;
  *vphi = Omega_over_Omega_crit * std::sqrt(gm_/s);


}


// void Oscillating_ix1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
//                    FaceField &bb, Real time, Real dt,
//                    int is, int ie, int js, int je, int ks, int ke, int ngh){
//     // copy hydro variables into ghost zones
//  //   for (int n=0; n<(NHYDRO); ++n) {


//                 }
//               }
//             }

//           // copy face-centered magnetic fields into ghost zones
//   if (MAGNETIC_FIELDS_ENABLED) {
//     for (int k=ks; k<=ke; ++k) {
//     for (int j=js; j<=je; ++j) {
// #pragma omp simd
//       for (int i=1; i<=ngh; ++i) {
//         bb.x1f(k,j,(is-i)) = 0.0;
//       }
//     }}

//     for (int k=ks; k<=ke; ++k) {
//     for (int j=js; j<=je+1; ++j) {
// #pragma omp simd
//       for (int i=1; i<=ngh; ++i) {
//         bb.x2f(k,j,(is-i)) = 0.0;
//       }
//     }}

//     for (int k=ks; k<=ke+1; ++k) {
//     for (int j=js; j<=je; ++j) {
// #pragma omp simd
//       for (int i=1; i<=ngh; ++i) {
//         bb.x3f(k,j,(is-i)) = B0;
//       }
//     }}
//   }

//         }


    static Real Ax_func(Real x,Real y, Real z){
      Real r,th,phi;
      GetSphericalCoordinates(x,y,z,&r,&th,&phi);

      return -y/std::pow(r,3.0) * field_norm;
    }
    static Real Ay_func(Real x, Real y, Real z){
      Real r,th,phi;
      GetSphericalCoordinates(x,y,z,&r,&th,&phi);
      return  x/std::pow(r,3.0) * field_norm;   //x 
    }
    static Real Az_func(Real x, Real y, Real z){
      return 0.0 * field_norm;
    }


    static Real Bx_func(Real x,Real y, Real z){
      Real r,th,phi;
      GetSphericalCoordinates(x,y,z,&r,&th,&phi);

      return 3.0 * y * z /std::pow(r,5.0) * field_norm;
    }
    static Real By_func(Real x, Real y, Real z){
      Real r,th,phi;
      GetSphericalCoordinates(x,y,z,&r,&th,&phi);
      return  0.0 * field_norm;   //x 
    }
    static Real Bz_func(Real x, Real y, Real z){
      Real r,th,phi;
      GetSphericalCoordinates(x,y,z,&r,&th,&phi);
      return -(1.0/std::pow(r,3.0) - 3.0*SQR(z)/std::pow(r,5.0) ) * field_norm;
    }



void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim){
 

  int m = -4;
  Real phi_0 = 0.0;
  Real P = 0.65; 
  Real time = pmb->pmy_mesh->time;
  //if (pmb->pmy_mesh->ncycle > 0) return;

   for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {

        Real r_tmp,th,phi;
         GetSphericalCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r_tmp, &th, &phi);

          if (r_tmp < r_star){

                  Real rho,press,vphi;
                  Real r = r_star;
                  get_stellar_solution(r,th,&rho,&press,&vphi);

                  Real rho_max = 1.1 * rho;
                  Real vkep = std::sqrt(gm_/r);

                  Real delta_v = vkep - vphi;
                  Real vphi_pert_amp = delta_v;

                  Real Plm = std::pow(std::sin(th),4.0);

                  Real rho_pert_arg = std::log10(rho/rho_max) * std::sin( 2.0 * PI * time/P + m*phi) * Plm;
                  Real phi_pert = vphi_pert_amp * std::sin( 2.0 * PI * time/P + m*phi + phi_0) * Plm;

                  Real vx = -std::sin(phi) * (vphi + phi_pert);
                  Real vy =  std::cos(phi) * (vphi + phi_pert);
                  Real vz = 0.0  ;

                  Real cs = cs_ratio * vkep;
                  Real P0 = SQR(cs)*(rho) *gm1/gam;
                  Real kappa = P0/std::pow(rho,gam);
      
                  
                  prim(IDN,k,j,i) = rho * std::pow(10,rho_pert_arg);
                  prim(IVX,k,j,i) = vx;
                  prim(IVY,k,j,i) = vy;
                  prim(IVZ,k,j,i) = vz;
                  prim(IPR,k,j,i) = kappa * std::pow(prim(IDN,k,j,i),gam);

              
              
          }



}}}


if (MAGNETIC_FIELDS_ENABLED){
    // for 1,2,3-D
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
         Real r_tmp,th,phi;
         GetSphericalCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2f(j),pmb->pcoord->x3v(k), &r_tmp, &th, &phi);

          if (r_tmp < r_star){
            pfield->b.x2f(k,j,i) = By_func(pcoord->x1v(i),pcoord->x2f(j),pcoord->x3v(k));
            if (area(i)==0.0) pfield->b.x2f(k,j,i) = 0;
          }
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=il; i<=iu; ++i) {
          Real r_tmp,th,phi;
          GetSphericalCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3f(k), &r_tmp, &th, &phi);

          if (r_tmp < r_star){
            pfield->b.x3f(k,j,i) = Bz_func(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3f(k));
          }

         }
        }
      }
    }
    // for 2D and 3D
    if (block_size.nx2 > 1) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=il; i<=iu+1; ++i) {
            Real r_tmp,th,phi;
            GetSphericalCoordinates(pmb->pcoord->x1f(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r_tmp, &th, &phi);

            if (r_tmp < r_star){
              pfield->b.x1f(k,j,i) = Bx_func(pcoord->x1f(i),pcoord->x2v(j),pcoord->x3v(k));
            }
          }
        }
      }
}

}


void Mesh::InitUserMeshData(ParameterInput *pin) {
    //EnrollUserBoundaryFunction(INNER_X1, Oscillating_ix1);
        EnrollUserTimeStepFunction(time_step);
        EnrollUserRadSourceFunction(integrate_cool);

        field_norm =  pin->GetOrAddReal("problem", "field_norm",0.0);




}
static void integrate_cool(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim )

{
  int i, j, k;
  int is, ie, js, je, ks, ke;



    // Allocate memory for primitive/conserved variables
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if (pmb->block_size.nx2 > 1) ncells2 = pmb->block_size.nx2 + 2*(NGHOST);
  if (pmb->block_size.nx3 > 1) ncells3 = pmb->block_size.nx3 + 2*(NGHOST);

  // //prim.InitWithShallowCopy(pmb->phydro->w);

  // /* ath_pout(0, "integrating cooling using Townsend (2009) algorithm.\n"); */

  is = pmb->is;  ie = pmb->ie;
  js = pmb->js;  je = pmb->je;
  ks = pmb->ks;  ke = pmb->ke;
  Real igm1 = 1.0/(gm1);
  Real gamma = gm1+1.;

  //pmb->peos->ConservedToPrimitive(cons, prim_old, pmb->pfield->b, prim, pmb->pfield->bcc,
           //pmb->pcoord, pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);


  
  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {

        Real cs_max = 10 * std::sqrt(gm_/r_star);
        Real v_s = sqrt(gamma*prim(IPR,k,j,i)/prim(IDN,k,j,i));

        if (v_s>cs_max) v_s = cs_max;
        if ( fabs(prim(IVX,k,j,i)) > cs_max) prim(IVX,k,j,i) = cs_max * ( (prim(IVX,k,j,i) >0) - (prim(IVX,k,j,i)<0) ) ;
        if ( fabs(prim(IVY,k,j,i)) > cs_max) prim(IVY,k,j,i) = cs_max * ( (prim(IVY,k,j,i) >0) - (prim(IVY,k,j,i)<0) ) ;
        if ( fabs(prim(IVZ,k,j,i)) > cs_max) prim(IVZ,k,j,i) = cs_max * ( (prim(IVZ,k,j,i) >0) - (prim(IVZ,k,j,i)<0) ) ;

         prim(IPR,k,j,i) = SQR(v_s) *prim(IDN,k,j,i)/gamma ;
       // cons(IEN,k,j,i) = prim(IPR,k,j,i)*igm1 + 0.5*prim(IDN,k,j,i)*( SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)) );
          
      }
    }
  }




  apply_inner_boundary_condition(pmb,prim);

  return;
}


void MeshBlock::UserWorkInLoop(void){

//   int il = is - NGHOST;
//   int iu = ie + NGHOST;
//   int jl = js;
//   int ju = je;
//   if (block_size.nx2 > 1) {
//     jl -= (NGHOST);
//     ju += (NGHOST);
//   }
//   int kl = ks;
//   int ku = ke;
//   if (block_size.nx3 > 1) {
//     kl -= (NGHOST);
//     ku += (NGHOST);
//   }


//   for (int k = ks; k <= ke; ++k) {
//     for (int j = js; j <= je; ++j) {
//       for (int i = is; i <= ie; ++i) {


//         Real vmax = 10 * std::sqrt(gm_/r_star);

//         Real v1 = phydro->w(IVX,k,j,i);
//         Real v2 = phydro->w(IVY,k,j,i);
//         Real v3 = phydro->w(IVZ,k,j,i);

//         if (std::fabs(v1)>vmax) v1 = v1/std::fabs(v1)*vmax;
//         if (std::fabs(v2)>vmax) v2 = v2/std::fabs(v2)*vmax;
//         if (std::fabs(v3)>vmax) v3 = v3/std::fabs(v3)*vmax;

//         // if (std::isnan(v1)) v1 = 0.0;
//         // if (std::isnan(v2)) v2 = 0.0;
//         // if (std::isnan(v3)) v3 = 0.0;

//         phydro->w(IVX,k,j,i) =  v1;
//         phydro->w(IVY,k,j,i) =  v2;
//         phydro->w(IVZ,k,j,i) =  v3; 

//         Real v_s = sqrt(gam*phydro->w(IPR,k,j,i)/phydro->w(IDN,k,j,i));

//         if (v_s>vmax) v_s = vmax;
        
//         phydro->w(IPR,k,j,i) = SQR(v_s) * phydro->w(IDN,k,j,i)/gam ;


//       }
//     }
//   }

//   apply_inner_boundary_condition(pcoord->pmy_block,phydro->w);

//   AthenaArray<Real> bb;
//   if (MAGNETIC_FIELDS_ENABLED) {

//   } else {
//     bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
//   }

//   // Initialize conserved values
// #if (MAGNETIC_FIELDS_ENABLED)
//     peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
//         kl, ku);
// #else 
//     peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
//     bb.DeleteAthenaArray();
// #endif


}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
    

    gam = peos->GetGamma();
    gm1 = peos->GetGamma() - 1.0;

    density_floor_ = peos->GetDensityFloor();
    pressure_floor_ = peos->GetPressureFloor();

    gm_ = pin->GetOrAddReal("problem","GM",0.0);
    r_star = pin->GetOrAddReal("problem","r_star",1.0);
    Omega_over_Omega_crit = pin->GetOrAddReal("problem","Omega_frac",0.95);
    cs_ratio = pin->GetOrAddReal("problem","cs_ratio",0.1);
    beta_kep = pin->GetOrAddReal("problem","beta_kep",10.0);

    B0 = std::sqrt(1.0 * gm_/r_star / beta_kep*2.0);

    
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Liska & Wendroff implosion test problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {


  // Set initial conditions
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // Real x= pcoord->x1v(i);
        // Real y = pcoord->x2v(j);
        // Real z = pcoord->x3v(k);
        // Real r = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
        // Real phi = std::atan2(y,x);
        // Real th = std::acos(z/r);
    
        Real rho,press,vphi;
        // get_stellar_solution(r,th,&rho,&press,&vphi);

        rho = 1e-5;
        press = 1e-7;
        vphi = 0.0;
  
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = rho* vphi;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IEN,k,j,i) = press/gm1 + 0.5 * rho * SQR(vphi) ;


        phydro->w(IVX,k,j,i) = 0.0;
        phydro->w(IVY,k,j,i) = 0.0;
        phydro->w(IVZ,k,j,i) = 0.0;
        phydro->w(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = press;

    }
  }
}

<<<<<<< HEAD

  if (MAGNETIC_FIELDS_ENABLED){

    // for 1,2,3-D
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          pfield->b.x2f(k,j,i) = By_func(pcoord->x1v(i),pcoord->x2f(j),pcoord->x3v(k));
          if (area(i)==0.0) pfield->b.x2f(k,j,i) = 0;
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=il; i<=iu; ++i) {
          pfield->b.x3f(k,j,i) = Bz_func(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3f(k));

         }
        }
      }
    }
    // for 2D and 3D
    if (block_size.nx2 > 1) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=il; i<=iu+1; ++i) {
            pfield->b.x1f(k,j,i) = Bx_func(pcoord->x1f(i),pcoord->x2v(j),pcoord->x3v(k));;
          }
        }
      }

    // Calculate cell-centered magnetic field
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
        ku);
}

UserWorkInLoop();
apply_inner_boundary_condition(pcoord->pmy_block,phydro->w);


=======
UserWorkInLoop();
apply_inner_boundary_condition(pcoord->pmy_block,phydro->w);

>>>>>>> 048dc4b413d4deb38d6ae748be731c24f92af9cb
  return;
}
