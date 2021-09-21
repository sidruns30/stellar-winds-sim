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




#if MAGNETIC_FIELDS_ENABLED
#error "This problem generator does not support magnetic fields"
#endif

Real gam, gm1,Omega_over_Omega_crit,r_star,gm_,kappa,density_floor_,pressure_floor_;

void Fixed_ix1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);



Real get_stellar_solution(Real r, Real th, Real *rho, Real* press, Real *vphi){
    Real Omega_crit = 2.0* std::sqrt(6.0)/9.0 * std::sqrt(gm_/std::pow(r_star,3));
    Real Omega = Omega_over_Omega_crit * Omega_crit;
    Real C = -gm_/r_star;
    Real s = r * std::sin(th);



    Real RHS =C + gm_/r + SQR(Omega * s)/2.0;

    //fprintf(stderr,"r: %g th: %g RHS: %g Omega: %g\n gm_: %g Omega_rat: %g r_star: %g\n",r,th,RHS,Omega,gm_,Omega_over_Omega_crit, r_star);
    if (RHS >0 && r < 1.5 * r_star) *rho = std::pow( (RHS*gm1/gam/kappa),(1./gm1) );
    else *rho = density_floor_;
    *press = kappa * std::pow(*rho,gam);
    *vphi = Omega * s;


}


void Fixed_ix1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh){
    // copy hydro variables into ghost zones
 //   for (int n=0; n<(NHYDRO); ++n) {
        for (int k=ks; k<=ke; ++k) {
            for (int j=js; j<=je; ++j) {
#pragma simd
                for (int i=1; i<=(NGHOST); ++i) {
                  Real r = pmb->pcoord->x1v(is-i);
                  Real th = pmb->pcoord->x2v(j);
                  Real rho,press,vphi;
                  get_stellar_solution(r,th,&rho,&press,&vphi);

                  prim(IVX,k,j,is-i) = 0.0;
                  prim(IVY,k,j,is-i) = 0.0;
                  prim(IVZ,k,j,is-i) = vphi;
                  prim(IDN,k,j,is-i) = rho;
                  prim(IPR,k,j,is-i) = press ;





                }
              }
            }

        }


void Mesh::InitUserMeshData(ParameterInput *pin) {
    EnrollUserBoundaryFunction(INNER_X1, Fixed_ix1);

}

void MeshBlock::UserWorkInLoop(void){



    int64_t iseed = -1 - gid;
    Real t = pmy_mesh->time;
    Real dt = pmy_mesh->dt;
    Real kicks_per_orbit = 4.0;
    Real t_orbit = 2.0*PI/std::sqrt(std::pow(r_star,3.0)/gm_);
    int time_step_kick_frequency = 400;

    Real t_kick = t_orbit/kicks_per_orbit;

    

    Real Omega_crit = 2.0* std::sqrt(6.0)/9.0 * std::sqrt(gm_/std::pow(r_star,3));
    Real Omega = Omega_over_Omega_crit * Omega_crit;
      // Go through all cells
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {


        Real r = pcoord->x1v(i);
        Real th = pcoord->x2v(j);
        Real s = r * std::sin(th);

        Real amp = 0.0 ;
        Real rho,press,vphi;
        get_stellar_solution(r,th,&rho,&press,&vphi);
        //if ( (std::fabs(std::fmod(t,t_kick)) <= dt) &&(std::fabs(std::fmod(t,t_kick)) < std::fabs(std::fmod(t+dt,t_kick))) ) {
        if ( (pmy_mesh->ncycle % time_step_kick_frequency ==0) && ( pmy_mesh->ncycle > time_step_kick_frequency-1) &&
          (rho>0) && (r<1.5*r_star) )
        {
          amp =0.1 * Omega *s ; 
          //if (i==is && k==ks && j==js) fprintf(stderr,"applying momentum kicks \n");
        }
        Real  rvel1 = amp*(ran2(&iseed) - 0.5)*2.0 ;
        Real  rvel2 = amp*(ran2(&iseed) - 0.5)*2.0 ;



        phydro->u(IM2,k,j,i) += phydro->u(IDN,k,j,i) * rvel1;
        phydro->u(IM3,k,j,i) += phydro->u(IDN,k,j,i) * rvel2; 


      }
    }
  }



}


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
    

    gam = peos->GetGamma();
    gm1 = peos->GetGamma() - 1.0;

    density_floor_ = peos->GetDensityFloor();
    pressure_floor_ = peos->GetPressureFloor();

    gm_ = pin->GetOrAddReal("problem","GM",0.0);
    r_star = pin->GetOrAddReal("problem","r_star",1.0);
    kappa  = pin->GetOrAddReal("problem","kappa",1.0);
    Omega_over_Omega_crit = pin->GetOrAddReal("problem","Omega_frac",0.95);
    
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
        Real r = pcoord->x1v(i);
        Real th =pcoord->x2v(j) ;
        Real rho,press,vphi;
        get_stellar_solution(r,th,&rho,&press,&vphi);
  
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = rho* vphi;
        phydro->u(IDN,k,j,i) = rho;
        phydro->u(IEN,k,j,i) = press/gm1 + 0.5 * rho * SQR(vphi) ;

    }
  }
}

  return;
}
