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
#include "../field/field.hpp"              // Field
#include "../mesh/mesh.hpp"
#include "../bvals/bvals.hpp"              // BoundaryValues
#include "../utils/utils.hpp" //ran2()





Real gam, gm1,Omega_over_Omega_crit,r_star,gm_,density_floor_,pressure_floor_,cs_ratio,beta_kep,B0;
Real delta_v_frac,field_norm,rho_0;
Real rho_flr_pwr, press_flr_pwr, rho_flr_0, press_flr_0;
Real period,rho_amp,va_ratio;
Real h_slope; 
bool isothermal_;
int m_phi;

void Oscillating_ix1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);
void Outflow_ox1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);
void Outflow_ox2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);
void Outflow_ix2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh);
void set_bboundary_array(const RegionSize block_size, Coordinates *pcoord, 
  const int is, const int ie, const int js, const int je, const int ks, const int ke, FaceField &b_bound);
void apply_user_floors(MeshBlock *pmb,AthenaArray<Real> &prim);
static void user_floors(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim );


Real time_step(MeshBlock *pmb);

Real time_step(MeshBlock *pmb){

  if (pmb->pmy_mesh->ncycle==0) return 1e-20;
  else return 1e10;

}
void GetSphericalCoordinates(Real x1, Real x2, Real x3, Real *pr, Real *ptheta, Real *pphi){

  if (COORDINATE_SYSTEM=="cartesian"){
    *pr = std::sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
    *pphi = std::atan2(x2,x1);
    *ptheta = std::acos(x3/(*pr));
  }
  else if (COORDINATE_SYSTEM == "cylindrical"){
    *pr = std::sqrt( SQR(x1) + SQR(x3));
    *pphi = x2;
    *ptheta = std::acos(x3/(*pr));

  }
  else if (COORDINATE_SYSTEM == "spherical_polar"){
    *pr = x1;
    *pphi = x3;
    *ptheta = x2;
  }


  return;
}


static Real Aphi_func(Real x,Real y, Real z){
      Real r,th,phi;
      GetSphericalCoordinates(x,y,z,&r,&th,&phi);

      if (r<=r_star) return field_norm * std::sin(th)/std::pow(r,2) * std::sin(2.0*PI * r/r_star);
      else return 0.0 ;
}

Real CompressedX2(Real x, RegionSize rs)
{

    Real h = 0.2;
    Real th_max = rs.x2max; 
    Real th_min = rs.x2min;
    return (th_max-th_min) * x + (th_min) + 0.5*(1.0-h)*std::sin(2.0*((th_max-th_min) * x + (th_min)));
    //return PI*x + 0.5*(1.0-h) * std::sin(2.0*PI*x);

}

void get_stellar_solution(Real r, Real th, Real *rho, Real* press, Real *vphi){

  Real s = r * std::sin(th);

  Real vkep_star = std::sqrt(gm_/r_star);
  Real cs = cs_ratio * vkep_star;


  *rho = rho_0;
  *press = SQR(cs)*(*rho)/gam;
  *vphi = Omega_over_Omega_crit * s * std::sqrt(gm_/std::pow(r_star,3.0));

  return;


}

//----------------------------------------------------------------------------------------
//! \fn void OutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                         FaceField &b, Real time, Real dt,
//                         int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x1 boundary

void Outflow_ox1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(n,k,j,ie+i) = prim(n,k,j,ie);
      }
    }}
  }
      for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        if (prim(IVX,k,j,ie+i)<0) prim(IVX,k,j,ie+i) = 0.0;
      }
    }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(ie+i+1)) = b.x1f(k,j,(ie+1));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(ie+i)) = b.x2f(k,j,ie);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(ie+i)) = b.x3f(k,j,ie);
      }
    }}
  }

  return;
}

void Oscillating_ix1(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh){
    // copy hydro variables into ghost zones
 //   for (int n=0; n<(NHYDRO); ++n) {


 	Real phi_0 = 0.0;
 	Real P = period; 
        for (int k=ks; k<=ke; ++k) {
            for (int j=js; j<=je; ++j) {
#pragma simd
                for (int i=1; i<=(NGHOST); ++i) {

                  Real r,th,phi;
                  GetSphericalCoordinates(pmb->pcoord->x1v(is-i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r, &th, &phi);

                  Real rho,press,vphi;
                  get_stellar_solution(r_star,th,&rho,&press,&vphi);

                  Real rho_max = rho_amp * rho;
                  Real delta_v = std::sqrt(gm_/r_star) * (1.0 - Omega_over_Omega_crit);
                  Real vphi_pert_amp = delta_v_frac*delta_v;

                  Real Plm = std::pow(std::sin(th),std::abs(m_phi*1.0));

                  Real rho_pert_arg = std::log10(rho_max/rho) * std::sin( 2.0 * PI * time/P + m_phi*phi) * Plm;
                  Real phi_pert = vphi_pert_amp * std::sin( 2.0 * PI * time/P + m_phi*phi + phi_0) * Plm;


                  if (COORDINATE_SYSTEM == "cylindrical")
                  {
                    prim(IVY,k,j,is-i) = vphi + phi_pert;
                    prim(IVZ,k,j,is-i) = 0.0;
                    prim(IVX,k,j,is-i) = prim(IVX,k,j,is);
                    if (prim(IVX,k,j,is-i) < 0) prim(IVX,k,j,is-i) = 0.0;
                  }
                  else if (COORDINATE_SYSTEM == "spherical_polar"){
                    //prim(IVX,k,j,is-i) = prim(IVX,k,j,is);
                    //if (prim(IVX,k,j,is-i) < 0) prim(IVX,k,j,is-i) = 0.0;
                    prim(IVX,k,j,is-i) = 0.0;
                    prim(IVY,k,j,is-i) = 0.0;
                    prim(IVZ,k,j,is-i) = vphi + phi_pert;

                    // prim(IVX,k,j,is-i) = 1e-4;
                  }
                  prim(IDN,k,j,is-i) = rho * std::pow(10,rho_pert_arg);
                  Real vkep = std::sqrt(gm_/r_star);
                  Real cs = cs_ratio * vkep;
            #ifndef ISOTHERMAL

                  if (isothermal_) prim(IPR,k,j,is-i) = SQR(cs)*(prim(IDN,k,j,is-i));
                  else prim(IPR,k,j,is-i) = SQR(cs)*(prim(IDN,k,j,is-i)) /gam ;
            #endif

                  // fprintf(stderr,"r: %g th: %g bsq: %g Br: %g bth: %g \n vphi: %g rho: %g cs: %g \n",r,th,
                  //   SQR(pmb->pfield->b_bound.x1f(k,j,is-i)) + SQR(pmb->pfield->b_bound.x2f(k,j,is-i)),
                  //   pmb->pfield->b_bound.x1f(k,j,is-i), pmb->pfield->b_bound.x2f(k,j,is-i),vphi + phi_pert,prim(IDN,k,j,is-i), cs );


                }
              }
            }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        bb.x1f(k,j,(is-i)) = pmb->pfield->b_bound.x1f(k,j,is-i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        bb.x2f(k,j,(is-i)) = pmb->pfield->b_bound.x2f(k,j,is-i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        bb.x3f(k,j,(is-i)) = pmb->pfield->b_bound.x3f(k,j,is-i);
      }
    }}
  }

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


}


//----------------------------------------------------------------------------------------
//! \fn void OutflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x2 boundary

void Outflow_ix2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(n,k,js-j,i) = prim(n,k,js,i);
      }
    }}
  }

    // prevent inflow from ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        if (prim(IVY,k,js-j,i) > 0 ) prim(IVY,k,js-j,i) = 0.0 ;
      }
    }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(js-j),i) = b.x1f(k,js,i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(js-j),i) = b.x2f(k,js,i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(js-j),i) = b.x3f(k,js,i);
      }
    }}
  }

  return;
}
//----------------------------------------------------------------------------------------
//! \fn void OutflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x2 boundary

void Outflow_ox2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(n,k,je+j,i) = prim(n,k,je,i);
      }
    }}
  }

    // copy hydro variables into ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        if (prim(IVY,k,je+j,i) <0 )prim(IVY,k,je+j,i) =0.0;
      }
    }}

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(je+j  ),i) = b.x1f(k,(je  ),i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(je+j+1),i) = b.x2f(k,(je+1),i);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(je+j  ),i) = b.x3f(k,(je  ),i);
      }
    }}
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void OutflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, inner x2 boundary

void Fixed_ix2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {

        Real r,th,phi;
        GetSphericalCoordinates(pco->x1v(i), pco->x2v(js-j),pco->x3v(k), &r, &th, &phi);

        Real rho = rho_flr_0 * std::pow(r,rho_flr_pwr);
        Real press = press_flr_0 * std::pow(r,press_flr_pwr);
        if (isothermal_) press = rho * SQR(cs_ratio);

        prim(IDN,k,js-j,i) = rho;
    #ifndef ISOTHERMAL
        prim(IPR,k,js-j,i) = press;
    #endif
        prim(IVX,k,js-j,i) = 0.0;
        prim(IVY,k,js-j,i) = 0.0;
        prim(IVZ,k,js-j,i) = 0.0;
      }
    }}


  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(js-j),i) = 0.0;
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(js-j),i) = 0.0;
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(js-j),i) = 0.0;
      }
    }}
  }

  return;
}
//----------------------------------------------------------------------------------------
//! \fn void OutflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief OUTFLOW boundary conditions, outer x2 boundary

void Fixed_ox2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {

        Real r,th,phi;
        GetSphericalCoordinates(pco->x1v(i), pco->x2v(je+j),pco->x3v(k), &r, &th, &phi);

        Real rho = rho_flr_0 * std::pow(r,rho_flr_pwr);
        Real press = press_flr_0 * std::pow(r,press_flr_pwr);
        if (isothermal_) press = rho * SQR(cs_ratio);

        prim(IDN,k,je+j,i) = rho;
    #ifndef ISOTHERMAL
        prim(IPR,k,je+j,i) = press;
    #endif
        prim(IVX,k,je+j,i) = 0.0;
        prim(IVY,k,je+j,i) = 0.0;
        prim(IVZ,k,je+j,i) = 0.0;
      }
    }}


  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie+1; ++i) {
        b.x1f(k,(je+j  ),i) = 0.0;
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x2f(k,(je+j+1),i) = 0.0;
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        b.x3f(k,(je+j  ),i) = 0.0;
      }
    }}
  }

  return;
}

void Mesh::InitUserMeshData(ParameterInput *pin) {
    EnrollUserBoundaryFunction(INNER_X1, Oscillating_ix1);
    EnrollUserBoundaryFunction(OUTER_X1, Outflow_ox1);

    if (pin->GetString("mesh","ix2_bc") == "user") EnrollUserBoundaryFunction(INNER_X2,Outflow_ix2);
    if (pin->GetString("mesh","ox2_bc") == "user") EnrollUserBoundaryFunction(OUTER_X2,Outflow_ox2);


    EnrollUserTimeStepFunction(time_step);
    EnrollUserRadSourceFunction(user_floors);



    if (COORDINATE_SYSTEM == "spherical_polar" && mesh_size.x2rat<0) EnrollUserMeshGenerator(X2DIR, CompressedX2);



}


static void user_floors(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim )
{


  apply_user_floors(pmb,prim);


  return;
}


void apply_user_floors(MeshBlock *pmb,AthenaArray<Real> &prim){


  Real v_ff = std::sqrt(2.*gm_/(r_star));
  Real va_max; /* Maximum Alfven speed allowed */
  Real bsq,bsq_rho_ceiling;

  va_max = 5.0*v_ff ;
  Real vmax = va_max;


  Real r,th,phi;
   for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {

          GetSphericalCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2f(j),pmb->pcoord->x3v(k), &r, &th, &phi);


          Real rho_flr = rho_flr_0 * std::pow(r,rho_flr_pwr);
          Real p_flr = press_flr_0 *   std::pow(r,press_flr_pwr);

          if (prim(IDN,k,j,i) < rho_flr) prim(IDN,k,j,i) = rho_flr;

          #ifndef ISOTHERMAL
          if (prim(IPR,k,j,i) < p_flr)   prim(IPR,k,j,i) = p_flr;
          #endif



          #ifndef ISOTHERMAL

          Real v_s = sqrt(gam*prim(IPR,k,j,i)/prim(IDN,k,j,i));


          if (v_s>vmax) v_s = vmax;
          prim(IPR,k,j,i) = SQR(v_s) *prim(IDN,k,j,i)/gam ;
          #endif

          Real v1 = prim(IVX,k,j,i);
          Real v2 = prim(IVY,k,j,i);
          Real v3 = prim(IVZ,k,j,i);



          if (std::fabs(v1)>vmax) v1 = v1/std::fabs(v1)*vmax;
          if (std::fabs(v2)>vmax) v2 = v2/std::fabs(v2)*vmax;
          if (std::fabs(v3)>vmax) v3 = v3/std::fabs(v3)*vmax;


          prim(IVX,k,j,i) = v1;
          prim(IVY,k,j,i) = v2;
          prim(IVZ,k,j,i) = v3; 


          if (MAGNETIC_FIELDS_ENABLED){

            bsq = SQR(pmb->pfield->bcc(IB1,k,j,i)) + SQR(pmb->pfield->bcc(IB2,k,j,i)) + SQR(pmb->pfield->bcc(IB3,k,j,i));
            bsq_rho_ceiling = SQR(va_max);

          

            if (prim(IDN,k,j,i) < bsq/bsq_rho_ceiling){
              prim(IDN,k,j,i) = bsq/bsq_rho_ceiling;
             }

            }


        #ifndef ISOTHERMAL

            if (isothermal_){
              prim(IPR,k,j,i) = prim(IDN,k,j,i) * SQR(cs_ratio);

            }
        #endif
              
      
              
        


}}}
}

// void MeshBlock::UserWorkInLoop(void){

// //       // Go through all cells
//   for (int k = ks; k <= ke; ++k) {
//     for (int j = js; j <= je; ++j) {
//       for (int i = is; i <= ie; ++i) {

//         Real r,th,phi;
//         GetSphericalCoordinates(pcoord->x1v(i), pcoord->x2f(j),pcoord->x3v(k), &r, &th, &phi);
//         Real s = r * std::sin(th);


//         Real vmax = 10 * std::sqrt(gm_/r_star);

//         Real v_s = sqrt(gam*phydro->w(IPR,k,j,i)/phydro->w(IDN,k,j,i));

//         if (v_s>vmax) v_s = vmax;
//         phydro->w(IPR,k,j,i) = SQR(v_s) *phydro->w(IDN,k,j,i)/gam ;

//         Real v1 = phydro->w(IVX,k,j,i);
//         Real v2 = phydro->w(IVY,k,j,i);
//         Real v3 = phydro->w(IVZ,k,j,i);



//         if (std::fabs(v1)>vmax) v1 = v1/std::fabs(v1)*vmax;
//         if (std::fabs(v2)>vmax) v2 = v2/std::fabs(v2)*vmax;
//         if (std::fabs(v3)>vmax) v3 = v3/std::fabs(v3)*vmax;


//         phydro->w(IVX,k,j,i) = v1;
//         phydro->w(IVY,k,j,i) = v2;
//         phydro->w(IVZ,k,j,i) = v3; 

//         //impose magnetic floors on density

//         if (MAGNETIC_FIELDS_ENABLED){

//           Real bsq = SQR(pfield->bcc(IB1,k,j,i)) + SQR(pfield->bcc(IB2,k,j,i)) + SQR(pfield->bcc(IB3,k,j,i)) ;
//           Real va = std::sqrt(bsq/phydro->w(IDN,k,j,i));
//           if (va > vmax){
//             va = vmax;
//             phydro->w(IDN,k,j,i) = bsq/SQR(vmax);
//           }
//         }



//       }
//     }
//   }


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

//   AthenaArray<Real> bb;
//   if (MAGNETIC_FIELDS_ENABLED) {

//   } else {
//     bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
//   }

//   // Initialize conserved values
// if (MAGNETIC_FIELDS_ENABLED) peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
//         kl, ku);
// else {
//     peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
//     bb.DeleteAthenaArray();
// }



// }


// void MeshBlock::UserWorkInLoop(void){

// //       // Go through all cells
//   for (int k = ks; k <= ke; ++k) {
//     for (int j = js; j <= je; ++j) {
//       for (int i = is; i <= ie; ++i) {

//         Real r,th,phi;
//         GetSphericalCoordinates(pcoord->x1v(i), pcoord->x2f(j),pcoord->x3v(k), &r, &th, &phi);
//         Real s = r * std::sin(th);


//         Real vmax = 10 * std::sqrt(gm_/r_star);

//         Real v_s = sqrt(gam*phydro->w(IPR,k,j,i)/phydro->w(IDN,k,j,i));

//         if (v_s>vmax) v_s = vmax;
//         phydro->w(IPR,k,j,i) = SQR(v_s) *phydro->w(IDN,k,j,i)/gam ;

//         Real v1 = phydro->w(IVX,k,j,i);
//         Real v2 = phydro->w(IVY,k,j,i);
//         Real v3 = phydro->w(IVZ,k,j,i);



//         if (std::fabs(v1)>vmax) v1 = v1/std::fabs(v1)*vmax;
//         if (std::fabs(v2)>vmax) v2 = v2/std::fabs(v2)*vmax;
//         if (std::fabs(v3)>vmax) v3 = v3/std::fabs(v3)*vmax;


//         phydro->w(IVX,k,j,i) = v1;
//         phydro->w(IVY,k,j,i) = v2;
//         phydro->w(IVZ,k,j,i) = v3; 

//         //impose magnetic floors on density

//         if (MAGNETIC_FIELDS_ENABLED){

//           Real bsq = SQR(pfield->bcc(IB1,k,j,i)) + SQR(pfield->bcc(IB2,k,j,i)) + SQR(pfield->bcc(IB3,k,j,i)) ;
//           Real va = std::sqrt(bsq/phydro->w(IDN,k,j,i));
//           if (va > vmax){
//             va = vmax;
//             phydro->w(IDN,k,j,i) = bsq/SQR(vmax);
//           }
//         }



//       }
//     }
//   }


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

//   AthenaArray<Real> bb;
//   if (MAGNETIC_FIELDS_ENABLED) {

//   } else {
//     bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
//   }

//   // Initialize conserved values
// if (MAGNETIC_FIELDS_ENABLED) peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
//         kl, ku);
// else {
//     peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
//     bb.DeleteAthenaArray();
// }



// }


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
    

    gam = peos->GetGamma();
    gm1 = peos->GetGamma() - 1.0;

    density_floor_ = peos->GetDensityFloor();
    pressure_floor_ = peos->GetPressureFloor();

    gm_ = pin->GetOrAddReal("problem","GM",0.0);
    r_star = pin->GetOrAddReal("problem","r_star",1.0);
    Omega_over_Omega_crit = pin->GetOrAddReal("problem","Omega_frac",0.95);
    cs_ratio = pin->GetOrAddReal("problem","cs_ratio",0.1);
    rho_0 = pin->GetOrAddReal("problem","rho_0",1.0);
    // beta_kep = pin->GetOrAddReal("problem","beta_kep",10.0);

    // B0 = std::sqrt(1.0 * gm_/r_star / beta_kep*2.0);

    delta_v_frac = pin->GetOrAddReal("problem","pert_amp",1.0);

    //ratio between Alfven speed and v_kep at stellar surface (pole)
    va_ratio =  pin->GetOrAddReal("problem", "va_ratio",0.0);

    //b at equator, r= r_star  = 2 pi /(r_star**3)
    field_norm = va_ratio * std::pow(r_star,3.0)/(2.0*PI) * std::sqrt(rho_0);

    m_phi = pin->GetOrAddInteger("problem","m_phi",-4);

    rho_flr_0 = pin->GetOrAddReal("problem","rho_flr_0",1e-5);
    press_flr_0 = pin->GetOrAddReal("problem","press_flr_0",1e-7);
    rho_flr_pwr = pin->GetOrAddReal("problem","rho_flr_pwr",-3.5);
    press_flr_pwr = pin->GetOrAddReal("problem","press_flr_pwr",-35.0/6.0);

    h_slope = pin->GetOrAddReal("problem","h_slope",0.2);

    rho_amp = pin->GetOrAddReal("problem","rho_amp",1.1);

    period = pin->GetOrAddReal("problem","period",0.65);

    isothermal_ = pin->GetOrAddBoolean("problem","isothermal",false);



    if (ALLOCATE_FIELD_BOUNDARY_ARRAYS) set_bboundary_array(block_size, pcoord, is, ie, js, je, ks, ke, pfield->b_bound);
    else {
      fprintf(stderr,"FIELDS NOT ALLOCATED FOR BOUNDARY...RECONFIGURE \n");
      exit(0);
    }
}


void set_bboundary_array(const RegionSize block_size, Coordinates *pcoord, 
  const int is, const int ie, const int js, const int je, const int ks, const int ke, FaceField &b_bound)
{

int nx1 = (ie-is)+1 + 2*(NGHOST);
int nx2 = (je-js)+1 + 2*(NGHOST);
int nx3 = (ke-ks)+1 + 2*(NGHOST);
AthenaArray<Real> area,len,len_p1;
AthenaArray<Real>A1,A2,A3;
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


if (MAGNETIC_FIELDS_ENABLED && COORDINATE_SYSTEM=="spherical_polar"){

    A1.NewAthenaArray(nx3+1,nx2+1,nx1+1);
    A2.NewAthenaArray(nx3+1,nx2+1,nx1+1);
    A3.NewAthenaArray(nx3+1,nx2+1,nx1+1);

    for (int k=kl; k<=ku+1; k++) {
      for (int j=jl; j<=ju+1; j++) {
        for (int i=il; i<=iu+1; i++) {
          if (i<=iu) A1(k,j,i) = 0.0;
          if (j<=ju) A2(k,j,i) = 0.0;
          if (k<=ku) A3(k,j,i) = Aphi_func(pcoord->x1f(i), pcoord->x2f(j), pcoord->x3v(k));
        }
      }
    }

    int ncells1 = block_size.nx1 + 2*(NGHOST);
    int ncells2 = 1, ncells3 = 1;
    if (block_size.nx2 > 1) ncells2 = block_size.nx2 + 2*(NGHOST);
    if (block_size.nx3 > 1) ncells3 = block_size.nx3 + 2*(NGHOST);


    // Initialize interface fields
    area.NewAthenaArray(ncells1+1);
    len.NewAthenaArray(ncells1+1);
    len_p1.NewAthenaArray(ncells1+1);

 // for 1,2,3-D
    for (int k=ks; k<=ke; ++k) {
      // reset loop limits for polar boundary
      int jl=js; int ju=je+1;
      if (pcoord->pmy_block->pbval->block_bcs[INNER_X2] == 5) jl=js+1;
      if (pcoord->pmy_block->pbval->block_bcs[OUTER_X2] == 5) ju=je;
      for (int j=jl; j<=ju; ++j) {
        pcoord->Face2Area(k,j,il,iu,area);
        pcoord->Edge3Length(k,j,il,iu+1,len);
        for (int i=il; i<=iu; ++i) {
          b_bound.x2f(k,j,i) = -1.0*(len(i+1)*A3(k,j,i+1) - len(i)*A3(k,j,i))/area(i);
          if (area(i)==0.0) b_bound.x2f(k,j,i) = 0.0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        pcoord->Face3Area(k,j,il,iu,area);
        pcoord->Edge2Length(k,j,il,iu+1,len);
        for (int i=il; i<=iu; ++i) {
          b_bound.x3f(k,j,i) = (len(i+1)*A2(k,j,i+1) - len(i)*A2(k,j,i))/area(i);
        }
      }
    }

    // for 2D and 3D
    if (block_size.nx2 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          pcoord->Face1Area(k,j,il,iu+1,area);
          pcoord->Edge3Length(k,j  ,il,iu+1,len);
          pcoord->Edge3Length(k,j+1,il,iu+1,len_p1);
          for (int i=il; i<=iu+1; ++i) {
            b_bound.x1f(k,j,i) = (len_p1(i)*A3(k,j+1,i) - len(i)*A3(k,j,i))/area(i);
          }
        }
      }
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=js; j<=je; ++j) {
          pcoord->Face3Area(k,j,il,iu,area);
          pcoord->Edge1Length(k,j  ,il,iu,len);
          pcoord->Edge1Length(k,j+1,il,iu,len_p1);
          for (int i=il; i<=iu; ++i) {
            b_bound.x3f(k,j,i) -= (len_p1(i)*A1(k,j+1,i) - len(i)*A1(k,j,i))/area(i);
          }
        }
      }
    }
    // for 3D only
    if (block_size.nx3 > 1) {
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          pcoord->Face1Area(k,j,il,iu+1,area);
          pcoord->Edge2Length(k  ,j,il,iu+1,len);
          pcoord->Edge2Length(k+1,j,il,iu+1,len_p1);
          for (int i=il; i<=iu+1; ++i) {
            b_bound.x1f(k,j,i) -= (len_p1(i)*A2(k+1,j,i) - len(i)*A2(k,j,i))/area(i);
          }
        }
      }
      for (int k=ks; k<=ke; ++k) {
        // reset loop limits for polar boundary
        int jl=js; int ju=je+1;
        if (pcoord->pmy_block->pbval->block_bcs[INNER_X2] == 5) jl=js+1;
        if (pcoord->pmy_block->pbval->block_bcs[OUTER_X2] == 5) ju=je;
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face2Area(k,j,il,iu,area);
          pcoord->Edge1Length(k  ,j,il,iu,len);
          pcoord->Edge1Length(k+1,j,il,iu,len_p1);
          for (int i=il; i<=iu; ++i) {
            b_bound.x2f(k,j,i) += (len_p1(i)*A1(k+1,j,i) - len(i)*A1(k,j,i))/area(i);
            if (area(i)==0.0) b_bound.x2f(k,j,i) = 0.0;
          }
        }
      }
    }
 

  area.DeleteAthenaArray();
  len.DeleteAthenaArray();
  len_p1.DeleteAthenaArray();
  A1.DeleteAthenaArray();
  A2.DeleteAthenaArray();
  A3.DeleteAthenaArray();


}
else if (MAGNETIC_FIELDS_ENABLED && COORDINATE_SYSTEM=="cylindrical"){


 // for 1,2,3-D
    for (int k=ks; k<=ke; ++k) {
      // reset loop limits for polar boundary
      int jl=js; int ju=je+1;
      if (pcoord->pmy_block->pbval->block_bcs[INNER_X2] == 5) jl=js+1;
      if (pcoord->pmy_block->pbval->block_bcs[OUTER_X2] == 5) ju=je;
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          b_bound.x2f(k,j,i) = 0.0;
        }
      }
    }
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=il; i<=iu; ++i) {

          Real r, th,phi;
          GetSphericalCoordinates(pcoord->x1v(i), pcoord->x2v(j),pcoord->x3f(k), &r, &th, &phi);

          if (r<r_star) b_bound.x3f(k,j,i) = va_ratio * std::sqrt(gm_/r_star)*std::sqrt(rho_0);
          else b_bound.x3f(k,j,i) = 0.0;
        }
      }
    }

      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=il; i<=iu+1; ++i) {
            b_bound.x1f(k,j,i) = 0.0;
          }
        }
      }


}

}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {


  // Set initial conditions
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {

        Real r,th,phi;
        GetSphericalCoordinates(pcoord->x1v(i), pcoord->x2v(j),pcoord->x3v(k), &r, &th, &phi);
        Real z = r * std::cos(th);

        Real rho,press,vphi;
        get_stellar_solution(r,th,&rho,&press,&vphi);

        rho = rho_flr_0 * std::pow(r,rho_flr_pwr);
        press = press_flr_0 * std::pow(r,press_flr_pwr);
        if (isothermal_) press = rho * SQR(cs_ratio);

        vphi = 0.0;
  
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IDN,k,j,i) = rho;
  #ifndef ISOTHERMAL
        phydro->u(IEN,k,j,i) = press/gm1 ;
  #endif

    }
  }
}
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

if (MAGNETIC_FIELDS_ENABLED){

    // for 1,2,3-D
    for (int k=kl; k<=ku; ++k) {
      // reset loop limits for polar boundary
      int jl=js; int ju=je+1;
      if (pcoord->pmy_block->pbval->block_bcs[INNER_X2] == 5) jl=js+1;
      if (pcoord->pmy_block->pbval->block_bcs[OUTER_X2] == 5) ju=je;
      for (int j=jl; j<=ju; ++j) {
        for (int i=il; i<=iu; ++i) {
          pfield->b.x2f(k,j,i) = pfield->b_bound.x2f(k,j,i);
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=il; i<=iu; ++i) {
          pfield->b.x3f(k,j,i) = pfield->b_bound.x3f(k,j,i);
         }
        }
      }
    
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je; ++j) {
          for (int i=il; i<=iu+1; ++i) {
            pfield->b.x1f(k,j,i) = pfield->b_bound.x1f(k,j,i);
          }
        }
      }

    // Calculate cell-centered magnetic field
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
        ku);

  #ifndef ISOTHERMAL

  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IEN,k,j,i) += 
          0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
               SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
               SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i))));
      }
    }}

    #endif


}


  return;
}
