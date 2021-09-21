//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//  \brief Adds source terms due to point mass AT ORIGIN

// Athena++ headers
#include "hydro_srcterms.hpp"
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../mesh/mesh.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../hydro.hpp"
#include <stdio.h>
#include <stdlib.h>

//----------------------------------------------------------------------------------------
//! \fn void HydroSourceTerms::PointMass
//  \brief Adds source terms due to point mass AT ORIGIN

void HydroSourceTerms::PointMass(const Real dt, const AthenaArray<Real> *flux,
  const AthenaArray<Real> &prim, AthenaArray<Real> &cons)
{
  MeshBlock *pmb = pmy_hydro_->pmy_block;
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        if (COORDINATE_SYSTEM == "cylindrical" || COORDINATE_SYSTEM == "spherical_polar"){
            Real den = prim(IDN,k,j,i);
            Real src = dt*den*pmb->pcoord->coord_src1_i_(i)*gm_/pmb->pcoord->x1v(i);
            cons(IM1,k,j,i) -= src;
            if (NON_BAROTROPIC_EOS) cons(IEN,k,j,i) -=
              dt*0.5*(pmb->pcoord->phy_src1_i_(i)*flux[X1DIR](IDN,k,j,i)*gm_
                     +pmb->pcoord->phy_src2_i_(i)*flux[X1DIR](IDN,k,j,i+1)*gm_);
          }
          else{
              Real r =0.;
              Real r_cut_off = 1e-10;  /* Make potential go to zero below this radius to avoid singularity */


              if (pmb->block_size.nx1>1) r += SQR(pmb->pcoord->x1v(i));
              if (pmb->block_size.nx2>1) r += SQR(pmb->pcoord->x2v(j));
              if (pmb->block_size.nx3>1) r += SQR(pmb->pcoord->x3v(k));
              r = sqrt(r);
              if (r<r_cut_off) r = r_cut_off ;
              Real den = prim(IDN,k,j,i);
              Real src = dt*den*gm_/SQR(r);
              if (pmb->block_size.nx1>1) cons(IM1,k,j,i) -= src * pmb->pcoord->x1v(i)/r;
              if (pmb->block_size.nx2>1) cons(IM2,k,j,i) -= src * pmb->pcoord->x2v(j)/r;
              if (pmb->block_size.nx3>1) cons(IM3,k,j,i) -= src * pmb->pcoord->x3v(k)/r;


              if (NON_BAROTROPIC_EOS) {
                  if (pmb->block_size.nx1>1) cons(IEN,k,j,i) -= dt*0.5*gm_/SQR(r) *
                        (flux[X1DIR](IDN,k,j,i) + flux[X1DIR](IDN,k,j,i+1)) *pmb->pcoord->x1v(i)/r;
                  if (pmb->block_size.nx2>1) cons(IEN,k,j,i) -= dt*0.5*gm_/SQR(r) *
                        (flux[X2DIR](IDN,k,j,i) + flux[X2DIR](IDN,k,j+1,i)) * pmb->pcoord->x2v(j)/r;
                  if (pmb->block_size.nx3>1) cons(IEN,k,j,i) -= dt*0.5*gm_/SQR(r) *
                        (flux[X3DIR](IDN,k,j,i) + flux[X3DIR](IDN,k+1,j,i)) * pmb->pcoord->x3v(k)/r;
      
              }

          }
          
      }
    }
  }

  return;
}
