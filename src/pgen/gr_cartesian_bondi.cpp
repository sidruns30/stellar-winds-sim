//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gr_bondi.cpp
//  \brief Problem generator for spherically symmetric black hole accretion.

// C++ headers
#include <cmath>  // abs(), NAN, pow(), sqrt()

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
#include <cfloat>

// Configuration checking
#if not GENERAL_RELATIVITY
#error "This problem generator must be used with general relativity"
#endif

// Declarations
void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);
void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
static void inner_boundary(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim );


static void GetBoyerLindquistCoordinates(Real x1, Real x2, Real x3, Real *pr,
                                         Real *ptheta, Real *pphi);
static void TransformVector(Real a0_bl, Real a1_bl, Real a2_bl, Real a3_bl, Real r,
                     Real theta, Real phi, Real *pa0, Real *pa1, Real *pa2, Real *pa3);
static void CalculatePrimitives(Real r, Real temp_min, Real temp_max, Real *prho,
                                Real *ppgas, Real *put, Real *pur);
static Real TemperatureMin(Real r, Real t_min, Real t_max);
static Real TemperatureBisect(Real r, Real t_min, Real t_max);
static Real TemperatureResidual(Real t, Real r);
void Cartesian_GR(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);

Real DivergenceB(MeshBlock *pmb, int iout);
Real Flux_Calc(MeshBlock *pmb, int iout);

// Global variables
Real m, a;          // black hole mass and spin
Real rh;                /* Horizon radius */
Real risco;             /* ISCO radius */
Real dfloor,pfloor;
static Real n_adi, k_adi;  // hydro parameters
static Real r_crit;        // sonic point radius
static Real c1, c2;        // useful constants
static Real bsq_over_rho;  // b^2/rho at inner radius

//----------------------------------------------------------------------------------------
// Function for initializing global mesh properties
// Inputs:
//   pin: input parameters (unused)
// Outputs: (none)

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read problem parameters
  k_adi = pin->GetReal("hydro", "k_adi");
  r_crit = pin->GetReal("problem", "r_crit");
  bsq_over_rho = 0.0;
  if (MAGNETIC_FIELDS_ENABLED) {
    bsq_over_rho = pin->GetReal("problem", "bsq_over_rho");
  }

  // Enroll boundary functions
  EnrollUserBoundaryFunction(INNER_X1, FixedBoundary);
  EnrollUserBoundaryFunction(OUTER_X1, FixedBoundary);
  EnrollUserBoundaryFunction(INNER_X2, FixedBoundary);
  EnrollUserBoundaryFunction(OUTER_X2, FixedBoundary);
  EnrollUserBoundaryFunction(INNER_X3, FixedBoundary);
  EnrollUserBoundaryFunction(OUTER_X3, FixedBoundary);

  //Enroll metric
  EnrollUserMetric(Cartesian_GR);

  EnrollUserRadSourceFunction(inner_boundary);



  // AllocateUserHistoryOutput(6);

  // EnrollUserHistoryOutput(0,Flux_Calc,"mdot");
  // EnrollUserHistoryOutput(1,Flux_Calc,"jdot");
  // EnrollUserHistoryOutput(2,Flux_Calc,"edot");
  // EnrollUserHistoryOutput(3,Flux_Calc,"vol");

    

  //   if (MAGNETIC_FIELDS_ENABLED){
  //     EnrollUserHistoryOutput(4,Flux_Calc,"phibh");
  //     EnrollUserHistoryOutput(5, DivergenceB, "divB");
  //   }
    
  return;
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
  const Real temp_min = 1.0e-2;  // lesser temperature root must be greater than this
  const Real temp_max = 1.0e1;   // greater temperature root must be less than this

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


  // Get ratio of specific heats
  Real gamma_adi = peos->GetGamma();
  n_adi = 1.0/(gamma_adi-1.0);

  // Prepare scratch arrays
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);

  // Prepare various constants for determining primitives
  Real u_crit_sq = m/(2.0*r_crit);                                          // (HSW 71)
  Real u_crit = -std::sqrt(u_crit_sq);
  Real t_crit = n_adi/(n_adi+1.0) * u_crit_sq/(1.0-(n_adi+3.0)*u_crit_sq);  // (HSW 74)
  c1 = std::pow(t_crit, n_adi) * u_crit * SQR(r_crit);                      // (HSW 68)
  c2 = SQR(1.0 + (n_adi+1.0) * t_crit) * (1.0 - 3.0*m/(2.0*r_crit));        // (HSW 69)

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i = il; i <= iu; ++i) {
        Real r, theta, phi;
        GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &r,
            &theta, &phi);
        Real rho, pgas, ut, ur;
        CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
        Real u0, u1, u2, u3;
        TransformVector(ut, ur, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &u0, &u1, &u2, &u3);

        Real u_0,u_1,u_2,u_3;
        pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);

        Real usq = u0*u_0 + u1*u_1 + u2*u_2 + u3*u_3;



        Real g_raised[4][4];

        g_raised[0][0] = g(I00,i)*gi(I00,i) + g(I01,i)*gi(I01,i) + g(I02,i)*gi(I02,i) + g(I03,i)*gi(I03,i);
        g_raised[0][1] = g(I00,i)*gi(I01,i) + g(I01,i)*gi(I11,i) + g(I02,i)*gi(I12,i) + g(I03,i)*gi(I13,i);
        g_raised[1][0] = g(I01,i)*gi(I00,i) + g(I11,i)*gi(I01,i) + g(I12,i)*gi(I02,i) + g(I13,i)*gi(I03,i);
        g_raised[0][2] = g(I00,i)*gi(I02,i) + g(I01,i)*gi(I12,i) + g(I02,i)*gi(I22,i) + g(I03,i)*gi(I23,i);
        g_raised[2][0] = g(I02,i)*gi(I00,i) + g(I12,i)*gi(I01,i) + g(I22,i)*gi(I02,i) + g(I23,i)*gi(I03,i);
        g_raised[0][3] = g(I00,i)*gi(I03,i) + g(I01,i)*gi(I13,i) + g(I02,i)*gi(I23,i) + g(I03,i)*gi(I33,i);
        g_raised[0][3] = g(I03,i)*gi(I00,i) + g(I13,i)*gi(I01,i) + g(I23,i)*gi(I02,i) + g(I33,i)*gi(I03,i);
        g_raised[1][1] = g(I01,i)*gi(I01,i) + g(I11,i)*gi(I11,i) + g(I12,i)*gi(I12,i) + g(I13,i)*gi(I13,i);
        g_raised[2][1] = g(I02,i)*gi(I01,i) + g(I12,i)*gi(I11,i) + g(I22,i)*gi(I12,i) + g(I23,i)*gi(I13,i);
        g_raised[1][2] = g(I01,i)*gi(I02,i) + g(I11,i)*gi(I12,i) + g(I12,i)*gi(I22,i) + g(I13,i)*gi(I23,i);     
        g_raised[2][2] = g(I02,i)*gi(I02,i) + g(I12,i)*gi(I12,i) + g(I22,i)*gi(I22,i) + g(I23,i)*gi(I23,i);  
        g_raised[2][3] = g(I02,i)*gi(I03,i) + g(I12,i)*gi(I13,i) + g(I22,i)*gi(I23,i) + g(I23,i)*gi(I33,i);
        g_raised[3][2] = g(I03,i)*gi(I02,i) + g(I13,i)*gi(I12,i) + g(I23,i)*gi(I22,i) + g(I33,i)*gi(I23,i);
        g_raised[3][1] = g(I03,i)*gi(I01,i) + g(I13,i)*gi(I11,i) + g(I23,i)*gi(I12,i) + g(I33,i)*gi(I13,i);
        g_raised[1][3] = g(I01,i)*gi(I03,i) + g(I11,i)*gi(I13,i) + g(I12,i)*gi(I23,i) + g(I13,i)*gi(I33,i);
        g_raised[3][3] = g(I03,i)*gi(I03,i) + g(I13,i)*gi(I13,i) + g(I23,i)*gi(I23,i) + g(I33,i)*gi(I33,i);

        // for (int mu =0; mu<=3; ++mu){
        //   for (int nu = 0; nu<=3; ++nu){

        //     fprintf(stderr,"mu: %d nu: %d g_mu^nu: %g \n",mu, nu,g_raised[mu][nu]);

        //   }
        // }
        // if ((std::fabs( 1.0 + usq )<1e-5 )) fprintf(stderr,"usq: %g u0: %g u1: %g u2: %g u3: %g \n",usq,u0,u1,u2,u3);  

        Real uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
        Real uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
        Real uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;
        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas;
        phydro->w(IM1,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
        phydro->w(IM2,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
        phydro->w(IM3,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;


        Real tmp = g(I11,i)*uu1*uu1 + 2.0*g(I12,i)*uu1*uu2 + 2.0*g(I13,i)*uu1*uu3
                 + g(I22,i)*uu2*uu2 + 2.0*g(I23,i)*uu2*uu3
                 + g(I33,i)*uu3*uu3;
        Real gamma_new = std::sqrt(1.0 + tmp);

        Real gi00_bl = -1.0/(1.0 - 2.0*m/r);
        Real gi01_bl = 0.0;
        Real g11_bl = 1.0/(1.0 - 2.0*m/r);
        Real uur = ur - gi(I01,i)/gi(I00,i) * ut;

        tmp = g11_bl*uur*uur;
        Real gamma_pre = std::sqrt(1.0 + tmp);


        //fprintf(stderr,"r: %g gamma_pre: %g gamma_new: %g \n",r,gamma_pre,gamma_new);



        //fprintf(stderr,"r: %g rho: %g pgas: %g uu1: %g uu2: %g uu3: %g\n",r,rho,pgas,uu1,uu2,uu3);

        //sanity checks
        // g_mu a g^a nu = eta_

        //Real g(I00,i)*gi(I00) + g(I01)*gi(I01)
      }
    }
  }

  // Initialize magnetic field
  if (MAGNETIC_FIELDS_ENABLED) {

    // Find normalization
    Real r, theta, phi;
    GetBoyerLindquistCoordinates(pcoord->x1f(is), pcoord->x2v((jl+ju)/2),
        pcoord->x3v((kl+ku)/2), &r, &theta, &phi);
    Real rho, pgas, ut, ur;
    CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
    Real bbr = 1.0/SQR(r);
    Real bt = 1.0/(1.0-2.0*m/r) * bbr * ur;
    Real br = (bbr + bt * ur) / ut;
    Real bsq = -(1.0-2.0*m/r) * SQR(bt) + 1.0/(1.0-2.0*m/r) * SQR(br);
    Real bsq_over_rho_actual = bsq/rho;
    Real normalization = std::sqrt(bsq_over_rho/bsq_over_rho_actual);

    // Set face-centered field
    for (int k = kl; k <= ku+1; ++k) {
      for (int j = jl; j <= ju+1; ++j) {
        for (int i = il; i <= iu+1; ++i) {

          // Set B^1
          if (j != ju+1 and k != ku+1) {
            GetBoyerLindquistCoordinates(pcoord->x1f(i), pcoord->x2v(j), pcoord->x3v(k),
                &r, &theta, &phi);
            CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(r);
            bt = 1.0/(1.0-2.0*m/r) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &u0, &u1, &u2, &u3);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &b0, &b1, &b2, &b3);
            pfield->b.x1f(k,j,i) = b1 * u0 - b0 * u1;
          }

          // Set B^2
          if (i != iu+1 and k != ku+1) {
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2f(j), pcoord->x3v(k),
                &r, &theta, &phi);
            CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(r);
            bt = 1.0/(1.0-2.0*m/r) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &u0, &u1, &u2, &u3);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &b0, &b1, &b2, &b3);
            pfield->b.x2f(k,j,i) = b2 * u0 - b0 * u2;
          }

          // Set B^3
          if (i != iu+1 and j != ju+1) {
            GetBoyerLindquistCoordinates(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3f(k),
                &r, &theta, &phi);
            CalculatePrimitives(r, temp_min, temp_max, &rho, &pgas, &ut, &ur);
            bbr = normalization/SQR(r);
            bt = 1.0/(1.0-2.0*m/r) * bbr * ur;
            br = (bbr + bt * ur) / ut;
            Real u0, u1, u2, u3;
            TransformVector(ut, ur, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &u0, &u1, &u2, &u3);
            Real b0, b1, b2, b3;
            TransformVector(bt, br, 0.0, 0.0, pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &b0, &b1, &b2, &b3);
            pfield->b.x3f(k,j,i) = b3 * u0 - b0 * u3;
          }
        }
      }
    }

    // Calculate cell-centered magnetic field
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
        ku);
  }

  // Initialize conserved variables
  peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
      kl, ku);

  // for (int k = kl; k <= ku; ++k) {
  //   for (int j = jl; j <= ju; ++j) {
  //     for (int i = il; i <= iu; ++i) {
  //         fprintf(stderr,"rho: %g rho_u0: %g\n",phydro->w(IDN,k,j,i),phydro->u(IDN,k,j,i));
  //       }
  //     }
  //   }
  // Free scratch arrays
  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();

    UserWorkInLoop();

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
              

              //set uu assuming u is zero
              Real gamma = 1.0;
              Real alpha = std::sqrt(-1.0/gi(I00,i));
              Real u0 = gamma/alpha;
              Real uu1 = - gi(I01,i)/gi(I00,i) * u0;
              Real uu2 = - gi(I02,i)/gi(I00,i) * u0;
              Real uu3 = - gi(I03,i)/gi(I00,i) * u0;
              
              prim(IDN,k,j,i) = dfloor;
              prim(IVX,k,j,i) = 0.;
              prim(IVY,k,j,i) = 0.;
              prim(IVZ,k,j,i) = 0.;
              prim(IPR,k,j,i) = pfloor;
            
              
              
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


void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){

    
  int N_user_vars = 6;
  if (MAGNETIC_FIELDS_ENABLED) {
    AllocateUserOutputVariables(N_user_vars);
  } else {
    AllocateUserOutputVariables(N_user_vars);
  }
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie+1);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie+1);


  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));


  a = pcoord->GetSpin();
  m = pcoord->GetMass();
  rh = m * ( 1.0 + std::sqrt(1.0-SQR(a)) );

  r_inner_boundary = rh/2.0;
  Real Z1 = 1.0 + std::pow(1.0-a*a,1.0/3.0) * ( std::pow(1.0+a,1.0/3.0) + std::pow(1.0-a,1.0/3.0) ) ;
  Real Z2 = std::sqrt(3.0*a*a + Z1*Z1);
  int sgn = 1;
  if (a>0) sgn = -1;
  risco = 3.0 + Z2 + sgn*std::sqrt((3.0-Z1) * (3.0+Z1 + 2.0*Z2));
  risco *= m;


  std::string init_file_name;

  int ncells1 = block_size.nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if (block_size.nx2 > 1) ncells2 = block_size.nx2 + 2*(NGHOST);
  if (block_size.nx3 > 1) ncells3 = block_size.nx3 + 2*(NGHOST);


  
    
}

/* Store some useful variables like mdot and vr */

Real DivergenceB(MeshBlock *pmb, int iout)
{
  Real divb=0;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  AthenaArray<Real> face1, face2p, face2m, face3p, face3m;
  FaceField &b = pmb->pfield->b;

  face1.NewAthenaArray((ie-is)+2*NGHOST+2);
  face2p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face2m.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3p.NewAthenaArray((ie-is)+2*NGHOST+1);
  face3m.NewAthenaArray((ie-is)+2*NGHOST+1);

  for(int k=ks; k<=ke; k++) {
      int jl=js; int ju=je+1;
      if (pmb->pbval->block_bcs[INNER_X2] == 5) jl=js+1;
      if (pmb->pbval->block_bcs[OUTER_X2] == 5) ju=je;
    for(int j=jl; j<=ju; j++) {
      pmb->pcoord->Face1Area(k,   j,   is, ie+1, face1);
      pmb->pcoord->Face2Area(k,   j+1, is, ie,   face2p);
      pmb->pcoord->Face2Area(k,   j,   is, ie,   face2m);
      pmb->pcoord->Face3Area(k+1, j,   is, ie,   face3p);
      pmb->pcoord->Face3Area(k,   j,   is, ie,   face3m);
      for(int i=is; i<=ie; i++) {

        divb+= (face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
              +face2p(i)*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i)
              +face3p(i)*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));
      }
    }
  }

  face1.DeleteAthenaArray();
  face2p.DeleteAthenaArray();
  face2m.DeleteAthenaArray();
  face3p.DeleteAthenaArray();
  face3m.DeleteAthenaArray();

  return divb;
}


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

/* Compute Fluxes Near Horizon */

Real Flux_Calc(MeshBlock *pmb, int iout){

  Real mdot = 0;
  Real jdot = 0;
  Real edot = 0;
  Real phibh = 0;
  Real area = 0;
  Real volume = 0;
  Real gam = pmb->peos->GetGamma();
  Real gm1 = gam -1.0;

  AthenaArray<Real> g, gi;
  g.InitWithShallowCopy(pmb->ruser_meshblock_data[0]);
  gi.InitWithShallowCopy(pmb->ruser_meshblock_data[1]);
  Real a = pmb->pcoord->GetSpin();
  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      pmb->pcoord->CellMetric(k, j, pmb->is, pmb->ie, g, gi);
      for(int i=pmb->is; i<=pmb->ie; i++) {

        Real r,th,ph;
        GetBoyerLindquistCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r, &th, &ph);

        if ( (r<5.0) && (r>rh) ){
          Real det = (SQR(r) + SQR(a) * SQR(std::cos(th))) * std::abs(std::sin(th));

          Real dA = det * pmb->pcoord->dx2f(j)*pmb->pcoord->dx3f(k) ;
          Real dV = dA * pmb->pcoord->dx1f(i);
          Real rho = pmb->phydro->w(IDN,k,j,i);
          Real p = pmb->phydro->w(IPR,k,j,i);
          Real ud0,ud1,ud2,ud3;
          Real uu0 = pmb->user_out_var(0,k,j,i);
          Real uu1 = pmb->user_out_var(1,k,j,i);
          Real uu2 = pmb->user_out_var(2,k,j,i);
          Real uu3 = pmb->user_out_var(3,k,j,i);

          pmb->pcoord->LowerVectorCell(uu0, uu1, uu2, uu3, k, j, i, &ud0, &ud1, &ud2, &ud3);

          mdot += rho * uu1 * dA;
          jdot += (rho + (gm1)/gam * p )*uu1 * ud3 *dA;
          edot += (rho + (gm1)/gam * p )*uu1 * ud0 *dA;
          volume  += dV;

          if (MAGNETIC_FIELDS_ENABLED){

                      // Calculate 4-magnetic field
            Real bb1 = pmb->pfield->bcc(IB1,k,j,i);
            Real bb2 = pmb->pfield->bcc(IB2,k,j,i);
            Real bb3 = pmb->pfield->bcc(IB3,k,j,i);
            Real b0 = g(I01,i)*uu0*bb1 + g(I02,i)*uu0*bb2 + g(I03,i)*uu0*bb3
                    + g(I11,i)*uu1*bb1 + g(I12,i)*uu1*bb2 + g(I13,i)*uu1*bb3
                    + g(I12,i)*uu2*bb1 + g(I22,i)*uu2*bb2 + g(I23,i)*uu2*bb3
                    + g(I13,i)*uu3*bb1 + g(I23,i)*uu3*bb2 + g(I33,i)*uu3*bb3;
            Real b1 = (bb1 + b0 * uu1) / uu0;
            Real b2 = (bb2 + b0 * uu2) / uu0;
            Real b3 = (bb3 + b0 * uu3) / uu0;
            Real b_0, b_1, b_2, b_3;
            pmb->pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);
            Real bsq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
            jdot += ( bsq * uu1 * ud3 - b1 * b_3 ) * dA ;
            edot += ( bsq * uu1 * ud0 - b1 * b_0 ) * dA; 
            phibh += pmb->pfield->bcc(IB1,k,j,i) * dA;

          }
        }
      }
    }
  }

 if (iout ==0) return mdot;
 else if (iout ==1) return jdot;
 else if (iout==2 ) return edot;
 else if (iout ==3) return  volume;
 else if (iout==4) return phibh;
 else {
  fprintf(stderr,"Invalid choice of iout in Flux_Calc.  iout = %d \n",iout);
  //exit(0);
 }
}

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh) {
  return;
}


//----------------------------------------------------------------------------------------
// Function for calculating primitives given radius
// Inputs:
//   r: Schwarzschild radius
//   temp_min,temp_max: bounds on temperature
// Outputs:
//   prho: value set to density
//   ppgas: value set to gas pressure
//   put: value set to u^t in Schwarzschild coordinates
//   pur: value set to u^r in Schwarzschild coordinates
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)

static void CalculatePrimitives(Real r, Real temp_min, Real temp_max, Real *prho,
                                Real *ppgas, Real *put, Real *pur) {
  // Calculate solution to (HSW 76)
  Real temp_neg_res = TemperatureMin(r, temp_min, temp_max);
  Real temp;
  if (r <= r_crit) {  // use lesser of two roots
    temp = TemperatureBisect(r, temp_min, temp_neg_res);
  } else {  // user greater of two roots
    temp = TemperatureBisect(r, temp_neg_res, temp_max);
  }

  // Calculate primitives
  Real rho = std::pow(temp/k_adi, n_adi);             // not same K as HSW
  Real pgas = temp * rho;
  Real ur = c1 / (SQR(r) * std::pow(temp, n_adi));    // (HSW 75)
  Real ut = std::sqrt(1.0/SQR(1.0-2.0*m/r) * SQR(ur)
      + 1.0/(1.0-2.0*m/r));

  // Set primitives
  *prho = rho;
  *ppgas = pgas;
  *put = ut;
  *pur = ur;
  return;
}

//----------------------------------------------------------------------------------------
// Function for finding temperature at which residual is minimized
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which minimum must occur
// Outputs:
//   returned value: some temperature for which residual of (HSW 76) is negative
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs golden section search (cf. Numerical Recipes, 3rd ed., 10.2)

static Real TemperatureMin(Real r, Real t_min, Real t_max) {
  // Parameters
  const Real ratio = 0.3819660112501051;  // (3+\sqrt{5})/2
  const int max_iterations = 30;          // maximum number of iterations

  // Initialize values
  Real t_mid = t_min + ratio * (t_max - t_min);
  Real res_mid = TemperatureResidual(t_mid, r);

  // Apply golden section method
  bool larger_to_right = true;  // flag indicating larger subinterval is on right
  for (int n = 0; n < max_iterations; ++n) {
    if (res_mid < 0.0) {
      return t_mid;
    }
    Real t_new;
    if (larger_to_right) {
      t_new = t_mid + ratio * (t_max - t_mid);
      Real res_new = TemperatureResidual(t_new, r);
      if (res_new < res_mid) {
        t_min = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_max = t_new;
        larger_to_right = false;
      }
    } else {
      t_new = t_mid - ratio * (t_mid - t_min);
      Real res_new = TemperatureResidual(t_new, r);
      if (res_new < res_mid) {
        t_max = t_mid;
        t_mid = t_new;
        res_mid = res_new;
      } else {
        t_min = t_new;
        larger_to_right = true;
      }
    }
  }
  return NAN;
}

//----------------------------------------------------------------------------------------
// Bisection root finder
// Inputs:
//   r: Schwarzschild radius
//   t_min,t_max: bounds between which root must occur
// Outputs:
//   returned value: temperature that satisfies (HSW 76)
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs bisection search

static Real TemperatureBisect(Real r, Real t_min, Real t_max) {
  // Parameters
  const int max_iterations = 20;
  const Real tol_residual = 1.0e-6;
  const Real tol_temperature = 1.0e-6;

  // Find initial residuals
  Real res_min = TemperatureResidual(t_min, r);
  Real res_max = TemperatureResidual(t_max, r);
  if (std::abs(res_min) < tol_residual) {
    return t_min;
  }
  if (std::abs(res_max) < tol_residual) {
    return t_max;
  }
  if ((res_min < 0.0 and res_max < 0.0) or (res_min > 0.0 and res_max > 0.0)) {
    return NAN;
  }

  // Iterate to find root
  Real t_mid;
  for (int i = 0; i < max_iterations; ++i) {
    t_mid = (t_min + t_max) / 2.0;
    if (t_max - t_min < tol_temperature) {
      return t_mid;
    }
    Real res_mid = TemperatureResidual(t_mid, r);
    if (std::abs(res_mid) < tol_residual) {
      return t_mid;
    }
    if ((res_mid < 0.0 and res_min < 0.0) or (res_mid > 0.0 and res_min > 0.0)) {
      t_min = t_mid;
      res_min = res_mid;
    } else {
      t_max = t_mid;
      res_max = res_mid;
    }
  }
  return t_mid;
}

//----------------------------------------------------------------------------------------
// Function whose value vanishes for correct temperature
// Inputs:
//   t: temperature
//   r: Schwarzschild radius
// Outputs:
//   returned value: residual that should vanish for correct temperature
// Notes:
//   implements (76) from Hawley, Smarr, & Wilson 1984, ApJ 277 296

static Real TemperatureResidual(Real t, Real r) {
  return SQR(1.0 + (n_adi+1.0) * t)
      * (1.0 - 2.0*m/r + SQR(c1) / (SQR(SQR(r)) * std::pow(t, 2.0*n_adi))) - c2;
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

    Real x = x1;
    Real y = x2;
    Real z = x3;
    Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);

    *pr = r;
    *ptheta = std::acos(z/r);
    *pphi = std::atan2( (r*y-a*x)/(SQR(r)+SQR(a) ), (a*y+r*x)/(SQR(r) + SQR(a) )  );
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
  Real R = std::sqrt(SQR(x) + SQR(y) + SQR(z));
  Real r = SQR(R) - SQR(a) + std::sqrt( SQR( SQR(R) - SQR(a) ) + 4.0*SQR(a)*SQR(z) );
  r = std::sqrt(r/2.0);


  Real eta[4],l_lower[4],l_upper[4];

  Real f = 2.0 * SQR(r)*r / (SQR(SQR(r)) + SQR(a)*SQR(z));
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
  Real dl1_dx1 = x*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
  // x r *(-r^2 x)/(r^6) + 1/r = -x^2/r^3 + 1/r
  Real dl1_dx2 = y*r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( SQR(rsq_p_asq) * ( sqrt_term ) )+ a/( rsq_p_asq );
  Real dl1_dx3 = z/r * ( SQR(a)*x - 2.0*a*r*y - SQR(r)*x )/( (rsq_p_asq) * ( sqrt_term ) ) ;
  Real dl2_dx1 = x*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) - a/( rsq_p_asq );
  Real dl2_dx2 = y*r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( SQR(rsq_p_asq) * ( sqrt_term ) ) + r/( rsq_p_asq );
  Real dl2_dx3 = z/r * ( SQR(a)*y + 2.0*a*r*x - SQR(r)*y )/( (rsq_p_asq) * ( sqrt_term ) );
  Real dl3_dx1 = - x*z/(r) /( sqrt_term );
  Real dl3_dx2 = - y*z/(r) /( sqrt_term );
  Real dl3_dx3 = - SQR(z)/(SQR(r)*r) * ( rsq_p_asq )/( sqrt_term ) + 1.0/r;

  Real dl0_dx1 = 0.0;
  Real dl0_dx2 = 0.0;
  Real dl0_dx3 = 0.0;


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