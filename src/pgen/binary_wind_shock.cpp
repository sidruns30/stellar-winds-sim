/*
 * Function star_wind.c
 *
 * Problem generator for stars with solar wind output, with gravity included
 */
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


#define TYPE_1 (0)
#define TYPE_2 (1)
#define TYPE_3 (2)
#define FIELD_SOURCE TYPE_1
/* cooling */
/* -------------------------------------------------------------------------- */
static int cooling;
static void integrate_cool(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim );
static Real Lambda_T(const Real T);
static Real Yinv(Real Y1);
static Real Y(const Real T);
static Real tcool(const Real d, const Real T);


Real mp_over_kev = 9.994827;   //mp * (pc/kyr)^2/kev
Real UnitDensity = 6.767991e-23; // solar mass pc^-3
Real UnitEnergyDensity = 6.479592e-7; //solar mass /(pc ky^2)
Real UnitTime = 3.154e10;  //kyr
Real Unitlength = 3.086e+18; //parsec
Real UnitB = Unitlength/UnitTime * std::sqrt(4. * PI* UnitDensity);
Real UnitLambda_times_mp_times_kev = 1.255436328493696e-21 ;//  UnitEnergyDensity/UnitTime*Unitlength**6.*mp*kev/(solar_mass**2. * Unitlength**2./UnitTime**2.)
Real keV_to_Kelvin = 1.16045e7;
Real dlogkT,T_max_tab,T_min_tab;
Real X = 1e-15; //0.7;   // Hydrogen Fraction
//Real Z_sun = 0.02;  //Metalicity
Real muH = 1./X;
Real mue = 2./(1. + X);


//Lodders et al 2003
Real Z_o_X_sun = 0.0177;
Real X_sun = 0.7491;
Real Y_sun =0.2246 + 0.7409 * (Z_o_X_sun);
Real Z_sun = 1.0 - X_sun - Y_sun;
Real muH_sun = 1./X_sun;


#define CUADRA_COOL (0)
#if (CUADRA_COOL==1)
Real Z = 3.*Z_sun;
#else
Real Z = 3.*Z_sun;
#endif
#if (CUADRA_COOL==1)
Real mu_highT = 0.5;
#else
Real mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.);  //mean molecular weight in proton masses
#endif

 void cons_force(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
 void emf_source(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim,  const AthenaArray<Real> &bcc, const AthenaArray<Real> &cons, EdgeField &e);
 void star_update_function(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
 void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
 void Dirichlet_Boundary(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
 void DirichletInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);

void DirichletOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void DirichletInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void DirichletOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void DirichletInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void DirichletOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);
void interp_inits(const Real x, const Real y, const Real z, Real *rho, Real *vx, Real *vy, Real *vz, Real *p);

Real time_step(MeshBlock *pmb);

int RefinementCondition(MeshBlock *pmb);
Real DivergenceB(MeshBlock *pmb, int iout);


/* A structure defining the properties of each of the source 'stars' */
typedef struct Stars_s{
  Real M;     /* mass of the star */
  Real Mdot;    /* mass loss rate from solar wind (in M_solar/kyr) */
  Real Vwind;   /* speed of solar wind (in pc/kyr) */
  Real x1;      /* position in X,Y,Z (in pc) */
  Real x2;
  Real x3;
  int i;      /* i,j,k of x,y,z cell the star is located in */
  int j;
  int k;
  Real v1;      /* velocity in X,Y,Z */
  Real v2;
  Real v3;
  Real alpha;     /* euler angles for ZXZ rotation*/
  Real beta;
  Real gamma;
  Real tau;
  Real mean_angular_motion;
  Real eccentricity;
  Real rotation_matrix[3][3];
  Real period;
  Real radius;  /* effective radius of star */
  Real r_cut;   /* cut-off radius of star to prevent r=0 divergence */
  Real volume;   /* effective volume of star */
    Real B_A;
    Real r_A;
  RegionSize block_size;   /* block size of the mesh block in which the star is located */
  AthenaArray<Real> spin_axis;      /* angular momentum unit vector for star */
  AthenaArray<Real> x_axis;      /* x_axis for star frame */
  AthenaArray<Real> y_axis;      /* y_axis for star frame */
}Stars;


/* Initialize a couple of the key variables used throughout */
//Real r_inner_boundary = 0.;         /* remove mass inside this radius */
Real r_min_inits = 1e15; /* inner radius of the grid of initial conditions */
Stars star[500];          /* The stars structure used throughout */
int nstars;             /* Number of stars in the simulation */
static Real G = 4.48e-9;      /* Gravitational constant G (in problem units) */
Real gm_;               /* G*M for point mass at origin */
Real gm1;               /* \gamma-1 (adiabatic index) */
Real N_cells_per_radius;  /* Number of cells that are contained in one stellar radius (this is defined in terms of the longest length 
across a cell ) */
double SMALL = 1e-20;       /* Small number for numerical purposes */
LogicalLocation *loclist;              /* List of logical locations of meshblocks */
int max_refinement_level = 0;    /*Maximum allowed level of refinement for AMR */

Real beta_star;  /* beta for each star, defined wrt rho v^2 */


int N_r =128;  /* Number of points to sample in r for radiale profile */
int N_user_vars = 27; /* Number of user defined variables in UserWorkAfterLoop */
int N_user_history_vars = 26; /* Number of user defined variables which have radial profiles */
int N_user_vars_field = 6; /* Number of user defined variables related to magnetic fields */
Real r_dump_min,r_dump_max; /* Range in r to sample for radial profile */


Real yr = 31556926.0, pc = 3.09e18;    /* yr and parsec in code units */
Real cl = 2.99792458e10 * (1e3 * yr)/pc ;      /* speed of light in code units */
Real cs_max = cl ; //0.023337031 * cl;  /*sqrt(me/mp) cl....i.e. sound speed of electrons is ~ c */

// int nx_inits,ny_inits,nz_inits; /* size of initial condition arrays */
// AthenaArray<Real> x_inits,y_inits,z_inits,v1_inits,v2_inits,v3_inits,press_inits,rho_inits; /* initial condition arrays*/


#if (CUADRA_COOL==0)
Real kbTfloor_kev = 8.00000000e-04;
#else
Real kbTfloor_kev = 8.61733130e-4;
#endif

/* global definitions for the cooling curve using the
   Townsend (2009) exact integration scheme */
//#define nfit_cool 11

#define nfit_cool 13
//Real kbT_cc[nfit_cool],Lam_cc[nfit_cool],
Real exp_cc[nfit_cool],Lam_cc[nfit_cool];

// #if (CUADRA_COOL ==0)
// static const Real kbT_cc[nfit_cool] = {
//   8.61733130e-06,   8.00000000e-04,   1.50000000e-03,
//   2.50000000e-03,   7.50000000e-03,   2.00000000e-02,
//   3.10000000e-02,   1.25000000e-01,   3.00000000e-01,
//   2.26000000e+00,   1.00000000e+02};

// static const Real Lam_cc[nfit_cool] = {
//   1.24666909e-27,   3.99910139e-26,   1.47470970e-22,
//   1.09120314e-22,   4.92195285e-22,   5.38853593e-22,
//   2.32144473e-22,   1.38278507e-22,   3.66863203e-23,
//   2.15641313e-23,   9.73848346e-23};

// static const Real exp_cc[nfit_cool] = {
//    0.76546122,  13.06493514,  -0.58959508,   1.37120661,
//    0.09233853,  -1.92144798,  -0.37157016,  -1.51560627,
//   -0.26314206,   0.39781441,   0.39781441};
// #else
// static const Real kbT_cc[nfit_cool] = {
//     8.61733130e-06,   8.00000000e-04,   1.50000000e-03,
//     2.50000000e-03,   7.50000000e-03,   2.00000000e-02,
//     3.10000000e-02,   1.25000000e-01,   3.00000000e-01,
//     2.26000000e+00,   1.00000000e+02};

// static const Real Lam_cc[nfit_cool] = {
//     1.89736611e-19,   7.95699530e-21,   5.12446122e-21,
//     3.58388517e-21,   1.66099838e-21,   8.35970776e-22,
//     6.15118667e-22,   2.31779533e-22,   1.25581948e-22,
//     3.05517566e-23,   2.15234602e-24};

// static const Real exp_cc[nfit_cool] = {
//     -0.7,  -0.7,  -0.7,   -0.7,
//     -0.7,  -0.7,  -0.7,  -0.7,
//     -0.7,   -0.7,  -0.7};
// #endif

static const Real kbT_cc[nfit_cool] = {
    8.00000000e-04,   1.50000000e-03, 2.50000000e-03,
    7.50000000e-03,   2.00000000e-02, 3.10000000e-02,
    1.25000000e-01,   3.00000000e-01, 8.22000000e-01,
    2.26000000e+00, 3.010000000e+00,  3.4700000000e+01,
    1.00000000e+02};
static const Real Lam_cc_H[nfit_cool] = {
    6.16069200e-24,   4.82675600e-22,   1.17988800e-22,
    1.08974000e-23,   3.59794100e-24,   2.86297800e-24,
    2.85065300e-24,   3.73480000e-24,   5.58385400e-24,
    8.75574600e-24,   1.00022900e-23,   3.28378800e-23,
    5.49397977e-23};
static const Real Lam_cc_He[nfit_cool] = {
    6.21673000e-29,   5.04222000e-26,   4.57500800e-24,
     1.06434700e-22,   1.27271300e-23,   6.06418700e-24,
     1.81856400e-24,   1.68631400e-24,   2.10823800e-24,
     3.05093700e-24,   3.43240700e-24,   1.02736900e-23,
     1.65141824e-23};
static const Real Lam_cc_Metals[nfit_cool] = {
    5.52792700e-26,   5.21197100e-24,   1.15916400e-22,
     1.17551800e-21,   1.06054500e-21,   2.99407900e-22,
     1.04016100e-22,   2.40264000e-23,   2.14343200e-23,
     5.10608800e-24,   4.29634900e-24,   4.12127500e-24,
     4.04771017e-24 };

static Real Yk[nfit_cool];
/* -- end piecewise power-law fit */


/* must call init_cooling() in both problem() and read_restart() */
static void init_cooling();
static void init_cooling_tabs(std::string filename);
static void test_cooling();
static Real newtemp_townsend(const Real d, const Real T, const Real dt_hydro);



static void init_cooling_tabs(std::string filename){
//  FILE *input_file;
//    if ((input_file = fopen(filename.c_str(), "r")) == NULL)  { 
//           fprintf(stderr, "Cannot open %s, %s\n", "input_file",filename.c_str());
//           exit(0);
//         }

  Real T_Kelvin, tmp, Lam_H, Lam_metals ; 
  int j;
  for (j=0; j<nfit_cool; j++) {
//   fscanf(input_file, "%lf %lf %lf \n",&T_Kelvin,&Lam_H,&Lam_metals);

//   kbT_cc[j] = T_Kelvin /keV_to_Kelvin;

//Lam_cc[j] = Lam_H + Lam_metals * Z/Z_sun;
   Lam_cc[j] = X/X_sun * Lam_cc_H[j] + (1.-X-Z)/Y_sun * Lam_cc_He[j] + Z/Z_sun * Lam_cc_Metals[j];
 }

 //fclose(input_file);
 for (j=0; j<nfit_cool-1; j++) exp_cc[j] = std::log(Lam_cc[j+1]/Lam_cc[j])/std::log(kbT_cc[j+1]/kbT_cc[j]) ; 

  exp_cc[nfit_cool-1] = exp_cc[nfit_cool -2];
    
    T_min_tab = kbT_cc[0];
    T_max_tab = kbT_cc[nfit_cool-1];
    dlogkT = std::log(kbT_cc[1]/kbT_cc[0]);


//for (j=0; j<nfit_cool; j++) fprintf(stderr,"kbT: %g Lam: %g exp_cc: %g \n", kbT_cc[j], Lam_cc[j],exp_cc[j]);

   
   return ;
}


static void init_cooling()
{
  int k, n=nfit_cool-1;
  Real term;

  /* populate Yk following equation A6 in Townsend (2009) */
  Yk[n] = 0.0;
  for (k=n-1; k>=0; k--){
    term = (Lam_cc[n]/Lam_cc[k]) * (kbT_cc[k]/kbT_cc[n]);

    if (exp_cc[k] == 1.0)
      term *= log(kbT_cc[k]/kbT_cc[k+1]);
    else
      term *= ((1.0 - std::pow(kbT_cc[k]/kbT_cc[k+1], exp_cc[k]-1.0)) / (1.0-exp_cc[k]));

    Yk[k] = Yk[k+1] - term;
  }
  return;
}

/* piecewise power-law fit to the cooling curve with temperature in
   keV and L in erg cm^3 / s */
static Real Lambda_T(const Real T)
{
  int k, n=nfit_cool-1;

  /* first find the temperature bin */
  for(k=n; k>=1; k--){
    if (T >= kbT_cc[k])
      break;
  }

  /* piecewise power-law; see equation A4 of Townsend (2009) */
  /* (factor of 1.311e-5 takes lambda from units of 1e-23 erg cm^3 /s
     to code units.) */
  return (Lam_cc[k] * std::pow(T/kbT_cc[k], exp_cc[k]));
}

/* see Lambda_T() or equation A1 of Townsend (2009) for the
   definition */
static Real Y(const Real T)
{
  int k, n=nfit_cool-1;
  Real term;

  /* first find the temperature bin */
  for(k=n; k>=1; k--){
    if (T >= kbT_cc[k])
      break;
  }

  /* calculate Y using equation A5 in Townsend (2009) */
  term = (Lam_cc[n]/Lam_cc[k]) * (kbT_cc[k]/kbT_cc[n]);

  if (exp_cc[k] == 1.0)
    term *= log(kbT_cc[k]/T);
  else
    term *= ((1.0 - std::pow(kbT_cc[k]/T, exp_cc[k]-1.0)) / (1.0-exp_cc[k]));

  return (Yk[k] + term);
}

static Real Yinv(const Real Y1)
{
  int k, n=nfit_cool-1;
  Real term;

  /* find the bin i in which the final temperature will be */
  for(k=n; k>=1; k--){
    if (Y(kbT_cc[k]) >= Y1)
      break;
  }


  /* calculate Yinv using equation A7 in Townsend (2009) */
  term = (Lam_cc[k]/Lam_cc[n]) * (kbT_cc[n]/kbT_cc[k]);
  term *= (Y1 - Yk[k]);

  if (exp_cc[k] == 1.0)
    term = exp(-1.0*term);
  else{
    term = std::pow(1.0 - (1.0-exp_cc[k])*term,
               1.0/(1.0-exp_cc[k]));
  }

  return (kbT_cc[k] * term);
}

static Real newtemp_townsend(const Real d, const Real T, const Real dt_hydro)
{
  Real term1, Tref;
  int n=nfit_cool-1;

  Tref = kbT_cc[n];

  term1 = (T/Tref) * (Lambda_T(Tref)/Lambda_T(T)) * (dt_hydro/tcool(d, T));

  return Yinv(Y(T) + term1);
}

static Real tcool(const Real d, const Real T)
{
  // T is in keV, d is in g/cm^3
#if (CUADRA_COOL==0)
  //return  (T) * (muH * muH) / ( gm1 * d * mu_highT * Lambda_T(T)/UnitLambda_times_mp_times_kev );
    return  (T) * (muH_sun * mue) / ( gm1 * d * mu_highT * Lambda_T(T)/UnitLambda_times_mp_times_kev );
#else
  return  (T) * (mu_highT) / ( gm1 * d *             Lambda_T(T)/UnitLambda_times_mp_times_kev );
#endif
}


static void integrate_cool(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim )
{
  int i, j, k;
  int is, ie, js, je, ks, ke;

  Real kbT_keV;
  //AthenaArray<Real> prim;
  //prim.InitWithShallowCopy(pmb->phydro->w);

  /* ath_pout(0, "integrating cooling using Townsend (2009) algorithm.\n"); */

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

        /* find temp in keV */
        kbT_keV = mu_highT*mp_over_kev*(prim(IPR,k,j,i)/prim(IDN,k,j,i));
        // ath_pout(0, "temperature before = %e ", kbT_keV);
        kbT_keV = newtemp_townsend(prim(IDN,k,j,i), kbT_keV, dt_hydro);
        // ath_pout(0, "temperature after = %e \n", kbT_keV);
        /* apply a temperature floor (nans tolerated) */
        if (isnan(kbT_keV) || kbT_keV < kbTfloor_kev)
          kbT_keV = kbTfloor_kev;

        prim(IPR,k,j,i) = prim(IDN,k,j,i) * kbT_keV / (mu_highT * mp_over_kev);

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

  //apply_inner_boundary_condition(pmb,prim);
   // pmb->peos->PrimitiveToConserved(prim, pmb->pfield->bcc,
      // cons, pmb->pcoord,
      // pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);
  return;
}



/******************************************/
/*        Some Vector Functions           */
/******************************************/


void cross(const AthenaArray<Real> &A , const AthenaArray<Real> &B, AthenaArray<Real> &result){
    
    result(0) = A(1)*B(2) - A(2)*B(1);
    result(1) = A(2)*B(0) - A(0)*B(2);
    result(2) = A(0)*B(1) - A(1)*B(0);
    return;
    
}

Real dot(const AthenaArray<Real> &A , const AthenaArray<Real> &B){
    return A(0) * B(0) + A(1) * B(1) + A(2) * B(2);
}

Real norm_calc(const AthenaArray<Real> &A ){
    return std::sqrt( SQR(A(0)) + SQR(A(1)) + SQR(A(2)) );
}
void norm_vector(AthenaArray<Real> &A){
    Real norm = norm_calc(A);
    for (int i=0; i<=2; ++i) A(i) *= 1./norm;
    return;
}

void add_vectors(const int i_sign, const AthenaArray<Real> &A , const AthenaArray<Real> &B, AthenaArray<Real> &result){
  for (int i=0; i<=2; ++i) result(i) = A(i) + i_sign* B(i);
  return;
}
void scale_vector(const AthenaArray<Real> &A , const Real alpha , AthenaArray<Real> &result){
  for (int i=0; i<=2; ++i) result(i) = A(i)*alpha;
  return;
}
/******************************************/
/*        Rotation Functions              */
/******************************************/


void pre_compute_rotation_matrix(Stars *star, int i_star) {
    Real alpha,beta,gamma;
    alpha = star[i_star].alpha;
    beta = star[i_star].beta;
    gamma = star[i_star].gamma;
    
    double X_rot[3][3];
    double Z_rot[3][3];
    double Z_rot2[3][3];
    double tmp[3][3],rot[3][3];
    int i,j,k;
    
    
    Z_rot2[0][0] = std::cos(gamma);
    Z_rot2[0][1] = -std::sin(gamma);
    Z_rot2[0][2] = 0.;
    Z_rot2[1][0] = std::sin(gamma);
    Z_rot2[1][1] = std::cos(gamma);
    Z_rot2[1][2] = 0.;
    Z_rot2[2][0] = 0.;
    Z_rot2[2][1] = 0.;
    Z_rot2[2][2] = 1.;
    
    X_rot[0][0] = 1.;
    X_rot[0][1] = 0.;
    X_rot[0][2] = 0.;
    X_rot[1][0] = 0.;
    X_rot[1][1] = std::cos(beta);
    X_rot[1][2] = -std::sin(beta);
    X_rot[2][0] = 0.;
    X_rot[2][1] = std::sin(beta);
    X_rot[2][2] = std::cos(beta);
    
    Z_rot[0][0] = std::cos(alpha);
    Z_rot[0][1] = -std::sin(alpha);
    Z_rot[0][2] = 0.;
    Z_rot[1][0] = std::sin(alpha);
    Z_rot[1][1] = std::cos(alpha);
    Z_rot[1][2] = 0.;
    Z_rot[2][0] = 0.;
    Z_rot[2][1] = 0.;
    Z_rot[2][2] = 1.;
    
    
    for (i=0; i<3; i++){
        for (j=0; j<3; j++) {
            rot[i][j] = 0.;
            tmp[i][j] = 0.;
        }
    }
    
    for (i=0; i<3; i++) for (j=0; j<3; j++) for (k=0; k<3; k++) tmp[i][j] += X_rot[i][k] * Z_rot[k][j] ;
    for (i=0; i<3; i++) for (j=0; j<3; j++) for (k=0; k<3; k++) rot[i][j] += Z_rot2[i][k] * tmp[k][j] ;
    
    
    for (i=0; i<3; i++){
        for (j=0; j<3; j++) {
            star[i_star].rotation_matrix[i][j] = rot[i][j] ;
        }
    }


    
}
void rotate_orbit(Stars *star, int i_star, const Real x1_prime, const Real x2_prime, Real * x1, Real * x2, Real * x3)
{
  Real alpha,beta,gamma;
  alpha = star[i_star].alpha;
  beta = star[i_star].beta;
  gamma = star[i_star].gamma;

  double X_rot[3][3];
  double Z_rot[3][3];
  double Z_rot2[3][3];
  double tmp[3][3],rot[3][3];
  double x_prime[3], x_result[3];
  int i,j,k;

  x_prime[0] = x1_prime;
  x_prime[1] = x2_prime;
  x_prime[2] = 0.;



  for (i=0; i<3; i++) x_result[i] = 0.;

  
  for (i=0; i<3; i++) for (j=0; j<3; j++) x_result[i] += star[i_star].rotation_matrix[j][i]*x_prime[j] ;   /*Note this is inverse rotation so rot[j,i] instead of rot[i,j] */


    *x1 = x_result[0];
    *x2 = x_result[1];
    *x3 = x_result[2];


}

/* 
Simple function to get Cartesian Coordinates

*/
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

Real get_dV(const Coordinates *pcoord, const Real x1, const Real x2, const Real x3,const Real dx1, const Real dx2, const Real dx3){
    
    if (COORDINATE_SYSTEM == "cartesian"){
        if (pcoord->pmy_block->block_size.nx3>1){
            return dx1 * dx2 * dx3;
        }
        else{
            return dx1 * dx2;
        }
    }
    else if (COORDINATE_SYSTEM == "cylindrical"){
        
        if (pcoord->pmy_block->block_size.nx3>1){
            return dx1 * x1 * dx2 * dx3;
        }
        else{
            return dx1 * x1 * dx2 ;
        }
        
    }
    else if (COORDINATE_SYSTEM == "spherical_polar"){
        
        return dx1 * x1 * dx2 * x1 * std::sin(x2) *dx3 ;
    }
    
}

/*
Convert vector in cartesian coords to code coords
*/
void convert_cartesian_vector_to_code_coords(const Real vx, const Real vy, const Real vz, const Real x, const Real y, const Real z, Real *vx1, Real *vx2, Real *vx3){

  if (COORDINATE_SYSTEM == "cartesian"){
    *vx1 = vx;
    *vx2 = vy;
    *vx3 = vz;
  }
  else if (COORDINATE_SYSTEM == "cylindrical"){

    Real s = sqrt( SQR(x) + SQR(y) );

    *vx1 = vx * x/s + vy * y/s;
    *vx2 =(-y * vx + x * vy) / (s);
    *vx3 = vz;

  }
  else if (COORDINATE_SYSTEM == "spherical_polar"){
    Real r = sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real s = sqrt( SQR(x) + SQR(y) );


    *vx1 = vx * x/r + vy * y/r + vz *z/r;
    *vx2 = ( (x * vx + y * vy) * z  - SQR(s) * vz ) / (r * s + SMALL) ;
    *vx3 = (-y * vx + x * vy) / (s + SMALL) ;
  }

}
/*
Returns approximate cell sizes if the grid was uniform
*/

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

/*
Function to get net wind velocity, assuming the wind velocity is radial in the frame of the star
*/
void get_net_wind_velocity(const Coordinates *pG, Stars *star, int i_star, const Real x, const Real y, const Real z ,Real *vx, Real *vy, Real *vz ){

  Real r;

  double dx = x - star[i_star].x1;
  double dy = y - star[i_star].x2;
  double dz = z - star[i_star].x3;
  if (pG->pmy_block->block_size.nx3 >1 ){
    r = sqrt( SQR(dx) + SQR(dy) + SQR(dz) ) ;
    *vx = dx/(r + SMALL) * star[i_star].Vwind + star[i_star].v1;
    *vy = dy/(r + SMALL) * star[i_star].Vwind + star[i_star].v2;
    *vz = dz/(r + SMALL) * star[i_star].Vwind + star[i_star].v3;
    return;
  }
  else{
    r = sqrt( SQR(dx) + SQR(dy) ) ;
    *vx = dx/(r + SMALL) * star[i_star].Vwind + star[i_star].v1;
    *vy = dy/(r + SMALL) * star[i_star].Vwind + star[i_star].v2;
    *vz = 0.;
        return;
  }



}

// /*
// Function to get net wind velocity times r, assuming the wind velocity is radial in the frame of the star
// */
// void get_net_wind_velocity_times_r(const Coordinates *pG, Stars *star, int i_star, const Real x, const Real y, const Real z ,Real *vx, Real *vy, Real *vz ){

//   Real r;

//   double dx = x - star[i_star].x1;
//   double dy = y - star[i_star].x2;
//   double dz = z - star[i_star].x3;
//   if (pG->pmy_block->block_size.nx3 >1 ){
//     r = sqrt( SQR(dx) + SQR(dy) + SQR(dz) ) ;
//     *vx = dx * star[i_star].Vwind + r * star[i_star].v1;
//     *vy = dy * star[i_star].Vwind + r * star[i_star].v2;
//     *vz = dz * star[i_star].Vwind + r * star[i_star].v3;
//     return;
//   }
//   else{
//     r = sqrt( SQR(dx) + SQR(dy) ) ;
//     *vx = dx * star[i_star].Vwind + r * star[i_star].v1;
//     *vy = dy * star[i_star].Vwind + r * star[i_star].v2;
//     *vz = 0.;
//         return;
//   }



// }
/*
Solve Kepler's equation for a given star in the plane of the orbit and then rotate
to the lab frame
*/
void update_star(Stars *star, int i_star, const Real t)
{

  // Real mean_anomaly = star[i_star].mean_angular_motion * (t - star[i_star].tau);
  // Real a = std::pow(gm_/SQR(star[i_star].mean_angular_motion),1./3.);    //mean_angular_motion = np.sqrt(mu/(a*a*a));
  //   Real b;
  //   if (star[i_star].eccentricity <1){
  //       b =a * sqrt(1. - SQR(star[i_star].eccentricity) );
  //       mean_anomaly = fmod(mean_anomaly, 2*PI);
  //       if (mean_anomaly >  PI) mean_anomaly = mean_anomaly- 2.0*PI;
  //       if (mean_anomaly < -PI) mean_anomaly = mean_anomaly + 2.0*PI;
  //   }
  //   else{
  //       b = a * sqrt(SQR(star[i_star].eccentricity) -1. );
  //   }


  //   //Construct the initial guess.
  //   Real E;
  //   if (star[i_star].eccentricity <1){
  //     Real sgn = 1.0;
  //     if (std::sin(mean_anomaly) < 0.0) sgn = -1.0;
  //     E = mean_anomaly + sgn*(0.85)*star[i_star].eccentricity;
  //    }
  //   else{
  //     Real sgn = 1.0;
  //     if (std::sinh(-mean_anomaly) < 0.0) sgn = -1.0;
  //     E = mean_anomaly;
  //   }

  //   //Solve kepler's equation iteratively to improve the solution E.
  //   Real error = 1.0;
  //   Real max_error = 1e-6;
  //   int i_max = 100;
  //   int i;

  //   if (star[i_star].eccentricity <1){
  //     for(i = 0; i < i_max; i++){
  //       Real es = star[i_star].eccentricity*std::sin(E);
  //       Real ec = star[i_star].eccentricity*std::cos(E);
  //       Real f = E - es - mean_anomaly;
  //       error = fabs(f);
  //       if (error < max_error) break;
  //       Real df = 1.0 - ec;
  //       Real ddf = es;
  //       Real dddf = ec;
  //       Real d1 = -f/df;
  //       Real d2 = -f/(df + d1*ddf/2.0);
  //       Real d3 = -f/(df + d2*ddf/2.0 + d2*d2*dddf/6.0);
  //       E = E + d3;
  //     }
  //   }
  //   else{
  //     for(i = 0; i < i_max; i++){
  //       Real es = star[i_star].eccentricity*std::sinh(E);
  //       Real ec = star[i_star].eccentricity*std::cosh(E);
  //       Real f = E - es + mean_anomaly;
  //       error = fabs(f);
  //       if (error < max_error) break;
  //       Real df = 1.0 - ec;
  //       Real ddf = -es;
  //       Real dddf = -ec;
  //       Real d1 = -f/df;
  //       Real d2 = -f/(df + d1*ddf/2.0);
  //       Real d3 = -f/(df + d2*ddf/2.0 + d2*d2*dddf/6.0);
  //       E = E + d3;
  //     }
  //   }

  //    //Warn if solution did not converge.
  //    if (error > max_error)
  //      std::cout << "***Warning*** Orbit::keplers_eqn() failed to converge***\n";

  //    Real x1_prime,x2_prime,v1_prime,v2_prime;
  //   if (star[i_star].eccentricity<1){
  //     x1_prime= a * (std::cos(E) - star[i_star].eccentricity) ;
  //     x2_prime= b * std::sin(E) ;
      
  //     /* Time Derivative of E */
  //     Real Edot = star[i_star].mean_angular_motion/ (1.-star[i_star].eccentricity * std::cos(E));
      
  //     v1_prime = - a * std::sin(E) * Edot;
  //     v2_prime =   b * std::cos(E) * Edot;
  //   }
  //   else{
  //     x1_prime = a * ( star[i_star].eccentricity - std::cosh(E) );
  //     x2_prime = b * std::sinh(E);

  //     /* Time Derivative of E */  
  //     Real Edot = -star[i_star].mean_angular_motion/ (1. - star[i_star].eccentricity * std::cosh(E));

  //     v1_prime = a * (-std::sinh(E)*Edot);
  //     v2_prime = b * std::cosh(E) * Edot;
  //   }

  //   Real x1,x2,x3;

  //   rotate_orbit(star,i_star, x1_prime, x2_prime,&x1,&x2,&x3 );
    
  //   star[i_star].x1 = x1;
  //   star[i_star].x2 = x2;
  //   star[i_star].x3 = x3;
    
  //   Real v1,v2,v3;
  //   rotate_orbit(star,i_star,v1_prime,v2_prime,&v1, &v2, &v3);
    
    
  //   star[i_star].v1 = v1;
  //   star[i_star].v2 = v2;
  //   star[i_star].v3 = v3;


  // star[i_star].x1 = 0.4* std::cos(2.5*t);
  // star[i_star].x2 = 0.4 * std::sin(2.5*t);
  // star[i_star].x3 = 0;

  // star[i_star].v1 = - 1.0 * std::sin(2.5*t) ;
  // star[i_star].v2 =  1.0 * std::cos(2.5*t);
  // star[i_star].v3 = 0;

}

/*
* -------------------------------------------------------------------
*   Function to read the information about each star from the file
*     'filename' (specified in the athinput.star_wind file). The 
*     file should be output by the program star_dump.c
* -------------------------------------------------------------------
*/
void read_stardata(Stars *star,std::string starfile)
{
  int n_stars,n_dim, gauss_check, j;
  float M, x1, x2, x3, v1, v2, v3, Mdot, Vwind, box_half_length;
  float alpha,beta,gamma,tau,mean_angular_motion,eccentricity;
  float simulation_start_time,gm_tmp;

  
  FILE *input_file;
    if ((input_file = fopen(starfile.c_str(), "r")) == NULL)   
           fprintf(stderr, "Cannot open %s, %s\n", "input_file",starfile.c_str());

  fscanf(input_file, "%i %g %g \n", &n_stars, &simulation_start_time, &gm_tmp);

  if (fabs(gm_-gm_tmp)/gm_>1e-4){
    fprintf(stderr,"Mismatched M_BH, fix in input file.  GM_input: %g GM_star_file: %g \n", gm_,gm_tmp);
    exit(0);
  }
  nstars = n_stars;


  for (j=0; j<n_stars; j++) {
   fscanf(input_file, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f \n",
                      &M,&Mdot,&Vwind,&x1,&x2,&x3,&v1,&v2,&v3,&alpha,&beta,&gamma,&tau,&mean_angular_motion,&eccentricity);
   star[j].M = M;
   star[j].Mdot = Mdot;
   star[j].Vwind = Vwind;
   star[j].x1 = x1;
   star[j].x2 = x2;
   star[j].x3 = x3;
   star[j].v1 = v1;
   star[j].v2 = v2;
   star[j].v3 = v3;
   star[j].alpha = alpha;
   star[j].beta = beta;
   star[j].gamma = gamma;
   star[j].tau = tau;
   star[j].mean_angular_motion = mean_angular_motion;
   star[j].period = 2.*PI/mean_angular_motion;
   star[j].eccentricity = eccentricity;
  
   star[j].spin_axis.NewAthenaArray(3);
   star[j].x_axis.NewAthenaArray(3);
   star[j].y_axis.NewAthenaArray(3);


   star[j].spin_axis(0) = 0;
   star[j].spin_axis(1) = 0;
   star[j].spin_axis(2) = 1.0;
      
  
   star[j].B_A = 3000.0/UnitB ;//3000/UnitB;
   star[j].r_A = 6.957e10 / Unitlength ;

   Real BA_times_rA = std::sqrt( 1.0/beta_star * star[j].Mdot * star[j].Vwind/(2.0*PI) );

   star[j].B_A = BA_times_rA/star[j].r_A;
      
      
  AthenaArray<Real> tmp;
  tmp.NewAthenaArray(3);
  //assume that spin_axis is not aligned with y, then cross y with spin_axis to get x direction
  tmp(0) = 0;
  tmp(1) = 1.;
  tmp(2) = 0.;
  if ( std::fabs(dot(tmp,star[j].spin_axis)/(norm_calc(tmp)*norm_calc(star[j].spin_axis)))  > 0.8 ){  //if y direction is too much in line with spin axis, use x
      tmp(0) = 1.;
      tmp(1) = 0.;
      tmp(2) = 0.;
  }
  cross(tmp,star[j].spin_axis,star[j].x_axis);
  cross(star[j].spin_axis,star[j].x_axis,star[j].y_axis);
  norm_vector(star[j].x_axis);
  norm_vector(star[j].y_axis);
  
  tmp.DeleteAthenaArray();
   

   pre_compute_rotation_matrix(star, j);

   // update_star(star,j, -simulation_start_time);

   // Real r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
   // Real v = sqrt( SQR(v1) + SQR(v2) + SQR(v3) );
   // if ( (fabs(1.-star[j].x1/x1) * fabs(x1/r) >1e-4) || (fabs(1.-star[j].x2/x2) * fabs(x2/r) > 1e-4) || 
   //      (fabs(1.-star[j].x3/x3) * fabs(x3/r) >1e-4) || (fabs(1.-star[j].v1/v1) * fabs(v1/v) >1e-4) || 
   //      (fabs(1.-star[j].v2/v2) * fabs(v2/v) >1e-4) || (fabs(1.-star[j].v3/v3) * fabs(v3/v) >1e-4) ){

   //  fprintf(stderr,"Error in solving Kepler\n x1: %g x2: %g x3: %g \n x1_orbit: %g x2_orbit %g x3_orbit %g \n v1: %g v2: %g v3: %g \n v1_orbit: %g v2_orbit %g v3_orbit %g ecc: %g mean_angular_motion: %g tau: %g \n alpha: %g beta: %g gamma: %g \n simulation_start_time: %g \n",
   //    x1,x2,x3,star[j].x1,star[j].x2,star[j].x3,v1,v2,v3,star[j].v1,star[j].v2,star[j].v3,star[j].eccentricity,star[j].mean_angular_motion,star[j].tau,star[j].alpha,star[j].beta,star[j].gamma,simulation_start_time);

   //  exit(0);
   // }
        
  update_star(star,j, 0.);

        



  }
    fclose(input_file);
  return;
}

/*
* -------------------------------------------------------------------
*   Function to read in the initial conditions 
*     'init_filename' (specified in the athinput.star_wind file).
* -------------------------------------------------------------------
*/
// void read_inits(std::string initfile)
// {
//   FILE *input_file;
//     if ((input_file = fopen(initfile.c_str(), "r")) == NULL)   
//            fprintf(stderr, "Cannot open %s, %s\n", "input_file",initfile.c_str());

//   fscanf(input_file, "%i %i %i \n", &nx_inits, &ny_inits, &nz_inits);

   

// x_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
// y_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
// z_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
// rho_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
// v1_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
// v2_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
// v3_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
// press_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);



// int i,j,k;
//   for (k=0; k<nx_inits; k++) {
//   for (j=0; j<ny_inits; j++) {
//   for (i=0; i<nz_inits; i++) {

// fread( &x_inits(k,j,i), sizeof( Real ), 1, input_file );
// fread( &y_inits(k,j,i), sizeof( Real ), 1, input_file );
// fread( &z_inits(k,j,i), sizeof( Real ), 1, input_file );
// fread( &rho_inits(k,j,i), sizeof( Real ), 1, input_file );
// fread( &v1_inits(k,j,i), sizeof( Real ), 1, input_file );
// fread( &v2_inits(k,j,i), sizeof( Real ), 1, input_file );
// fread( &v3_inits(k,j,i), sizeof( Real ), 1, input_file );
// fread( &press_inits(k,j,i), sizeof( Real ), 1, input_file );

// r_min_inits = std::min(r_min_inits,std::sqrt(SQR(x_inits(k,j,i)) + SQR(y_inits(k,j,i)) + SQR(z_inits(k,j,i)) ));

// }
// }
// }
//     fclose(input_file);
//   return;
// }

void set_boundary_arrays(std::string initfile, const RegionSize block_size, const Coordinates *pcoord, const int is, const int ie, const int js, const int je, const int ks, const int ke,
  AthenaArray<Real> &prim_bound){
      FILE *input_file;
        if ((input_file = fopen(initfile.c_str(), "r")) == NULL)   
               fprintf(stderr, "Cannot open %s, %s\n", "input_file",initfile.c_str());


      int nx_inits,ny_inits,nz_inits; /* size of initial condition arrays */
      AthenaArray<Real> x_inits,y_inits,z_inits,v1_inits,v2_inits,v3_inits,press_inits,rho_inits; /* initial condition arrays*/

      fscanf(input_file, "%i %i %i \n", &nx_inits, &ny_inits, &nz_inits);

       

    x_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    y_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    z_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    rho_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    v1_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    v2_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    v3_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    press_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);



    int i,j,k;
      for (k=0; k<nx_inits; k++) {
      for (j=0; j<ny_inits; j++) {
      for (i=0; i<nz_inits; i++) {

    fread( &x_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &y_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &z_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &rho_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &v1_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &v2_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &v3_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &press_inits(k,j,i), sizeof( Real ), 1, input_file );

    r_min_inits = std::min(r_min_inits,std::sqrt(SQR(x_inits(k,j,i)) + SQR(y_inits(k,j,i)) + SQR(z_inits(k,j,i)) ));

    }
    }
    }
        fclose(input_file);

    
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
    
    //fprintf(stderr,"nz: %d ny: %d nz: %d ijk lims: %d %d %d %d %d %d\n",nz,ny,nz,il,iu, kl,ku,jl,ju);

    //read_inits(initfile);

      for (int k=kl; k<=ku; ++k) {
#pragma omp parallel for schedule(static)
        for (int j=jl; j<=ju; ++j) {
#pragma simd
            for (int i=il; i<=iu; ++i) {

              Real x,y,z; 
              Real rho,p,vx,vy,vz;
              get_cartesian_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &x, &y, &z);
             // interp_inits(x, y, z, &rho, &vx,&vy,&vz,&p);

              Real dx = x_inits(0,0,1) - x_inits(0,0,0);
              Real dy = y_inits(0,1,0) - y_inits(0,0,0);
              Real dz = z_inits(1,0,0) - z_inits(0,0,0);

              Real x0 = x_inits(0,0,0);
              Real y0 = y_inits(0,0,0);
              Real z0 = z_inits(0,0,0);

              int i0 = (int) ((x - x0) / dx + 0.5 + 1000) - 1000;
              int j0 = (int) ((y - y0) / dy + 0.5 + 1000) - 1000;
              int k0 = (int) ((z - z0) / dz + 0.5 + 1000) - 1000;



              //fprintf(stderr,"i,j,k: %d %d %d \n i0 j0 k0: %d %d %d \n",i,j,k,i0,j0,k0);

              //fprintf(stderr,"x y z: %g %g %g \n dx dy dz: %g %g %g \n x0 y0 z0: %g %g %g \n i j k: %d %d %d \n",x,y,z,dx,dy,dz,x0,y0,z0,i,j,k);
              //fprintf(stderr,"nx ny nz: %d %d %d\n", nx_inits,ny_inits,nz_inits);

              //fprintf(stderr,"x y z: %g %g %g \n x_inits y_inits z_inits: %g %g %g \n", x,y,z, x_inits(k,j,i),y_inits(k,j,i),z_inits(k,j,i));

              Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
              if (i0<0 || i0>=nx_inits || j0<0 || j0>=ny_inits || k0<0 || k0>=nz_inits || r<r_min_inits){
                  rho = 1e-7;
                  vx = 0.;
                  vy = 0.;
                  vz = 0.;
                  p = 1e-10;
              }
              else{
                  rho = rho_inits(k0,j0,i0);
                  vx = v1_inits(k0,j0,i0);
                  vy = v2_inits(k0,j0,i0);
                  vz = v3_inits(k0,j0,i0);
                  p = press_inits(k0,j0,i0);

              }

                

              //fprintf(stderr,"ijk %d %d %d rho: %g\n",i,j,k,rho);
              prim_bound(IDN,k,j,i) = rho;
              prim_bound(IVX,k,j,i) = vx;
              prim_bound(IVY,k,j,i) = vy;
              prim_bound(IVZ,k,j,i) = vz;
              prim_bound(IPR,k,j,i) = p;


            }}}
    

       x_inits.DeleteAthenaArray();
       y_inits.DeleteAthenaArray();
       z_inits.DeleteAthenaArray();
       rho_inits.DeleteAthenaArray();
       v1_inits.DeleteAthenaArray();
       v2_inits.DeleteAthenaArray();
       v3_inits.DeleteAthenaArray();
       press_inits.DeleteAthenaArray();

}



/* 
* -------------------------------------------------------------------
*  Another basic function, this one converts the initial units 
*   in the Stars struct to those of the problem:
*   t_0 = 10^3 yr = 10^3 * (31556926 s)
*   d_0 = pc = 3.09 * 10^13 km
*   M_0 = M_solar = 1.99 * 10^33 g
*
* These units also give:
*   G = 6.67 * 10^-11 m^3 / (kg*s^2)
*     = 4.48 * 10^-9 d_0^3 / (M_0 * t_0^2)
* -------------------------------------------------------------------
*/
void star_unitconvert(Stars *star)
{
  int i;
  //Real yr = 31556926.0, pc = 3.09e13, r0;
  
  for (i=0; i<nstars; i++) {
  /* Convert velocities from km/s to d_0/t_0
      (This gives a wind velocity ~1 for ~1000 km/s) */
  //star[i].Vwind *= 1.0e3 * yr / pc;

  /* Need to think about initial units for star velocities
    - proper motions? */
  /*
  star[i].v1 *= 1.0e3 * yr / pc;
  star[i].v2 *= 1.0e3 * yr / pc;
  star[i].v3 *= 1.0e3 * yr / pc;
  */
  
  /* Convert location from arcsec to pc 
    1 arcsec = PI/(648000) radians 
    d = r0 (in parsec) * theta (radians) 
    r0 is problem-dependent, here probably using
    the distance to the galactic center, 8.5 kpc */
  /*
  r0 = par_getd("problem","r0");
  r0 *= PI/648000.0;
  star[i].x1 *= r0;
  star[i].x2 *= r0;
  star[i].x3 *= r0;
  */
  
  /* Convert mass loss rate from 10^-5 M_solar/yr
      to M_0/t_0 = M_solar/t_0
      (This gives ~1 for Mdot = 10^-4 M_solar/yr) */
  //star[i].Mdot *= 0.01;
  }
  return;
}




/* 
* --------------------------------------------------------------------------
*  Brute-force mechanism for figuring out how much of a cell is filled 
* by the overlay of a star effective radius mask. There is an analytical 
* 2-D solution, but it doesn't extend to 3-D and is almost as 
* computationaly demanding 
* --------------------------------------------------------------------------
*/

Real starmask(const Real r, const Real r_eff, const Real r_cut){
  if (r<r_cut) return 0;
  else return 1.;
}

Real mask_volume_func(const Real r_eff, const Real r_cut, const int ndim){
  if (ndim ==2){
    return PI * SQR(r_eff) - PI * SQR(r_cut);
  }
  else if (ndim ==3){
    return 4.0/3.0 * PI * std::pow(r_eff,3.) - 4.0/3.0 * PI * std::pow(r_cut,3.);
  }
  else{
    fprintf(stderr,"Error, %d dimensions not supported for mask_volume\n",ndim);
  }


}

float mask(const Coordinates *pG, const int i, const int j, const int k,
        const int i_star, const int n_depth, const AthenaArray<Real> &bcc, Real *vx_avg, Real *vy_avg, Real *vz_avg, Real *vsq_avg, Real *bsq_func_avg, Real *P_func_avg)
{
  int i_x1, i_x2, i_x3, n_points_total;
  Real n_in_overlay,dx1,dx2,dx3,r,di_x1,di_x2,di_x3;
  Real dV,V;
  Real v_wind_x,v_wind_y,v_wind_z;
  Real fac;
  Real tmp = 0.;

  AthenaArray<Real> x_vec,B_vec;
  x_vec.NewAthenaArray(3);
  B_vec.NewAthenaArray(3);

  *vx_avg = 0.;
  *vy_avg = 0.;
  *vz_avg = 0.;
  *vsq_avg = 0.;
  *bsq_func_avg = 0.;
  *P_func_avg = 0.;

  int is_3D = (pG->pmy_block->block_size.nx3 >1);
  V = 0.;

  n_in_overlay = 0.0;
  n_points_total = n_depth*n_depth;
  // Use face spacing since that is the physical length of cell //

  // Discretize the cell into a subgrid n_depth x n_depth//
  di_x1 = pG->dx1f(i) / ((float)(n_depth-1));
  di_x2 = pG->dx2f(j) / ((float)(n_depth-1));
  di_x3 = pG->dx3f(k) / ((float)(n_depth-1));

  //integrating from left (e.g.; i) to right (e.g.; i+1) face in both directions //

  double x,y,z;
    for (i_x1 = 0; i_x1 < n_depth; i_x1++) {
      for (i_x2 = 0; i_x2 < n_depth; i_x2++) {
        for (i_x3 = 0*is_3D; i_x3 <= (n_depth-1)*is_3D; i_x3++) {

        Real x1 = pG->x1f(i) + i_x1*di_x1;
        Real x2 = pG->x2f(j) + i_x2*di_x2;
        Real x3 = pG->x3f(k) + i_x3*di_x3;
        get_cartesian_coords(x1,x2,x3,&x, &y, &z);
        get_net_wind_velocity(pG,star,i_star,x,y,z,&v_wind_x,&v_wind_y,&v_wind_z);

      dx1 = x - star[i_star].x1;
      dx2 = y - star[i_star].x2;
      dx3 = z - star[i_star].x3;
          
      r = sqrt(dx1*dx1 + dx2*dx2 + dx3*dx3*is_3D);
      x_vec(0) = dx1;
      x_vec(1) = dx2;
      x_vec(2) = dx3;

      if (MAGNETIC_FIELDS_ENABLED){
          B_vec(0) = bcc(IB1,k,j,i);
          B_vec(1) = bcc(IB2,k,j,i);
          B_vec(2) = bcc(IB3,k,j,i);
      }
      else{
        B_vec(0) = 0;
        B_vec(1) = 0;
        B_vec(2) = 0;
      }

      dV = get_dV(pG,x1,x2,x3,di_x1, di_x2, di_x3);


      /* Account for endpoints of integral */
      fac = 1.;

      if (i_x1 ==0 || i_x1 == n_depth- 1) fac = fac * 0.5;
            if (i_x2 ==0 || i_x2 == n_depth -1) fac = fac * 0.5;
      if ( (i_x3 ==0 || i_x3 == n_depth-1) && (is_3D) ) fac = fac*0.5;

      dV = dV * fac;
                
            Real f = starmask(r,star[i_star].radius,star[i_star].r_cut);
                

      if (r <= star[i_star].radius) {
        n_in_overlay += f * dV;
        *vx_avg += v_wind_x * f * dV;
        *vy_avg += v_wind_y * f * dV;
        *vz_avg += v_wind_z * f * dV;

        *vsq_avg += ( SQR(v_wind_x) + SQR(v_wind_y) + SQR(v_wind_z)*is_3D ) * f * dV;

        Real x_prime = dot(star[i_star].x_axis,x_vec);
        Real y_prime = dot(star[i_star].y_axis,x_vec);
        Real s_sq = SQR(x_prime) + SQR(y_prime) ; /* This is r^2 sin(theta)^2 */
        Real s = std::sqrt(s_sq);

        Real Bx = dot(star[i_star].x_axis,B_vec);
        Real By = dot(star[i_star].y_axis,B_vec);

        Real Bphi = Bx * -y_prime/(s + SMALL) + By * x_prime/(s+ SMALL); 

    #if (FIELD_SOURCE == TYPE_1)
        Real Bdot_phi = s/(r+SMALL) * std::sin(r/star[i_star].radius * PI) *PI * star[i_star].B_A * star[i_star].r_A * star[i_star].Vwind / (SQR(star[i_star].radius)); 
    #elif (FIELD_SOURCE == TYPE_2)
        Real Bdot_phi = SQR(PI)/(2.0*PI-4.0) * s/(r+SMALL) * std::cos(r/star[i_star].radius * PI/2.0) * star[i_star].B_A * star[i_star].r_A * star[i_star].Vwind / (SQR(star[i_star].radius)); 
    #else
        Real Bdot_phi = std::sin(r/star[i_star].radius * PI) *PI * star[i_star].B_A * star[i_star].r_A * star[i_star].Vwind / (SQR(star[i_star].radius)); 
    #endif
        //*bsq_func_avg += s_sq/(SQR(r)+SMALL)* SQR(std::sin(r/star[i_star].radius * PI)) * f * dV;
        *bsq_func_avg += Bphi * Bdot_phi * f * dV;

        Real z_prime = dot(star[i_star].spin_axis,x_vec);
        Real Edot_p = (gm1+1.0)/gm1 * 12.0*SQR(PI)/(2.0*SQR(PI)-3) * SQR(star[i_star].B_A * star[i_star].r_A)*star[i_star].Vwind / (std::pow(star[i_star].radius,3.0));
        *P_func_avg += Edot_p * SQR(z/(r+SMALL)) * SQR(std::sin(r/star[i_star].radius * PI)) *  f * dV;

      }
      V += dV;
      }}}

  *vx_avg = *vx_avg/(n_in_overlay +SMALL);
  *vy_avg = *vy_avg/(n_in_overlay +SMALL);
  *vz_avg = *vz_avg/(n_in_overlay +SMALL);
  *vsq_avg = *vsq_avg/(n_in_overlay +SMALL);
  *bsq_func_avg = *bsq_func_avg/V;
  *P_func_avg = *P_func_avg/V;

  x_vec.DeleteAthenaArray();
  B_vec.DeleteAthenaArray();

  return (n_in_overlay/V);
}


/*
Give box spacing, calculate cell r and V
*/
void get_star_size(const Coordinates *pcoord,const Real DX, const Real DY, const Real DZ, Real *r_eff, Real *r_cut, Real *mask_volume){

  if (pcoord->pmy_block->block_size.nx3 > 1){
        *r_eff = N_cells_per_radius * sqrt(SQR(DX) + SQR(DY) + SQR(DZ));
        *r_cut = 0; //sqrt(SQR(DX) + SQR(DY) + SQR(DZ));
        *mask_volume = mask_volume_func(*r_eff,*r_cut,3);
    }
    else{
        *r_eff = N_cells_per_radius * sqrt(SQR(DX) + SQR(DY));
        *r_cut = 0; //sqrt(SQR(DX) + SQR(DY));
        *mask_volume = mask_volume_func(*r_eff,*r_cut,2);
    }

}



// float cos_mask(float r) {return (cos((PI/2.0)*(r/r_eff)));}
// void set_cos_mask_2d(float (**mask)(float)) {
//  mask_volume = 4.0*r_eff*r_eff*(1.0 - 2.0/PI); *mask = cos_mask; }
// void set_cos_mask_3d(float (**mask)(float)) {
//  mask_volume = 8.0*r_eff*r_eff*r_eff*(1.0 - 8.0/PI/PI); *mask = cos_mask; }

// float box_mask(float r) {return (1.0);}
// void set_box_mask_2d(float (**mask)(float)) {
//  mask_volume = PI*r_eff*r_eff; *mask = box_mask; }
// void set_box_mask_3d(float (**mask)(float)) {
//  mask_volume = (4.0/3.0)*PI*r_eff*r_eff*r_eff; *mask = box_mask; }


int is_in_block(RegionSize block_size, const Real x, const Real y, const Real z){
  Real x1, x2, x3;

  if (COORDINATE_SYSTEM =="cartesian"){
    x1 = x;
    x2 = y;
    x3 = z;
  }
  else if (COORDINATE_SYSTEM == "cylindrical"){
    x1 = sqrt(x*x + y*y);
    x2 = std::atan2(y,x);
    x3 = z;
  }
  else if (COORDINATE_SYSTEM == "spherical_polar"){
    x1 = sqrt(x*x + y*y + z*z);
    x2 = std::acos(z/x1);
    x3 = std::atan2(y,x);
  }

  int is_in_x1 = (block_size.x1min <= x1) && (block_size.x1max >= x1);
  int is_in_x2 = (block_size.x2min <= x2) && (block_size.x2max >= x2);
  int is_in_x3 = (block_size.x3min <= x3) && (block_size.x3max >= x3);

  if (block_size.nx3>1) return is_in_x3 * is_in_x2 * is_in_x1;
  else return is_in_x2 * is_in_x1;


}

/*For use with SMR.  Assumes Cartesian coordinates
  computes radius and volume of star based on the meshblock it finds
  itself in*/
void update_star_size(const Coordinates * pcoord, Stars *star, const int i_star){

    
    RegionSize block_size;
    enum BoundaryFlag block_bcs[6];
    int n_mb = pcoord->pmy_block->pmy_mesh->nbtotal;

    

    
  //fprintf(stderr,"star_block: %d n_mb: %d \n",star[i_star].iblock,n_mb);
  block_size = pcoord->pmy_block->block_size;

  Real DX,DY,DZ,r_eff,r_cut,mask_volume;


  /* First Check if the star is even in the mesh.  If not, give it the coarse grid radius + volume */
  if (!is_in_block(pcoord->pmy_block->pmy_mesh->mesh_size,star[i_star].x1,star[i_star].x2,star[i_star].x3) ){
        block_size = pcoord->pmy_block->pmy_mesh->mesh_size;
          get_uniform_box_spacing(block_size,&DX,&DY,&DZ);
          get_star_size(pcoord,DX,DY,DZ,&r_eff,&r_cut,&mask_volume);

            star[i_star].radius = r_eff;
            star[i_star].r_cut = r_cut;
            star[i_star].volume = mask_volume;
        return;

  }

  /* First check if star has moved from its previous block.  If not the radius and 
     volume should already be set */
  if ( is_in_block(star[i_star].block_size,star[i_star].x1,star[i_star].x2,star[i_star].x3) ){
        return ;
  }

  /* Now loop over all mesh blocks by reconstructing the block sizes to find the block that the 
     star is located in */
    for (int j=0; j<n_mb; j++) {
        pcoord->pmy_block->pmy_mesh->SetBlockSizeAndBoundaries(loclist[j], block_size, block_bcs);

        if (is_in_block(block_size,star[i_star].x1,star[i_star].x2,star[i_star].x3) ) break;
        
        //fprintf(stderr,"level: %d %ld %ld %ld\n", j,loclist[j].lx1,loclist[j].lx2,loclist[j].lx3);

    }

    star[i_star].block_size = block_size;

    get_uniform_box_spacing(block_size,&DX,&DY,&DZ);
    get_star_size(pcoord,DX,DY,DZ,&r_eff,&r_cut,&mask_volume);

    star[i_star].radius = r_eff;
    star[i_star].r_cut = r_cut;
    star[i_star].volume = mask_volume;
    return;
  
}

void get_minimum_cell_lengths(const Coordinates * pcoord, Real *dx_min, Real *dy_min, Real *dz_min){
    
        
        RegionSize block_size;
        enum BoundaryFlag block_bcs[6];
        int n_mb = pcoord->pmy_block->pmy_mesh->nbtotal;
    
        *dx_min = 1e15;
        *dy_min = 1e15;
        *dz_min = 1e15;

        
        
        
        block_size = pcoord->pmy_block->block_size;
        
        Real DX,DY,DZ;
    
        
        /* Loop over all mesh blocks by reconstructing the block sizes to find the block that the
         star is located in */
        for (int j=0; j<n_mb; j++) {
            pcoord->pmy_block->pmy_mesh->SetBlockSizeAndBoundaries(loclist[j], block_size, block_bcs);
            
            get_uniform_box_spacing(block_size,&DX,&DY,&DZ);
            
            if (DX < *dx_min)  *dx_min = DX;
            if (DY < *dy_min) *dy_min = DY;
            if (DZ < *dz_min) *dz_min = DZ;
            
            
            
        }
        
    
}
/* Make sure i,j,k are in the domain */
void bound_ijk(Coordinates *pcoord, int *i, int *j, int*k){
    
    int is,js,ks,ie,je,ke;
    is = pcoord->pmy_block->is;
    js = pcoord->pmy_block->js;
    ks = pcoord->pmy_block->ks;
    ie = pcoord->pmy_block->ie;
    je = pcoord->pmy_block->je;
    ke = pcoord->pmy_block->ke;
    
    
    if (*i<is) *i = is;
    if (*j<js) *j = js;
    if (*k<ks) *k = ks;
    
    if (*i>ie) *i = ie;
    if (*j>je) *j = je;
    if (*k>ke) *k = ke;
    
    return; 
}


/* Set maximum temperature such that v_th <= c */
void limit_temperature(MeshBlock *pmb,AthenaArray<Real> &cons,const AthenaArray<Real> &prim_old){

  AthenaArray<Real> prim;
  prim.InitWithShallowCopy(pmb->phydro->w);
  pmb->peos->ConservedToPrimitive(cons, prim_old, pmb->pfield->b, prim, pmb->pfield->bcc,
           pmb->pcoord, pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);
    Real gamma = gm1 +1.;

for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {

    Real v_s = sqrt(gamma*prim(IPR,k,j,i)/prim(IDN,k,j,i));

        if (v_s>cs_max) v_s = cs_max;
          
        if ( fabs(prim(IVX,k,j,i)) > cs_max) prim(IVX,k,j,i) = cs_max * ( (prim(IVX,k,j,i) >0) - (prim(IVX,k,j,i)<0) ) ;
        if ( fabs(prim(IVY,k,j,i)) > cs_max) prim(IVY,k,j,i) = cs_max * ( (prim(IVY,k,j,i) >0) - (prim(IVY,k,j,i)<0) ) ;
        if ( fabs(prim(IVZ,k,j,i)) > cs_max) prim(IVZ,k,j,i) = cs_max * ( (prim(IVZ,k,j,i) >0) - (prim(IVZ,k,j,i)<0) ) ;

    prim(IPR,k,j,i) = SQR(v_s) *prim(IDN,k,j,i)/gamma ;



  }}}


  pmb->peos->PrimitiveToConserved(prim, pmb->pfield->bcc,
       cons, pmb->pcoord,
       pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);

}


/* Do nothing for Dirichlet Bounds */
 void Dirichlet_Boundary(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke){
     
     //if (is==2)fprintf(stderr,"den: %g %g is: %d %d %d %d %d %d \n",prim(IDN,ks,js,is-2),prim(IDN,ke,je,ie+2),is,ie,js,je,ks,ke);

    //      if (pmb->block_size.x1min==pmb->pmy_mesh->mesh_size.x1min && pmb->block_size.x2min==pmb->pmy_mesh->mesh_size.x2min && 
    //   pmb->block_size.x3min==pmb->pmy_mesh->mesh_size.x3min){

    //       int i,j,k;

    // for (int k=0; k<=1; ++k) {
    // for (int j=0; j<=1; ++j) {
    //   for (int i=0; i<=1; ++i) {

    //   fprintf(stderr,"boundaries: x y z: %g %g %g \n prims: %g %g %g %g %g \n ijk: %d %d %d\n",pco->x1v(i),pco->x2v(j),pco->x3v(k),prim(IDN,k,j,i),prim(IVX,k,j,i),prim(IVY,k,j,i),prim(IVZ,k,j,i),
    //     prim(IPR,k,j,i),i,j,k);
    //     }}}
    // }

  return; 
 }

 //----------------------------------------------------------------------------------------
//!\f: User-defined boundary Conditions: sets solution in ghost zones to initial values
// 

// void DirichletInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
//                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
// {
//   Real x,y,z;
//   Real vx,vy,vz,p,rho;
//   for (int k=ks; k<=ke; ++k) {
//     for (int j=js; j<=je; ++j) {
//       for (int i=1; i<=(NGHOST); ++i) {


//         prim(IDN,k,j,is-i) = pmb->phydro->w_bound(IDN,k,j,is-i);
//         prim(IVX,k,j,is-i) = pmb->phydro->w_bound(IVX,k,j,is-i);
//         prim(IVY,k,j,is-i) = pmb->phydro->w_bound(IVY,k,j,is-i);
//         prim(IVZ,k,j,is-i) = pmb->phydro->w_bound(IVZ,k,j,is-i);
//         prim(IPR,k,j,is-i) = pmb->phydro->w_bound(IPR,k,j,is-i);

//         if (prim(IVX,k,j,is-i)<0) prim(IVX,k,j,is-i) = 0;
//     }
//   }
// }
// }

// void DirichletOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
//                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
// {
//   Real x,y,z;
//   Real vx,vy,vz,p,rho;
//   for (int k=ks; k<=ke; ++k) {
//     for (int j=js; j<=je; ++j) {
//       for (int i=1; i<=(NGHOST); ++i) {


//           prim(IDN,k,j,ie+i) = pmb->phydro->w_bound(IDN,k,j,ie+i);
//           prim(IVX,k,j,ie+i) = pmb->phydro->w_bound(IVX,k,j,ie+i);
//           prim(IVY,k,j,ie+i) = pmb->phydro->w_bound(IVY,k,j,ie+i);
//           prim(IVZ,k,j,ie+i) = pmb->phydro->w_bound(IVZ,k,j,ie+i);
//           prim(IPR,k,j,ie+i) = pmb->phydro->w_bound(IPR,k,j,ie+i);

//           if (prim(IVX,k,j,ie+i)>0) prim(IVX,k,j,ie+i) = 0;


//       }
//     }
//   }
// }

// void DirichletInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
//                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
// {
//   Real x,y,z;
//   Real vx,vy,vz,p,rho;
//   for (int k=ks; k<=ke; ++k) {
//     for (int j=1; j<=(NGHOST); ++j) {
//       for (int i=is; i<=ie; ++i) {


//           prim(IDN,k,js-j,i) = pmb->phydro->w_bound(IDN,k,js-j,i);
//           prim(IVX,k,js-j,i) = pmb->phydro->w_bound(IVX,k,js-j,i);
//           prim(IVY,k,js-j,i) = pmb->phydro->w_bound(IVY,k,js-j,i);
//           prim(IVZ,k,js-j,i) = pmb->phydro->w_bound(IVZ,k,js-j,i);
//           prim(IPR,k,js-j,i) = pmb->phydro->w_bound(IPR,k,js-j,i);


//           if (prim(IVY,k,js-j,i)<0)  prim(IVY,k,js-j,i) = 0;

//       }
//     }
//   }
// }

// void DirichletOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
//                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
// {
//   Real x,y,z;
//   Real vx,vy,vz,p,rho;
//   for (int k=ks; k<=ke; ++k) {
//     for (int j=1; j<=(NGHOST); ++j) {
//       for (int i=is; i<=ie; ++i) {


//           prim(IDN,k,je+j,i) = pmb->phydro->w_bound(IDN,k,je+j,i);
//           prim(IVX,k,je+j,i) = pmb->phydro->w_bound(IVX,k,je+j,i);
//           prim(IVY,k,je+j,i) = pmb->phydro->w_bound(IVY,k,je+j,i);
//           prim(IVZ,k,je+j,i) = pmb->phydro->w_bound(IVZ,k,je+j,i);
//           prim(IPR,k,je+j,i) = pmb->phydro->w_bound(IPR,k,je+j,i);

//          if (prim(IVY,k,je+j,i)>0) prim(IVY,k,je+j,i) = 0.;
//       }
//     }
//   }
// }

// void DirichletInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
//                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
// {
//   Real x,y,z;
//   Real vx,vy,vz,p,rho;
//   for (int k=1; k<=(NGHOST); ++k) {
//     for (int j=js; j<=je; ++j) {
//       for (int i=is; i<=ie; ++i) {


//           prim(IDN,ks-k,j,i) = pmb->phydro->w_bound(IDN,ks-k,j,i);
//           prim(IVX,ks-k,j,i) = pmb->phydro->w_bound(IVX,ks-k,j,i);
//           prim(IVY,ks-k,j,i) = pmb->phydro->w_bound(IVY,ks-k,j,i);
//           prim(IVZ,ks-k,j,i) = pmb->phydro->w_bound(IVZ,ks-k,j,i);
//           prim(IPR,ks-k,j,i) = pmb->phydro->w_bound(IPR,ks-k,j,i);


//       }
//     }
//   }
// }

// void DirichletOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
//                  Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
// {
//   Real x,y,z;
//   Real vx,vy,vz,p,rho;
//   for (int k=1; k<=(NGHOST); ++k) {
//     for (int j=js; j<=je; ++j) {
//       for (int i=is; i<=ie; ++i) {

//           prim(IDN,ke+k,j,i) = pmb->phydro->w_bound(IDN,ke+k,j,i);
//           prim(IVX,ke+k,j,i) = pmb->phydro->w_bound(IVX,ke+k,j,i);
//           prim(IVY,ke+k,j,i) = pmb->phydro->w_bound(IVY,ke+k,j,i);
//           prim(IVZ,ke+k,j,i) = pmb->phydro->w_bound(IVZ,ke+k,j,i);
//           prim(IPR,ke+k,j,i) = pmb->phydro->w_bound(IPR,ke+k,j,i);

//           if(prim(IVZ,ke+k,j,i)>0) prim(IVZ,ke+k,j,i) = 0.;

//       }
//     }
//   }
// }

Real time_step(MeshBlock *pmb){

  if (pmb->pmy_mesh->ncycle<3) return 1e-10;
  else {

    Real dt = 1e10;

  int is = pmb->is; int js = pmb->js; int ks = pmb->ks;
  int ie = pmb->ie; int je = pmb->je; int ke = pmb->ke;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          Real kbT_keV = mu_highT*mp_over_kev*(pmb->phydro->w(IPR,k,j,i)/pmb->phydro->w(IDN,k,j,i));

          Real t_cool = tcool(pmb->phydro->w(IDN,k,j,i),kbT_keV);

          dt = std::min(t_cool,dt);

  }
  }
  }

  return 0.5 * dt;


  }
}
int RefinementCondition(MeshBlock *pmb)
{
  int refine = 0;

    Real DX,DY,DZ;
    Real dx,dy,dz;
  get_uniform_box_spacing(pmb->pmy_mesh->mesh_size,&DX,&DY,&DZ);
  get_uniform_box_spacing(pmb->block_size,&dx,&dy,&dz);

  int current_level = int( std::log(DX/dx)/std::log(2.0) + 0.5);


  if (current_level >=max_refinement_level) return 0;


  for (int k = pmb->ks; k<=pmb->ke;k++){
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {
      
          
          for (int n_level = 1; n_level<=max_refinement_level; n_level++){
          
            Real x,y,z;
            Real new_r_in;
            get_cartesian_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &x, &y, &z);
            Real box_radius = 1./std::pow(2.,n_level)*0.9999;

          

                      // if (k==pmb->ks && j ==pmb->js && i ==pmb->is){
            //fprintf(stderr,"current level: %d n_level: %d box_radius: %g \n x: %g y: %g z: %g\n",current_level,n_level,box_radius,x,y,z);
            // }
            if (x<box_radius && x > -box_radius && y<box_radius
              && y > -box_radius && z<box_radius && z > -box_radius ){
              if (current_level < n_level){
                  new_r_in = 2* DX/std::pow(2.,n_level);
                  if (new_r_in<pmb->r_inner_boundary) pmb->r_inner_boundary = new_r_in;

                  //fprintf(stderr,"current level: %d n_level: %d box_radius: %g \n xmin: %g ymin: %g zmin: %g xmax: %g ymax: %g zmax: %g\n",current_level,
                    //n_level,box_radius,pmb->block_size.x1min,pmb->block_size.x2min,pmb->block_size.x3min,pmb->block_size.x1max,pmb->block_size.x2max,pmb->block_size.x3max);
                  return  1;
              }
            }

          
          }
  }
 }
}
  return 0;
}


/* Apply inner "inflow" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim){

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

          if (r < pmb->r_inner_boundary){
              
              // r_hat_x = x/r;
              // r_hat_y = y/r;
              // r_hat_z = z/r;
              // r1 = r;
              // ir = 1;


              // while (r1 <r_inner_boundary) {
                  
              //      i1 = int (i + ir*x/r + 0.5);   //Note: Addition of 0.5 ensures that "int" rounds up
              //      j1 = int (j + (ir*y/r + 0.5)*is_2D);
              //      k1 = int (k + (ir*z/r + 0.5)*is_3D);
                  
              //      bound_ijk(pmb->pcoord,&i1,&j1,&k1);
                  
              //      r1 = sqrt( SQR(pmb->pcoord->x1v(i1)) + SQR(pmb->pcoord->x2v(j1))*is_2D + SQR(pmb->pcoord->x3v(k1))*is_3D );
              //      ir = ir + 1;
              //  }

              // while (r1 <r_inner_boundary) {
                  
              //     i1 = int (i + ir*x/r + 0.5);   //Note: Addition of 0.5 ensures that "int" rounds up
              //     j1 = int (j + (ir*y/r + 0.5)*is_2D);
              //     k1 = int (k + (ir*z/r + 0.5)*is_3D);
                  
              //     bound_ijk(pmb->pcoord,&i1,&j1,&k1);
                  
              //     r1 = sqrt( SQR(pmb->pcoord->x1v(i1)) + SQR(pmb->pcoord->x2v(j1))*is_2D + SQR(pmb->pcoord->x3v(k1))*is_3D );
              //     ir = ir + 1;
              // }

              // //fprintf(stderr,"r: %g r_inner: %g r1: %g ijk: %d %d %d i1j1k1: %d %d %d \n", r, r_inner_boundary, r1,i,j,k,i1,j1,k1);

              
              // i2 = int (i + ir*x/r + 0.5);   //Note: Addition of 0.5 ensures that "int" rounds up
              // j2 = int (j + (ir*y/r + 0.5)*is_2D);
              // k2 = int (k + (ir*z/r + 0.5)*is_3D);
              
              // bound_ijk(pmb->pcoord,&i2,&j2,&k2);
              // r2 = sqrt( SQR(pmb->pcoord->x1v(i2)) + SQR(pmb->pcoord->x2v(j2))*is_2D + SQR(pmb->pcoord->x3v(k2))*is_3D );
              

              // for (int n=0; n<((NHYDRO+NFIELD)); ++n) {
                  
                  
              //     dU_dr =(cons(n,k2,j2,i2) - cons(n,k1,j1,i1)) /  (r2-r1 + SMALL);
                  
              //     cons(n,k,j,i) = cons(n,k1,j1,i1); //+ dU_dr * (r-r1);
                  
              // }
              
              

              Real rho_flr = 1e-7;
              Real p_floor = 1e-10;

              prim(IDN,k,j,i) = rho_flr;
              prim(IVX,k,j,i) = 0.;
              prim(IVY,k,j,i) = 0.;
              prim(IVZ,k,j,i) = 0.;
              prim(IPR,k,j,i) = p_floor;


              Real drho = prim(IDN,k,j,i) - rho_flr;
              pmb->user_out_var(N_user_history_vars,k,j,i) += drho;
              // cons(IDN,k,j,i) = rho_flr;
              // cons(IM1,k,j,i) = 0.;
              // cons(IM2,k,j,i) = 0.;
              // cons(IM3,k,j,i) = 0.;
              // cons(IEN,k,j,i) = p_floor/gm1;

              // for (int n=0; n<((NHYDRO+NFIELD)); ++n) {
                 
              //     cons(n,k,j,i) = cons(n,k1,j1,i1); 
                 
              // }

              /* Prevent outflow from inner boundary */ 
              if (prim(IVX,k,j,i)*x/r >0 ) prim(IVX,k,j,i) = 0.;
              if (prim(IVY,k,j,i)*y/r >0 ) prim(IVY,k,j,i) = 0.;
              if (prim(IVZ,k,j,i)*z/r >0 ) prim(IVZ,k,j,i) = 0.;

              
              
          }



}}}



}


/* 
* -------------------------------------------------------------------
*     Source term for B-field from stellar winds 
* -------------------------------------------------------------------
*/
/* curl g = phi_hat */
Real g_func(const Real x, const Real y, const Real z,const int dim,const int i_star, const int is_3D)
{
    
    //Primed in star frame, unprimed in lab frame
    
  AthenaArray<Real> x_vec,x_vec_prime,x_vec_star,tmp;
  x_vec.NewAthenaArray(3);
  x_vec_prime.NewAthenaArray(3);
  x_vec_star.NewAthenaArray(3);
  tmp.NewAthenaArray(3);

  x_vec(0) = x;
  x_vec(1) = y;
  x_vec(2) = z;

  x_vec_star(0) = star[i_star].x1;
  x_vec_star(1) = star[i_star].x2;
  x_vec_star(2) = star[i_star].x3;

  add_vectors(-1,x_vec,x_vec_star,tmp);

  x_vec_prime(0) = dot(star[i_star].x_axis,tmp);
  x_vec_prime(1) = dot(star[i_star].y_axis,tmp);
  x_vec_prime(2) = dot(star[i_star].spin_axis,tmp);


  Real r_prime ;

  if (is_3D==1) r_prime = norm_calc(x_vec_prime) ;
  else r_prime = std::sqrt(SQR(x_vec_prime(0)) + SQR(x_vec_prime(1)));

  // If outside stellar radius, return 0 //
  if (r_prime > star[i_star].radius || r_prime < star[i_star].r_cut) return 0.;


  Real theta_prime = std::acos( x_vec_prime(2)/(r_prime+TINY_NUMBER) );
    
    
  AthenaArray<Real> g_prime,g;
  g_prime.NewAthenaArray(3);
  g.NewAthenaArray(3);
  
#if (FIELD_SOURCE == TYPE_1)
  if (is_3D==1) for (int i=0; i<=2; ++i) g_prime(i) = - std::cos(theta_prime)  * x_vec_prime(i) * std::sin(r_prime/star[i_star].radius * PI); 
#elif (FIELD_SOURCE==TYPE_2)
  if (is_3D==1) for (int i=0; i<=2; ++i) g_prime(i) = - std::cos(theta_prime)  * x_vec_prime(i) * std::cos(r_prime/star[i_star].radius * PI/2.0); 
#else
  if (is_3D==1) for (int i=0; i<=2; ++i) g_prime(i) = theta_prime  * x_vec_prime(i) * std::sin(r_prime/star[i_star].radius * PI); 
#endif
  else {
    g_prime(0) = 0;
    g_prime(1) = 0;
    g_prime(2) = -r_prime; 
  }
  //for (int i=0; i<=2; ++i) g_prime(i) = theta_prime * x_vec_prime(i);
  //now rotate g_prime to get it in the lab frame
  for (int i=0; i<=2; ++i) g(i) = g_prime(0) * star[i_star].x_axis(i) + g_prime(1) * star[i_star].y_axis(i) + g_prime(2) * star[i_star].spin_axis(i);
  
  
  //if (dim==2) fprintf(stderr,"x: %g %g %g g: %g %g %g\n",x,y,z,g(0),g(1),g(2));
  //if (dim==2) fprintf(stderr,"x_prime: %g %g %g g: %g %g %g\n",x_vec_prime(0),x_vec_prime(1),x_vec_prime(2),g_prime(0),g_prime(1),g_prime(2));

  x_vec.DeleteAthenaArray();
  x_vec_prime.DeleteAthenaArray();
  x_vec_star.DeleteAthenaArray();
  g_prime.DeleteAthenaArray();
  tmp.DeleteAthenaArray();

  return g(dim);
}

void get_B_source( const Coordinates *pG, const int i, const int j, const int k, const int i_star, const int n_depth, 
  Real *Bx_source, Real *By_source,Real *Bz_source)
{
    


    *Bx_source = 0.0;
    *By_source = 0.0;
    *Bz_source = 0.0;

    Real dx3f;

    if (pG->pmy_block->block_size.nx3>1) dx3f = pG->dx3f(k);
    else dx3f = pG->pmy_block->pmy_mesh->mesh_size.x3max -  pG->pmy_block->pmy_mesh->mesh_size.x3min;
    
    
    int n_in_overlay = 0.0;
    // Use face spacing since that is the length of an edge //
    
    // Discretize the cell edge into a subgrid of size n_depth //
    Real di_x = pG->dx1f(i) / ((float)(n_depth-1));
    Real di_y = pG->dx2f(j) / ((float)(n_depth-1));
    Real di_z = dx3f / ((float)(n_depth-1));

    //Real di_z = 2.0/((float)(n_depth-1));
    Real xm,ym,zm;
    xm = pG->x1f(i);
    ym = pG->x2f(j);
    
    if (pG->pmy_block->block_size.nx3>1) zm = pG->x3f(k);
    else zm = pG->pmy_block->pmy_mesh->mesh_size.x3min;

    //zm = -1.0;
    
    //integrating from left (e.g.; i) to right (e.g.; i+1) face  //

    Real integrand_x,integrand_y,integrand_z;


    Real x,y,z; 
#if (FIELD_SOURCE == TYPE_1)
    Real Bdot_phi = PI * star[i_star].B_A * star[i_star].r_A * star[i_star].Vwind / (SQR(star[i_star].radius));
#elif (FIELD_SOURCE == TYPE_2)
    Real Bdot_phi = SQR(PI)/(2.0*PI-4.0) *star[i_star].B_A * star[i_star].r_A * star[i_star].Vwind / (SQR(star[i_star].radius));
#else
    Real Bdot_phi = PI * star[i_star].B_A * star[i_star].r_A * star[i_star].Vwind / (SQR(star[i_star].radius));
#endif
    for (int m = 0; m < n_depth; m++) {


        x = xm + m * di_x;
        y = ym + m * di_y;
        z = zm + m * di_z;


        //fprintf(stderr,"xyz: %g %g %g zm: %g i: %d di_z: %g \n", x,y,z,zm,m,di_z);

        /* Account for endpoints of integral */
        Real fac = 1.;

        if (m==0 || m== n_depth- 1) fac = fac * 0.5;
                        
              integrand_x = Bdot_phi * g_func(x,ym,zm,0,i_star,(pG->pmy_block->block_size.nx3>1));
              integrand_y = Bdot_phi * g_func(xm,y,zm,1,i_star,(pG->pmy_block->block_size.nx3>1));
              integrand_z = Bdot_phi * g_func(xm,ym,z,2,i_star,(pG->pmy_block->block_size.nx3>1));
              *Bx_source += integrand_x * di_x * fac / pG->dx1f(i);
              *By_source += integrand_y * di_y * fac / pG->dx2f(j);
              *Bz_source += integrand_z * di_z * fac / dx3f;

              if (isnan(*Bx_source) || isnan(*By_source) || isnan(*Bz_source) ){
                fprintf(stderr, "B: %g %g %g Bdot_phi: %g \n x y z : %g %g %g xm ym zm: %g %g %g \n g: %g %g %g \n ", *Bx_source, *By_source, *Bz_source, Bdot_phi, x,y,z,xm,ym,zm, 
                  g_func(x,ym,zm,0,i_star,(pG->pmy_block->block_size.nx3>1)),g_func(xm,y,zm,1,i_star,(pG->pmy_block->block_size.nx3>1)),g_func(xm,ym,z,2,i_star,(pG->pmy_block->block_size.nx3>1)));
                exit(0);
              }
        

    }


    //*Bx_source = g_func(xm,ym,zm)
    
    

}

/* 
* -------------------------------------------------------------------
*     The constant source terms for stars in a location i,j,k 
*      (This also includes the calls to the N-body routines
*         that evolve the star velocities and positions)
* -------------------------------------------------------------------
*/

void star_update_function(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons){
  /* Update the stellar positions/velocities based on Kepler Orbits. */

    int i_star;
    for (i_star=0; i_star<nstars; i_star++) {

     update_star(star,i_star, time);
     update_star_size(pmb->pcoord,star,i_star);

    }

}

void cons_force(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{



   for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
          
          
          
          if (isnan(cons(IDN,k,j,i)) || isnan(cons(IM1,k,j,i)) || isnan(cons(IM2,k,j,i)) || isnan(cons(IM3,k,j,i)) || isnan(cons(IEN,k,j,i))){
              
              for (int m=0; m < (NHYDRO+NFIELD); ++m){
                  fprintf(stderr,"m = %d \n ----------------\n",m);
              for (int k_tmp=-2; k_tmp<=2; ++k_tmp) {
                  for (int j_tmp=-2; j_tmp<=2; ++j_tmp) {
                      for (int i_tmp=-2; i_tmp<=2; ++i_tmp) {
                          
                          fprintf(stderr, "k,j,i: %d %d %d  cons: %g  \n", k+ k_tmp,j + j_tmp,i + i_tmp, cons(m,k+ k_tmp,j + j_tmp,i + i_tmp));
                      }}}}
              
              
              exit(0);
          }

      Real d0,dx1,dx2,dx3,r,a0,dr,frac_overlay;
      Real vx_avg,vy_avg,vz_avg,vsq_avg,bsq_func_avg, P_func_avg;
      Real x,y,z;
      int i_star, star_check, mask_depth;
      mask_depth = 5;
      dx1 = 0.0; dx2 = 0.0; dx3 = 0.0;
      dr = 0.0;



    


      star_check = 0;


      if (COORDINATE_SYSTEM == "cartesian"){

        if (pmb->block_size.nx1 > 1) dr += SQR(pmb->pcoord->dx1f(i));
        if (pmb->block_size.nx2 > 1) dr += SQR(pmb->pcoord->dx2f(j));
        if (pmb->block_size.nx3 > 1) dr += SQR(pmb->pcoord->dx3f(k));
        }
        else if (COORDINATE_SYSTEM == "cylindrical"){
          if (pmb->block_size.nx1 > 1) dr += SQR(pmb->pcoord->dx1f(i)) ;
          if (pmb->block_size.nx2 > 1) dr += SQR(pmb->pcoord->x1v(i) * pmb->pcoord->dx2f(j));
        if (pmb->block_size.nx3 > 1) dr += SQR(pmb->pcoord->dx3f(k));
        }
        else if  (COORDINATE_SYSTEM == "spherical_polar"){
          if (pmb->block_size.nx1 > 1) dr += SQR(pmb->pcoord->dx1f(i)) ;
          if (pmb->block_size.nx2 > 1) dr += SQR(pmb->pcoord->x1v(i)  * pmb->pcoord->dx2f(j));
        if (pmb->block_size.nx3 > 1) dr += SQR(pmb->pcoord->x1v(i) * std::sin(pmb->pcoord->x3v(k)) * pmb->pcoord->dx3f(k));
        }

          dr = sqrt(dr);

  for (i_star = 0; i_star < nstars; i_star++) {



  get_cartesian_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &x, &y, &z);

  if (pmb->block_size.nx1 > 1) dx1 = x - star[i_star].x1;
  if (pmb->block_size.nx2 > 1) dx2 = y - star[i_star].x2;
  if (pmb->block_size.nx3 > 1) dx3 = z - star[i_star].x3;

  r  = sqrt(dx1*dx1 + dx2*dx2 + dx3*dx3);
  

  /* Apply the star source conservative terms (density, energy) */
  /* First, check where cell is relative to star effective radius */
  if (r > star[i_star].radius + dr) {
    /* Cell entirely outside star mask - do nothing */
  } else {
    /* Some fraction of the cell may contain part of the mask */
    star_check = 1;


    frac_overlay = mask(pmb->pcoord,i,j,k,i_star,mask_depth,bcc,&vx_avg,&vy_avg,&vz_avg,&vsq_avg, &bsq_func_avg, &P_func_avg);

    //get_net_wind_velocity(pmb->pcoord,star,i_star,x,y,z,&vx_avg,&vy_avg,&vz_avg);


    d0 = frac_overlay * (star[i_star].Mdot / star[i_star].volume) *dt;

    Real d_old = cons(IDN,k,j,i);
    Real vsq_old = SQR(cons(IM1,k,j,i)/cons(IDN,k,j,i)) + SQR(cons(IM2,k,j,i)/cons(IDN,k,j,i)) + SQR(cons(IM3,k,j,i)/cons(IDN,k,j,i)) ;
    cons(IDN,k,j,i) += d0 ;
      
    Real vx1,vx2,vx3;
    convert_cartesian_vector_to_code_coords(vx_avg,vy_avg,vz_avg,star[i_star].x1,star[i_star].x2,star[i_star].x3, &vx1,&vx2,&vx3);


     cons(IM1,k,j,i) += d0 * vx1;
     cons(IM2,k,j,i) += d0 * vx2;
     cons(IM3,k,j,i) += d0 * vx3;


    Real d_new = cons(IDN,k,j,i);
    Real vsq_new = SQR(cons(IM1,k,j,i)/cons(IDN,k,j,i)) + SQR(cons(IM2,k,j,i)/cons(IDN,k,j,i)) + SQR(cons(IM3,k,j,i)/cons(IDN,k,j,i)) ;

    //cons(IEN,k,j,i) += 0.5 * d0 * (vsq_avg);  
    cons(IEN,k,j,i) +=0.5 * (d_new * vsq_new - d_old * vsq_old);

      
      
      /* bsq_func_avg is d/dt(B^2/2) */
      if (MAGNETIC_FIELDS_ENABLED) cons(IEN,k,j,i) += dt * (bsq_func_avg ); //+ P_func_avg); //0.5 * dt * bsq_func_avg * 12.0 * SQR(PI) * SQR(star[i_star].B_A*star[i_star].r_A) * star[i_star].Vwind / (std::pow(star[i_star].radius,3) * (2.0*SQR(PI) - 3.0));


    /* Rewrite solution to be the stellar wind solution */

    // Real dens = star[i_star].Mdot/star[i_star].Vwind/SQR(r)/(4.0*PI);

    //  cons(IDN,k,j,i) = dens;
    //  cons(IM1,k,j,i) = dens * vx1;
    //  cons(IM2,k,j,i) = dens * vx2;
    //  cons(IM3,k,j,i) = dens * vx3;
    //  cons(IEN,k,j,i) = 0.5 * dens * (SQR(vx1) + SQR(vx2) + SQR(vx3)); 


 
  }
  }

          
        


}}}


  return;
}


/* 
* -------------------------------------------------------------------
*     The source terms for the emf used in the induction equation provided by stars 
* -------------------------------------------------------------------
*/
void emf_source(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim,  const AthenaArray<Real> &bcc, const AthenaArray<Real> &cons, EdgeField &e)
{


   for (int k=pmb->ks; k<=pmb->ke+1; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je+1; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie+1; ++i) {
          
          
          
          if (isnan(cons(IDN,k,j,i)) || isnan(cons(IM1,k,j,i)) || isnan(cons(IM2,k,j,i)) || isnan(cons(IM3,k,j,i)) || isnan(cons(IEN,k,j,i))){
              
              for (int m=0; m < (NHYDRO+NFIELD); ++m){
                  fprintf(stderr,"m = %d \n ----------------\n",m);
              for (int k_tmp=-2; k_tmp<=2; ++k_tmp) {
                  for (int j_tmp=-2; j_tmp<=2; ++j_tmp) {
                      for (int i_tmp=-2; i_tmp<=2; ++i_tmp) {
                          
                          fprintf(stderr, "k,j,i: %d %d %d  cons: %g  \n", k+ k_tmp,j + j_tmp,i + i_tmp, cons(m,k+ k_tmp,j + j_tmp,i + i_tmp));
                      }}}}
              
              
              exit(0);
          }

      Real d0,dx1,dx2,dx3,r,a0,dr,frac_overlay;
      Real vx_avg,vy_avg,vz_avg,vsq_avg;
      Real x,y,z;
      int i_star, star_check, mask_depth;
      mask_depth = 5;
      dx1 = 0.0; dx2 = 0.0; dx3 = 0.0;
      dr = 0.0;



    


      star_check = 0;


      if (COORDINATE_SYSTEM == "cartesian"){

        if (pmb->block_size.nx1 > 1) dr += SQR(pmb->pcoord->dx1f(i));
        if (pmb->block_size.nx2 > 1) dr += SQR(pmb->pcoord->dx2f(j));
        if (pmb->block_size.nx3 > 1) dr += SQR(pmb->pcoord->dx3f(k));
        }
        else if (COORDINATE_SYSTEM == "cylindrical"){
          if (pmb->block_size.nx1 > 1) dr += SQR(pmb->pcoord->dx1f(i)) ;
          if (pmb->block_size.nx2 > 1) dr += SQR(pmb->pcoord->x1v(i) * pmb->pcoord->dx2f(j));
        if (pmb->block_size.nx3 > 1) dr += SQR(pmb->pcoord->dx3f(k));
        }
        else if  (COORDINATE_SYSTEM == "spherical_polar"){
          if (pmb->block_size.nx1 > 1) dr += SQR(pmb->pcoord->dx1f(i)) ;
          if (pmb->block_size.nx2 > 1) dr += SQR(pmb->pcoord->x1v(i)  * pmb->pcoord->dx2f(j));
        if (pmb->block_size.nx3 > 1) dr += SQR(pmb->pcoord->x1v(i) * std::sin(pmb->pcoord->x3v(k)) * pmb->pcoord->dx3f(k));
        }

          dr = sqrt(dr);

  for (i_star = 0; i_star < nstars; i_star++) {

  get_cartesian_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &x, &y, &z);

  if (pmb->block_size.nx1 > 1) dx1 = x - star[i_star].x1;
  if (pmb->block_size.nx2 > 1) dx2 = y - star[i_star].x2;
  if (pmb->block_size.nx3 > 1) dx3 = z - star[i_star].x3;

  r  = sqrt(dx1*dx1 + dx2*dx2 + dx3*dx3);
  

  /* Apply the star source conservative terms (density, energy) */
  /* First, check where cell is relative to star effective radius */
  if (r > star[i_star].radius + dr) {
    /* Cell entirely outside star mask - do nothing */
  } else {
    /* Some fraction of the cell may contain part of the mask */
    star_check = 1;


    if (MAGNETIC_FIELDS_ENABLED){
      Real Bx_source,By_source,Bz_source;

      //Note:  convention for the following is i,j,k
      //CHECK: is cons variable the right one to edit?
      //CHECK: is cons B the face B?  
      // Bx_source.NewAthenaArray(2,2);
      // By_source.NewAthenaArray(2,2);
      // Bz_source.NewAthenaArray(2,2);

      get_B_source(pmb->pcoord, i,j,k,i_star,mask_depth,&Bx_source,&By_source,&Bz_source);


      //fprintf(stderr," B source: %g %g %g \n", Bx_source,By_source,Bz_source);
     if (i < pmb->ie+1) e.x1e(k,j,i) += Bx_source;
     if (j < pmb->je+1) e.x2e(k,j,i) += By_source;
     if (k < pmb->ke+1) e.x3e(k,j,i) += Bz_source;


     if (r<=star[i_star].r_cut){
       if (i < pmb->ie+1) e.x1e(k,j,i) =0;
       if (j < pmb->je+1) e.x2e(k,j,i) =0;
       if (k < pmb->ke+1) e.x3e(k,j,i) =0;

       for (int d1=0; d1<=1; ++d1) for (int d2=0; d2<=1; ++d2){
        if ( (i < pmb->ie+1) && ( (k+d1) <= pmb->ke+1 ) && ( (j+d2) <= pmb->je+1)) e.x1e(k+d1,j+d2,i) = 0;
        if ( (j < pmb->je+1) && ( (k+d1) <= pmb->ke+1 ) && ( (i+d2) <= pmb->ie+1)) e.x2e(k+d1,j,i+d2) = 0;
        if ( (k < pmb->ke+1) && ( (j+d1) <= pmb->je+1 ) && ( (i+d2) <= pmb->ie+1)) e.x3e(k,j+d1,i+d2) = 0;
       }


     }

     
      //if ( (i < pmb->ie+1) && (j < pmb->je+1) && (k < pmb->ke+1))

     //fprintf(stderr,"grid spacing: x1: %g dx1: %g \n",pmb->pcoord->x1v(i),pmb->pcoord->dx1f(i));
        if (std::fabs(pmb->pcoord->x1v(i))<1e-10 && std::fabs(pmb->pcoord->x2v(j))<1e-10){

          if (pmb->block_size.nx3==1 || std::fabs(pmb->pcoord->x3v(k))<1e-10  ){

            //fprintf(stderr,"this is called!\n");
            //fprintf(stderr,"ijk: %d %d %d \n xyz: %g %g %g \n E: %g %g %g %g \n",
             // i,j,k,
             // pmb->pcoord->x1f(i),pmb->pcoord->x2f(j),pmb->pcoord->x3f(k),
             // e.x3e(k,j,i), e.x3e(k,j+1,i),e.x3e(k,j,i+1),e.x3e(k,j+1,i+1));

             // Real e_avg;
             // e_avg= (e.x3e(k,j,i) + e.x3e(k,j+1,i) + e.x3e(k,j,i+1) + e.x3e(k,j+1,i+1))/4.0;
             // // if (pmb->block_size.nx3 > 1) 
             // //  e_avg = (e_avg*4.0 + e.x3e(k+1,j,i) + e.x3e(k+1,j+1,i) + e.x3e(k+1,j,i+1) + e.x3e(k+1,j+1,i+1) )/8.0; 
             // e.x3e(k,j,i) = e_avg; 
             // e.x3e(k,j+1,i) = e_avg;
             // e.x3e(k,j,i+1) = e_avg;
             // e.x3e(k,j+1,i+1) = e_avg;

             // e_avg= (e.x2e(k,j,i) + e.x2e(k+1,j,i) + e.x2e(k,j,i+1) + e.x2e(k+1,j,i+1))/4.0;
             // // if (pmb->block_size.nx3 > 1) 
             // //  e_avg = (e_avg*4.0 + e.x2e(k+1,j,i) + e.x2e(k+1,j+1,i) + e.x2e(k+1,j,i+1) + e.x2e(k+1,j+1,i+1) )/8.0;

             // e.x2e(k,j,i) = e_avg; 
             // e.x2e(k+1,j,i) = e_avg;
             // e.x2e(k,j,i+1) = e_avg;
             // e.x2e(k+1,j,i+1) = e_avg;

             // e_avg= (e.x1e(k,j,i) + e.x1e(k+1,j,i) + e.x1e(k,j+1,i) + e.x1e(k+1,j+1,i))/4.0;
             // // if (pmb->block_size.nx3 > 1) 
             // //  e_avg = (e_avg*4.0 + e.x1e(k+1,j,i) + e.x1e(k+1,j+1,i) + e.x1e(k+1,j,i+1) + e.x1e(k+1,j+1,i+1) )/8.0;

             // e.x1e(k,j,i) = e_avg; 
             // e.x1e(k+1,j,i) = e_avg;
             // e.x1e(k,j+1,i) = e_avg;
             // e.x1e(k+1,j+1,i) = e_avg;

     }
     }



    

      // Real dx = pmb->pcoord->dx1f(i);
      // Real dy = pmb->pcoord->dx2f(j);
      // Real dz = pmb->pcoord->dx3f(k);

      // pmb->pfield->b.x1f(k,j,i) += - dt/dy * ( Bz_source(0,1) - Bz_source(0,0) )
      //                      + dt/dz * ( By_source(0,1) - By_source(0,0) );
      // pmb->pfield->b.x2f(k,j,i)   +=   dt/dx * ( Bz_source(1,0) - Bz_source(0,0) )
      //                      - dt/dz * ( Bx_source(0,1) - Bx_source(0,0) );
      // pmb->pfield->b.x3f(k,j,i)   += - dt/dx * ( By_source(1,0) - By_source(0,0) )
      //                      + dt/dy * ( Bx_source(1,0) - Bx_source(0,0) );
        

    }
 
  }
  }

          
        


}}}



return;
}
/* 
* -------------------------------------------------------------------
*     Get Stellar Positions For History Output
* -------------------------------------------------------------------
*/
Real star_position(MeshBlock *pmb, int iout)
{
/*Total number of position variables is nstars * 3 */

  int i_star = iout/3;
  int i_pos  = iout % 3;

  int n_meshes = pmb->pmy_mesh->nbtotal; // (pmb->pmy_mesh->mesh_size.nx1 * pmb->pmy_mesh->mesh_size.nx2 * pmb->pmy_mesh->mesh_size.nx3 ) / (pmb->block_size.nx3 * pmb->block_size.nx2 * pmb->block_size.nx1);

  if (i_pos==0) return star[i_star].x1 / (1. * n_meshes);
  else if (i_pos==1) return star[i_star].x2 / (1. * n_meshes);
  else return star[i_star].x3 / (1. * n_meshes);

}


/* Convert position to location of cell.  Note well: assumes a uniform grid in each meshblock */
void get_ijk(MeshBlock *pmb,const Real x, const Real y, const Real z , int *i, int *j, int *k){
    Real dx = pmb->pcoord->dx1f(0);
    Real dy = pmb->pcoord->dx2f(0);
    Real dz = pmb->pcoord->dx3f(0);

   *i = int ( (x-pmb->block_size.x1min)/dx) + pmb->is;
   *j = int ( (y-pmb->block_size.x2min)/dy) + pmb->js;
   *k = pmb->ks;

   if (*i>pmb->ie) *i = pmb->ie;
   if (*j>pmb->je) *j = pmb->je;

   if (pmb->block_size.nx3>1) *k = int ( (z-pmb->block_size.x3min)/dz) + pmb->ks;

   if (*k>pmb->ke) *k = pmb->ke;
   // if ( (x < pmb->pcoord->x1f(*i) ) || (x > pmb->pcoord->x1f(*i+1) ) ||
   //      (y < pmb->pcoord->x2f(*j) ) || (y > pmb->pcoord->x2f(*j+1) ) ){
   //        fprintf(stderr,"Error in get_ijk, ijk : %d %d %d outside of cell for xyz: %g %g %g \n",*i,*j,*k,x,y,z);
   //        fprintf(stderr,"Error in get_ijk, x1_bound : %g %g \n x2_bound : %g %g\n",pmb->pcoord->x1f(*i),pmb->pcoord->x1f(*i+1),pmb->pcoord->x2f(*j),pmb->pcoord->x2f(*j+1));

   //        exit(0);
   //      }

   //  if (pmb->block_size.nx3>1){
   //    if ( (z < pmb->pcoord->x3f(*k) ) || (z > pmb->pcoord->x3f(*k+1) ) ){
   //      fprintf(stderr,"Error in get_ijk, ijk : %d %d %d outside of cell for xyz: %g %g %g",*i,*j,*k,x,y,z);
   //        exit(0);
   //    }
   //  } 
    
}

/* Compute the total mass removed from the inner boundary.  Cumulative over the whole simulation */
Real compute_mass_removed(MeshBlock *pmb, int iout){
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  AthenaArray<Real> vol;
  vol.NewAthenaArray(ncells1);

  Real sum = 0;
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
      for(int i=is; i<=ie; i++) {
        sum += pmb->user_out_var(N_user_history_vars,k,j,i) * vol(i);
      }
    }
  }

  vol.DeleteAthenaArray();

  return sum;
}

/* Compute the total mass enclosed in the inner boundary.   */
Real compute_mass_in_boundary(MeshBlock *pmb, int iout){
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  AthenaArray<Real> vol;
  vol.NewAthenaArray(ncells1);

  Real sum = 0;
  for(int k=ks; k<=ke; k++) {
    for(int j=js; j<=je; j++) {
      pmb->pcoord->CellVolume(k,j,pmb->is,pmb->ie,vol);
      for(int i=is; i<=ie; i++) {
        Real x,y,z;
        
        int is_3D = (pmb->block_size.nx3>1);
        int is_2D = (pmb->block_size.nx2>1);
        
        get_cartesian_coords(pmb->pcoord->x1v(i),pmb->pcoord->x2v(j),pmb->pcoord->x3v(k),&x,&y,&z);
        
        Real r = std::sqrt( SQR(x) + SQR(y)*is_2D + SQR(z)*is_3D);


        if (r<pmb->r_inner_boundary) sum += pmb->phydro->u(IDN,k,j,i) * vol(i);
      }
    }
  }

  vol.DeleteAthenaArray();

  return sum;
}

/* Compute the average of primitives and user_out_vars over angle   */
Real radial_profile(MeshBlock *pmb, int iout){
    int i_r = iout % N_r;
    int i_prim =  iout / N_r + IDN;
    
    int i_user_var = i_prim - (NHYDRO) - IDN -1;


    r_dump_min = pmb->r_inner_boundary/2.;
    r_dump_max = pmb->pmy_mesh->mesh_size.x1max;
    
    Real r = r_dump_min * std::pow(10., (i_r * std::log10(r_dump_max/r_dump_min)/(1.*N_r-1.)) );


    if (i_prim == (NHYDRO)) return r/(1.*pmb->pmy_mesh->nbtotal);

    
    int N_phi = 128 ;
    int N_theta = 1;
    
    if (pmb->block_size.nx3>1) N_theta = 128;
    
    Real dphi = 2.*PI/(N_phi*1.-1.);
    
    Real dtheta = 1.;
    Real Omega = 2.*PI;

    if (pmb->block_size.nx3>1) {
      dtheta = PI/(N_theta*1.-1.);
      Omega = 4.*PI;
    }
    

    Real result = 0.;
        for (int i_theta=0; i_theta<N_theta; ++i_theta) {
            for (int i_phi=0; i_phi<N_phi; ++i_phi) {
                Real phi = i_phi * dphi;
                Real theta = PI/2.;
                int i,j,k;
                
                if (pmb->block_size.nx3>1) theta = i_theta * dtheta;
                
                Real x = r * std::cos(phi) * std::sin(theta);
                Real y = r * std::sin(phi) * std::sin(theta);
                Real z = r * std::cos(theta);
                Real fac = 1.;
                if ( is_in_block(pmb->block_size,x,y,z) ){
                    
                    get_ijk(pmb,x,y,z,&i,&j,&k);
                    

                    if (i_phi == 0 || i_phi == N_phi-1) fac = fac*0.5;
                    if ( (i_theta ==0 || i_theta == N_theta-1) && (pmb->block_size.nx3>1) ) fac = fac*0.5;

                    Real dOmega =  std::sin(theta)*dtheta*dphi * fac;
                    if (i_prim<=IPR){
                        result += pmb->phydro->w(i_prim,k,j,i) * dOmega / Omega;
                    }
                    else{
                        result += pmb->user_out_var(i_user_var,k,j,i) * dOmega / Omega;
                        
                    }
                    
                    
                }
                
                
                
            }
        }
    
    return result;
    
    
}

// /* Interpolate fluid inital conditions to computational grid */
// void interp_inits(const Real x, const Real y, const Real z, Real *rho, Real *vx, Real *vy, Real *vz, Real *p){


//  Real dx = x_inits(0,0,1) - x_inits(0,0,0);
//  Real dy = y_inits(0,1,0) - y_inits(0,0,0);
//  Real dz = z_inits(1,0,0) - z_inits(0,0,0);

//  Real x0 = x_inits(0,0,0);
//  Real y0 = y_inits(0,0,0);
//  Real z0 = z_inits(0,0,0);

//  int i = (int) ((x - x0) / dx + 0.5 + 1000) - 1000;
//  int j = (int) ((y - y0) / dy + 0.5 + 1000) - 1000;
//  int k = (int) ((z - z0) / dz + 0.5 + 1000) - 1000;
    

    
//     //fprintf(stderr,"x y z: %g %g %g \n dx dy dz: %g %g %g \n x0 y0 z0: %g %g %g \n i j k: %d %d %d \n",x,y,z,dx,dy,dz,x0,y0,z0,i,j,k);
//    //fprintf(stderr,"nx ny nz: %d %d %d\n", nx_inits,ny_inits,nz_inits);

//  //fprintf(stderr,"x y z: %g %g %g \n x_inits y_inits z_inits: %g %g %g \n", x,y,z, x_inits(k,j,i),y_inits(k,j,i),z_inits(k,j,i));

//  Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
//  if (i<0 || i>=nx_inits || j<0 || j>=ny_inits || k<0 || k>=nz_inits || r<r_min_inits){
//   *rho = 1e-8;
//   *vx = 0.;
//   *vy = 0.;
//   *vz = 0.;
//   *p = 1e-10;
//  }
//  else{
//   *rho = rho_inits(k,j,i);
//   *vx = v1_inits(k,j,i);
//   *vy = v2_inits(k,j,i);
//   *vz = v3_inits(k,j,i);
//   *p = press_inits(k,j,i);

//  }


// }

/*
 * -------------------------------------------------------------------
 *     Initialize Mesh
 * -------------------------------------------------------------------
 */
void Mesh::InitUserMeshData(ParameterInput *pin)
{



    // if (pin->GetString("mesh","ix1_bc") == "user") EnrollUserBoundaryFunction(INNER_X1, DirichletInnerX1);
    // if (pin->GetString("mesh","ox1_bc") == "user") EnrollUserBoundaryFunction(OUTER_X1, DirichletOuterX1);
    // if (pin->GetString("mesh","ix2_bc") == "user") EnrollUserBoundaryFunction(INNER_X2, DirichletInnerX2);
    // if (pin->GetString("mesh","ox2_bc") == "user") EnrollUserBoundaryFunction(OUTER_X2, DirichletOuterX2);
    // if (pin->GetString("mesh","ix3_bc") == "user") EnrollUserBoundaryFunction(INNER_X3, DirichletInnerX3);
    // if (pin->GetString("mesh","ox3_bc") == "user") EnrollUserBoundaryFunction(OUTER_X3, DirichletOuterX3);
    
    if(adaptive==true) EnrollUserRefinementCondition(RefinementCondition);


    beta_star = pin->GetOrAddReal("problem","beta_star",1000.0);

    EnrollUserTimeStepFunction(time_step);
    EnrollUserExplicitSourceFunction(cons_force);
    if (MAGNETIC_FIELDS_ENABLED) EnrollUserExplicitEMFSourceFunction(emf_source);
    EnrollUserStarUpdateFunction(star_update_function);
    EnrollUserRadSourceFunction(integrate_cool);
    
    int i = 0;
    if (MAGNETIC_FIELDS_ENABLED){
        N_user_history_vars += N_user_vars_field;
        N_user_vars += N_user_vars_field;
    }
    AllocateUserHistoryOutput(N_r*((NHYDRO)+N_user_history_vars +1) + 3.);
    if (COORDINATE_SYSTEM == "cartesian"){
        for (i = 0; i<N_r*((NHYDRO)+N_user_history_vars +1); i++){

            EnrollUserHistoryOutput(i, radial_profile, "rad_profiles"); 
        }
    }
    EnrollUserHistoryOutput(i,compute_mass_removed,"mass_removed");
    EnrollUserHistoryOutput(i+1,compute_mass_in_boundary,"boundary_mass");
    if (MAGNETIC_FIELDS_ENABLED) EnrollUserHistoryOutput(i+2, DivergenceB, "divB");
    // AllocateUserHistoryOutput(33*3);
    // for (int i = 0; i<33*3; i++){
    //     int i_star = i/3;
    //     int i_pos  = i % 3;
    //     EnrollUserHistoryOutput(i, star_position, "star_"); 
    // }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
    
    
    AllocateUserOutputVariables(N_user_vars);

    r_inner_boundary = 0.;
    loclist = pmy_mesh->loclist;
    gm_ = pin->GetOrAddReal("problem","GM",0.0);
    gm1 = peos->GetGamma() - 1.0;

    N_cells_per_radius = pin->GetOrAddReal("problem", "star_radius", 2.0);
    
    std::string file_name,cooling_file_name;
    file_name =  pin->GetString("problem","filename");
    cooling_file_name = pin->GetOrAddString("problem","cooling_file","lambda.tab");

    max_refinement_level = pin->GetOrAddReal("mesh","numlevel",0);

    if (max_refinement_level>0) max_refinement_level = max_refinement_level -1;
    
    Real dx_min,dy_min,dz_min;
    get_minimum_cell_lengths(pcoord, &dx_min, &dy_min, &dz_min);
    
    if (block_size.nx3>1)       r_inner_boundary = 2* std::max(std::max(dx_min,dy_min),dz_min); // r_inner_boundary = 2*sqrt( SQR(dx_min) + SQR(dy_min) + SQR(dz_min) );
    else if (block_size.nx2>1)  r_inner_boundary = 2 * std::max(dx_min,dy_min); //2*sqrt( SQR(dx_min) + SQR(dy_min)               );
    else                        r_inner_boundary = 2 * dx_min;
    
    Real v_ff = std::sqrt(2.*gm_/(r_inner_boundary+SMALL))*10.;
   // cs_max = std::min(cs_max,v_ff);
    
    
    read_stardata(star,file_name);
    init_cooling_tabs(cooling_file_name);
    init_cooling();




    
    
    /* Switch over to problem units and locate the star cells */
    //star_unitconvert(star);
    
    Real DX,DY,DZ, r_eff, r_cut,mask_volume; //Length of entire simulation
    get_uniform_box_spacing(pcoord->pmy_block->pmy_mesh->mesh_size,&DX,&DY,&DZ);
    get_uniform_box_spacing(pcoord->pmy_block->block_size,&DX,&DY,&DZ);
    get_star_size(pcoord,DX,DY,DZ,&r_eff,&r_cut,&mask_volume);

    

    for (int i_star=0; i_star<nstars; i_star++) {
      star[i_star].block_size = block_size;
      star[i_star].radius = r_eff;
      star[i_star].r_cut = r_cut;
      star[i_star].volume = mask_volume;
    }


    
    // if (pin->GetOrAddString("problem","zoom_in","no") == "yes"){
    //     std::string init_file_name;
    //     init_file_name =  pin->GetOrAddString("problem","init_filename", "inits.in");
    //     //read_inits(init_file_name);
    //     //set_boundary_arrays(init_file_name,block_size,pcoord,is,ie,js,je,ks,ke,phydro->w_bound);
        

    // }

    
    
    
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
    for(int j=js; j<=je; j++) {
      pmb->pcoord->Face1Area(k,   j,   is, ie+1, face1);
      pmb->pcoord->Face2Area(k,   j+1, is, ie,   face2p);
      pmb->pcoord->Face2Area(k,   j,   is, ie,   face2m);
      pmb->pcoord->Face3Area(k+1, j,   is, ie,   face3p);
      pmb->pcoord->Face3Area(k,   j,   is, ie,   face3m);
      for(int i=is; i<=ie; i++) {
        divb+=(face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
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
    for (int k=ks; k<=ke; ++k) {
#pragma omp parallel for schedule(static)
        for (int j=js; j<=je; ++j) {
#pragma simd
            for (int i=is; i<=ie; ++i) {
                
                Real x,y,z;
                
                int is_3D = (block_size.nx3>1);
                int is_2D = (block_size.nx2>1);
                
                
                get_cartesian_coords(pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k),&x,&y,&z);
                
                Real r = std::sqrt( SQR(x) + SQR(y)*is_2D + SQR(z)*is_3D);
                
                Real v_r = phydro->w(IVX,k,j,i) * x/r + phydro->w(IVY,k,j,i) * y/r * is_2D + phydro->w(IVZ,k,j,i) * z/r * is_3D;
                
                Real mdot = 2. * PI * r * phydro->w(IDN,k,j,i) * v_r ;
                if (is_3D) mdot = mdot * 2. * r;
                
                Real bernoulli = (SQR(phydro->w(IVX,k,j,i)) + SQR(phydro->w(IVY,k,j,i)) + SQR(phydro->w(IVZ,k,j,i)) )/2. +   (gm1+1.)/gm1 * phydro->w(IPR,k,j,i)/phydro->w(IDN,k,j,i) - gm_/r;
                
                Real lx1,lx2,lx3; //Angular momentum per unit mass  l = r x v
                Real kappa =phydro->w(IPR,k,j,i)/std::pow(phydro->w(IDN,k,j,i),gm1+1.);
                
                lx1 = pcoord->x2v(j) * phydro->w(IVZ,k,j,i) - pcoord->x3v(k) * phydro->w(IVY,k,j,i) ;
                lx2 = pcoord->x3v(k) * phydro->w(IVX,k,j,i) - pcoord->x1v(i) * phydro->w(IVZ,k,j,i) ;
                lx3 = pcoord->x1v(i) * phydro->w(IVY,k,j,i) - pcoord->x2v(j) * phydro->w(IVX,k,j,i) ;

                user_out_var(0,k,j,i) = mdot;
                user_out_var(1,k,j,i) = v_r;  
                user_out_var(2,k,j,i) = phydro->w(IDN,k,j,i) * lx1;
                user_out_var(3,k,j,i) = phydro->w(IDN,k,j,i) * lx2;
                user_out_var(4,k,j,i) = phydro->w(IDN,k,j,i) * lx3;

                user_out_var(5,k,j,i) = mdot * (v_r>0);
                user_out_var(6,k,j,i) = mdot * (v_r<0);
                
                user_out_var(7,k,j,i) = bernoulli * mdot;
                user_out_var(8,k,j,i) = bernoulli * mdot * (v_r>0);
                user_out_var(9,k,j,i) = bernoulli * mdot * (v_r<0);

                user_out_var(10,k,j,i) = lx1 * mdot;
                user_out_var(11,k,j,i) = lx2 * mdot;
                user_out_var(12,k,j,i) = lx3 * mdot;


                user_out_var(13,k,j,i) = lx1 * mdot * (v_r>0);
                user_out_var(14,k,j,i) = lx2 * mdot * (v_r>0);
                user_out_var(15,k,j,i) = lx3 * mdot * (v_r>0);


                user_out_var(16,k,j,i) = lx1 * mdot * (v_r<0);
                user_out_var(17,k,j,i) = lx2 * mdot * (v_r<0);
                user_out_var(18,k,j,i) = lx3 * mdot * (v_r<0);
                
                user_out_var(19,k,j,i) = lx1/std::sqrt(SQR(y) + SQR(z));
                user_out_var(20,k,j,i) = lx2/std::sqrt(SQR(x) + SQR(z));
                user_out_var(21,k,j,i) = lx3/std::sqrt(SQR(x) + SQR(y));
                
                user_out_var(22,k,j,i) = kappa * mdot;
                user_out_var(23,k,j,i) = phydro->w(IDN,k,j,i) * kappa ;
                
                Real kbT_kev = mu_highT*mp_over_kev*(phydro->w(IPR,k,j,i)/phydro->w(IDN,k,j,i));
                Real Lambda =Lambda_T(kbT_kev);
                
                user_out_var(24,k,j,i) = SQR(phydro->w(IDN,k,j,i))/(muH_sun * mue) * Lambda/UnitLambda_times_mp_times_kev/mp_over_kev;
                user_out_var(25,k,j,i) = (kbT_kev) * (muH_sun * mue) / ( gm1 *phydro->w(IDN,k,j,i)  * mu_highT * Lambda/UnitLambda_times_mp_times_kev );
                
                
                if (MAGNETIC_FIELDS_ENABLED){
                    Real bsq = SQR(pfield->bcc(IB1,k,j,i)) + SQR(pfield->bcc(IB2,k,j,i)) + SQR(pfield->bcc(IB3,k,j,i));
                    user_out_var(26,k,j,i) = pfield->bcc(IB1,k,j,i);
                    user_out_var(27,k,j,i) = pfield->bcc(IB2,k,j,i);
                    user_out_var(28,k,j,i) = pfield->bcc(IB3,k,j,i);
   
                    user_out_var(29,k,j,i) = bsq;

                    Real Br = pfield->bcc(IB1,k,j,i) * x/r + pfield->bcc(IB2,k,j,i) * y/r * is_2D + pfield->bcc(IB3,k,j,i) * z/r * is_3D;
                    user_out_var(30,k,j,i) = Br;
                    user_out_var(31,k,j,i) = pfield->bcc(IB1,k,j,i) * (-y) / std::sqrt( SQR(x) + SQR(y) )  +  pfield->bcc(IB2,k,j,i) * (x) / std::sqrt( SQR(x) + SQR(y) )  ;

                }

                
            }
        }
    }
}

/* 
* -------------------------------------------------------------------
*     The actual problem / initial condition setup file
* -------------------------------------------------------------------
*/
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  int i=0,j=0,k=0;

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
  Real pressure,b0,da,pa,ua,va,wa,bxa,bya,bza,x1,x2;
  Real T_dt,T_dmin,T_dmax;

  int is_zoom_in = 0;

    
    if (pin->GetOrAddString("problem","zoom_in","no") == "yes") {
        is_zoom_in = 1;
    }
  



  Real gm1 = peos->GetGamma() - 1.0;
  /* Set up a uniform medium */
  /* For now, make the medium almost totally empty */
  da = 1.0e-8;
  pa = 1.0e-10;
  ua = 0.0;
  va = 0.0;
  wa = 0.0;
  bxa = 0; //1e-4;
  bya = 0; //1e-4;
  bza = 0.0;
  Real x,y,z;

  for (k=kl; k<=ku; k++) {
  for (j=jl; j<=ju; j++) {
  for (i=il; i<=iu; i++) {

    if (is_zoom_in==1){
        
        // da = phydro->w_bound(IDN,k,j,i);
        // ua = phydro->w_bound(IVX,k,j,i);
        // va = phydro->w_bound(IVY,k,j,i);
        // wa = phydro->w_bound(IVZ,k,j,i);
        // pa = phydro->w_bound(IPR,k,j,i);

      // phydro->w(IDN,k,j,i) = da;
      // phydro->w(IVX,k,j,i) = ua;
      // phydro->w(IVY,k,j,i) = va;
      // phydro->w(IVZ,k,j,i) = wa;
      // phydro->w(IPR,k,j,i) = pa;

        
      }

    phydro->u(IDN,k,j,i) = da;
    phydro->u(IM1,k,j,i) = da*ua;
    phydro->u(IM2,k,j,i) = da*va;
    phydro->u(IM3,k,j,i) = da*wa;

    for (int i_user=0;i_user<N_user_vars; i_user ++){
      user_out_var(i_user,k,j,i) = 0;
    }

if (MAGNETIC_FIELDS_ENABLED){
    pfield->b.x1f(k,j,i) = bxa;
    pfield->b.x2f(k,j,i) = bya;
    pfield->bcc(IB1,k,j,i) = bxa;
    pfield->bcc(IB2,k,j,i) = bya;
    pfield->bcc(IB3,k,j,i) = bza;
    if (i == ie) pfield->b.x1f(k,j,i+1) = bxa;
    if (j == je) pfield->b.x2f(k,j+1,i) = bya;


    if (COORDINATE_SYSTEM == "spherical_polar"){

      Real theta = pcoord->x2v(j);
      Real phi = pcoord->x3v(k);

      Real bra  = bxa * std::sin(theta)*std::cos(phi) + bya * std::sin(theta)*std::cos(phi);
      Real btha = bxa * std::cos(theta)*std::cos(phi) + bya * std::cos(theta)*std::cos(phi);
      Real bphia = bxa * std::sin(phi)*-1 + bya * std::cos(phi);

      pfield->b.x1f(k,j,i) = bra;
      pfield->b.x2f(k,j,i) = btha;
      pfield->bcc(IB1,k,j,i) = bra;
      pfield->bcc(IB2,k,j,i) = btha;
      pfield->bcc(IB3,k,j,i) = bphia;
      if (i == ie) pfield->b.x1f(k,j,i+1) = bra;
      if (j == je) pfield->b.x2f(k,j+1,i) = btha;
    }
}

    pressure = pa;
#ifndef ISOTHERMAL
    phydro->u(IEN,k,j,i) = pressure/gm1;
if (MAGNETIC_FIELDS_ENABLED){
      phydro->u(IEN,k,j,i) +=0.5*(bxa*bxa + bya*bya + bza*bza);
}
     phydro->u(IEN,k,j,i) += 0.5*da*(ua*ua + va*va + wa*wa);
#endif /* ISOTHERMAL */

      if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
        phydro->u(IEN,k,j,i) += da;
 
  }}}
    



  UserWorkInLoop();


  

}



