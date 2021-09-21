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


/* cooling */
/* -------------------------------------------------------------------------- */
static int cooling;
static void integrate_cool(MeshBlock *pmb, const AthenaArray<Real> &prim_old, AthenaArray<Real> &cons, const Real dt_hydro );
static Real Lambda_T(const Real T);
static Real Yinv(Real Y1);
static Real Y(const Real T);
static Real tcool(const Real d, const Real T);


Real r_inner_boundary = 0.; // remove mass inside this radius
Real density_floor_, pressure_floor_; //floors

Real mp_over_kev = 9.994827;   //mp * (pc/kyr)^2/kev
Real UnitDensity = 6.767991e-23; // solar mass pc^-3
Real UnitEnergyDensity = 6.479592e-7; //solar mass /(pc ky^2)
Real UnitTime = 3.154e10;  //kyr
Real Unitlength = 3.086e+18; //parsec
Real UnitLambda_times_mp_times_kev = 1.255436328493696e-21 ;//
Real X = 0.7;   // Hydrogen Fraction
Real Z = 0.02;  //Metalicity
Real muH = 1./X;
Real mu_highT = 1./(2.*X + 3.*(1.-X-Z)/4. + Z/2.);  //mean molecular weight in proton masses

 void cons_force(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);
  void bondi_interp(const Real r, Real *rho, Real *v, Real *a);
  void get_Bondi(const Real r, Real *rho, Real *v, Real *a );



/* A structure defining the properties of each of the source 'stars' */
typedef struct Stars_s{
  Real M;			/* mass of the star */
  Real Mdot;		/* mass loss rate from solar wind (in 10^-5 M_solar/yr) */
  Real Vwind;		/* speed of solar wind (in km/s) */
  Real x1;			/* position in X,Y,Z (in arcseconds) */
  Real x2;
  Real x3;
  int i;			/* i,j,k of x,y,z cell the star is located in */
  int j;
  int k;
  Real v1;			/* velocity in X,Y,Z */
  Real v2;
  Real v3;
  Real alpha;   	/* euler angles for ZXZ rotation*/
  Real beta;
  Real gamma;
  Real tau;
  Real mean_angular_motion;
  Real eccentricity;
  Real rotation_matrix[3][3];
  Real period;
  Real radius;  /* effective radius of star */
  Real volume;   /* effective volume of star */
  RegionSize block_size;   /* block size of the mesh block in which the star is located */
  AthenaArray<Real> d_dt_source;
  AthenaArray<Real> E_dt_source;
  AthenaArray<Real> P_dt_source;
}Stars;


/* Initialize a couple of the key variables used throughout */
Stars star[100];					/* The stars structure used throughout */
int nstars;							/* Number of stars in the simulation */
static Real G = 4.48e-9;			/* Gravitational constant G (in problem units) */
Real dtau;							/* Spacing in t/period for each star orbit in tabulation */
Real gm_; 							/* G*M for point mass at origin */
Real gm1; 							/* \gamma-1 (adiabatic index) */
Real N_cells_per_radius;  /* Number of cells that are contained in one stellar radius (this is defined in terms of the longest length 
across a cell ) */
Real *rho_bondi,*v_bondi,*a_bondi;
Real r_min_tab,r_max_tab;
int N_bondi;
Real r_a = 0.2; /* = gm_/a_inf**2. */
Real rho_inf,a_inf,qs; /* bondi solution params */
Real r_inner_boundary_bondi = 0.;
Real r_outer_boundary_bondi;
double SMALL = 1e-20;				/* Small number for numerical purposes */
LogicalLocation *loclist;              /* List of logical locations of meshblocks */

Real yr = 31556926.0, pc = 3.09e18;    /* yr and parsec in code units */
Real cl = 2.99792458e10 * 1e3*yr/pc ;      /* speed of light in code units */
Real kbTfloor_kev = 1e-20; 

Real r_circ = 1e-2;
Real l_bondi; 

/* global definitions for the cooling curve using the
   Townsend (2009) exact integration scheme */
#define nfit_cool 11

static const Real kbT_cc[nfit_cool] = {
  8.61733130e-06,   8.00000000e-04,   1.50000000e-03,
  2.50000000e-03,   7.50000000e-03,   2.00000000e-02,
  3.10000000e-02,   1.25000000e-01,   3.00000000e-01,
  2.26000000e+00,   1.00000000e+02};

static const Real Lam_cc[nfit_cool] = {
  1.24666909e-27,   3.99910139e-26,   1.47470970e-22,
  1.09120314e-22,   4.92195285e-22,   5.38853593e-22,
  2.32144473e-22,   1.38278507e-22,   3.66863203e-23,
  2.15641313e-23,   9.73848346e-23};

static const Real exp_cc[nfit_cool] = {
   0.76546122,  13.06493514,  -0.58959508,   1.37120661,
   0.09233853,  -1.92144798,  -0.37157016,  -1.51560627,
  -0.26314206,   0.39781441,   0.39781441};

static Real Yk[nfit_cool];
/* -- end piecewise power-law fit */


/* must call init_cooling() in both problem() and read_restart() */
static void init_cooling();
static void test_cooling();
static Real newtemp_townsend(const Real d, const Real T, const Real dt_hydro);



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
      term *= ((1.0 - pow(kbT_cc[k]/kbT_cc[k+1], exp_cc[k]-1.0)) / (1.0-exp_cc[k]));

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
  for(k=n; k>=0; k--){
    if (T >= kbT_cc[k])
      break;
  }

  /* piecewise power-law; see equation A4 of Townsend (2009) */
  /* (factor of 1.311e-5 takes lambda from units of 1e-23 erg cm^3 /s
     to code units.) */
  return (Lam_cc[k] * pow(T/kbT_cc[k], exp_cc[k]));
}

/* see Lambda_T() or equation A1 of Townsend (2009) for the
   definition */
static Real Y(const Real T)
{
  int k, n=nfit_cool-1;
  Real term;

  /* first find the temperature bin */
  for(k=n; k>=0; k--){
    if (T >= kbT_cc[k])
      break;
  }

  /* calculate Y using equation A5 in Townsend (2009) */
  term = (Lam_cc[n]/Lam_cc[k]) * (kbT_cc[k]/kbT_cc[n]);

  if (exp_cc[k] == 1.0)
    term *= log(kbT_cc[k]/T);
  else
    term *= ((1.0 - pow(kbT_cc[k]/T, exp_cc[k]-1.0)) / (1.0-exp_cc[k]));

  return (Yk[k] + term);
}

static Real Yinv(const Real Y1)
{
  int k, n=nfit_cool-1;
  Real term;

  /* find the bin i in which the final temperature will be */
  for(k=n; k>=0; k--){
    if (Y(kbT_cc[k]) >= Y1)
      break;
  }


  /* calculate Yinv using equation A7 in Townsend (2009) */
  term = (Lam_cc[k]/Lam_cc[n]) * (kbT_cc[n]/kbT_cc[k]);
  term *= (Y1 - Yk[k]);

  if (exp_cc[k] == 1.0)
    term = exp(-1.0*term);
  else{
    term = pow(1.0 - (1.0-exp_cc[k])*term,
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
  return  (T) * (muH * muH) / ( gm1 * d * mu_highT * Lambda_T(T)/UnitLambda_times_mp_times_kev );
}


static void integrate_cool(MeshBlock *pmb, const AthenaArray<Real> &prim_old, AthenaArray<Real> &cons, const Real dt_hydro )
{
  int i, j, k;
  int is, ie, js, je, ks, ke;

  Real kbT_keV;
  AthenaArray<Real> prim;
  prim.InitWithShallowCopy(pmb->phydro->w);

  /* ath_pout(0, "integrating cooling using Townsend (2009) algorithm.\n"); */

  is = pmb->is;  ie = pmb->ie;
  js = pmb->js;  je = pmb->je;
  ks = pmb->ks;  ke = pmb->ke;
  Real igm1 = 1.0/(gm1);
  Real gamma = gm1+1.;

  pmb->peos->ConservedToPrimitive(cons, prim_old, pmb->pfield->b, prim, pmb->pfield->bcc,
           pmb->pcoord, pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);

  for (k=ks; k<=ke; k++) {
    for (j=js; j<=je; j++) {
      for (i=is; i<=ie; i++) {

        /* find temp in keV */
        kbT_keV = mu_highT*mp_over_kev*(prim(IPR,k,j,i)/prim(IDN,k,j,i)); ;
        // ath_pout(0, "temperature before = %e ", kbT_keV);
        kbT_keV = newtemp_townsend(prim(IDN,k,j,i), kbT_keV, dt_hydro);
        // ath_pout(0, "temperature after = %e \n", kbT_keV);
        /* apply a temperature floor (nans tolerated) */
        if (isnan(kbT_keV) || kbT_keV < kbTfloor_kev)
          kbT_keV = kbTfloor_kev;

        prim(IPR,k,j,i) = prim(IDN,k,j,i) * kbT_keV / (mu_highT * mp_over_kev);

        Real v_s = sqrt(gamma*prim(IPR,k,j,i)/prim(IDN,k,j,i));

		if (v_s>cl) v_s = cl;

		prim(IPR,k,j,i) = SQR(v_s) *prim(IDN,k,j,i)/gamma ;
        cons(IEN,k,j,i) = prim(IPR,k,j,i)*igm1 + 0.5*prim(IDN,k,j,i)*( SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)) );
      }
    }
  }
  return;
}






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
	}



}
/*
Solve Kepler's equation for a given star in the plane of the orbit and then rotate
to the lab frame
*/
void update_star(Stars *star, int i_star, const Real t)
{

	Real mean_anomaly = star[i_star].mean_angular_motion * (t - star[i_star].tau);
	Real a = pow(gm_/SQR(star[i_star].mean_angular_motion),1./3.);    //mean_angular_motion = np.sqrt(mu/(a*a*a));
	Real b = a * sqrt(1. - SQR(star[i_star].eccentricity) );

	mean_anomaly = fmod(mean_anomaly, 2*PI);
    if (mean_anomaly >  PI) mean_anomaly = mean_anomaly- 2.0*PI;
    if (mean_anomaly < -PI) mean_anomaly = mean_anomaly + 2.0*PI;

    //Construct the initial guess.
    Real sgn = 1.0;
    if (std::sin(mean_anomaly) < 0.0) sgn = -1.0;
    Real E = mean_anomaly + sgn*(0.85)*star[i_star].eccentricity;

    //Solve kepler's equation iteratively to improve the solution E.
    Real error = 1.0;
    Real max_error = 1e-5;
    int i_max = 100;
    int i;
    for(i = 0; i < i_max; i++){
      Real es = star[i_star].eccentricity*std::sin(E);
      Real ec = star[i_star].eccentricity*std::cos(E);
      Real f = E - es - mean_anomaly;
      error = fabs(f);
      if (error < max_error) break;
      Real df = 1.0 - ec;
      Real ddf = es;
      Real dddf = ec;
      Real d1 = -f/df;
      Real d2 = -f/(df + d1*ddf/2.0);
      Real d3 = -f/(df + d2*ddf/2.0 + d2*d2*dddf/6.0);
      E = E + d3;
    }

     //Warn if solution did not converge.
     if (error > max_error)
       std::cout << "***Warning*** Orbit::keplers_eqn() failed to converge***\n";

    Real x1_prime= a * (std::cos(E) - star[i_star].eccentricity) ;
	Real x2_prime= b * std::sin(E) ;
    
    /* Time Derivative of E */
    Real Edot = star[i_star].mean_angular_motion/ (1.-star[i_star].eccentricity * std::cos(E));
    
    Real v1_prime = - a * std::sin(E) * Edot;
    Real v2_prime =   b * std::cos(E) * Edot;

    Real x1,x2,x3;

    rotate_orbit(star,i_star, x1_prime, x2_prime,&x1,&x2,&x3 );
    
    star[i_star].x1 = x1;
    star[i_star].x2 = x2;
    star[i_star].x3 = x3;
    
    Real v1,v2,v3;
    rotate_orbit(star,i_star,v1_prime,v2_prime,&v1, &v2, &v3);
    
    
    star[i_star].v1 = v1;
    star[i_star].v2 = v2;
    star[i_star].v3 = v3;


	
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

	
	FILE *input_file;
    if ((input_file = fopen(starfile.c_str(), "r")) == NULL)   
           fprintf(stderr, "Cannot open %s, %s\n", "input_file",starfile.c_str());

	fscanf(input_file, "%i \n", &n_stars);
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
	 

     pre_compute_rotation_matrix(star, j);

	 update_star(star,j, 0.);

	 if (fabs(1.-star[j].x1/x1)>1e-4 || fabs(1.-star[j].x2/x2)>1e-4 || fabs(1.-star[j].x3/x3)>1e-4 || fabs(1.-star[j].v1/v1)>1e-4 || fabs(1.-star[j].v2/v2)>1e-4 || fabs(1.-star[j].v3/v3)>1e-4){
	 	fprintf(stderr,"Error in solving Kepler\n x1: %g x2: %g x3: %g \n x1_orbit: %g x2_orbit %g x3_orbit %g \n v1: %g v2: %g v3: %g \n v1_orbit: %g v2_orbit %g v3_orbit %g ecc: %g mean_angular_motion: %g tau: %g \n alpha: %g beta: %g gamma: %g \n",
	 		x1,x2,x3,star[j].x1,star[j].x2,star[j].x3,v1,v2,v3,star[j].v1,star[j].v2,star[j].v3,star[j].eccentricity,star[j].mean_angular_motion,star[j].tau,star[j].alpha,star[j].beta,star[j].gamma);

	 	exit(0);
	 }
        



	}
    fclose(input_file);
	return;
}


/* 
* -------------------------------------------------------------------
*  Another basic function, this one converts the initial units 
* 	in the Stars struct to those of the problem:
*		t_0 = 10^3 yr = 10^3 * (31556926 s)
*		d_0 = pc = 3.09 * 10^13 km
*		M_0 = M_solar = 1.99 * 10^33 g
*
*	These units also give:
*	  G = 6.67 * 10^-11 m^3 / (kg*s^2)
*	    = 4.48 * 10^-9 d_0^3 / (M_0 * t_0^2)
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
*	by the overlay of a star effective radius mask. There is an analytical 
*	2-D solution, but it doesn't extend to 3-D and is almost as 
*	computationaly demanding 
* --------------------------------------------------------------------------
*/

Real starmask(const Real r, const Real r_eff){
	return 1.;
}

Real mask_volume_func(const Real r_eff, const int ndim){
	if (ndim ==2){
		return PI * SQR(r_eff);
	}
	else if (ndim ==3){
		return 4.0/3.0 * PI * pow(r_eff,3.);
	}
	else{
		fprintf(stderr,"Error, %d dimensions not supported for mask_volume\n",ndim);
	}


}

float mask(const Coordinates *pG, const int i, const int j, const int k,
				const int i_star, const int n_depth, Real *vx_avg, Real *vy_avg, Real *vz_avg, Real *vsq_avg)
{
	int i_x1, i_x2, i_x3, n_points_total;
	Real n_in_overlay,dx1,dx2,dx3,r,di_x1,di_x2,di_x3;
	Real dV,V;
	Real v_wind_x,v_wind_y,v_wind_z;
	Real fac;

	*vx_avg = 0.;
	*vy_avg = 0.;
	*vz_avg = 0.;
	*vsq_avg = 0.;

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

			dV = get_dV(pG,x1,x2,x3,di_x1, di_x2, di_x3);


			/* Account for endpoints of integral */
			fac = 1.;

			if (i_x1 ==0 || i_x1 == n_depth- 1) fac = fac * 0.5;
      if (i_x2 ==0 || i_x2 == n_depth -1) fac = fac * 0.5;
			if ( (i_x3 ==0 || i_x3 == n_depth-1) && (is_3D) ) fac = fac*0.5;

			dV = dV * fac;

			if (r <= star[i_star].radius) {
				n_in_overlay += starmask(r,star[i_star].radius) * dV;
				*vx_avg += v_wind_x * starmask(r,star[i_star].radius) * dV;
				*vy_avg += v_wind_y * starmask(r,star[i_star].radius) * dV;
				*vz_avg += v_wind_z * starmask(r,star[i_star].radius) * dV;
				*vsq_avg += ( SQR(v_wind_x) + SQR(v_wind_y) + SQR(v_wind_z)*is_3D ) * starmask(r,star[i_star].radius) * dV;
			}
			V += dV;
			}}}

	*vx_avg = *vx_avg/(n_in_overlay +SMALL);
	*vy_avg = *vy_avg/(n_in_overlay +SMALL);
	*vz_avg = *vz_avg/(n_in_overlay +SMALL);
	*vsq_avg = *vsq_avg/(n_in_overlay +SMALL);

	return (n_in_overlay/V);
}


/*
Give box spacing, calculate cell r and V
*/
void get_star_size(const Coordinates *pcoord,const Real DX, const Real DY, const Real DZ, Real *r_eff, Real *mask_volume){

	if (pcoord->pmy_block->block_size.nx3 > 1){
        *r_eff = N_cells_per_radius * sqrt(SQR(DX) + SQR(DY) + SQR(DZ));
        *mask_volume = mask_volume_func(*r_eff,3);
    }
    else{
        *r_eff = N_cells_per_radius * sqrt(SQR(DX) + SQR(DY));
        *mask_volume = mask_volume_func(*r_eff,2);
    }

}



// float cos_mask(float r) {return (cos((PI/2.0)*(r/r_eff)));}
// void set_cos_mask_2d(float (**mask)(float)) {
// 	mask_volume = 4.0*r_eff*r_eff*(1.0 - 2.0/PI); *mask = cos_mask; }
// void set_cos_mask_3d(float (**mask)(float)) {
// 	mask_volume = 8.0*r_eff*r_eff*r_eff*(1.0 - 8.0/PI/PI); *mask = cos_mask; }

// float box_mask(float r) {return (1.0);}
// void set_box_mask_2d(float (**mask)(float)) {
// 	mask_volume = PI*r_eff*r_eff; *mask = box_mask; }
// void set_box_mask_3d(float (**mask)(float)) {
// 	mask_volume = (4.0/3.0)*PI*r_eff*r_eff*r_eff; *mask = box_mask; }


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

	Real DX,DY,DZ,r_eff,mask_volume;


	/* First Check if the star is even in the mesh.  If not, give it the coarse grid radius + volume */
	if (!is_in_block(pcoord->pmy_block->pmy_mesh->mesh_size,star[i_star].x1,star[i_star].x2,star[i_star].x3) ){
				block_size = pcoord->pmy_block->pmy_mesh->mesh_size;
					get_uniform_box_spacing(block_size,&DX,&DY,&DZ);
					get_star_size(pcoord,DX,DY,DZ,&r_eff,&mask_volume);

				    star[i_star].radius = r_eff;
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
		get_star_size(pcoord,DX,DY,DZ,&r_eff,&mask_volume);

    star[i_star].radius = r_eff;
    star[i_star].volume = mask_volume;
		return;
	
}


/* Get cell lengths in highest level of refinement */
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

/*M Make sure i,j,k are in the domain */
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
/* Get the cell location of a position in a given meshblock */
void pos_to_ijk(Coordinates *pcoord, const Real x,const Real y, const Real z, int *i, int *j, int *k){

  Real dx = pcoord->dx1f(0);
  Real dy = pcoord->dx2f(0);
  Real dz = pcoord->dx3f(0);

  *i = int ( (x - pcoord->pmy_block->block_size.x1min)/dx - 0.5 +1000 ) -1000;
  *j = int ( (y - pcoord->pmy_block->block_size.x2min)/dy - 0.5 +1000 ) -1000;
  *k = int ( (z - pcoord->pmy_block->block_size.x3min)/dz - 0.5 +1000 ) -1000;

  if (*i<0) *i = 0;
  if (*j<0) *j = 0;
  if (*k<0) *k = 0;

  if (*i>=pcoord->pmy_block->block_size.nx1) *i = pcoord->pmy_block->block_size.nx1 -1;
  if (*j>=pcoord->pmy_block->block_size.nx2) *j = pcoord->pmy_block->block_size.nx2 -1;
  if (*k>=pcoord->pmy_block->block_size.nx3) *k = pcoord->pmy_block->block_size.nx3 -1;

  return; 

}

/* Apply inner "inflow" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &cons){

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
          int di,dj,dk,sgn;
          Real r_tmp;
          
          is_3D = (pmb->block_size.nx3>1);
          is_2D = (pmb->block_size.nx2>1);

          r = sqrt( SQR(x) + SQR(y)*is_2D + SQR(z)*is_3D);

          if (r < r_inner_boundary){
              
              r_hat_x = x/r;
              r_hat_y = y/r;
              r_hat_z = z/r;
              r1 = r;
              ir = 1;

              di = 0.;
              dj = 0.;
              dk = 0.;
//              while (r1 <r_inner_boundary){
//                sgn = (x>0) - (x<0);
//                if (sgn==0){
//                  fprintf(stderr,"Boundary condition not specified for x= 0 \n");
//                }
//                di = di + sgn*1;
//                r1 = sqrt( SQR(pmb->pcoord->x1v(i+di)) + SQR(pmb->pcoord->x2v(j))*is_2D + SQR(pmb->pcoord->x3v(k))*is_3D );
//              }
//              r1 = r;
//              if (is_2D) {
//                while (r1 < r_inner_boundary){
//                sgn = (y>0) - (y<0);
//                if (sgn==0){
//                  fprintf(stderr,"Boundary condition not specified for y= 0 \n");
//                }
//                dj = dj + sgn*1;
//                r1 = sqrt( SQR(pmb->pcoord->x1v(i)) + SQR(pmb->pcoord->x2v(j+dj))*is_2D + SQR(pmb->pcoord->x3v(k))*is_3D );
//              }
//            }
//            r1 = r;
//            if (is_3D) {
//                while (r1 < r_inner_boundary){
//                sgn = (z>0) - (z<0);
//                if (sgn==0){
//                  fprintf(stderr,"Boundary condition not specified for z= 0 \n");
//                }
//                dk = dk + sgn*1;
//                r1 = sqrt( SQR(pmb->pcoord->x1v(i)) + SQR(pmb->pcoord->x2v(j))*is_2D + SQR(pmb->pcoord->x3v(k+dk))*is_3D );
//              }
//            }
               while (r1 <r_inner_boundary) {
                  
                   i1 = int (i + ir*x/r + 0.5);   //Note: Addition of 0.5 ensures that "int" rounds up
                   j1 = int (j + (ir*y/r + 0.5)*is_2D);
                   k1 = int (k + (ir*z/r + 0.5)*is_3D);
                  
                   bound_ijk(pmb->pcoord,&i1,&j1,&k1);
                  
                   r1 = sqrt( SQR(pmb->pcoord->x1v(i1)) + SQR(pmb->pcoord->x2v(j1))*is_2D + SQR(pmb->pcoord->x3v(k1))*is_3D );
                   ir = ir + 1;
               }

              
              //if ((fabs(pmb->block_size.x1max/0.0625-1.) <1e-3) && (fabs(pmb->block_size.x2max/0.0625-1.) <1e-3) )fprintf(stderr,"r: %g r_inner: %g r1: %g ijk: %d %d %d i1j1k1: %d %d %d \n mesh_block: x1min: %g x1max: %g x2min: %g x2max; %g  \n", r, r_inner_boundary, r1,i,j,k,i1,j1,k1,pmb->block_size.x1min,pmb->block_size.x1max,pmb->block_size.x2min,pmb->block_size.x2max );

              
              // i2 = int (i + ir*x/r + 0.5);   //Note: Addition of 0.5 ensures that "int" rounds up
              // j2 = int (j + (ir*y/r + 0.5)*is_2D);
              // k2 = int (k + (ir*z/r + 0.5)*is_3D);
              
              // bound_ijk(pmb->pcoord,&i2,&j2,&k2);
              // r2 = sqrt( SQR(pmb->pcoord->x1v(i2)) + SQR(pmb->pcoord->x2v(j2))*is_2D + SQR(pmb->pcoord->x3v(k2))*is_3D );
              
//              Real faci,facj,fack,norm;
//
//              if (is_3D){
//                norm = fabs(di*dj) + fabs(dj*dk) + fabs(dk*di);
//                faci = fabs(dj*dk) / norm;
//                facj = fabs(di*dk) / norm;
//                fack = fabs(di*dj) / norm;
//
//              }
//              else if (is_2D){
//                norm = fabs(di)+fabs(dj);
//                faci = fabs(dj) / norm;
//                facj = fabs(di) / norm;
//                fack = 0.;
//              }
//              else {
//                faci = 1;
//                facj = 0.;
//                fack = 0.;
//              }


              //fprintf(stderr,"r: %g r_inner: %g ijk: %d %d %d didjdk: %d %d %d fac: %g %g %g \n mesh_block: x1min: %g x1max: %g x2min: %g x2max; %g  \n", r, r_inner_boundary,
               // i,j,k,di,dj,dk,faci,facj,fack,pmb->block_size.x1min,pmb->block_size.x1max,pmb->block_size.x2min,pmb->block_size.x2max );

             // for (int n=0; n<(NHYDRO); ++n) {
                 
                 
             //     //dU_dr =(cons(n,k2,j2,i2) - cons(n,k1,j1,i1)) /  (r2-r1 + SMALL);
                 
             //     cons(n,k,j,i) = cons(n,k1,j1,i1); //+ dU_dr * (r-r1);

             //     //cons(n,k,j,i) = (cons(n,k+dk,j,i)*fack + is_2D*cons(n,k,j+dj,i)*facj + cons(n,k,j,i+di)*faci);
                 
             // }
              
              Real rho_flr = 1e-8;
              Real p_floor = 1e-10;
              cons(IDN,k,j,i) = rho_flr;
              cons(IM1,k,j,i) = 0.;
              cons(IM2,k,j,i) = 0.;
              cons(IM3,k,j,i) = 0.;
              cons(IEN,k,j,i) = p_floor/gm1;
              /* Prevent outflow from inner boundary */ 
              if (cons(IM1,k,j,i)*x/r >0 ) cons(IM1,k,j,i) = 0.;
              if (cons(IM2,k,j,i)*y/r >0 ) cons(IM2,k,j,i) = 0.;
              if (cons(IM3,k,j,i)*z/r >0 ) cons(IM3,k,j,i) = 0.;
              
              
          }



}}}



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

		if (v_s>cl) v_s = cl;

		prim(IPR,k,j,i) = SQR(v_s) *prim(IDN,k,j,i)/gamma ;



	}}}


	pmb->peos->PrimitiveToConserved(prim, pmb->pfield->bcc,
       cons, pmb->pcoord,
       pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);

}


/* Apply inner "inflow" boundary conditions */

void apply_bondi_boundary(MeshBlock *pmb,AthenaArray<Real> &cons){

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
          Real rho,v,a;
          Real m_r;
          
          is_3D = (pmb->block_size.nx3>1);
          is_2D = (pmb->block_size.nx2>1);

          r = sqrt( SQR(x) + SQR(y)*is_2D + SQR(z)*is_3D);

          if (r <= r_inner_boundary_bondi || r>= r_outer_boundary_bondi){

              
              //bondi_interp(r,&rho,&v,&a);
              get_Bondi(r,&rho,&v,&a);

              Real v_init_x =  cons(IM1,k,j,i)/ cons(IDN,k,j,i);
              Real v_init_y =  cons(IM2,k,j,i)/ cons(IDN,k,j,i);
              Real v_init_z =  cons(IM3,k,j,i)/ cons(IDN,k,j,i);

              cons(IDN,k,j,i) = rho ;


              // cons(IM1,k,j,i) = rho * v_init_x;
              // cons(IM2,k,j,i) = rho * v_init_y;
              // cons(IM3,k,j,i) = rho * v_init_z;

              if (COORDINATE_SYSTEM == "cartesian"){
                  cons(IM1,k,j,i) = rho * v*  x/r;
                  cons(IM2,k,j,i) = rho * v*  y/r;
                  if (is_3D) cons(IM3,k,j,i) = rho * v*  z/r;

              }
              else if (COORDINATE_SYSTEM == "cylindrical"){
                  Real v_phi = l_bondi/std::sqrt(SQR(x) + SQR(y) + 1e-20);
                  cons(IM1,k,j,i) = rho * v;
                  cons(IM2,k,j,i) = rho * v_phi;
                  v = std::sqrt(SQR(v) + SQR(v_phi));
              }
              else{
                Real v_phi = l_bondi/std::sqrt(SQR(x) + SQR(y) + 1e-20);
                cons(IM1,k,j,i) = rho * v;
                cons(IM3,k,j,i) = rho * v_phi;
                v = std::sqrt(SQR(v) + SQR(v_phi));
              }

              
              Real press = SQR(a)/(gm1+1.) * rho  ;
              cons(IEN,k,j,i) = 0.5 * rho * SQR(v) + press/gm1;
             // cons(IEN,k,j,i) = 0.5 * rho * (SQR(v_init_x) + SQR(v_init_y) + SQR(v_init_z)*is_3D ) + press/gm1;



              
          }
          



}}}



}

/* 
* -------------------------------------------------------------------
*     The constant source terms for stars in a location i,j,k 
*      (This also includes the calls to the N-body routines
*         that evolve the star velocities and positions)
* -------------------------------------------------------------------
*/


void cons_force(MeshBlock *pmb,const Real time, const Real dt,const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons)
{

	/* Update the stellar positions/velocities based on Kepler Orbits. */



//    for (int k=pmb->ks; k<=pmb->ke; ++k) {
// #pragma omp parallel for schedule(static)
//     for (int j=pmb->js; j<=pmb->je; ++j) {
// #pragma simd
//       for (int i=pmb->is; i<=pmb->ie; ++i) {


//       Real q; 

// 		  Real d0,r;
// 		  Real vwind,mdot;
//           Real rin, rout,A;
// 		  Real x,y,z,v,cs;
//           int eta;

//           eta = 3;
//           rin = 0.080479124;
//           rout = 0.40239562;
//           mdot = 1.;
//           vwind = 1000. * 0.001022;
//           v = sqrt( SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)) );
//           cs = sqrt(prim(IPR,k,j,i)/prim(IDN,k,j,i));
          
          

//       get_cartesian_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &x, &y, &z);
//       if (pmb->block_size.nx3 >1) {
//         r = sqrt(SQR(x) + SQR(y) + SQR(z));
//         v = sqrt( SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)) );
//         if (eta == 3) A = mdot/(4.*PI) * 1./(log(rout/rin));
//         else A = mdot * (3. - eta)/(4. * PI) * 1. / (pow(rout,3.-eta) - pow(rin,3.-eta));
//       }
//       else if (pmb->block_size.nx2 >1) {
//         eta = 2;
//         mdot = mdot /(pmb->pcoord->dx3f(0));
//         r = sqrt(SQR(x) + SQR(y));
//         v = sqrt( SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)));
//         if (eta == 2) A = mdot/(2.*PI) * 1./(log(rout/rin));
//         else A = mdot * (2. - eta)/(2. * PI) * 1. / (pow(rout,2.-eta) - pow(rin,2.-eta)) ;
//       }
//       else{
//         eta = 1;
//         mdot = mdot/(pmb->pcoord->dx2f(0)*pmb->pcoord->dx3f(0))/2.;
//         r = sqrt(SQR(x));
//         v = sqrt( SQR(prim(IVX,k,j,i)));
//         if (eta == 1) A = mdot/(log(rout/rin));
//         else A = mdot * (1. - eta) / (pow(rout,1.-eta) - pow(rin,1.-eta)) ;
//       }
//           q = A * pow(r,-eta);
//           d0 = dt*q;


//           if ( (r>=rin) && (r<=rout) ){
//               cons(IDN,k,j,i) += d0;
//               cons(IM1,k,j,i) += -d0* prim(IVX,k,j,i) ;
//               cons(IM2,k,j,i) += -d0* prim(IVY,k,j,i) ;
//               cons(IM3,k,j,i) += -d0* prim(IVZ,k,j,i) ;
                  
//               cons(IEN,k,j,i) += 0.5 * d0 * ( SQR(vwind)); // + SQR(v) ); //0.5 * d0 * ( SQR(vwind) - SQR(v) - 2.*(gm1 +1.)/gm1 * SQR(cs) );
//           }

  

  		
// }}}


  apply_inner_boundary_condition(pmb,cons);
  apply_bondi_boundary(pmb,cons);


	//integrate_cool(pmb, prim, cons, dt );
	limit_temperature(pmb,cons,prim);

  return;
}



void tabulate_wind_source(MeshBlock *pmb ){

	int N_points_orbit = 200;
	dtau = 1./(1.*N_points_orbit-1.); //tau = t/Period for each star


	for (int i_star =0; i_star<nstars; i_star ++){
		star[i_star].d_dt_source.NewAthenaArray(N_points_orbit,pmb->block_size.nx3,pmb->block_size.nx2,pmb->block_size.nx1);
		star[i_star].E_dt_source.NewAthenaArray(N_points_orbit,pmb->block_size.nx3,pmb->block_size.nx2,pmb->block_size.nx1);
		star[i_star].P_dt_source.NewAthenaArray(3,N_points_orbit,pmb->block_size.nx3,pmb->block_size.nx2,pmb->block_size.nx1);
	}

   for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {


      	  Real d0_dt,dx1,dx2,dx3,r,a0,dr,frac_overlay;
      	  Real vx_avg,vy_avg,vz_avg,vsq_avg;
		  Real x,y,z;
		  int star_check, mask_depth;
		  mask_depth = 10;
		  dx1 = 0.0; dx2 = 0.0; dx3 = 0.0;
		  dr = 0.0;

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
			  	if (pmb->block_size.nx3 > 1) dr += SQR(pmb->pcoord->x1v(i) * std::sin(pmb->pcoord->x3v(k)) *pmb->pcoord->dx3f(k));
	        }

        	dr = sqrt(dr);

        	get_cartesian_coords(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j), pmb->pcoord->x3v(k), &x, &y, &z);

      	for (int it = 0; it<N_points_orbit; it ++){
      		star_check = 0;
      	for (int i_star =0; i_star<nstars; i_star ++){

      		Real tau = 0. + it * dtau;
      		Real t = tau * star[i_star].period;
      		update_star(star,i_star,t);

 
	  		



			if (pmb->block_size.nx1 > 1) dx1 = x - star[i_star].x1;
			if (pmb->block_size.nx2 > 1) dx2 = y - star[i_star].x2;
			if (pmb->block_size.nx3 > 1) dx3 = z - star[i_star].x3;

			r  = sqrt(dx1*dx1 + dx2*dx2 + dx3*dx3);
		

			/* Apply the star source conservative terms (density, energy) */
			/* First, check where cell is relative to star effective radius */
			if (r > star[i_star].radius + dr) {
				/* Cell entirely outside star mask - do nothing */

				star[i_star].d_dt_source(it,k,j,i) = 0.;
				star[i_star].E_dt_source(it,k,j,i) = 0. ;


				star[i_star].P_dt_source(0,it,k,j,i) = 0.;
				star[i_star].P_dt_source(1,it,k,j,i) = 0.;
				star[i_star].P_dt_source(2,it,k,j,i) = 0.;
			} else {
				/* Some fraction of the cell may contain part of the mask */
				star_check = 1;

				frac_overlay = mask(pmb->pcoord,i,j,k,i_star,mask_depth,&vx_avg,&vy_avg,&vz_avg,&vsq_avg);

				d0_dt= frac_overlay * (star[i_star].Mdot / star[i_star].volume) ;
				star[i_star].d_dt_source(it,k,j,i) = d0_dt;
				star[i_star].E_dt_source(it,k,j,i) = 0.5 * d0_dt *  ( SQR(star[i_star].Vwind) + SQR(star[i_star].v1) + SQR(star[i_star].v2) + SQR(star[i_star].v3) ) ;

				Real vx1,vx2,vx3;
				convert_cartesian_vector_to_code_coords(star[i_star].v1,star[i_star].v2,star[i_star].v3,star[i_star].x1,star[i_star].x2,star[i_star].x3, &vx1,&vx2,&vx3);


				star[i_star].P_dt_source(0,it,k,j,i) = d0_dt * vx1;
				star[i_star].P_dt_source(1,it,k,j,i) = d0_dt * vx2;
				star[i_star].P_dt_source(2,it,k,j,i) = d0_dt * vx3;

			}
  


      }}}}}


      /* reset v and x of star to initial time */
      for (int i_star =0; i_star<nstars; i_star ++) update_star(star,i_star,0);



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

static void *malloc_rank1(int n1, int size)
{
  void *A;

  if ((A = (void *) malloc(n1 * size)) == NULL) {
    fprintf(stderr, "malloc failure in malloc_rank1\n");
    exit(123);
  }

  return A;
}

 /* return scalar in cgs units */
// double interp_scalar(Coordinates *pcoord ,const AthenaArray<Real> &prim_old, int i, int j, int k, int i_new, int j_new, int k_new,
//   double del[4], AthenaArray<Real> &prim)
// {
//   double interp, interpk, interpkp1;
//   int kp1,ip1,jp1;


//   coeff[0] = (1. - del[1]) * (1. - del[2]);
//   coeff[1] = (1. - del[1]) * del[2];
//   coeff[2] = del[1] * (1. - del[2]);
//   coeff[3] = del[1] * del[2];
//   coeff[4] = del[3];


//   kp1 = k;
//   ip1 = i;
//   jp1 = j;

//   if (pcoord->pmy_block->block_size.nx1>1) ip1 = i+1;
//   if (pcoord->pmy_block->block_size.nx2>1) jp1 = j+1;
//   if (pcoord->pmy_block->block_size.nx3>1) kp1 + k+1;
    

//     for (int n=0; n<(NHYDRO); ++n) {
//         interpk =
//           prim_old(n,k,j,i)  * coeff[0] +
//           prim_old(n,k,jp1,i) * coeff[1] +
//           prim_old(n,k,j,ip1) * coeff[2] + prim_old(n,k,jp1,ip1) * coeff[3];

//         interpkp1 =
//           prim_old(n,kp1,j,i)   * coeff[0] +
//           prim_old(n,kp1,jp1,i) * coeff[1] +
//           prim_old(n,kp1,j,ip1) * coeff[2] + prim_old(n,kp1,jp1,ip1) * coeff[3];
      
//         interp = (1.-coeff[4]) * interpk + coeff[4] * interpkp1;

//         prim(n,k_new,j_new,i_new) = interp; 

//     }


//   /* use bilinear interpolation to find rho; piecewise constant
//      near the boundaries */

//   return interp ;

// }
void read_in_bondi(){

  FILE *frho,*fa,*fv;

  frho = fopen("rho_bondi.txt","r");
  if (frho == NULL) {
  fprintf(stderr, "trouble opening rho_bondi.txt\n");
  exit(0);
}

  fv = fopen("v_bondi.txt","r");
  if (fv == NULL) {
  fprintf(stderr, "trouble opening v_bondi.txt\n");
  exit(0);
}
  fa= fopen("a_bondi.txt","r");
  if (fa == NULL) {
  fprintf(stderr, "trouble opening a_bondi.txt\n");
  exit(0);
}


  fscanf(frho, "%d", &N_bondi        );
  fscanf(frho, "%lf", &r_min_tab        );
  fscanf(frho, "%lf", &r_max_tab        );
  fscanf(fv, "%d", &N_bondi        );
  fscanf(fv, "%lf", &r_min_tab        );
  fscanf(fv, "%lf", &r_max_tab        );
  fscanf(fa, "%d", &N_bondi        );
  fscanf(fa, "%lf", &r_min_tab        );
  fscanf(fa, "%lf", &r_max_tab        );

  rho_bondi = (Real *)malloc_rank1(N_bondi,sizeof(double));
  v_bondi =  (Real *)malloc_rank1(N_bondi,sizeof(double));
  a_bondi =  (Real *)malloc_rank1(N_bondi,sizeof(double));

  r_min_tab *= r_a;
  r_max_tab *= r_a;

  for (int i =0; i<N_bondi; i++){
      fscanf(frho, "%lf", &rho_bondi[i]        );
      rho_bondi[i] *= rho_inf;

  }

  for (int i =0; i<N_bondi; i++){
      fscanf(fv, "%lf", &v_bondi[i]        );
      v_bondi[i] *= a_inf;

  }

  for (int i =0; i<N_bondi; i++){
      fscanf(fa, "%lf", &a_bondi[i]        );
      a_bondi[i] *= a_inf;

  }


  fclose(frho);
  fclose(fv);
  fclose(fa);
}

void bondi_interp(const Real r, Real *rho, Real *v, Real *a){
  double dlogr = log10(r_max_tab/r_min_tab)/(1.*N_bondi -1.);

 int i_r  = (int) ((log10(r/r_min_tab))   / dlogr + 1000) - 1000;
 Real rat;
if (i_r < 0) {
    i_r = 0;
    rat = 1.;
} else if (i_r > N_bondi - 2) {
    i_r = N_bondi - 2;
    rat = r/(r_min_tab * pow(10.,i_r*dlogr)); 
} else {
    rat =  r/(r_min_tab * pow(10.,i_r*dlogr)); 
}


*rho = rho_bondi[i_r] * pow(rat,log10(rho_bondi[i_r+1]/rho_bondi[i_r])/dlogr);
*v = v_bondi[i_r] * pow(rat,log10(v_bondi[i_r+1]/v_bondi[i_r])/dlogr);
*a = a_bondi[i_r] * pow(rat,log10(a_bondi[i_r+1]/a_bondi[i_r])/dlogr);


// Real bernoulli = SQR(v_bondi[i_r])/2. + SQR(a_bondi[i_r])/(gm1) - gm_/(r_min_tab * pow(10.,i_r*dlogr)) - SQR(a_inf)/gm1;
// Real bernoulli_interp = SQR(*v)/2. + SQR(*a)/(gm1) - gm_/(r) - SQR(a_inf)/gm1;
//fprintf(stderr, "r: %g r_tab: %g rho: %g v: %g a: %g bernoulli: %g bernoulli_interp: %g \n", r,r_min_tab * pow(10.,i_r*dlogr),*rho,*v,*a,bernoulli, bernoulli_interp);

fprintf(stderr, "r: %g r_tab1: %g rtab2: %g v: %g vtab1: %g vtab2: %g \n", r,r_min_tab * pow(10.,i_r*dlogr),r_min_tab * pow(10.,(i_r+1)*dlogr),*v,v_bondi[i_r],v_bondi[i_r+1]);

return ;


}


//Relationship between v,a, and Mach number, x = v/a
// for bondi

Real v_of_a( const Real a, const Real r){
    return -qs /(r) * pow(a,(-2./(gm1)));
  }
Real a_of_x(const Real x,const Real r){
    //#v/a = x -> -qs/r * (a) **(-2/(gam-1) -1.)
    Real x_pow = -2./(gm1) -1.;
    return pow( (x*r/qs) , (1./x_pow) );
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

static Real MachResidual(Real x, Real r)
{

    Real gam = gm1+1.;

    Real a = a_of_x(x,r);
    Real v = v_of_a(a,r);
    return SQR(v)/2. + SQR(a)/(gm1) -1./r - 1./(gm1) ;

}



//----------------------------------------------------------------------------------------
// Bisection root finder
// Inputs:
//   r: Schwarzschild radius
//   x_min,x_max: bounds between which root must occur
// Outputs:
//   returned value: temperature that satisfies (HSW 76)
// Notes:
//   references Hawley, Smarr, & Wilson 1984, ApJ 277 296 (HSW)
//   performs bisection search

static Real MachBisect(Real r, Real x_min, Real x_max)
{
  // Parameters
  const int max_iterations = 100;
  const Real tol_residual = 1.0e-8;
  const Real tol_x = 1.0e-8;

  // Find initial residuals
  Real res_min = MachResidual(x_min, r);
  Real res_max = MachResidual(x_max, r);
  if (std::abs(res_min) < tol_residual) {
    return x_min;
  }
  if (std::abs(res_max) < tol_residual) {
    return x_max;
  }
  if ((res_min < 0.0 and res_max < 0.0) or (res_min > 0.0 and res_max > 0.0)) {
    return NAN;
  }

  // Iterate to find root
  Real x_mid;
  for (int i = 0; i < max_iterations; ++i) {
    x_mid = sqrt(x_min*x_max); //(x_min + x_max) / 2.0;
    if (x_max - x_min < tol_x) {
      return x_mid;
    }
    Real res_mid = MachResidual(x_mid, r);
    if (std::abs(res_mid) < tol_residual) {
      return x_mid;
    }
    if ((res_mid < 0.0 and res_min < 0.0) or (res_mid > 0.0 and res_min > 0.0)) {
      x_min = x_mid;
      res_min = res_mid;
    } else {
      x_max = x_mid;
      res_max = res_mid;
    }
  }
  return x_mid;
}

void get_Bondi(const Real r, Real *rho, Real *v, Real *a ){


   a_inf = sqrt(gm_/r_a);
   rho_inf = 1.;
   Real gam =gm1+1.;
   Real r_scaled = r/r_a;
   qs = pow( 2./(3.-gam) ,  (3.-gam)/(2.*gam-2.)   );
   Real mdot  = 2.*PI * qs;
   Real rs = (3.-gam)/2.;
   Real x;

   if (r_scaled<= rs){
    x = MachBisect(r_scaled,1.,1e2);
   }
   else{
    x = MachBisect(r_scaled,1e-10,1.);
   }

   *a = a_of_x(x,r_scaled) * a_inf;
   *v = v_of_a(*a/a_inf,r_scaled) * a_inf;
   *rho = mdot /(2.*PI *-(*v/a_inf)*r_scaled) * rho_inf;
}

/*
 * -------------------------------------------------------------------
 *     Initialize Mesh
 * -------------------------------------------------------------------
 */
void Mesh::InitUserMeshData(ParameterInput *pin)
{

    EnrollUserExplicitSourceFunction(cons_force);
    
    AllocateUserHistoryOutput(33*3);
    for (int i = 0; i<33*3; i++){
        int i_star = i/3;
        int i_pos  = i % 3;
        EnrollUserHistoryOutput(i, star_position, "star_"); 
    }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
    

    init_cooling();

    loclist = pmy_mesh->loclist;
    gm_ = pin->GetOrAddReal("problem","GM",0.0);
    gm1 = peos->GetGamma() - 1.0;
    l_bondi = std::sqrt(gm_ * r_circ);


    density_floor_ = peos->GetDensityFloor();
    pressure_floor_ = peos->GetPressureFloor();

    N_cells_per_radius = pin->GetOrAddReal("problem", "star_radius", 2.0);
    
    std::string file_name;
    file_name =  pin->GetString("problem","filename");


    Real dx_min,dy_min,dz_min;
    get_minimum_cell_lengths(pcoord, &dx_min, &dy_min, &dz_min);

    if (block_size.nx3>1)       r_inner_boundary = 2* std::max(std::max(dx_min,dy_min),dz_min); // r_inner_boundary = 2*sqrt( SQR(dx_min) + SQR(dy_min) + SQR(dz_min) );
    else if (block_size.nx2>1)  r_inner_boundary = 2 * std::max(dx_min,dy_min); //2*sqrt( SQR(dx_min) + SQR(dy_min)               );
    else                        r_inner_boundary = 2 * dx_min;
    //read_stardata(star,file_name);


    //read_in_bondi();


    Real DX,DY,DZ;
    get_uniform_box_spacing(pcoord->pmy_block->pmy_mesh->mesh_size,&DX,&DY,&DZ);

    r_inner_boundary_bondi = r_inner_boundary;
    r_outer_boundary_bondi = pcoord->pmy_block->pmy_mesh->mesh_size.x1max - 2.*DX ;

    if (COORDINATE_SYSTEM == "cylindrical" || COORDINATE_SYSTEM == "spherical_polar"){
      r_inner_boundary_bondi = 0.; //pcoord->pmy_block->block_size.x1min + 2.*(pcoord->pmy_block->block_size.x1max -pcoord->pmy_block->block_size.x1min)/pcoord->pmy_block->block_size.nx1;
      r_outer_boundary_bondi = pcoord->pmy_block->block_size.x1max - 2.*(pcoord->pmy_block->block_size.x1max -pcoord->pmy_block->block_size.x1min)/pcoord->pmy_block->block_size.nx1;
    }



    
    
    /* Switch over to problem units and locate the star cells */
    //star_unitconvert(star);
    
//    Real DX,DY,DZ, r_eff, mask_volume; //Length of entire simulation
//    get_uniform_box_spacing(pcoord->pmy_block->pmy_mesh->mesh_size,&DX,&DY,&DZ);
//    get_uniform_box_spacing(pcoord->pmy_block->block_size,&DX,&DY,&DZ);
//    get_star_size(pcoord,DX,DY,DZ,&r_eff,&mask_volume);

    

//    for (int i_star=0; i_star<nstars; i_star++) {
//      star[i_star].block_size = block_size;
//      star[i_star].radius = r_eff;
//      star[i_star].volume = mask_volume;
//    }
    
    
    
}

/* 
* -------------------------------------------------------------------
*     The actual problem / initial condition setup file
* -------------------------------------------------------------------
*/
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  int i=0,j=0,k=0;

  //read_in_bondi();
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
  Real x,y,z,r;
  Real rho,v,a;



  Real gm1 = peos->GetGamma() - 1.0;
  /* Set up a uniform medium */
  /* For now, make the medium almost totally empty */
  da = 1.0e-8;
  pa = 1.0e-10;
  ua = 0.0;
  va = 0.0;
  wa = 0.0;
  bxa = 0.0;
  bya = 0.0;
  bza = 0.0;
  

  for (k=kl; k<=ku; k++) {
  for (j=jl; j<=ju; j++) {
  for (i=il; i<=iu; i++) {

    get_cartesian_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &x, &y, &z);

    if ((block_size.nx3)>1) r = sqrt( SQR(x) + SQR(y) + SQR(z));
    else r = sqrt(SQR(x) + SQR(y));

    get_Bondi(r,&rho,&v,&a);
    //v = 0;
    
    phydro->u(IDN,k,j,i) = rho ;
      
      if (COORDINATE_SYSTEM == "cartesian"){
        phydro->u(IM1,k,j,i) = rho * v * x/r;
        phydro->u(IM2,k,j,i) = rho * v * y/r;
        if ((block_size.nx3)>1) phydro->u(IM3,k,j,i) = rho  * v* z/r;
        else phydro->u(IM3,k,j,i) = 0.;

      }
      else if (COORDINATE_SYSTEM == "cylindrical"){
          Real v_phi = l_bondi/std::sqrt(SQR(x) + SQR(y) + 1e-20);
          phydro->u(IM1,k,j,i) = rho * v;
          phydro->u(IM2,k,j,i) = rho * v_phi;
          v = std::sqrt(SQR(v) + SQR(v_phi));
      }
      else{
          Real v_phi = l_bondi/std::sqrt(SQR(x) + SQR(y) + 1e-20);
          phydro->u(IM1,k,j,i) = rho * v;
          phydro->u(IM2,k,j,i) = 0.;
          phydro->u(IM3,k,j,i) = rho * v_phi;
          v = std::sqrt(SQR(v) + SQR(v_phi));
      }

if (MAGNETIC_FIELDS_ENABLED){
    pfield->b.x1f(k,j,i) = bxa;
    pfield->b.x2f(k,j,i) = bya;
    pfield->bcc(IB1,k,j,i) = bxa;
    pfield->bcc(IB2,k,j,i) = bya;
    pfield->bcc(IB3,k,j,i) = bza;
    if (i == ie) pfield->b.x1f(k,j,i+1) = bxa;
    if (j == je) pfield->b.x2f(k,j+1,i) = bya;
}

    pressure = SQR(a) * rho  /(gm1+1.);
#ifndef ISOTHERMAL
    phydro->u(IEN,k,j,i) = pressure/gm1;
if (MAGNETIC_FIELDS_ENABLED){
      phydro->u(IEN,k,j,i) +=0.5*(bxa*bxa + bya*bya + bza*bza);
}
     phydro->u(IEN,k,j,i) += 0.5*rho*(SQR(v));
#endif /* ISOTHERMAL */

      if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
        phydro->u(IEN,k,j,i) += da;
 
  }}}
  

}

