#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
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

// Problem specific variables
namespace prob{
    // Test temperature and density arrays for the cooling function, 
    // should span most of the simulation domain
    const double GAMMA = 2.e-26;    // Used in ISM cooling (ergs/s)
    double current_n, current_T, dt;
}

// Root finding 
namespace root_find{
    double IQI(double (*f)(double), double a, double b, double c);
    double brendt(double (*f) (double), double a, double b);
    double bisect(double (*f) (double), double a, double b);
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
Real erg_to_code = g_to_msun*SQR(cm_to_pc/s_to_kyr);
Real icm3_to_code = 2.94e55;
// Taken from star cluster
Real UnitDensity = 6.767991e-23; // solar mass pc^-3
Real UnitEnergyDensity = 6.479592e-7; //solar mass /(pc ky^2)
Real UnitTime = 3.154e10;  //kyr
Real Unitlength = 3.086e+18; //parsec
Real UnitB = Unitlength/UnitTime * std::sqrt(4. * PI* UnitDensity);
// Problem 
Real pfloor, rhofloor;
Real vmax = 3/std::sqrt(3);        // ~1700 km/s
Real gas_gamma; 
Real kb = 1.38e-16*erg_to_code;    // per Kelvin
Real kb_cgs = 1.38e-16;
Real mh2 = 3.32e-24*g_to_msun;
Real Tfloor = 3.e2;
Real GM;
// Torus parameters (calculared using Python)
Real R_in = 1.4; 
Real R_k = 1.9;
Real q = 1.56; 
Real T_tar;
Real n_tar; //target number density in 1/cm3
Real rho_tar = n_tar*icm3_to_code*mh2;
// More parameters
Real C, K, P_amb, A; 
// Inner boundary
Real r_inner_boundary=0.0;
int n_mb = 0;
double SMALL = 1e-20;
bool amr_increase_resolution;
LogicalLocation *loc_list;
int max_refinement_level = 0; 

// Cooling source function
void Cooling_source(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons);

/* Disk pressure functions*/
// Gravitational potential
Real potential(Real r){
  return -GM/r;
}

// Centrifugal term
Real centrifugal(Real R){
  return SQR(C) * std::pow(R, -2*q + 2) / (2*q - 2);
}

// Integration constant 
Real integration_cons(){
  Real t1 = centrifugal(R_in);
  Real t2 = gas_gamma * P_amb*std::pow(K, 1/gas_gamma) / ((gas_gamma - 1) * std::pow(P_amb, 1/gas_gamma));
  Real t3 = potential(R_in);
  return -(t1 + t2 + t3);
}

/* Functions to deal with the inner boundary*/

// Floors out variables at the center of the simulation
void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
// Source function wrapper around 'apply_inner_boundary_condition'
static void integrate_cool(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim)
{
  apply_inner_boundary_condition(pmb,prim);
}

// Initialize variables
void Mesh::InitUserMeshData(ParameterInput *pin){
  pfloor = pin->GetReal("hydro", "pfloor");
  rhofloor = pin->GetReal("hydro", "dfloor");
  GM = pin->GetReal("problem", "GM");
  gas_gamma = pin->GetReal("hydro", "gamma");
  T_tar = pin->GetReal("problem","tar_T");
  n_tar = pin->GetReal("problem", "tar_n");
  //EnrollUserExplicitSourceFunction(gravity);
  EnrollUserExplicitSourceFunction(Cooling_source);
  EnrollUserRadSourceFunction(integrate_cool);
  C = std::sqrt(GM) * std::pow(R_k, q-1.5);
  K = (rho_tar * kb * T_tar) / (mh2 * std::pow(rho_tar, gas_gamma));
  P_amb = pfloor;
  A = integration_cons();
}

/* Functions to calculate the disk pressure */ 

// Disk pressure
Real disk_press(Real r_vec[3]){
  // r_vec must be in cartesian coords i.e. <x,y,z>
  Real r = std::sqrt(SQR(r_vec[0]) + SQR(r_vec[1]) + SQR(r_vec[2]));
  Real R = std::sqrt(SQR(r_vec[0]) + SQR(r_vec[1]));
  Real factor = (gas_gamma - 1) / (gas_gamma * std::pow(K, 1/gas_gamma));
  Real terms  = -(A + potential(r) + centrifugal(R));
  // To avoid nans:
  if (terms < 0){
    terms = 0;
  }

  Real P = std::pow(factor * terms, gas_gamma / (gas_gamma-1));
  return std::max(P, P_amb);
}
/* Root finding functions
*/

// Inverse quadratic interpolation (Required for brendt)
double root_find::IQI(double (*f)(double), double a, double b, double c){
    double fa = (*f)(a);
    double fb = (*f)(b);
    double fc = (*f)(c);
    return (a*fb*fc/((fa-fb) * (fa-fc)) +
            b*fa*fc/((fb-fa) * (fb-fc)) +
            c*fa*fb/((fc-fa) * (fc-fb)) );
}

// Implicit root finder Brendt
double root_find::brendt(double (*f) (double), double a, double b){
    // Will make these global variables later
    double eps = 1.e-3;     // bracket tolerance
    int maxiter = 20;       // maximum allowed iterations
    // Check if a and b bracket at least one root
    if(((*f)(a)) * ((*f)(b)) >= 0)
        {   
            std::cout   <<"products of roots not positive (a,b,f(a),f(b)): "
                        <<a<<","<<b<<","<<f(a)<<","<<f(b)<<std::endl;
            //throw std::invalid_argument("f(a) * f(b) must be negative") ;
        }
    // Want b to be closer to the root
    if (std::abs((*f)(a)) < std::abs((*f)(b))) {std::swap(a, b);}

    double c = a, s = b, d;
    int iter = 1;
    bool mflag = true;
    // Start main loop
    while (((*f)(b) != 0) && ((*f)(s) != 0) && (std::abs(b-a) > eps) && (iter <= maxiter)){
        //std::cout<<"iter: "<< iter << ", root guess: " << b <<std::endl;
        //if (std::abs(b-a) > eps) {std::cout<<"iter: "<< iter << std::endl;}
        // Perform IQI as the first try if 3 unique function evals exist
        if (((*f)(a) != (*f)(c)) && ((*f)(b) != (*f)(c))) {s = IQI(f, a, b, c);}
        // Otherwise try the secant method
        else {s = b - (*f)(b) * (b-a)/((*f)(b) - (*f)(a));}
        // Use bisection if necessary
        if (
            ((s > (3*a + b)/4 && s < b) || (s < (3*a + b)/4 && s > b)) ||
            (mflag && std::abs(s-b) >= std::abs(b-c)/2) ||
            ((!mflag) && std::abs(s-b) >= std::abs(c-d)/2) ||
            (mflag && std::abs(b-c) < eps) ||
            ((!mflag) && std::abs(c-d) < eps)
        ) {
            s = (a+b)/2;
            mflag = true;
        }
        else {mflag = false;}
        // Update bounds
        d = c;
        c = b;
        if (((*f)(a)) * ((*f)(s)) < 0){b = s;}
        else {a = s;}
        // Ensure that b is closer to the root
        if (std::abs((*f)(a)) < std::abs((*f)(b))) {std::swap(a, b);}
    iter ++;
    if (iter == maxiter){
        b = -1.;    // Use a different scheme if root is not found after maxiter
                    // Expect a positive return value if successful
        //std::cout<<"Reached max iterations, terminating loop " << std::endl;
        break;}
    }
    return b;
}

// Only works on monotonic functions
double root_find::bisect(double (*f) (double), double a, double b){
    double eps = 1.e-6;     // Tolerance bw bounds
    // 'full_cool' is monotonic so set current Temp as the upper bound
    // f(b) is guarenteed to be > 0
    double lower = a, upper = b;
    double mid = 0.5*(lower + upper), root;
    if (f(lower) > 0){
        root = lower;
        //std::cout<<"Temperature not bracketed in bounds"<<std::endl;
    }
    else if (upper - lower < eps){root = mid;}
    else {
        int iter=0, maxiter=20;
        while ((upper - lower > eps) && iter<maxiter){
            mid = 0.5*(upper + lower);
            if (f(mid) > 0){upper = mid;}
            else if (f(mid) < 0){lower = mid;}
            iter ++;
        }
    root = mid;
    //std::cout<<"Total iterations: "<<iter<<std::endl;
    //if (iter>=maxiter){std::cout<<"Reached max iterations"<<std::endl;}
    }
    return root;
}


/* ISM cooling function Koyama & Inutsuka (2002)
*/
// NOT the cooling rate, returns in erg * cm^3
// Full cooling is given by n(n gamma - Lam(T))
double Lam(double logT){
    // Only act on T < 10^(4.2)
    if (logT < 4.2 && logT >=2. && prob::current_n > 1.e-1){
        double T = pow(10., logT);
        return (2.e-19*exp(-1.184e5/(T+1.e3)) + 2.8e-28*sqrt(T)*exp(-92./T));
    }
    else {return -1.;}
}

// Need to find the ZERO of this func. (future: keeping stuff in log might help with minimzation?)
// f = T_(n+1) - T_n + (n * Lam(T_(n+1)) - GAMMA) * dt * gm1/k
// Make sure current n, current T and dt are set beforehand - can set n to cgs
double full_cool(double logT){
    // Quit if temperature range is outside the Koyama domain
    if (Lam(logT) == -1.){return 0;}
    else{
        double T = pow(10.,logT);
        return  (T - prob::current_T) + 
                prob::dt*(gas_gamma-1)/kb_cgs * (prob::current_n*Lam(logT) -
                prob::GAMMA);
    }
}

void Cooling_source(MeshBlock *pmb, const Real time, const Real dt,
  const AthenaArray<Real> &prim, const AthenaArray<Real> &bcc, AthenaArray<Real> &cons){
  // Get the dt, current temperature and ndens in cgs
  prob::dt = dt * 3.154e10;
  int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
  const double rho_to_n = mh2*icm3_to_code;   // Factor to convert to ndens from code rho
  double Tup, Tlow, Tf;
  double thermal, kinetic;
  for (int i=is;i<=ie;i++){
    for (int j=js;j<=je;j++){
      for (int k=ks;k<=ke;k++){
        // Obtain the final temperature from the cooling function
        prob::current_T = prim(IPR,k,j,i)/prim(IDN,k,j,i) * (mh2/kb);
        prob::current_n = prim(IDN,k,j,i)/rho_to_n;
        Tup = prob::current_T;
        Tlow = Tfloor;
        // NOTE: Calling full_cool requires setting {current_T, current_n, dt}
        Tf = pow(10.,root_find::bisect(full_cool,log10(Tlow),log10(Tup)));
        if (Tf>Tup){Tf = Tup;}  // Don't want gas to heat from rounding error in cooling
        // Set the new internal energy after cooling
        thermal = (Tf * kb * prob::current_n * icm3_to_code)/(gas_gamma - 1);
        kinetic = 0.5 * prim(IDN,k,j,i) * (SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)));
        cons(IEN,k,j,i) = kinetic + thermal;
        if (MAGNETIC_FIELDS_ENABLED){
          thermal = (Tf * kb * prob::current_n * icm3_to_code)/(gas_gamma - 1);
          kinetic = 0.5 * prim(IDN,k,j,i) * (SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)));
          double magnetic = 0.5*(SQR(0.5*(pmb->pfield->b.x1f(k,j,i) + pmb->pfield->b.x1f(k,j,i+1))) +
                            SQR(0.5*(pmb->pfield->b.x2f(k,j,i) + pmb->pfield->b.x2f(k,j+1,i))) +
                            SQR(0.5*(pmb->pfield->b.x3f(k,j,i) + pmb->pfield->b.x3f(k+1,j,i))));
          cons(IEN,k,j,i) = thermal + kinetic + magnetic;
        }
      }
    }
  }
}

/* Functions for disk orientation*/
// Find the index of the cell with coordinate x,y,z
// Try setting the input parameter as an Athena array
void disk_get_index(double x, double y, double z, int *i, int *j, int *k){
  MeshBlock *pmb;
  double is = pmb->is, ie = pmb->ie;
  double js = pmb->js, je = pmb->je;
  double ks = pmb->ks, ke = pmb->ke;
  Coordinates *pcoord = pmb->pcoord;
  
  /* Iterate through x */
  for (int tmp=is+1; tmp<ie-1; tmp++){
    if  (abs(x - pcoord->x1v(tmp)) < abs(x - pcoord->x1v(tmp-1)) &&
         abs(x - pcoord->x1v(tmp)) < abs(x - pcoord->x1v(tmp+1))){
          *i = tmp;
          break;
    }
  }
  /* Iterate through y */
  for (int tmp=js+1; tmp<je-1; tmp++){
    if  (abs(y - pcoord->x2v(tmp)) < abs(y - pcoord->x2v(tmp-1)) &&
         abs(y - pcoord->x2v(tmp)) < abs(y - pcoord->x2v(tmp+1))){
          *j = tmp;
          break;
    }
  }
    /* Iterate through z */
  for (int tmp=ks+1; tmp<ke-1; tmp++){
    if  (abs(z - pcoord->x3v(tmp)) < abs(z - pcoord->x3v(tmp-1)) &&
         abs(z - pcoord->x3v(tmp)) < abs(z - pcoord->x3v(tmp+1))){
          *k = tmp;
          break;
    }
  }
}

// Rotate by inclination and angle of periapse (Lena's matrix)
// Same matrix can be used to get the velocity
void disk_rotate(double x, double y, double z, double *xr, double *yr, double *zr){
  double ii = 66.*PI/180.;
  double Omega = 25*PI/180.;
  double w = 0; // arbitrary for disk
  *xr = cos(ii)*cos(Omega)*(y*cos(w)+x*sin(w))+sin(Omega)*(x*cos(w)-y*sin(w));
  *yr = -sin(w)*(y*cos(Omega)+x*cos(ii)*sin(Omega))+cos(w)*(x*cos(Omega)-y*cos(ii)*sin(Omega));
  *zr = z + sin(ii)*(y*cos(w)+x*sin(w));
}


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

void get_cylindrical_coords(const Real x1, const Real x2, const Real x3, Real *r, Real *phi, Real *z){  if (COORDINATE_SYSTEM == "cartesian"){
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

// Convert a point in spherical coords to cartesian coords
void spherical_to_cart(const Real Ar, const Real Ath, const Real Aph, const Real x, const Real y, const const Real z, Real *ax, Real *ay, Real *az){
  Real r, a, cos_theta, sin_theta, cos_phi, sin_phi;
  r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
  a = std::sqrt(SQR(x) + SQR(y));
  cos_theta = z/r;
  sin_theta = a/r;
  cos_phi = x/a;
  sin_phi = y/a;

  *ax = sin_theta*cos_phi*Ar + cos_theta*cos_phi*Ath - sin_phi*Aph;
  *ay = sin_theta*sin_phi*Ar + cos_theta*sin_phi*Ath + cos_phi*Aph;
  *az = cos_theta*Ar - sin_theta*Aph;

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
              // pmb->user_out_var(16,k,j,i) += bsq/bsq_rho_ceiling - prim(IDN,k,j,i);
              prim(IDN,k,j,i) = bsq/bsq_rho_ceiling;
             }
           }
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
  Real R, phi;
  // Add in the magnetic fields in the torus
  AthenaArray<Real> Ax, Ay, Az;
  int nx1 = ie-is + 1 + 2*NGHOST;
  int nx2 = je-js + 1 + 2*NGHOST;
  int nx3 = ke-ks + 1 + 2*NGHOST;
  Ax.NewAthenaArray(nx3,nx2,nx1);
  Ay.NewAthenaArray(nx3,nx2,nx1);
  Az.NewAthenaArray(nx3,nx2,nx1);

  // Set vector potential to be 0 for now
  for (k=0;k<nx3;k++){
    for(j=0;j<nx2;j++){
      for(i=0;i<nx1;i++){
        Ax(k,j,i) = 0.;
        Ay(k,j,i) = 0.;
        Az(k,j,i) = 0.;
      }
    }
  }

  Real ar, ath, aph, ax, ay, az;

  // Arrays to store the data of the rotated disk
  AthenaArray<Real> Rho_Disk, Press_Disk, Velx_Disk, Vely_Disk, Velz_Disk;
  nx1 = ie - is;
  nx2 = je - js;
  nx3 = ke - ks;
  Rho_Disk.NewAthenaArray(nx3,nx2,nx1);
  Press_Disk.NewAthenaArray(nx3,nx2,nx1);
  Velx_Disk.NewAthenaArray(nx3,nx2,nx1);
  Vely_Disk.NewAthenaArray(nx3,nx2,nx1);
  Velz_Disk.NewAthenaArray(nx3,nx2,nx1);

  for (k=ks; k<=ke+1; k++){
    for (j=js; j<=je+1; j++){
      for (i=is; i<=ie+1; i++){
        /***NON ROTATED VARIABLES********/
        get_cartesian_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &x, &y, &z);
        Real r_vec[3] = {x,y,z};
        R = std::sqrt(SQR(x) + SQR(y));
        Real press = disk_press(r_vec);
        Real torus_rho = std::pow(press/K, 1/gas_gamma);
        Real int_energy = press / (gas_gamma - 1);
        Real torus_omega = C * std::pow(R, -q);
        Real torus_v = torus_omega * R;
        phi = std::atan2(y,x);

        if (disk_press(r_vec) > pfloor){
          torus_rho = std::pow(press/K, 1/gas_gamma);
        }
        // When pressure = pfloor, v = 0 & rho = rhofloor
        else if (disk_press(r_vec) <= pfloor){ 
           torus_rho = rhofloor;
           torus_v = 0;
        }

        /*******Disk rotation stuff **************/
        double xr=0, yr=0, zr=0;    // rotated positions
        double vxr=0, vyr=0, vzr=0; // rotated velocities
        int ir=0, jr=0, kr=0;       // indices of rotated coords
        double vx = - torus_v * std::sin(phi), vy = torus_v * std::cos(phi), vz = 0;  // current velocities

        disk_rotate(x,y,z,&xr,&yr,&zr);
        disk_rotate(vx,vy,vz,&vxr,&vyr,&vzr);
        //disk_get_index(xr,yr,zr,&ir,&jr,&kr);
        printf("Old and new coordinates: (%f,%f,%f), (%f,%f,%f)",x,y,z,xr,yr,zr);

        /* Iterate through x */
        for (int tmp=is+1; tmp<ie-1; tmp++){
          if  ( (abs(x - pcoord->x1v(tmp)) < abs(x - pcoord->x1v(tmp-1))) &&
              (abs(x - pcoord->x1v(tmp)) < abs(x - pcoord->x1v(tmp+1)))){
                printf("Index found for x \n");
                ir = tmp;
                break;
          }
        }
        /* Iterate through y */
        for (int tmp=js+1; tmp<je-1; tmp++){
          if  ( (abs(y - pcoord->x2v(tmp)) < abs(y - pcoord->x2v(tmp-1))) &&
              (abs(y - pcoord->x2v(tmp)) < abs(y - pcoord->x2v(tmp+1)))){
                printf("Index found for y \n");
                jr = tmp;
                break;
          }
        }
          /* Iterate through z */
        for (int tmp=ks+1; tmp<ke-1; tmp++){
          if  ( (abs(z - pcoord->x3v(tmp)) < abs(z - pcoord->x3v(tmp-1))) &&
              (abs(z - pcoord->x3v(tmp)) < abs(z - pcoord->x3v(tmp+1)))){
                printf("Index found for z \n");
                kr = tmp;
                break;
          }
        }
        // printf("Old indices (%d, %d, %d), New indices (%d, %d, %d) \n", i, j, k, ir, jr, kr);

        // Now replace with rotated values
        Rho_Disk(kr,jr,ir) = torus_rho;
        Press_Disk(kr,jr,ir) = press;
        Velx_Disk(kr,jr,ir) = vxr;
        Vely_Disk(kr,jr,ir) = vyr;
        Velz_Disk(kr,jr,ir) = vzr;

        /********Vector potential***********************/

        // Set the vector potential in cartesian coords
        // We have Ar = Krho, Atheta = Krho, Aphi = 0, K is some constant
        ar = 0.;
        ath = 0.;
        aph = torus_rho;
        spherical_to_cart(ar, ath, aph, x, y, z, &ax, &ay, &az);
        Ax(k,j,i) = ax;
        Ay(k,j,i) = ay;
        Az(k,j,i) = az;
        // Copy vector potential into the ghost zones
        if (i == is){
          Ax(i-1,j,k) = Ax(i,j,k);
          Ay(i-1,j,k) = Ay(i,j,k);
          Az(i-1,j,k) = Az(i,j,k);
        }
        if (j == js){
          Ax(i,j-1,k) = Ax(i,j,k);
          Ay(i,j-1,k) = Ay(i,j,k);
          Az(i,j-1,k) = Az(i,j,k);
        }
        if (k == ks){
          Ax(i,j,k-1) = Ax(i,j,k);
          Ay(i,j,k-1) = Ay(i,j,k);
          Az(i,j,k-1) = Az(i,j,k);
        }
        if (i == ie+1){
          Ax(i+1,j,k) = Ax(i,j,k);
          Ay(i+1,j,k) = Ay(i,j,k);
          Az(i+1,j,k) = Az(i,j,k);
        }
        if (j == je+1){
          Ax(i,j+1,k) = Ax(i,j,k);
          Ay(i,j+1,k) = Ay(i,j,k);
          Az(i,j+1,k) = Az(i,j,k);
        }
        if (k == ke+1){
          Ax(i,j,k+1) = Ax(i,j,k);
          Ay(i,j,k+1) = Ay(i,j,k);
          Az(i,j,k+1) = Az(i,j,k);
        }

        // Don't set conservatives in ghost zones 
        if (k>=ks && k<=ke){
          if (j>=js && j<=je){    
            if (i>=is && i<=ke){
              // Outer regions
              if (R >= 0.1){
                phydro->u(IDN,k,j,i) = torus_rho;
                phydro->u(IM1,k,j,i) = -torus_rho * torus_v * std::sin(phi);
                phydro->u(IM2,k,j,i) = torus_rho * torus_v * std::cos(phi);
                phydro->u(IM3,k,j,i) = 0;
                phydro->u(IEN,k,j,i) = (0.5 * torus_rho) * SQR(torus_v) + int_energy;
              }
              // Assign floors and 0 velocity to the inner region
              // pressure is already at floor in this region... 
              else if (R <0.1){
                phydro->u(IDN,k,j,i) = rhofloor;
                phydro->u(IM1,k,j,i) = 0;
                phydro->u(IM2,k,j,i) = 0;
                phydro->u(IM3,k,j,i) = 0;
                phydro->u(IEN,k,j,i) = int_energy;
              }
         }}}
      }
    }
  }
  Real bmag = 1.e-3/UnitB, bnorm = 1.e5; 
  
  if (MAGNETIC_FIELDS_ENABLED){
    // Add in the magnetic fields
    for (k=ks; k<=ke; k++){
      for (j=js; j<=je; j++){
        for (i=is; i<=ie+1; i++){
          pfield->b.x1f(k,j,i) = (bmag/bnorm)*((Az(k,j+1,i)-Az(k,j,i))/pcoord->dx2f(j) -
                                  (Ay(k+1,j,i)-Ay(k,j,i))/pcoord->dx3f(k));         
        }}}
    for (k=ks; k<=ke; k++){
      for (j=js; j<=je+1; j++){
        for (i=is; i<=ie; i++){
          pfield->b.x2f(k,j,i) = (bmag/bnorm)*((Ax(k+1,j,i)-Ax(k,j,i))/pcoord->dx3f(k) -
                                  (Az(k,j,i+1)-Az(k,j,i))/pcoord->dx1f(i));         
        }}}
    for (k=ks; k<=ke+1; k++){
      for (j=js; j<=je; j++){
        for (i=is; i<=ie; i++){
          pfield->b.x3f(k,j,i) = (bmag/bnorm)*((Ay(k,j,i+1)-Ay(k,j,i))/pcoord->dx1f(i) -
                                  (Ax(k,j+1,i)-Ax(k,j,i))/pcoord->dx2f(j));         
        }}}
    
     // Change the energy from B fields
    for (k=ks; k<=ke; k++){
      for (j=js; j<=je; j++){
        for (i=is; i<=ie; i++){
           phydro->u(IEN,k,j,i) +=
            0.5*(SQR(0.5*(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))) +
                 SQR(0.5*(pfield->b.x2f(k,j,i) + pfield->b.x2f(k,j+1,i))) +
                 SQR(0.5*(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i))));   
     }}}
   } 
  Ax.DeleteAthenaArray();
  Ay.DeleteAthenaArray();
  Az.DeleteAthenaArray();

  // Replace old hydro variables with rotated ones
  for (k=ks; k<=ke+1; k++){
    for (j=js; j<=je+1; j++){
      for (i=is; i<=ie+1; i++){
        int ir = i-NGHOST;
        int jr = j-NGHOST;
        int kr = k-NGHOST;
        get_cartesian_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &x, &y, &z);
        double r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        // Only rotate coordinates outside 0.5 pc
        if (r >= 0.5){
          phydro->u(IDN,k,j,i) = Rho_Disk(kr,jr,ir);
          phydro->u(IM1,k,j,i) = Rho_Disk(kr,jr,ir)*Velx_Disk(kr,jr,ir);
          phydro->u(IM2,k,j,i) = Rho_Disk(kr,jr,ir)*Vely_Disk(kr,jr,ir);
          phydro->u(IM3,k,j,i) = Rho_Disk(kr,jr,ir)*Velz_Disk(kr,jr,ir);;
          phydro->u(IEN,k,j,i) = (0.5 * Rho_Disk(kr,jr,ir)) * std::sqrt(
                                 SQR(Velx_Disk(kr,jr,ir)) + SQR(Vely_Disk(kr,jr,ir)) + SQR(Velz_Disk(kr,jr,ir))) 
                                 + Press_Disk(kr,jr,ir)/gm1;
        }

      }
    }
  }

  Rho_Disk.DeleteAthenaArray();
  Press_Disk.DeleteAthenaArray();
  Velx_Disk.DeleteAthenaArray();
  Vely_Disk.DeleteAthenaArray();
  Velz_Disk.DeleteAthenaArray();
}
