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
#include <cfloat>

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


/* cooling */
/* -------------------------------------------------------------------------- */
void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim);
static void inner_boundary(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim );



 void FixedBoundary(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                    Real time, Real dt, int is, int ie, int js, int je, int ks, int ke);

void interp_inits(const Real x, const Real y, const Real z, Real *rho, Real *vx, Real *vy, Real *vz, Real *p);

Real DivergenceB(MeshBlock *pmb, int iout);
void Cartesian_GR(Real x1, Real x2, Real x3, ParameterInput *pin,
    AthenaArray<Real> &g, AthenaArray<Real> &g_inv, AthenaArray<Real> &dg_dx1,
    AthenaArray<Real> &dg_dx2, AthenaArray<Real> &dg_dx3);
/* Initialize a couple of the key variables used throughout */
//Real r_inner_boundary = 0.;         /* remove mass inside this radius */
Real r_min_inits = 1e15; /* inner radius of the grid of initial conditions */
Real gm_;               /* G*M for point mass at origin */
Real gm1;               /* \gamma-1 (adiabatic index) */
Real rh;                /* Horizon radius */
Real risco;             /* ISCO radius */
Real a,m;
static Real dfloor,pfloor;                         // density and pressure floors

double SMALL = 1e-20;       /* Small number for numerical purposes */
LogicalLocation *loc_list;              /* List of logical locations of meshblocks */
int n_mb = 0; /* Number of meshblocks */
int max_smr_level = 0;
int max_refinement_level = 0;    /*Maximum allowed level of refinement for AMR */
Real beta_star;  /* beta for each star, defined wrt rho v^2 */

int N_r =128;  /* Number of points to sample in r for radiale profile */
int N_user_vars = 6; //10; /* Number of user defined variables in UserWorkAfterLoop */
int N_user_vars_field = 6; /* Number of user defined variables related to magnetic fields */
Real r_dump_min,r_dump_max; /* Range in r to sample for radial profile */


Real yr = 31556926.0, pc = 3.09e18;    /* yr and parsec in code units */
Real cl = 2.99792458e10 * (1e3 * yr)/pc ;      /* speed of light in code units */
Real cs_max = cl ; //0.023337031 * cl;  /*sqrt(me/mp) cl....i.e. sound speed of electrons is ~ c */

bool amr_increase_resolution; /* True if resolution is to be increased from restarted run */


    AthenaArray<Real> w_inits;
    FaceField b_inits;
    AthenaArray<Real> A1_bound,A2_bound,A3_bound;
    AthenaArray<Real> divb_array;


void GetKSCoordinates(Real x1, Real x2, Real x3, Real *pr, Real *ptheta, Real *pphi){

  if (COORDINATE_SYSTEM == "kerr-schild" or COORDINATE_SYSTEM == "schwarzschild"){
      *pr = x1;
      *ptheta = x2;
      *pphi = x3;
  }
  else{


    Real R = std::sqrt(SQR(x1) + SQR(x2) + SQR(x3));
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(x3) )  )/std::sqrt(2.0);

    *pr = r;
    *ptheta = std::acos(x3/r);
    *pphi = std::atan2( (r*x2 - a*x1) , (a*x2+r*x1)  );   
  }


  return;
}



Real interp_scalar(const AthenaArray<Real> &var, int i, int j, int k, double coeff[5], int N3)
{

  double interp, interpk, interpkp1;
  int kp1;

  k = k%N3;
  kp1 = (k+1)%N3;
    
  interpk =
      var(k,j,i) * coeff[0] +
      var(k,j+1,i) * coeff[1] +
      var(k,j,i+1) * coeff[2] + var(k,j+1,i+1) * coeff[3];

  interpkp1 =
      var(kp1,j,i) * coeff[0] +
      var(kp1,j+1,i) * coeff[1] +
      var(kp1,j,i+1) * coeff[2] + var(kp1,j+1,i+1) * coeff[3];
  
  interp = (1.-coeff[4]) * interpk + coeff[4] * interpkp1;

  return interp;
}



void Get_interp_quantities(const Real r,const Real th,const Real ph, const Real r0, const Real th0, const Real ph0,
  const Real dlogr, const Real dth, const Real dph, const int nx_inits, const int ny_inits, const int nz_inits,
    int *i0, int *j0, int *k0, Real del[4], Real coeff[5]){

                *i0 = (int) ((std::log(r) - std::log(r0)) / dlogr - 0.5 + 1000) - 1000;
                *j0 = (int) ((th - th0) / dth - 0.5 + 1000) - 1000;
                *k0 = (int) ((ph - ph0) / dph - 0.5 + 1000) - 1000;

                *k0 = ((*k0 % nz_inits) + nz_inits) %nz_inits;
                

  // *i = (int) ((X[1] - startx[1]) / dx[1] - 0.5 + 1000) - 1000;
  // *j = (int) ((X[2] - startx[2]) / dx[2] - 0.5 + 1000) - 1000;
  // *k = (int) ((x3 - startx[3]) / dx[3] - 0.5 + 1000) - 1000;


                  if (*i0>(nx_inits-2)) {
                    *i0=nx_inits-2;
                    del[1]=1.0;
                  }
                  else if (*i0<0){
                    *i0=0;
                    del[1] = 0.0;
                  }
                  else{
                    del[1] = (std::log(r) - ((*i0 + 0.5) * dlogr + std::log(r0))) / dlogr;
                  }
                  if (*j0>ny_inits-2){
                    *j0 = ny_inits-2;
                    del[2] = 1.0;
                  }
                  else if (j0<0){
                    *j0 = 0;
                    del[2] = 0.0;
                  }
                  else{
                    del[2] = (th - ((*j0 + 0.5) * dth)) / dth;
                  }
                  //double ph0 = ((k0 + 0.5) * dph);
                  //if (ph0>)
                  if (*k0>nz_inits-2){
                    *k0 = nz_inits-2;
                    del[3] = 1.0;
                  }
                  else if (*k0<0){
                    *k0 = 0;
                    del[3]=0.0;
                  }
                  else{
                    del[3] = (ph - ((*k0 + 0.5) * dph)) / dph;
                  }

                  coeff[0] = (1. - del[1]) * (1. - del[2]);
                  coeff[1] = (1. - del[1]) * del[2];
                  coeff[2] = del[1] * (1. - del[2]);
                  coeff[3] = del[1] * del[2];
                  coeff[4] = del[3];

                  return;

}



void set_boundary_arrays(std::string initfile, const RegionSize block_size, Coordinates *pcoord, const int is, const int ie, const int js, const int je, const int ks, const int ke,
  AthenaArray<Real> &prim_bound, FaceField &b_bound){
      FILE *input_file;
        if ((input_file = fopen(initfile.c_str(), "r")) == NULL)   
               fprintf(stderr, "Cannot open %s, %s\n", "input_file",initfile.c_str());

    int ncells1 = block_size.nx1 + 2*(NGHOST);
    int ncells2 = 1, ncells3 = 1;
    bool RESCALE = false;
    if (block_size.nx2 > 1) ncells2 = block_size.nx2 + 2*(NGHOST);
    if (block_size.nx3 > 1) ncells3 = block_size.nx3 + 2*(NGHOST);

      int nx_inits,ny_inits,nz_inits; /* size of initial condition arrays */
      AthenaArray<Real> x_inits,y_inits,z_inits,v1_inits,v2_inits,v3_inits,press_inits,rho_inits; /* initial condition arrays*/

      AthenaArray<Real> bx1f_inits,bx2f_inits,bx3f_inits, A1_inits, A2_inits, A3_inits; //, A1_bound, A2_bound,A3_bound;

      Real rho_max = 0;

      fscanf(input_file, "%i %i %i \n", &nx_inits, &ny_inits, &nz_inits);

       

    x_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    y_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    z_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    rho_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    v1_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    v2_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    v3_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    press_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);

  Real L_unit = gm_/SQR(cl);
  Real r_in_9 = 2.0*2.0/128.0/std::pow(2.0,9.0)/L_unit;

  Real scale_fac_radius =  (rh/r_in_9);
  Real scale_fac_density =  scale_fac_radius;
  Real scale_fac_pressure = SQR(scale_fac_radius);
  Real scale_fac_velocity = std::sqrt(scale_fac_radius);


  if (RESCALE == false){
    scale_fac_radius = 1.0;
    scale_fac_density = 1.0;
    scale_fac_pressure = 1.0;
    scale_fac_velocity = 1.0;
  }


if (MAGNETIC_FIELDS_ENABLED){

    A1_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    A2_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);
    A3_inits.NewAthenaArray(nz_inits,ny_inits,nx_inits);

    A1_bound.NewAthenaArray( ncells3  +2,ncells2+2, ncells1+2   );
    A2_bound.NewAthenaArray( ncells3  +2,ncells2+2, ncells1+2   );
    A3_bound.NewAthenaArray( ncells3  +2,ncells2+2, ncells1+2   );
}




    int i,j,k;
      for (k=0; k<nz_inits; k++) {
      for (j=0; j<ny_inits; j++) {
      for (i=0; i<nx_inits; i++) {


    fread( &x_inits(k,j,i), sizeof( Real ), 1, input_file );
    x_inits(k,j,i) = x_inits(k,j,i)/L_unit * scale_fac_radius;
    fread( &y_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &z_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &rho_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &v1_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &v2_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &v3_inits(k,j,i), sizeof( Real ), 1, input_file );


    v1_inits(k,j,i) = v1_inits(k,j,i) * scale_fac_velocity;
    v2_inits(k,j,i) = v2_inits(k,j,i) * scale_fac_velocity;
    v3_inits(k,j,i) = v3_inits(k,j,i) * scale_fac_velocity;
    fread( &press_inits(k,j,i), sizeof( Real ), 1, input_file );

    Real vsq = SQR(v1_inits(k,j,i)) + SQR(v2_inits(k,j,i)) + SQR(v3_inits(k,j,i));

    if (std::sqrt(vsq)>=cl){
      v1_inits(k,j,i) *= cl/std::sqrt(vsq)*0.9;
      v2_inits(k,j,i) *= cl/std::sqrt(vsq)*0.9;
      v3_inits(k,j,i) *= cl/std::sqrt(vsq)*0.9;
    }

    vsq = SQR(v1_inits(k,j,i)) + SQR(v2_inits(k,j,i)) + SQR(v3_inits(k,j,i));

    //Real gamma = 1.0/std::sqrt(1.0-vsq/SQR(cl));


      // v1_inits(k,j,i) *= 1.0/gamma;
      // v2_inits(k,j,i) *= 1.0/gamma;
      // v3_inits(k,j,i) *= 1.0/gamma;


    //fprintf(stderr,"i,j,k %d %d %d \n  x y z  %g %g %g \n ",i,j,k,x_inits(k,j,i),y_inits(k,j,i),z_inits(k,j,i));

    Real tmp;

  if (MAGNETIC_FIELDS_ENABLED){
    fread( &tmp, sizeof( Real ), 1, input_file );
    fread( &tmp, sizeof( Real ), 1, input_file );
    fread( &tmp, sizeof( Real ), 1, input_file );

    fread( &A1_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &A2_inits(k,j,i), sizeof( Real ), 1, input_file );
    fread( &A3_inits(k,j,i), sizeof( Real ), 1, input_file );


    //fprintf(stderr,"bx,by,bz: %g %g %g \n",bx1f_inits(k,j,i),bx2f_inits(k,j,i),bx3f_inits(k,j,i));
  }

    r_min_inits = std::min(r_min_inits,x_inits(k,j,i));
    rho_max = std::max(rho_max,rho_inits(k,j,i));

    }
    }
    }
        fclose(input_file);

    
    r_min_inits = 50.0;
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
  Real rho_unit = rho_max;
  Real v_unit = cl;
  Real P_unit = SQR(cl) * rho_unit;
  Real B_unit = cl * std::sqrt(rho_unit);
  Real A_unit = B_unit*L_unit; 


//  if (Globals::my_rank==0) {
  fprintf(stderr,"rho_unit: %g \nv_unit: %g \n P_unit: %g \n B_unit: %g \n A_unit: %g \n r_min_inits: %g\n", rho_unit,v_unit,P_unit,B_unit,A_unit,r_min_inits);
//}

      for (int k=kl; k<=ku+2; ++k) {
#pragma omp parallel for schedule(static)
        for (int j=jl; j<=ju+2; ++j) {
#pragma simd
            for (int i=il; i<=iu+2; ++i) {

              Real rho,p,vx,vy,vz,bx1f,bx2f,bx3f, A1,A2,A3;
             // interp_inits(x, y, z, &rho, &vx,&vy,&vz,&p);

              int i0,j0,k0;
                //Real rg = gm_/std::pow(cl,2.0);

                Real dlogr = std::log(x_inits(0,0,1)) - std::log(x_inits(0,0,0)) ;
                Real dth   = y_inits(0,1,0) - y_inits(0,0,0);
                Real dph   = z_inits(1,0,0) - z_inits(0,0,0);

                Real r0 = x_inits(0,0,0);
                Real th0 = y_inits(0,0,0);
                Real ph0 = z_inits(0,0,0);

                Real r,th, ph;
                

  

                Real x1_coord, x2_coord, x3_coord; 

                if (i<=iu) x1_coord= pcoord->x1v(i);
                else if (i==iu+1) x1_coord = pcoord->x1v(iu) + pcoord->dx1v(iu);
                else x1_coord =  pcoord->x1v(iu) + 2.0*pcoord->dx1v(iu);

                if (j<=ju) x2_coord= pcoord->x2v(j);
                else if (j==ju+1) x2_coord = pcoord->x2v(ju) + pcoord->dx2v(ju);
                else x2_coord =  pcoord->x2v(ju) + 2.0*pcoord->dx2v(ju);

                if (k<=ku) x3_coord= pcoord->x3v(k);
                else if (k==ku+1) x3_coord = pcoord->x3v(ku) + pcoord->dx3v(ku);
                else x3_coord =  pcoord->x3v(ku) + 2.0*pcoord->dx3v(ku);
      

                // if (i<=iu && j<= ju && k<=){

                GetKSCoordinates(x1_coord, x2_coord,
                                                     x3_coord, &r, &th, &ph);

                ph = std::fmod((std::fmod(ph,2.0*PI) + 2.0*PI),2.0*PI);

                Real del[4],coeff[5];

                Get_interp_quantities(r,th,ph,r0,th0,ph0,dlogr,dth,dph,nx_inits,ny_inits,nz_inits,&i0,&j0,&k0,del,coeff);

                  rho = interp_scalar(rho_inits,i0,j0,k0,coeff,nz_inits)/rho_unit * scale_fac_density; //rho_inits(k0,j0,i0)/rho_unit;
                  vx = interp_scalar(v1_inits,i0,j0,k0,coeff,nz_inits)/v_unit ;
                  vy = interp_scalar(v2_inits,i0,j0,k0,coeff,nz_inits)/v_unit ;
                  vz = interp_scalar(v3_inits,i0,j0,k0,coeff,nz_inits)/v_unit;
                  p = interp_scalar(press_inits,i0,j0,k0,coeff,nz_inits)/P_unit * scale_fac_pressure;

                  if (r<r_min_inits){
                    rho = 0;
                    vx = 0;
                    vy = 0;
                    vz = 0;
                    p = 0;
                  }

                  if (rho<0) fprintf(stderr,"del: %g %g %g \n i j k: %d %d %d\n r th phi: %g %g %g\n", del[1],del[2],del[3],i0,j0,k0,r,th,ph);

                  // if (fabs(vx)>1 || fabs(vy)> 1 || fabs(vz)>1 ){
                  //   fprintf(stderr,"Superluminal speeds!!!! \n vx vy vz: %g %g %g \n r th v_unit: %g %g %g\n",vx,vy,vz,r,th,v_unit);
                  // }

                  // fprintf(stderr,"x y z: %g %g %g \n r th phi: %g %g %g i j k: %d %d %d\n rho: %g \n",
                  //   x1_coord,x2_coord,x3_coord,r,th,ph,i,j,k, rho);


                  if (MAGNETIC_FIELDS_ENABLED){
                    //Note this is A_\mu not A^\mu */ 


                    Real scale_fac = 1.0 ;//std::sqrt(5.0);

                    if (i==is && j==js && k==ks) fprintf(stderr,"ARBITRARY SCALE FACTOR TO B_FIELD: %g \n",scale_fac);

                    Real x1_coord, x2_coord, x3_coord; 
                    Real exp_fac;

                    if (i<=iu) x1_coord= pcoord->x1v(i);
                    else if (i==iu+1) x1_coord = pcoord->x1v(iu) + pcoord->dx1v(iu);
                    else x1_coord =  pcoord->x1v(iu) + 2.0*pcoord->dx1v(iu);

                    if (j<=ju+1) x2_coord= pcoord->x2f(j);
                    else x2_coord = pcoord->x2f(ju) + pcoord->dx2f(ju);

                    if (k<=ku+1) x3_coord= pcoord->x3f(k);
                    else x3_coord = pcoord->x3f(ku) + pcoord->dx3f(ku);

                    GetKSCoordinates(x1_coord, x2_coord,x3_coord, &r, &th, &ph);
                    ph = std::fmod((std::fmod(ph,2.0*PI) + 2.0*PI),2.0*PI);
                    Get_interp_quantities(r,th,ph,r0,th0,ph0,dlogr,dth,dph,nx_inits,ny_inits,nz_inits,
                      &i0,&j0,&k0,del,coeff);

                    if (r<=r_min_inits) exp_fac = std::exp(5 * (r-r_min_inits)/r);
                    else exp_fac = 1.0;

                    A1 =  exp_fac * interp_scalar(A1_inits,i0,j0,k0,coeff,nz_inits)*scale_fac                 /A_unit;

                    if (i<=iu+1) x1_coord= pcoord->x1f(i);
                    else if (i==iu+2) x1_coord = pcoord->x1f(iu+1) + pcoord->dx1f(iu+1);

                    if (j<=ju) x2_coord= pcoord->x2v(j);
                    else if (j==ju+1) x2_coord = pcoord->x2v(ju) + pcoord->dx2v(ju);
                    else x2_coord =  pcoord->x2v(ju) + 2.0*pcoord->dx2v(ju);

                    if (k<=ku+1) x3_coord= pcoord->x3f(k);
                    else x3_coord = pcoord->x3f(ku) + pcoord->dx3f(ku);                    

                    GetKSCoordinates(x1_coord, x2_coord,x3_coord, &r, &th, &ph);

                    ph = std::fmod((std::fmod(ph,2.0*PI) + 2.0*PI),2.0*PI);
                    Get_interp_quantities(r,th,ph,r0,th0,ph0,dlogr,dth,dph,nx_inits,ny_inits,nz_inits,
                      &i0,&j0,&k0,del,coeff);

                    if (r<=r_min_inits) exp_fac = std::exp(5 * (r-r_min_inits)/r);
                    else exp_fac = 1.0;

                    A2 =  exp_fac * interp_scalar(A2_inits,i0,j0,k0,coeff,nz_inits) *scale_fac             /A_unit;

                    if (i<=iu+1) x1_coord= pcoord->x1f(i);
                    else if (i==iu+2) x1_coord = pcoord->x1f(iu+1) + pcoord->dx1f(iu+1);
                    if (j<=ju+1) x2_coord= pcoord->x2f(j);
                    else x2_coord = pcoord->x2f(ju) + pcoord->dx2f(ju);
                    if (k<=ku) x3_coord= pcoord->x3v(k);
                    else if (k==ku+1) x3_coord = pcoord->x3v(ku) + pcoord->dx3v(ku);
                    else x3_coord =  pcoord->x3v(ku) + 2.0*pcoord->dx3v(ku);

                    GetKSCoordinates(x1_coord, x2_coord,x3_coord, &r, &th, &ph);
                    ph = std::fmod((std::fmod(ph,2.0*PI) + 2.0*PI),2.0*PI);

                    Get_interp_quantities(r,th,ph,r0,th0,ph0,dlogr,dth,dph,nx_inits,ny_inits,nz_inits,
                      &i0,&j0,&k0,del,coeff);

                    if (r<=r_min_inits) exp_fac = std::exp(5 * (r-r_min_inits)/r);
                    else exp_fac = 1.0;

                    A3 =  exp_fac * interp_scalar(A3_inits,i0,j0,k0,coeff,nz_inits) * scale_fac/A_unit;

                }


              

            if (i<=iu && j<=ju && k<=ku){

              prim_bound(IDN,k,j,i) = rho;
              prim_bound(IVX,k,j,i) = vx;
              prim_bound(IVY,k,j,i) = vy;
              prim_bound(IVZ,k,j,i) = vz;
              prim_bound(IPR,k,j,i) = p;
            }

              if (MAGNETIC_FIELDS_ENABLED){

                A1_bound(k,j,i) = A1;
                A2_bound(k,j,i) = A2;
                A3_bound(k,j,i) = A3;


                                //fprintf(stderr,"bx,bx,bz: %g %g %g \n",b_bound.x1f(k,j,i),b_bound.x2f(k,j,i),b_bound.x3f(k,j,i));
              }


            }}}


if (MAGNETIC_FIELDS_ENABLED){

      // Initialize interface fields
    AthenaArray<Real> area,len,len_p1;
    area.NewAthenaArray(ncells1);
    len.NewAthenaArray(ncells1);
    len_p1.NewAthenaArray(ncells1);

    // for 1,2,3-D
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju+1; ++j) {
        pcoord->Face2Area(k,j,il,iu,area);
        for (int i=il; i<=iu; ++i) {
          b_bound.x2f(k,j,i) = -1.0*(pcoord->dx3f(k)*A3_bound(k,j,i+1) - pcoord->dx3f(k)*A3_bound(k,j,i))/area(i);
          if (area(i)==0.0) b_bound.x2f(k,j,i) = 0;
          //if (j==ju) fprintf(stderr,"B: %g area: %g theta: %g j: %d A3: %g %g \n",b_bound.x2f(k,j,i), area(i),pcoord->x2f(j),j, 
           // A3_bound(k,j,i+1), A3_bound(k,j,i));
        }
      }
    }
    for (int k=kl; k<=ku+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
        pcoord->Face3Area(k,j,il,iu,area);
        for (int i=il; i<=iu; ++i) {
          b_bound.x3f(k,j,i) = (pcoord->dx2f(j)*A2_bound(k,j,i+1) - pcoord->dx2f(j)*A2_bound(k,j,i))/area(i);
        }
      }
    }

    // for 2D and 3D
    if (block_size.nx2 > 1) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face1Area(k,j,il,iu+1,area);
          for (int i=il; i<=iu+1; ++i) {
            b_bound.x1f(k,j,i) = (pcoord->dx3f(k)*A3_bound(k,j+1,i) - pcoord->dx3f(k)*A3_bound(k,j,i))/area(i);
          }
        }
      }
      for (int k=kl; k<=ku+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face3Area(k,j,il,iu,area);
          for (int i=il; i<=iu; ++i) {
            b_bound.x3f(k,j,i) -= (pcoord->dx1f(i)*A1_bound(k,j+1,i) - pcoord->dx1f(i)*A1_bound(k,j,i))/area(i);
          }
        }
      }
    }
    // for 3D only
    if (block_size.nx3 > 1) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face1Area(k,j,il,iu+1,area);
          for (int i=il; i<=iu+1; ++i) {
            b_bound.x1f(k,j,i) -= (pcoord->dx2f(j)*A2_bound(k+1,j,i) - pcoord->dx2f(j)*A2_bound(k,j,i))/area(i);
          }
        }
      }
      for (int k=kl; k<=ku; ++k) {
        // reset loop limits for polar boundary
        for (int j=jl; j<=ju; ++j) {
          pcoord->Face2Area(k,j,il,iu,area);
          for (int i=il; i<=iu; ++i) {
            b_bound.x2f(k,j,i) += (pcoord->dx1f(i)*A1_bound(k+1,j,i) - pcoord->dx1f(i)*A1_bound(k,j,i))/area(i);
            if (area(i)==0.0) b_bound.x2f(k,j,i) = 0;
            //if ( ju==je && j==je) fprintf(stderr,"B: %g area: %g theta: %g j: %d A1: %g %g \n",b_bound.x2f(k,j,i), area(i),pcoord->x2f(j),j, 
            //A1_bound(k+1,j,i), A1_bound(k,j,i));
          }
        }
      }
    }

    area.DeleteAthenaArray();
    len.DeleteAthenaArray();
    len_p1.DeleteAthenaArray();    

}
       x_inits.DeleteAthenaArray();
       y_inits.DeleteAthenaArray();
       z_inits.DeleteAthenaArray();
       rho_inits.DeleteAthenaArray();
       v1_inits.DeleteAthenaArray();
       v2_inits.DeleteAthenaArray();
       v3_inits.DeleteAthenaArray();
       press_inits.DeleteAthenaArray();

       if (MAGNETIC_FIELDS_ENABLED){
        A1_inits.DeleteAthenaArray();
        A2_inits.DeleteAthenaArray();
        A3_inits.DeleteAthenaArray();

       A1_bound.DeleteAthenaArray();
       A2_bound.DeleteAthenaArray();
       A3_bound.DeleteAthenaArray();

     }

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


         GetKSCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r, &th, &ph);

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
//!\f: User-defined boundary Conditions: sets solution in ghost zones to initial values
// 

void FixedBoundary(MeshBlock *pmb, Coordinates *pcoord, AthenaArray<Real> &prim,
                   FaceField &bb, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh) {
  return;
}



/*
 * -------------------------------------------------------------------
 *     Initialize Mesh
 * -------------------------------------------------------------------
 */
void Mesh::InitUserMeshData(ParameterInput *pin)
{



    if (pin->GetString("mesh","ix1_bc") == "user") EnrollUserBoundaryFunction(INNER_X1, FixedBoundary);
    if (pin->GetString("mesh","ox1_bc") == "user") EnrollUserBoundaryFunction(OUTER_X1, FixedBoundary);
    if (pin->GetString("mesh","ix2_bc") == "user") EnrollUserBoundaryFunction(INNER_X2, FixedBoundary);
    if (pin->GetString("mesh","ox2_bc") == "user") EnrollUserBoundaryFunction(OUTER_X2, FixedBoundary);
    if (pin->GetString("mesh","ix3_bc") == "user") EnrollUserBoundaryFunction(INNER_X3, FixedBoundary);
    if (pin->GetString("mesh","ox3_bc") == "user") EnrollUserBoundaryFunction(OUTER_X3, FixedBoundary);
    

    
    AllocateUserHistoryOutput(1);
    

    if (MAGNETIC_FIELDS_ENABLED){
      EnrollUserHistoryOutput(0, DivergenceB, "divB");
    }


  EnrollUserMetric(Cartesian_GR);

  EnrollUserRadSourceFunction(inner_boundary);
    
    

}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){

    
  dfloor=pin->GetOrAddReal("hydro","dfloor",(1024*(FLT_MIN)));
  pfloor=pin->GetOrAddReal("hydro","pfloor",(1024*(FLT_MIN)));
  if (MAGNETIC_FIELDS_ENABLED) {
    AllocateUserOutputVariables(N_user_vars);
  } else {
    AllocateUserOutputVariables(N_user_vars);
  }
  AllocateRealUserMeshBlockDataField(2);
  ruser_meshblock_data[0].NewAthenaArray(NMETRIC, ie+1);
  ruser_meshblock_data[1].NewAthenaArray(NMETRIC, ie+1);


    r_inner_boundary = 0.;
    loc_list = pmy_mesh->loclist;
    n_mb = pmy_mesh->nbtotal;
    gm_ = 0.0191744; //pin->GetOrAddReal("problem","GM",0.0);
    gm1 = peos->GetGamma() - 1.0;


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

        // Real r,th,ph;

        // GetKSCoordinates(pmb->pcoord->x1f(i), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r, &th, &ph);
        // Real detx1f = (SQR(r) + SQR(a) * SQR(std::cos(th))) * std::abs(std::sin(th));
        // GetKSCoordinates(pmb->pcoord->x1f(i+1), pmb->pcoord->x2v(j),pmb->pcoord->x3v(k), &r, &th, &ph);
        // Real detx1fp1 = (SQR(r) + SQR(a) * SQR(std::cos(th))) * std::abs(std::sin(th));

        // GetKSCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2f(j),pmb->pcoord->x3v(k), &r, &th, &ph);
        // Real detx2f = (SQR(r) + SQR(a) * SQR(std::cos(th))) * std::abs(std::sin(th));
        // GetKSCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2f(j+1),pmb->pcoord->x3v(k), &r, &th, &ph);
        // Real detx2fp1 = (SQR(r) + SQR(a) * SQR(std::cos(th))) * std::abs(std::sin(th));

        // GetKSCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3f(k), &r, &th, &ph);
        // Real detx3f = (SQR(r) + SQR(a) * SQR(std::cos(th))) * std::abs(std::sin(th));
        // GetKSCoordinates(pmb->pcoord->x1v(i), pmb->pcoord->x2v(j),pmb->pcoord->x3f(k+1), &r, &th, &ph);
        // Real detx3fp1 = (SQR(r) + SQR(a) * SQR(std::cos(th))) * std::abs(std::sin(th));
        // divb+=(face1(i+1)*b.x1f(k,j,i+1)*detx1fp1-face1(i)*b.x1f(k,j,i)*detx1f
        //       +face2p(i)*b.x2f(k,j+1,i)*detx2fp1-face2m(i)*b.x2f(k,j,i)*detx2f
        //       +face3p(i)*b.x3f(k+1,j,i)*detx3fp1-face3m(i)*b.x3f(k,j,i)*detx3f);

        // divb_array(k,j,i) = (face1(i+1)*b.x1f(k,j,i+1)*detx1fp1-face1(i)*b.x1f(k,j,i)*detx1f
        //       +face2p(i)*b.x2f(k,j+1,i)*detx2fp1-face2m(i)*b.x2f(k,j,i)*detx2f
        //       +face3p(i)*b.x3f(k+1,j,i)*detx3fp1-face3m(i)*b.x3f(k,j,i)*detx3f);


        divb+= (face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
              +face2p(i)*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i)
              +face3p(i)*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));

        // divb_array(k,j,i)= (face1(i+1)*b.x1f(k,j,i+1)-face1(i)*b.x1f(k,j,i)
        //       +face2p(i)*b.x2f(k,j+1,i)-face2m(i)*b.x2f(k,j,i)
        //       +face3p(i)*b.x3f(k+1,j,i)-face3m(i)*b.x3f(k,j,i));
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


  DivergenceB(pcoord->pmy_block,0);

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

        // user_out_var(6,k,j,i) = A1_bound(k,j,i);
        // user_out_var(7,k,j,i) = A2_bound(k,j,i);
        // user_out_var(8,k,j,i) = A3_bound(k,j,i);
        // user_out_var(9,k,j,i) = divb_array(k,j,i);
      }
    }
  }
  return;
}


/* 
* -------------------------------------------------------------------
*     The actual problem / initial condition setup file
* -------------------------------------------------------------------
*/
void MeshBlock::ProblemGenerator(ParameterInput *pin)
{
  int i=0,j=0,k=0;

    std::string init_file_name;

    int ncells1 = block_size.nx1 + 2*(NGHOST);
    int ncells2 = 1, ncells3 = 1;
    if (block_size.nx2 > 1) ncells2 = block_size.nx2 + 2*(NGHOST);
    if (block_size.nx3 > 1) ncells3 = block_size.nx3 + 2*(NGHOST);
    w_inits.NewAthenaArray(NHYDRO,ncells3,ncells2,ncells1);
    b_inits.x1f.NewAthenaArray( ncells3   , ncells2   ,(ncells1+1));
    b_inits.x2f.NewAthenaArray( ncells3   ,(ncells2+1), ncells1   );
    b_inits.x3f.NewAthenaArray((ncells3+1), ncells2   , ncells1   );
    //divb_array.NewAthenaArray(ncells3,ncells2,ncells1);
    init_file_name =  pin->GetOrAddString("problem","init_filename", "inits.in");

    set_boundary_arrays(init_file_name,block_size,pcoord,is,ie,js,je,ks,ke,w_inits,b_inits);

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



  Real gm1 = peos->GetGamma() - 1.0;
  /* Set up a uniform medium */
  /* For now, make the medium almost totally empty */
  da = 1.0e-8;
  pa = 1.0e-10;
  ua = 0.0;
  va = 0.0;
  wa = 0.0;
  bxa = 1e-4;
  bya = 1e-4;
  bza = 0.0;
  Real x,y,z;

  // Prepare scratch arrays
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC, iu+1);
  gi.NewAthenaArray(NMETRIC, iu+1);
  Real gbl[4][4],vbl[4];
    
    
    Real a = pcoord->GetSpin();

  // Initialize primitive values
  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      pcoord->CellMetric(k, j, il, iu, g, gi);
      for (int i = il; i <= iu; ++i) {
        
        da = w_inits(IDN,k,j,i);
        ua = w_inits(IVX,k,j,i);
        va = w_inits(IVY,k,j,i);
        wa = w_inits(IVZ,k,j,i);
        pa = w_inits(IPR,k,j,i);
        bxa = b_inits.x1f(k,j,i);
        bya = b_inits.x2f(k,j,i);
        bza = b_inits.x3f(k,j,i);
        
        vbl[0] = 1.0;
        vbl[1] = ua;
        vbl[2] = va;
        vbl[3] = wa;

        /* Convert v to 4 velocity */
        Real u0,u1,u2,u3;
        
        Real r,th,ph;
        GetKSCoordinates(pcoord->x1v(i), pcoord->x2v(j),
                                             pcoord->x3v(k), &r, &th, &ph);
        
        
        /* compute 4 velocity given coordinate velocity */
        Real dt_dtau = g(I00,i)*SQR(vbl[0]) + g(I11,i) * SQR(vbl[1]) + g(I22,i)*SQR(vbl[2]) + g(I33,i)*SQR(vbl[3]) +
                  g(I01,i) * (2.0*vbl[1]*vbl[0]) + g(I02,i) * (2.0*vbl[0]*vbl[2]) + g(I03,i) * (2.0*vbl[3]*vbl[0]) + 
                  g(I13,i)*(2.0*vbl[3]*vbl[1]) + g(I12,i) * (2.0*vbl[1]*vbl[2]);
        dt_dtau = 1./std::sqrt(std::fabs(dt_dtau));
        
        u0 = vbl[0] * dt_dtau;
        u1 = vbl[1] * dt_dtau;
        u2 = vbl[2] * dt_dtau;
        u3 = vbl[3] * dt_dtau;
          
          Real ud0,ud1,ud2,ud3;
          pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &ud0, &ud1, &ud2, &ud3);
          Real usq = u0*ud0 + u1*ud1 + u2*ud2 + u3*ud3;
          
          // if (fabs(1.0 + usq)>1e-2 && (r>rh)){
          //     fprintf(stderr,"usq: %g r: %g th: %g\n i j k: %d %d %d  \n",usq,r,th, i,j,k);
          //     fprintf(stderr,"u^mu: %g %g %g %g \n",u0,u1,u2,u3 );
          //     fprintf(stderr,"rho: %g v:%g %g %g p: %g \n",da, vbl[1],vbl[2],vbl[3], pa);

          //     fprintf(stderr,"dt_dtau: %g \n", dt_dtau);

          //     exit(0);
          // }
          Real uu1,uu2,uu3;
          uu1 = u1 - gi(I01,i)/gi(I00,i) * u0;
          uu2 = u2 - gi(I02,i)/gi(I00,i) * u0;
          uu3 = u3 - gi(I03,i)/gi(I00,i) * u0;


        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = da;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pa;
        phydro->w(IVX,k,j,i) = phydro->w1(IM1,k,j,i) = uu1;
        phydro->w(IVY,k,j,i) = phydro->w1(IM2,k,j,i) = uu2;
        phydro->w(IVZ,k,j,i) = phydro->w1(IM3,k,j,i) = uu3;



    // fprintf(stderr,"x y z: %g %g %g \n r th phi: %g %g %g i j k: %d %d %d\n rho: %g \n",
    //               pcoord->x1v(i),pcoord->x2v(j),pcoord->x3v(k),r,th,ph,i,j,k, da);

 
    for (int i_user=0;i_user<N_user_vars; i_user ++){
      user_out_var(i_user,k,j,i) = 0;
    }

if (MAGNETIC_FIELDS_ENABLED){
    pfield->b.x1f(k,j,i) = bxa;
    pfield->b.x2f(k,j,i) = bya;
    pfield->b.x3f(k,j,i) = bza;
    pfield->bcc(IB1,k,j,i) = bxa;
    pfield->bcc(IB2,k,j,i) = bya;
    pfield->bcc(IB3,k,j,i) = bza;
    if (i == iu) pfield->b.x1f(k,j,i+1) = b_inits.x1f(k,j,i+1);;
    if (j == ju) pfield->b.x2f(k,j+1,i) = b_inits.x2f(k,j+1,i);;
    if (k == ku) pfield->b.x3f(k+1,j,i) = b_inits.x3f(k+1,j,i);;


        
}


  }}}

g.DeleteAthenaArray();
gi.DeleteAthenaArray();

  // Calculate cell-centered magnetic field
  AthenaArray<Real> bb;
  if (MAGNETIC_FIELDS_ENABLED) {
    pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, il, iu, jl, ju, kl,
        ku);
  } else {
    bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
  }

  // Initialize conserved values
  if (MAGNETIC_FIELDS_ENABLED) {
    peos->PrimitiveToConserved(phydro->w, pfield->bcc, phydro->u, pcoord, il, iu, jl, ju,
        kl, ku);
  } else {
    peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, il, iu, jl, ju, kl, ku);
    bb.DeleteAthenaArray();
  }

  UserWorkInLoop();

  w_inits.DeleteAthenaArray();
  b_inits.x1f.DeleteAthenaArray();
  b_inits.x2f.DeleteAthenaArray();
  b_inits.x3f.DeleteAthenaArray();


  ///exit(0);


  

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
  g_inv(I22) = eta[2] -f * l_upper[2]*l_upper[2];
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




   // Set x-derivatives of covariant components
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
//----------------------------------------------------------------------------------------
// Function for transforming 4-vector from Boyer-Lindquist to desired coordinates
// Inputs:
//   a0_bl,a1_bl,a2_bl,a3_bl: upper 4-vector components in Boyer-Lindquist coordinates
//   r,theta,phi: Boyer-Lindquist coordinates of point
// Outputs:
//   pa0,pa1,pa2,pa3: pointers to upper 4-vector components in desired coordinates
// Notes:
//   Schwarzschild coordinates match Boyer-Lindquist when a = 0


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

//Transform vector potential, A_\mu, from KS to CKS coordinates assuming A_r = A_theta = 0
// A_\mu (cks) = A_nu (ks)  dx^nu (ks)/dx^\mu (cks) = A_phi (ks) dphi (ks)/dx^\mu
// phi_ks = arctan((r*y + a*x)/(r*x - a*y) ) 
//
static void TransformAphi(Real a3_ks, Real x1,
                     Real x2, Real x3, Real *pa1, Real *pa2, Real *pa3) {

  if (COORDINATE_SYSTEM == "gr_user"){
    Real x = x1;
    Real y = x2;
    Real z = x3;

    Real R = std::sqrt( SQR(x) + SQR(y) + SQR(z) );
    Real r = std::sqrt( SQR(R) - SQR(a) + std::sqrt( SQR(SQR(R) - SQR(a)) + 4.0*SQR(a)*SQR(z) )  )/std::sqrt(2.0);
    Real delta = SQR(r) - 2.0*m*r + SQR(a);
    Real sqrt_term =  2.0*SQR(r)-SQR(R) + SQR(a);

    //dphi/dx =  partial phi/partial x + partial phi/partial r partial r/partial x 
    *pa1 = a3_ks * ( -y/(SQR(x)+SQR(y))  + a*x*r/( (SQR(a)+SQR(r))*sqrt_term ) ); 
    //dphi/dx =  partial phi/partial y + partial phi/partial r partial r/partial y 
    *pa2 = a3_ks * (  x/(SQR(x)+SQR(y))  + a*y*r/( (SQR(a)+SQR(r))*sqrt_term ) ); 
    //dphi/dx =   partial phi/partial r partial r/partial z 
    *pa3 = a3_ks * ( a*z/(r*sqrt_term) );
  }
  else{
          std::stringstream msg;
      msg << "### FATAL ERROR in TransformAphi\n"
          << "this function only works for CKS coordinates"
          <<  std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  return;
}

