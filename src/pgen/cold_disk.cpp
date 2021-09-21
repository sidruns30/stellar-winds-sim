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



/* Initialize a couple of the key variables used throughout */
//Real r_inner_boundary = 0.;         /* remove mass inside this radius */
Real r_min_inits = 1e15; /* inner radius of the grid of initial conditions */
static Real G = 4.48e-9;      /* Gravitational constant G (in problem units) */
Real gm_;               /* G*M for point mass at origin */
Real gm1;               /* \gamma-1 (adiabatic index) */
Real N_cells_per_radius;  /* Number of cells that are contained in one stellar radius (this is defined in terms of the longest length 
across a cell ) */
double SMALL = 1e-20;       /* Small number for numerical purposes */
LogicalLocation *loc_list;              /* List of logical locations of meshblocks */
int n_mb = 0; /* Number of meshblocks */
int max_smr_level = 0;
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

bool amr_increase_resolution; /* True if resolution is to be increased from restarted run */





static void integrate_cool(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim )
{
  int i, j, k, kprime;
  int is, ie, js, je, ks, ke;


  ///apply_inner_boundary_condition(pmb,prim);

  Real kbT_keV;
  AthenaArray<Real> prim_before,cons;


    // Allocate memory for primitive/conserved variables
  int ncells1 = pmb->block_size.nx1 + 2*(NGHOST);
  int ncells2 = 1, ncells3 = 1;
  if (pmb->block_size.nx2 > 1) ncells2 = pmb->block_size.nx2 + 2*(NGHOST);
  if (pmb->block_size.nx3 > 1) ncells3 = pmb->block_size.nx3 + 2*(NGHOST);
  prim_before.NewAthenaArray(NHYDRO,ncells3,ncells2,ncells1);
  cons.NewAthenaArray(NHYDRO,ncells3,ncells2,ncells1);
  prim_before = prim;
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

        /* find temp in keV */
        // kbT_keV = mu_highT*mp_over_kev*(prim(IPR,k,j,i)/prim(IDN,k,j,i));
        // // ath_pout(0, "temperature before = %e ", kbT_keV);
        // kbT_keV = newtemp_townsend(prim(IDN,k,j,i), kbT_keV, dt_hydro);
        // // ath_pout(0, "temperature after = %e \n", kbT_keV);
        // // apply a temperature floor (nans tolerated) 
        // if (isnan(kbT_keV) || kbT_keV < kbTfloor_kev)
        //   kbT_keV = kbTfloor_kev;

        // prim(IPR,k,j,i) = prim(IDN,k,j,i) * kbT_keV / (mu_highT * mp_over_kev);

        // Real v_s = sqrt(gamma*prim(IPR,k,j,i)/prim(IDN,k,j,i));

        // if (v_s>cs_max) v_s = cs_max;
        // if ( fabs(prim(IVX,k,j,i)) > cs_max) prim(IVX,k,j,i) = cs_max * ( (prim(IVX,k,j,i) >0) - (prim(IVX,k,j,i)<0) ) ;
        // if ( fabs(prim(IVY,k,j,i)) > cs_max) prim(IVY,k,j,i) = cs_max * ( (prim(IVY,k,j,i) >0) - (prim(IVY,k,j,i)<0) ) ;
        // if ( fabs(prim(IVZ,k,j,i)) > cs_max) prim(IVZ,k,j,i) = cs_max * ( (prim(IVZ,k,j,i) >0) - (prim(IVZ,k,j,i)<0) ) ;

        //  prim(IPR,k,j,i) = SQR(v_s) *prim(IDN,k,j,i)/gamma ;
       // cons(IEN,k,j,i) = prim(IPR,k,j,i)*igm1 + 0.5*prim(IDN,k,j,i)*( SQR(prim(IVX,k,j,i)) + SQR(prim(IVY,k,j,i)) + SQR(prim(IVZ,k,j,i)) );
          
      }
    }
  }




  apply_inner_boundary_condition(pmb,prim);



   // pmb->peos->PrimitiveToConserved(prim, pmb->pfield->bcc,
   //     cons, pmb->pcoord,
   //     pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);

   //  for (k=ks; k<=ke; k++) {
   //  for (j=js; j<=je; j++) {
   //    for (i=is; i<=ie; i++) {

   //                if ( isnan(cons(IDN,k,j,i)) || isnan(cons(IM1,k,j,i)) || isnan(cons(IM2,k,j,i)) || isnan(cons(IM3,k,j,i)) || isnan(cons(IEN,k,j,i)) ){
              
   //            int m_max;
   //            if (MAGNETIC_FIELDS_ENABLED) m_max = NHYDRO+ NFIELD;
   //            else m_max = NHYDRO;
   //            for (int m=0; m < (m_max); ++m){
   //                fprintf(stderr,"m = %d \n ----------------\n",m);
     
                          
   //                        fprintf(stderr, "k,j,i: %d %d %d  prim: %g prim_old: %g \n", k,j,i , prim(m,k,j,i),prim_before(m,k,j,i));
   //                }              
              
   //            exit(0);
   //        }

   //    }}}

      prim_before.DeleteAthenaArray();
      cons.DeleteAthenaArray();
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


void get_minimum_cell_lengths(const Coordinates * pcoord, Real *dx_min, Real *dy_min, Real *dz_min){
    
        //loc_list = pcoord->pmy_block->pmy_mesh->loclist; 
        RegionSize block_size;
        enum BoundaryFlag block_bcs[6];
        //int n_mb = pcoord->pmy_block->pmy_mesh->nbtotal;
    
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
         star is located in */
        for (int j=0; j<n_mb; j++) {
            pcoord->pmy_block->pmy_mesh->SetBlockSizeAndBoundaries(loc_list[j], block_size, block_bcs);
            
            get_uniform_box_spacing(block_size,&DX,&DY,&DZ);
            
            if (DX < *dx_min) *dx_min = DX;
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



 //----------------------------------------------------------------------------------------
//!\f: User-defined boundary Conditions: sets solution in ghost zones to initial values
// 

void DirichletInnerX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  Real x,y,z;
  Real vx,vy,vz,p,rho;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {


        prim(IDN,k,j,is-i) = pmb->phydro->w_bound(IDN,k,j,is-i);
        prim(IVX,k,j,is-i) = pmb->phydro->w_bound(IVX,k,j,is-i);
        prim(IVY,k,j,is-i) = pmb->phydro->w_bound(IVY,k,j,is-i);
        prim(IVZ,k,j,is-i) = pmb->phydro->w_bound(IVZ,k,j,is-i);
        prim(IPR,k,j,is-i) = pmb->phydro->w_bound(IPR,k,j,is-i);

        if (prim(IVX,k,j,is-i)<0) prim(IVX,k,j,is-i) = 0;
    }
  }
}
}

void DirichletOuterX1(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  Real x,y,z;
  Real vx,vy,vz,p,rho;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=(NGHOST); ++i) {


          prim(IDN,k,j,ie+i) = pmb->phydro->w_bound(IDN,k,j,ie+i);
          prim(IVX,k,j,ie+i) = pmb->phydro->w_bound(IVX,k,j,ie+i);
          prim(IVY,k,j,ie+i) = pmb->phydro->w_bound(IVY,k,j,ie+i);
          prim(IVZ,k,j,ie+i) = pmb->phydro->w_bound(IVZ,k,j,ie+i);
          prim(IPR,k,j,ie+i) = pmb->phydro->w_bound(IPR,k,j,ie+i);

          if (prim(IVX,k,j,ie+i)>0) prim(IVX,k,j,ie+i) = 0;


      }
    }
  }
}

void DirichletInnerX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  Real x,y,z;
  Real vx,vy,vz,p,rho;
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=(NGHOST); ++j) {
      for (int i=is; i<=ie; ++i) {


          prim(IDN,k,js-j,i) = pmb->phydro->w_bound(IDN,k,js-j,i);
          prim(IVX,k,js-j,i) = pmb->phydro->w_bound(IVX,k,js-j,i);
          prim(IVY,k,js-j,i) = pmb->phydro->w_bound(IVY,k,js-j,i);
          prim(IVZ,k,js-j,i) = pmb->phydro->w_bound(IVZ,k,js-j,i);
          prim(IPR,k,js-j,i) = pmb->phydro->w_bound(IPR,k,js-j,i);


          if (prim(IVY,k,js-j,i)<0)  prim(IVY,k,js-j,i) = 0;

      }
    }
  }
}

void DirichletOuterX2(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  Real x,y,z;
  Real vx,vy,vz,p,rho;
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=(NGHOST); ++j) {
      for (int i=is; i<=ie; ++i) {


          prim(IDN,k,je+j,i) = pmb->phydro->w_bound(IDN,k,je+j,i);
          prim(IVX,k,je+j,i) = pmb->phydro->w_bound(IVX,k,je+j,i);
          prim(IVY,k,je+j,i) = pmb->phydro->w_bound(IVY,k,je+j,i);
          prim(IVZ,k,je+j,i) = pmb->phydro->w_bound(IVZ,k,je+j,i);
          prim(IPR,k,je+j,i) = pmb->phydro->w_bound(IPR,k,je+j,i);

         if (prim(IVY,k,je+j,i)>0) prim(IVY,k,je+j,i) = 0.;
      }
    }
  }
}

void DirichletInnerX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  Real x,y,z;
  Real vx,vy,vz,p,rho;
  for (int k=1; k<=(NGHOST); ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {


          prim(IDN,ks-k,j,i) = pmb->phydro->w_bound(IDN,ks-k,j,i);
          prim(IVX,ks-k,j,i) = pmb->phydro->w_bound(IVX,ks-k,j,i);
          prim(IVY,ks-k,j,i) = pmb->phydro->w_bound(IVY,ks-k,j,i);
          prim(IVZ,ks-k,j,i) = pmb->phydro->w_bound(IVZ,ks-k,j,i);
          prim(IPR,ks-k,j,i) = pmb->phydro->w_bound(IPR,ks-k,j,i);


      }
    }
  }
}

void DirichletOuterX3(MeshBlock *pmb,Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                 Real time, Real dt, int is, int ie, int js, int je, int ks, int ke)
{
  Real x,y,z;
  Real vx,vy,vz,p,rho;
  for (int k=1; k<=(NGHOST); ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {

          prim(IDN,ke+k,j,i) = pmb->phydro->w_bound(IDN,ke+k,j,i);
          prim(IVX,ke+k,j,i) = pmb->phydro->w_bound(IVX,ke+k,j,i);
          prim(IVY,ke+k,j,i) = pmb->phydro->w_bound(IVY,ke+k,j,i);
          prim(IVZ,ke+k,j,i) = pmb->phydro->w_bound(IVZ,ke+k,j,i);
          prim(IPR,ke+k,j,i) = pmb->phydro->w_bound(IPR,ke+k,j,i);

          if(prim(IVZ,ke+k,j,i)>0) prim(IVZ,ke+k,j,i) = 0.;

      }
    }
  }
}

/* Apply inner "inflow" boundary conditions */

void apply_inner_boundary_condition(MeshBlock *pmb,AthenaArray<Real> &prim){


  Real v_ff = std::sqrt(2.*gm_/(pmb->r_inner_boundary+SMALL));
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
              pmb->user_out_var(16,k,j,i) += bsq/bsq_rho_ceiling - prim(IDN,k,j,i);
              prim(IDN,k,j,i) = bsq/bsq_rho_ceiling;
             }

            }

          if (r < pmb->r_inner_boundary){
              
              
              Real rho_flr = 1e-7;
              Real p_floor = 1e-10;
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
            

              /* Prevent outflow from inner boundary */ 
              if (prim(IVX,k,j,i)*x/r >0 ) prim(IVX,k,j,i) = 0.;
              if (prim(IVY,k,j,i)*y/r >0 ) prim(IVY,k,j,i) = 0.;
              if (prim(IVZ,k,j,i)*z/r >0 ) prim(IVZ,k,j,i) = 0.;

              
              
          }



}}}



}



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
    


    // EnrollUserRadSourceFunction(integrate_cool);


    
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
    
    
    r_inner_boundary = 0.;
    loc_list = pmy_mesh->loclist;
    n_mb = pmy_mesh->nbtotal;
    gm_ = pin->GetOrAddReal("problem","GM",0.0);
    gm1 = peos->GetGamma() - 1.0;


    int N_cells_per_boundary_radius = pin->GetOrAddInteger("problem", "boundary_radius", 2);

    

    // Real dx_min,dy_min,dz_min;
    // get_minimum_cell_lengths(pcoord, &dx_min, &dy_min, &dz_min);
    
    // if (block_size.nx3>1)       r_inner_boundary = N_cells_per_boundary_radius * std::max(std::max(dx_min,dy_min),dz_min); // r_inner_boundary = 2*sqrt( SQR(dx_min) + SQR(dy_min) + SQR(dz_min) );
    // else if (block_size.nx2>1)  r_inner_boundary = N_cells_per_boundary_radius * std::max(dx_min,dy_min); //2*sqrt( SQR(dx_min) + SQR(dy_min)               );
    // else                        r_inner_boundary = N_cells_per_boundary_radius * dx_min;
    
    // Real v_ff = std::sqrt(2.*gm_/(r_inner_boundary+SMALL))*10.;
    // cs_max = std::min(cs_max,v_ff);
    

    
    
    
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

  Real rmin_disk = 4.0;
  Real rmax_disk = 15.0; 


  Real gm1 = peos->GetGamma() - 1.0;
  /* Set up a uniform medium */
  /* For now, make the medium almost totally empty */
  da = 1.0e-8;
  pa = 1.0e-10;
  ua = 0.0;
  va = 0.0; //keplerian
  wa = 0.0;
  bxa = 0.0;
  bya = 0.0;
  bza = 0.0;
  Real x,y,z;

  for (k=kl; k<=ku; k++) {
  for (j=jl; j<=ju; j++) {
  for (i=il; i<=iu; i++) {

    // if (is_zoom_in==1){
        
    //     da = phydro->w_bound(IDN,k,j,i);
    //     ua = phydro->w_bound(IVX,k,j,i);
    //     va = phydro->w_bound(IVY,k,j,i);
    //     wa = phydro->w_bound(IVZ,k,j,i);
    //     pa = phydro->w_bound(IPR,k,j,i);

    //   phydro->w(IDN,k,j,i) = da;
    //   phydro->w(IVX,k,j,i) = ua;
    //   phydro->w(IVY,k,j,i) = va;
    //   phydro->w(IVZ,k,j,i) = wa;
    //   phydro->w(IPR,k,j,i) = pa;

        
    //   }

    if ( (pcoord->x1v(i) >= rmin_disk) && (pcoord->x1v(i)<=rmax_disk) ){
      da = 1.0;
      wa = std::sqrt(gm_/pcoord->x1v(i));
    } 
    else{
      da = 0.0;
      wa = 0.0;
    } 

    wa = std::sqrt(gm_/pcoord->x1v(i));


    phydro->u(IDN,k,j,i) = da;
    phydro->u(IM1,k,j,i) = da*ua;
    phydro->u(IM2,k,j,i) = da*va;
    phydro->u(IM3,k,j,i) = da*wa;



    pressure = pa;
#ifndef ISOTHERMAL
    phydro->u(IEN,k,j,i) = pressure/gm1;

     phydro->u(IEN,k,j,i) += 0.5*da*(ua*ua + va*va + wa*wa);
#endif /* ISOTHERMAL */

      if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
        phydro->u(IEN,k,j,i) += da;
 
  }}}
    



  UserWorkInLoop();


  

}



