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
void limit_temperature(MeshBlock *pmb,const Real t, const Real dt_hydro, const AthenaArray<Real> &prim_old, AthenaArray<Real> &prim ){

	//AthenaArray<Real> prim;
	//prim.InitWithShallowCopy(pmb->phydro->w);
	//pmb->peos->ConservedToPrimitive(cons, prim_old, pmb->pfield->b, prim, bcc,
          // pmb->pcoord, pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);
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


	//pmb->peos->PrimitiveToConserved(prim, bcc,
     //  cons, pmb->pcoord,
     //  pmb->is, pmb->ie, pmb->js, pmb->je, pmb->ks, pmb->ke);

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

  Real mdot = 1e-2;
  Real vwind = 1.0;
  Real slab_length = (5.0* pmb->pcoord->dx1f(0) );
  Real slab_vol = (pmb->pmy_mesh->mesh_size.x3max -  pmb->pmy_mesh->mesh_size.x3min) *(pmb->pmy_mesh->mesh_size.x2max -  pmb->pmy_mesh->mesh_size.x2min) * slab_length ; 

   for (int k=pmb->ks; k<=pmb->ke; ++k) {
#pragma omp parallel for schedule(static)
    for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma simd
      for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real x = pmb->pcoord->x1v(i);
          Real y = pmb->pcoord->x2v(j); 

          Real drho = mdot/slab_vol * dt ;
          Real m_slope = -0.5;
          Real y_0 = slab_length/2.0 * std::sqrt(1.0 + SQR(m_slope) );
          Real alpha = std::atan(-1.0/m_slope);

          if ( ( y < m_slope*x + y_0) && y >= m_slope*x ){
             cons(IDN,k,j,i) += drho;
             cons(IM1,k,j,i) += drho * vwind * std::cos(alpha);
             cons(IM2,k,j,i) += drho * vwind * std::sin(alpha);
             cons(IEN,k,j,i) += 0.5 * drho * SQR(vwind);
          }
          else if ( ( y > m_slope*x -y_0) && y<m_slope*x ){
             cons(IDN,k,j,i) += drho;
             cons(IM1,k,j,i) += -drho * vwind * std::cos(alpha);
             cons(IM2,k,j,i) += -drho * vwind * std::sin(alpha);
             cons(IEN,k,j,i) += 0.5 * drho * SQR(vwind);
          }
          



        }
      }
    }

  //apply_inner_boundary_condition(pmb,cons);
  //apply_bondi_boundary(pmb,cons);

	//limit_temperature(pmb,cons,bcc,prim);

  return;
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



/*
 * -------------------------------------------------------------------
 *     Initialize Mesh
 * -------------------------------------------------------------------
 */
void Mesh::InitUserMeshData(ParameterInput *pin)
{

    EnrollUserExplicitSourceFunction(cons_force);
    EnrollUserRadSourceFunction(limit_temperature);
    
    // AllocateUserHistoryOutput(33*3);
    // for (int i = 0; i<33*3; i++){
    //     int i_star = i/3;
    //     int i_pos  = i % 3;
    //     EnrollUserHistoryOutput(i, star_position, "star_"); 
    // }
}

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin){
    

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




    Real DX,DY,DZ;
    get_uniform_box_spacing(pcoord->pmy_block->pmy_mesh->mesh_size,&DX,&DY,&DZ);

    r_inner_boundary_bondi = r_inner_boundary;
    r_outer_boundary_bondi = pcoord->pmy_block->pmy_mesh->mesh_size.x1max - 2.*DX ;

    if (COORDINATE_SYSTEM == "cylindrical" || COORDINATE_SYSTEM == "spherical_polar"){
      r_inner_boundary_bondi = pcoord->pmy_block->block_size.x1min + 2.*(pcoord->pmy_block->block_size.x1max -pcoord->pmy_block->block_size.x1min)/pcoord->pmy_block->block_size.nx1;
      r_outer_boundary_bondi = pcoord->pmy_block->block_size.x1max - 2.*(pcoord->pmy_block->block_size.x1max -pcoord->pmy_block->block_size.x1min)/pcoord->pmy_block->block_size.nx1;
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
  bxa = 1e-4;
  bya = 1e-4;
  bza = 0.0;
  

  for (k=kl; k<=ku; k++) {
  for (j=jl; j<=ju; j++) {
  for (i=il; i<=iu; i++) {

    get_cartesian_coords(pcoord->x1v(i), pcoord->x2v(j), pcoord->x3v(k), &x, &y, &z);

    if ((block_size.nx3)>1) r = sqrt( SQR(x) + SQR(y) + SQR(z));
    else r = sqrt(SQR(x) + SQR(y));

    //get_Bondi(r,&rho,&v,&a);
    //v = 0;
    
    phydro->u(IDN,k,j,i) = da ;
    phydro->u(IM1,k,j,i) = da*ua;
    phydro->u(IM2,k,j,i) = da*va;
    phydro->u(IM3,k,j,i) = da*wa;
      
    

if (MAGNETIC_FIELDS_ENABLED){
    pfield->b.x1f(k,j,i) = bxa;
    pfield->b.x2f(k,j,i) = bya;
    pfield->bcc(IB1,k,j,i) = bxa;
    pfield->bcc(IB2,k,j,i) = bya;
    pfield->bcc(IB3,k,j,i) = bza;
    if (i == ie) pfield->b.x1f(k,j,i+1) = bxa;
    if (j == je) pfield->b.x2f(k,j+1,i) = bya;

      if (COORDINATE_SYSTEM == "cylindrical"){

      //Real theta = pcoord->x2v(j);
      Real phi = pcoord->x2v(j);

      Real bra  = bxa *std::cos(phi) + bya *std::cos(phi);
      //Real btha = bxa * std::cos(theta)*std::cos(phi) + bya * std::cos(theta)*std::cos(phi);
      Real bphia = bxa * std::sin(phi)*-1 + bya * std::cos(phi);

      pfield->b.x1f(k,j,i) = bra;
      pfield->b.x2f(k,j,i) = bphia;
      pfield->bcc(IB1,k,j,i) = bra;
      pfield->bcc(IB2,k,j,i) = bphia;
      pfield->bcc(IB3,k,j,i) = 0;
      if (i == ie) pfield->b.x1f(k,j,i+1) = bra;
      if (j == je) pfield->b.x2f(k,j+1,i) = bphia;
    }
}

    pressure = pa;
#ifndef ISOTHERMAL
    phydro->u(IEN,k,j,i) = pressure/gm1;
if (MAGNETIC_FIELDS_ENABLED){
      phydro->u(IEN,k,j,i) +=0.5*(bxa*bxa + bya*bya + bza*bza);
}
     phydro->u(IEN,k,j,i) += 0.5*rho*(SQR(va) + SQR(ua) + SQR(wa));
#endif /* ISOTHERMAL */

      if (RELATIVISTIC_DYNAMICS)  // this should only ever be SR with this file
        phydro->u(IEN,k,j,i) += da;
 
  }}}
  

}

