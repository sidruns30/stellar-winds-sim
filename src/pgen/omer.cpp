// C++ headers
#include <iostream>   // endl
#include <fstream>
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <cmath>      // sqrt
#include <algorithm>  // min
#include <cstdlib>    // srand
#include <cfloat>     // FLT_MIN

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../athena_arrays.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../hydro/hydro.hpp"
#include "../eos/eos.hpp"
#include "../bvals/bvals.hpp"
#include "../field/field.hpp"
#include "../coordinates/coordinates.hpp"



// Enforce density floor
static Real dfloor = 1e-5;

// Function to set density contour
void SetOmerDensity(int i, int j){
return 0;
}

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real rad, phi, z;
  Real v1, v2, v3;

  //  Initialize density and momenta
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        phydro->u(IDN,k,j,i) = SetOmerDensity(i,j);
        phydro->u(IM1,k,j,i) = 0;
        phydro->u(IM2,k,j,i) = 0;
        phydro->u(IM3,k,j,i) = 0;
        phydro->u(IEN,k,j,i) = 0;
      }
    } 
  }
  return;
}

