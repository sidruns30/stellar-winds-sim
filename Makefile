# Template for Athena++ Makefile
# The 'configure.py' python script uses this template to create the actual Makefile

# Files for conditional compilation

PROBLEM_FILE = torus_bfield.cpp
COORDINATES_FILE = cartesian.cpp
EOS_FILE = adiabatic_hydro.cpp
RSOLVER_FILE = hlle.cpp
RSOLVER_DIR = hydro/
MPIFFT_FILE =  

# General compiler specifications

CXX := mpicxx
CPPFLAGS := 
CXXFLAGS := -O3 -std=c++11 -ipo -xhost -inline-forceinline -qopenmp-simd -qopt-prefetch=4 -diag-disable 3180
LDFLAGS := 
LDLIBS :=  -lhdf5

# Preliminary definitions

EXE_DIR := bin/
EXECUTABLE := $(EXE_DIR)athena
SRC_FILES := $(wildcard src/*.cpp) \
	     $(wildcard src/bvals/*.cpp) \
	     $(wildcard src/coordinates/*.cpp) \
	     src/eos/$(EOS_FILE) \
	     $(wildcard src/field/*.cpp) \
	     $(wildcard src/hydro/*.cpp) \
	     $(wildcard src/hydro/srcterms/*.cpp) \
	     $(wildcard src/hydro/hydro_diffusion/*.cpp) \
	     $(wildcard src/field/field_diffusion/*.cpp) \
	     src/hydro/rsolvers/$(RSOLVER_DIR)$(RSOLVER_FILE) \
	     $(wildcard src/mesh/*.cpp) \
	     $(wildcard src/outputs/*.cpp) \
	     $(wildcard src/reconstruct/*.cpp) \
	     $(wildcard src/task_list/*.cpp) \
	     $(wildcard src/utils/*.cpp) \
	     $(wildcard src/fft/*.cpp) \
	     $(wildcard src/multigrid/*.cpp) \
	     $(wildcard src/gravity/*.cpp) \
	     $(MPIFFT_FILE) \
	     src/pgen/$(PROBLEM_FILE) \
	     src/pgen/default_pgen.cpp
OBJ_DIR := obj/
OBJ_FILES := $(addprefix $(OBJ_DIR),$(notdir $(SRC_FILES:.cpp=.o)))
SRC_DIR := $(dir $(SRC_FILES) $(PROB_FILES))
VPATH := $(SRC_DIR)

# Generally useful targets

.PHONY : all dirs clean

all : dirs $(EXECUTABLE)

objs : dirs $(OBJ_FILES)

dirs : $(EXE_DIR) $(OBJ_DIR)

$(EXE_DIR):
	mkdir -p $(EXE_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Link objects into executable

$(EXECUTABLE) : $(OBJ_FILES)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $(OBJ_FILES) $(LDFLAGS) $(LDLIBS)

# Create objects from source files

$(OBJ_DIR)%.o : %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Cleanup

clean :
	rm -rf $(OBJ_DIR)*
	rm -rf $(EXECUTABLE)
