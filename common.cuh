//
//  common.cuh
//  
//
//  Created by Pavel Zhuravlev on 5/8/14.
//
//

#ifndef _common_cuh
#define _common_cuh

#include<stdio.h>
#include "cuda.h"

int BLOCK_SIZE=512; //For launching GPU kernels

//For creating neighbor lists, bond list, native contact list
int const MaxBondsPerAtom=4;
int const MaxNCPerAtom=128;
int const MaxNeighbors=1024;
int const MaxSoftSphere=8192;

float h=0.05; //timestep
float zeta=50.; //friction coefficient
float kT=0.59; //temperature

float NumSteps=3e+4; //Total steps in trajectory
int const StartingStateEquilSteps=20000;//1000000; //For transition, starting structure
int const SwitchingSteps=10000;//10000; // Time for switching Hamiltionian
int const SwitchingStride=10;

int neighfreq=10;
int outputfreq=1000;
int trajfreq=10000;
int ntraj=1;
int seed=1234;

int extforces;  //Whether external forces are applied to any atoms

struct bond { //For bond map
    int i2;
    float l0;
};

struct __align__(16) nc { //For native contact map
    int i2;
    float r02;
    float factor; //12*epsilon/r0^2
    float epsilon;
};


struct BrDynPar {
	float kT;
	//float h;
	//float zeta;
	float hoz; //timestep over friction constant
	float Gamma;
};

BrDynPar bd_h;
__device__ __constant__ BrDynPar bd_c;

struct SoftSphere {
    static float const eps=1.;
    float Minus6eps;
    static float const Rcut=22.; //For neighbor list, 22 A
    static float const RcutOuter=66.; //For outer neighbor list
    static float const CutOffFactor=3.; // Interaction cut off at 3 sigma
    float Rcut2;
    float Rcut2Outer;
    float CutOffFactor2inv;
    float CutOffFactor6inv;
    float CutOffFactor8inv;
    float MaxSigma;
};

SoftSphere ss_h;
__device__ __constant__ SoftSphere ss_c;

struct ElStatPar {
    static float const kappainv=2.4; // Angstrom, from Changbong's 2006 PNASs
    //static float const prefactor=330.4; // kcal/mol/A = 1 Coulomb^2/4*pi*eps0 = 1 statCoulomb^2
    //static float const dielectricepsilon=10.; // from Changbong's 2006 PNASs
    static float const prefactor=33.04;
   // static float const Rcut=5.;
};

ElStatPar els_h;
__device__ __constant__ ElStatPar els_c;

struct FENE {
    static float const k=20.;
    static float const R0=2.;
    float kR0; //k*R_0
    float R02;  //R0^2
};

FENE fene_h;
__device__ __constant__ FENE fene_c;

int chainstarts_h[100];
__device__ __constant__ int chainstarts_c[100];


texture<float4, 1, cudaReadModeElementType> r_t;
texture<float, 1, cudaReadModeElementType> sig_t;
//texture<int, 1, cudaReadModeElementType> neibmap_t;

void checkCUDAError(const char *msg){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
        printf("CUDA error: %s: %s.\n", msg, cudaGetErrorString(error));
        exit(0);
	}
}


#endif
