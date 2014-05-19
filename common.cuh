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

int const BLOCK_SIZE=512;
int const MaxBondsPerAtom=4;
int const MaxNCPerAtom=128;
int const MaxNeighbors=1024;
int const MaxSoftSphere=16384;

float const NumSteps=2e+7;
int const StartingStateEquilSteps=1000000;
int const SwitchingSteps=10000;
int const SwitchingStride=10;

int const neighfreq=10;
int const outputfreq=1000;
int const trajfreq=10000;

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
	static float const kT=0.59;
	//float h;
	//float zeta;
	static float const hoz=0.001; //timestep over friction constant
	float Gamma;
};

BrDynPar bd_h;
__device__ __constant__ BrDynPar bd_c;

struct SoftSphere {
    static float const eps=1.;
    float Minus6eps;
    static float const Rcut=22.; //For neighbor list, 22 A
    static float const CutOffFactor=3.; // Interaction cut off at 3 sigma
    float Rcut2;
    float CutOffFactor2inv;
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
