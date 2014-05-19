//
//  InteractionList.cuh
//  
//
//  Created by Pavel Zhuravlev on 5/8/14.
//
//
/*
 Interaction lists for FENE, native contacs, electrostatics, neighbor lists etc.
 N is number of particles that can possibly engage in this interaction;
 Nmax is the maximum number of counterparts of each particle;
 Each i-th of the N particles has count[i] counterparts in interaction: first, second ... count[i]
 map is array organized as follows: N first counterparts (with parameters) for each of N particles; then N second counterparts etc.
 map[j*N+i] is thus the record for j-th counterpart of the i-th particle.
 map is allocated for the size N*Nmax*sizeof(T)
 */

#ifndef ____InteractionList__
#define ____InteractionList__

#include "common.cuh"

template<typename T>
class InteractionList {
    
public:
    T* map_h;
    int* count_h;
    
    T* map_d;
    int* count_d;
    
    int N;
    int Nmax;
    
    void AllocateInteractionListOnHost() {
        
        map_h=(T*)malloc(Nmax*N*sizeof(T));
        count_h=(int*)calloc(N, sizeof(int));
        
    }
    
    void FreeInteractionListOnHost() {
        
        free(map_h);
        free(count_h);
        
    }
    
    void AllocateInteractionListOnDevice(std::string s) {
        
        cudaMalloc((void**)&map_d, Nmax*N*sizeof(T));
        checkCUDAError(("Allocate "+s+" map").c_str());
        
        cudaMalloc((void**)&count_d, N*sizeof(int));
        checkCUDAError(("Allocate "+s+" count").c_str());
        
    }
    
    void FreeInteractionListOnDevice(std::string s) {
        
        cudaFree(map_d);
        checkCUDAError(("Free "+s+" map").c_str());
        
        cudaFree(count_d);
        checkCUDAError(("Free "+s+" count").c_str());
        
    }
    
    void CopyInteractionListToDevice(std::string s) {
        
        cudaMemcpy(map_d, map_h, Nmax*N*sizeof(T), cudaMemcpyHostToDevice);
        checkCUDAError(("Copy "+s+" map").c_str());
        
        cudaMemcpy(count_d, count_h, N*sizeof(int), cudaMemcpyHostToDevice);
        checkCUDAError(("Copy "+s+" count").c_str());
        
    }
    
    void CopyInteractionListToHost(std::string s) {
        
        cudaMemcpy(map_h, map_d, Nmax*N*sizeof(T), cudaMemcpyDeviceToHost);
        checkCUDAError(("Copy "+s+" map").c_str());
        
        cudaMemcpy(count_h, count_d, N*sizeof(int), cudaMemcpyDeviceToHost);
        checkCUDAError(("Copy "+s+" count").c_str());
        
    }
    
    void CheckNmaxHost (int i, std::string s) {
        if (count_h[i] > Nmax) {
            printf(("ERROR: Maximum number of "+s+" exceeded the limit of %d.\n").c_str(), Nmax);
            exit(0);
        }
    }
    
    InteractionList<T>(FILE *ind,int N_in, int Nmax_in, int Nb) {
        
        N=N_in; Nmax=Nmax_in;
        AllocateInteractionListOnDevice("bonds");
        AllocateInteractionListOnHost();
        
        // Read bonds and their lengths. Build bonds map. If each bead has a first, second etc... bond, the structure of array is: N first bonds, then N second bonds etc. (see InteractionList class description) This following loop constructs the map from list of bonds.
        
        printf("Reading bond list\n");
        for (int k=0; k<Nb; k++) {
            int i,j;
            bond bk;
            if (fscanf(ind,"%d %d %f", &i,&j,&(bk.l0))==EOF)
                printf("Premature end of file at line %d", k);
            
            bk.i2=j;
            map_h[N*count_h[i]+i]=bk;
            count_h[i]++;
            
            bk.i2=i;
            map_h[N*count_h[j]+j]=bk;
            count_h[j]++;
            
            CheckNmaxHost(i,"covalent bonds");
            CheckNmaxHost(j,"covalent bonds");
        }
        
        // Copy bonds data to device
        CopyInteractionListToDevice("bond");
        FreeInteractionListOnHost();
    }
    
    InteractionList<T>() {
        
    };

    
};

class InteractionListBond:InteractionList<bond> {
    
};

//__device__ __constant__ InteractionList<int> nl_c;

#endif /* defined(____InteractionList__) */
