//#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
#include <curand_kernel.h>
#include <string>
#include <iostream>

#include "common.cuh"
#include "InteractionList.cuh"
#include "kernels.cu"

void writexyz(FILE* traj, float4* r, int Naa);
void writexyz(FILE* traj, float4* r, float3 t,int Naa);
void writeforces(FILE* traj, float4* r, int Naa);


int main(int argc, char *argv[]){
    
    if (argc<2) {
        std::string progname=argv[0];
        printf("Usage: %s inputfilename\n",progname.c_str());
        exit(1);
    }
    
   cudaDeviceProp deviceProp;
   cudaGetDeviceProperties(&deviceProp, 0);
   cudaSetDevice(0);
   cudaDeviceReset();    

    std::string filename=argv[1];
    
    FILE *ind;
    if((ind = fopen(filename.c_str(), "r"))==NULL) {
        printf("Cannot open file %s \n",filename.c_str()) ;
        exit(1) ;
    }
    
    // Initialize trajectory output file
    FILE *traj;
    std::string trajfile=filename+"traj.xyz";
    if((traj = fopen(trajfile.c_str(), "w"))==NULL) {
        printf("Cannot open file %s \n",trajfile.c_str()) ;
        exit(1) ;
    }

    
//Total number of amino acid residues    
    int Naa;
    fscanf(ind,"%d",&Naa);
    int N=2*Naa; //Number of beads
    
//Number of bonds
    int Nb;
    fscanf(ind,"%d",&Nb);
    
    InteractionList<bond> bondlist(ind,N,MaxBondsPerAtom,Nb);
//    InteractionList<bond> bondlist;
//    bondlist.N=N;
//    bondlist.Nmax=MaxBondsPerAtom;
//    
//    bondlist.AllocateInteractionListOnDevice("bonds");
//    bondlist.AllocateInteractionListOnHost();
//    
//// Read bonds and their lengths. Build bonds map. If each bead has a first, second etc... bond, the structure of array is: N first bonds, then N second bonds etc. (see InteractionList class description) This following loop constructs the map from list of bonds.
//
//    printf("Reading bond list\n");
//    for (int k=0; k<Nb; k++) {
//        int i,j;
//        bond bk;
//        if (fscanf(ind,"%d %d %f", &i,&j,&(bk.l0))==EOF)
//            printf("Premature end of file at line %d", k);
//        
//        bk.i2=j;
//        bondlist.map_h[N*bondlist.count_h[i]+i]=bk;
//        bondlist.count_h[i]++;
//        
//        bk.i2=i;
//        bondlist.map_h[N*bondlist.count_h[j]+j]=bk;
//        bondlist.count_h[j]++;
//        
//        bondlist.CheckNmaxHost(i,"covalent bonds");
//        bondlist.CheckNmaxHost(j,"covalent bonds");
//    }
//    
//// Copy bonds data to device
//    bondlist.CopyInteractionListToDevice("bond");
//    bondlist.FreeInteractionListOnHost();

    fene_h.R02=fene_h.R0*fene_h.R0;
    fene_h.kR0=fene_h.R0*fene_h.k;
    cudaMemcpyToSymbol(fene_c, &fene_h, sizeof(FENE), 0, cudaMemcpyHostToDevice);
    checkCUDAError("Bonds init");
    
    
// Read native contacts and build map
    //Number of native contacts
    int Nnc;
    fscanf(ind,"%d",&Nnc);
    
    InteractionList<nc> nclist;
    nclist.N=N;
    nclist.Nmax=MaxNCPerAtom;
    
    nclist.AllocateInteractionListOnDevice("native contacts");
    nclist.AllocateInteractionListOnHost();
    
    printf("Reading native contacts\n");
    for (int k=0; k<Nnc; k++) {
        int i,j;
        float r0,eps;
        if (fscanf(ind,"%d %d %f %f", &i,&j,&r0,&eps)==EOF)
            printf("Premature end of file at line %d", k);
        
        nc nck;
        nck.r02=r0*r0;
        nck.factor=12.0*eps/r0/r0;
        nck.epsilon=eps;
        
        nck.i2=j;
        nclist.map_h[N*nclist.count_h[i]+i]=nck;
        nclist.count_h[i]++;
        
        nck.i2=i;
        nclist.map_h[N*nclist.count_h[j]+j]=nck;
        nclist.count_h[j]++;
        
        nclist.CheckNmaxHost(i,"native contacts");
        nclist.CheckNmaxHost(j,"native contacts");
    }
    
// Copy native contacts data to device
    nclist.CopyInteractionListToDevice("native contacts");
    nclist.FreeInteractionListOnHost();

// Read native contacts and build map
    //Number of native contacts
    int Nnc2;
    fscanf(ind,"%d",&Nnc2);
    
    InteractionList<nc> nclist2;
    nclist2.N=N;
    nclist2.Nmax=MaxNCPerAtom;
    
    nclist2.AllocateInteractionListOnDevice("native contacts 2");
    nclist2.AllocateInteractionListOnHost();
    
    printf("Reading %d native contacts 2\n", Nnc2);
    for (int k=0; k<Nnc2; k++) {
        int i,j;
        float r0,eps;
        if (fscanf(ind,"%d %d %f %f", &i,&j,&r0,&eps)==EOF)
                 printf("Premature end of file at line %d", k);

        nc nck;
        nck.r02=r0*r0;
        nck.factor=12.0*eps/r0/r0;
        nck.epsilon=eps;
        
        nck.i2=j;
        nclist2.map_h[N*nclist2.count_h[i]+i]=nck;
        nclist2.count_h[i]++;
        
        nck.i2=i;
        nclist2.map_h[N*nclist2.count_h[j]+j]=nck;
        nclist2.count_h[j]++;
        
        nclist2.CheckNmaxHost(i,"native contacts 2");
        nclist2.CheckNmaxHost(j,"native contacts 2");
    }
    
    // Copy native contacts data to device
    nclist2.CopyInteractionListToDevice("native contacts 2");
    nclist2.FreeInteractionListOnHost();
    
//Read sigmas for non-native and neighboring soft sphere repulsion
    float *sig_h, *sig_d;
    sig_h=(float*)malloc(N*sizeof(float));
    cudaMalloc((void**)&sig_d,N*sizeof(float));
    for (int i=0; i<N; i++) {
        if (fscanf(ind,"%f", &sig_h[i])==EOF)
        printf("Premature end of file at line %d", i);
    }
    cudaMemcpy(sig_d, sig_h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTexture(0, sig_t, sig_d, N*sizeof(float));
    
// Read salt bridges
    //Number of salt bridges
    int Nsb;
    fscanf(ind,"%d",&Nsb);
    
    InteractionList<bond> SaltBridgeList;
    SaltBridgeList.N=N;
    SaltBridgeList.Nmax=MaxNeighbors;
    
    SaltBridgeList.AllocateInteractionListOnDevice("electrostatic interactions");
    SaltBridgeList.AllocateInteractionListOnHost();
    
    printf("Reading % d salt bridges\n",Nsb);
    for (int k=0; k<Nsb; k++) {
        int i,j;
        float qiqj;
        bond sbk;
        if (fscanf(ind,"%d %d %f", &i,&j,&qiqj)==EOF)
            printf("Premature end of file at line %d", k);
        
        sbk.l0=els_h.prefactor*qiqj;
        sbk.i2=j;
        
        SaltBridgeList.map_h[N*SaltBridgeList.count_h[i]+i]=sbk;
        SaltBridgeList.count_h[i]++;
        
        sbk.i2=i;
        SaltBridgeList.map_h[N*SaltBridgeList.count_h[j]+j]=sbk;
        SaltBridgeList.count_h[j]++;
        
        SaltBridgeList.CheckNmaxHost(i,"electrostatic interactions");
        SaltBridgeList.CheckNmaxHost(j,"electrostatic interactions");
    }
    
    // Copy salt bridge data to device
    SaltBridgeList.CopyInteractionListToDevice("electrostatic interactions");
    SaltBridgeList.FreeInteractionListOnHost();
    
    cudaMemcpyToSymbol(els_c, &els_h, sizeof(ElStatPar), 0, cudaMemcpyHostToDevice);

    
    int THREADS=BLOCK_SIZE;
    int BLOCKS=N/THREADS+1;
    
    

//Initialize Brownian Dynamics integrator    
    bd_h.Gamma=sqrt(2*(bd_h.hoz)*(bd_h.kT));
    cudaMemcpyToSymbol(bd_c, &bd_h, sizeof(BrDynPar), 0, cudaMemcpyHostToDevice);
    
    curandStatePhilox4_32_10_t *RNGStates_d;
    
    cudaMalloc( (void **)&RNGStates_d, THREADS*BLOCKS*sizeof(curandStatePhilox4_32_10_t) );
    
    rand_init<<<BLOCKS,THREADS>>>(RNGStates_d);
    checkCUDAError("Random number initializion");
    
//Initialize Soft Sphere repulsion;
    ss_h.Minus6eps=-6.0*ss_h.eps;
    ss_h.Rcut2=ss_h.Rcut*ss_h.Rcut;
    ss_h.CutOffFactor2inv=1.0f/ss_h.CutOffFactor/ss_h.CutOffFactor;
    cudaMemcpyToSymbol(ss_c, &ss_h, sizeof(SoftSphere), 0, cudaMemcpyHostToDevice);
    
    
    
//Neighbor list allocate
    InteractionList<int> nl;
    nl.N=N;
    nl.Nmax=MaxNeighbors;
    nl.AllocateInteractionListOnDevice("neighbor list");
    nl.AllocateInteractionListOnHost();
    
    //cudaMemcpyToSymbol(nl_c, &nl, sizeof(InteractionList<int>), 0, cudaMemcpyHostToDevice);
    //cudaBindTexture(0, neibmap_t, nl.map_d, nl.N*nl.Nmax*sizeof(int));
    
//Allocate coordinates arrays on device and host
    float4 *r_h,*r_d;
    cudaMallocHost((void**)&r_h, N*sizeof(float4));
    cudaMalloc((void**)&r_d, N*sizeof(float4));
    
// Read starting coordinates
    for (int i=0;i<N;i++) {
        if (fscanf(ind,"%f %f %f", &r_h[i].x,&r_h[i].y,&r_h[i].z)==EOF)
            printf("Premature end of file at line %d", i);
        r_h[i].w=0.;
    }

//Copy coordinates to device
    cudaMemcpy(r_d, r_h, N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaBindTexture(0, r_t, r_d, N*sizeof(float4));
    
//Allocate and initialize forces arrays on device <and host>
    float4 *f_d;
	cudaMalloc((void**)&f_d, N*sizeof(float4));
    
    float4 *f_h;
    cudaMallocHost((void**)&f_h, N*sizeof(float4));

//Simulation
   
    printf("t\tE_TOTAL\t\tE_POTENTIAL\tE_SoftSpheres\tE_NatCont\tE_ElStat\tE_FENE\t\t~TEMP\n");
    float Delta=0.;
    int stride=neighfreq;
    for (int t=0;t<NumSteps;t+=stride) {
        
        if (t>StartingStateEquilSteps+SwitchingSteps)  {
            Delta=1.;
        } else if (t>StartingStateEquilSteps)  {
            Delta=(float)SwitchingStride/(float)SwitchingSteps*(float)((int)(t-StartingStateEquilSteps)/(int)SwitchingStride);
        } else {
            Delta=0.;
        }
        
        if (t==StartingStateEquilSteps) {
            bd_h.Gamma=sqrt(0.1*2*(bd_h.hoz)*(bd_h.kT));
            cudaMemcpyToSymbol(bd_c, &bd_h, sizeof(BrDynPar), 0, cudaMemcpyHostToDevice);
        }
        
        if (t==StartingStateEquilSteps+SwitchingSteps) {
            bd_h.Gamma=sqrt(2*(bd_h.hoz)*(bd_h.kT));
            cudaMemcpyToSymbol(bd_c, &bd_h, sizeof(BrDynPar), 0, cudaMemcpyHostToDevice);
        }
        
        
        bool CoordCopiedToHost=false;
        
        if ((t % neighfreq)==0) {
            SoftSphereNeighborList<<<BLOCKS,THREADS>>>(r_d,nl);
            checkCUDAError("Neighbor List");
        }
        
        
        if ((t % outputfreq)==0) {
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Ekin");
            CoordCopiedToHost=true;
            
            float Ekin=0.; for (int i=0; i<N; i++) Ekin+=r_h[i].w;
            
            FENEEnergy<<<BLOCKS,THREADS>>>(r_d,bondlist);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Efene");
            
            float Efene=0.; for (int i=0; i<N; i++) Efene+=r_h[i].w;
            
            SoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nl,sig_d);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Ess");
            
            float Ess=0.; for (int i=0; i<N; i++) Ess+=r_h[i].w;
            
            if (t<StartingStateEquilSteps+SwitchingSteps) {
            
                NativeSubtractSoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nclist,sig_d);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Enss");

                for (int i=0; i<N; i++) Ess+=(1.-Delta)*r_h[i].w;
            }
            
            if (t>StartingStateEquilSteps) {
            
                NativeSubtractSoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nclist2,sig_d);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Enss2");
            
                for (int i=0; i<N; i++) Ess+=(Delta)*r_h[i].w;
            }
            
            float Enat=0.;
            if (t<StartingStateEquilSteps+SwitchingSteps) {
            
                NativeEnergy<<<BLOCKS,THREADS>>>(r_d,nclist);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Enat");
            
                for (int i=0; i<N; i++) Enat+=(1.-Delta)*r_h[i].w;
            }
            
            if (t>StartingStateEquilSteps) {
            
                NativeEnergy<<<BLOCKS,THREADS>>>(r_d,nclist2);
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back for Enat2");
            
                for (int i=0; i<N; i++) Enat+=(Delta)*r_h[i].w;
            }

            DebyeHuckelEnergy<<<BLOCKS,THREADS>>>(r_d,SaltBridgeList);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Eel");
            
            float Eel=0.; for (int i=0; i<N; i++) Eel+=r_h[i].w;
            
            float Epot=(Efene+Ess+Enat+Eel)/2.;
            float Etot=Epot+Ekin;
            
            printf("%d\t%e\t%e\t%e\t%e\t%e\t%e\t%f\n",t,Etot,Epot,Ess/2.,Enat/2.,Eel/2.,Efene/2.,Ekin/(N*6.*bd_h.hoz/503.5));
            
        }
        
        if ((t % trajfreq)==0) {
            if (!CoordCopiedToHost) {
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back");
                CoordCopiedToHost=true;
            }
            float3 com=make_float3(0.,0.,0.);
            for (int i=0; i<N; i++) {
                com.x+=r_h[i].x;
                com.y+=r_h[i].y;
                com.z+=r_h[i].z;
            }
            com.x/=N;com.y/=N;com.z/=N;
            //writeforces(traj,f_h,Naa);
            writexyz(traj,r_h,Naa);
        }
        
        if (t>StartingStateEquilSteps+SwitchingSteps)  {
            for (int tongpu=0; tongpu<stride; tongpu++) {
                
                force_flush<<<BLOCKS,THREADS>>>(f_d,N);
                checkCUDAError("Force flush");
                
                FENEForce<<<BLOCKS,THREADS>>>(r_d,f_d,bondlist);
                checkCUDAError("FENE");
                
                SoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nl,sig_d);
                checkCUDAError("SoftSphere");
                
                NativeSubtractSoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist2,sig_d);
                checkCUDAError("Native subtract Soft Sphere 2");
                
                NativeForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist2);
                checkCUDAError("Native 2");
                
                DebyeHuckelForce<<<BLOCKS,THREADS>>>(r_d,f_d,SaltBridgeList);
                checkCUDAError("DebyeHuckel");
                
                integrate<<<BLOCKS,THREADS>>>(r_d,f_d,N,RNGStates_d);
                checkCUDAError("Integrate");
                
            }
        } else if (t>StartingStateEquilSteps)  {
            for (int tongpu=0; tongpu<stride; tongpu++) {
                
                force_flush<<<BLOCKS,THREADS>>>(f_d,N);
                checkCUDAError("Force flush");
                
                FENEForce<<<BLOCKS,THREADS>>>(r_d,f_d,bondlist);
                checkCUDAError("FENE");
                
                SoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nl,sig_d);
                checkCUDAError("SoftSphere");
                
                NativeSubtractSoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist2,sig_d,Delta);
                checkCUDAError("Native subtract Soft Sphere 2");
                
                NativeForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist2,Delta);
                checkCUDAError("Native 2");
                
                NativeSubtractSoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist,sig_d,1.-Delta);
                checkCUDAError("Native subtract Soft Sphere");
                
                NativeForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist,1.-Delta);
                checkCUDAError("Native");
                
                DebyeHuckelForce<<<BLOCKS,THREADS>>>(r_d,f_d,SaltBridgeList);
                checkCUDAError("DebyeHuckel");
                
                integrate<<<BLOCKS,THREADS>>>(r_d,f_d,N,RNGStates_d);
                checkCUDAError("Integrate");
                
            }
        } else {
            for (int tongpu=0; tongpu<stride; tongpu++) {
                
                force_flush<<<BLOCKS,THREADS>>>(f_d,N);
                checkCUDAError("Force flush");
                
                FENEForce<<<BLOCKS,THREADS>>>(r_d,f_d,bondlist);
                checkCUDAError("FENE");
                
                SoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nl,sig_d);
                checkCUDAError("SoftSphere");
                
                NativeSubtractSoftSphereForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist,sig_d);
                checkCUDAError("Native subtract Soft Sphere");
                
                NativeForce<<<BLOCKS,THREADS>>>(r_d,f_d,nclist);
                checkCUDAError("Native");
                
                DebyeHuckelForce<<<BLOCKS,THREADS>>>(r_d,f_d,SaltBridgeList);
                checkCUDAError("DebyeHuckel");
                
                integrate<<<BLOCKS,THREADS>>>(r_d,f_d,N,RNGStates_d);
                checkCUDAError("Integrate");
                
            }
        }
    }
    

    
    cudaFree(r_d);
    cudaFree(f_d);
    nclist.FreeInteractionListOnDevice("native contacts");
    bondlist.FreeInteractionListOnDevice("bonds");
    SaltBridgeList.FreeInteractionListOnDevice("salt bridges");
    nl.FreeInteractionListOnDevice("neighbor list");
    cudaDeviceReset();
}

