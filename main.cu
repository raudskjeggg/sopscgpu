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

void writexyz(FILE** traj, float4* r, int Naa, int ntraj);

void writexyz(FILE* traj, float4* r, float3 t,int Naa);

void writeforces(FILE* traj, float4* r, int Naa);

void readcoord(FILE* ind, float4* r, int N);

void readcoord(FILE* ind, float4* r, int N, int ntraj);

void readxyz(FILE* ind, float4* r, int N);

void readxyz(FILE* ind, float4* r, int N, int ntraj);




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

    
////////// READING INPUT FILE
    char comments[80];
    fscanf(ind,"%s %e",comments,&NumSteps);
    printf("%s %e\n",comments,NumSteps);
    fscanf(ind,"%s %f",comments,&h);
    printf("%s %f\n",comments,h);
    fscanf(ind,"%s %f",comments,&zeta);
    printf("%s %f\n",comments,zeta);
    fscanf(ind,"%s %f",comments,&kT);
    printf("%s %f\n",comments,kT);
    fscanf(ind,"%s %d",comments,&neighfreq);
    printf("%s %d\n",comments,neighfreq);
    fscanf(ind,"%s %d",comments,&outputfreq);
    printf("%s %d\n",comments,outputfreq);
    fscanf(ind,"%s %d",comments,&trajfreq);
    printf("%s %d\n",comments,trajfreq);
    fscanf(ind,"%s %d",comments,&ntraj);
    printf("%s %d\n",comments,ntraj);
    fscanf(ind,"%s %d",comments,&seed);
    printf("%s %d\n",comments,seed);
    fscanf(ind,"%s %d",comments,&BLOCK_SIZE);
    printf("%s %d\n",comments,BLOCK_SIZE);
    
    // Initialize trajectory output files
    FILE **traj;
    traj=(FILE**)malloc(ntraj*sizeof(FILE*));
    for (int itraj=0; itraj<ntraj; itraj++) {
        char itrajstr[3];
        sprintf(itrajstr, "%d", itraj);
        std::string trajfile=filename+"traj"+itrajstr+".xyz";
        if((traj[itraj] = fopen(trajfile.c_str(), "w"))==NULL) {
            printf("Cannot open file %s \n",trajfile.c_str()) ;
            exit(1) ;
        }
    }

    
    int Naa;    //Total number of amino acid residues
    fscanf(ind,"%d",&Naa);
    printf("Number of amino acid residues: %d\n",Naa);
    int N=2*Naa*ntraj; //Number of beads
    
// Read bonds and build map, allocate and copy to device
    int Nb; //Number of bonds
    fscanf(ind,"%d",&Nb);
    InteractionListBond bondlist(ind,N/ntraj,MaxBondsPerAtom,Nb,"covalent bonds",ntraj);

// Read native contacts and build map for initial structure, allocate and copy to device
    int Nnc;  //Number of native contacts (initial)
    fscanf(ind,"%d",&Nnc);
    InteractionListNC nclist(ind,N/ntraj,MaxNCPerAtom,Nnc,"native contacts (starting)",ntraj);
    
// Read native contacts and build map for target structure, allocate and copy to device
    int Nnc2;       //Number of native contacts (target)
    fscanf(ind,"%d",&Nnc2);
    InteractionListNC nclist2(ind,N/ntraj,MaxNCPerAtom,Nnc2,"native contacts (target)",ntraj);
    
//Read sigmas for non-native and neighboring soft sphere repulsion
    float *sig_h, *sig_d;
    sig_h=(float*)malloc(N*sizeof(float));
    cudaMalloc((void**)&sig_d,N*sizeof(float));
    ss_h.MaxSigma=0.;
    for (int i=0; i<N/ntraj; i++) {
        if (fscanf(ind,"%f", &sig_h[i])==EOF)
            printf("Premature end of file at line %d", i);
        for (int itraj=1; itraj<ntraj; itraj++)
            sig_h[itraj*N/ntraj+i]=sig_h[i];
        if (sig_h[i]>ss_h.MaxSigma)
            ss_h.MaxSigma=sig_h[i];
    }
    cudaMemcpy(sig_d, sig_h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaBindTexture(0, sig_t, sig_d, N*sizeof(float));
    
// Read salt bridges
    //Number of salt bridges
    int Nsb;
    fscanf(ind,"%d",&Nsb);
    InteractionListSB SaltBridgeList(ind,N/ntraj,MaxNeighbors,Nsb,"electrostatic interactions",ntraj);
    
//Allocate coordinates arrays on device and host
    float4 *r_h,*r_d;
    cudaMallocHost((void**)&r_h, N*sizeof(float4));
    cudaMalloc((void**)&r_d, N*sizeof(float4));
    
// Read starting coordinates
    
    ////readcoord(ind, r_h, N);
    //readcoord(ind, r_h, N/ntraj, ntraj);
    
//READ FROM SEPARATE FILE
    FILE *initl;
   std::string initlfilename="start.init";
    if((initl = fopen(initlfilename.c_str(), "r"))==NULL) {
        printf("Cannot open file %s \n",initlfilename.c_str()) ;
        exit(1) ;
    }
    //readxyz(initl, r_h, N);
    readxyz(initl, r_h, N/ntraj, ntraj);
    fclose(initl);
    
    //Copy coordinates to device
    cudaMemcpy(r_d, r_h, N*sizeof(float4), cudaMemcpyHostToDevice);
    cudaBindTexture(0, r_t, r_d, N*sizeof(float4));
    
//Allocate forces arrays on device <and host>
    float4 *f_d;
	cudaMalloc((void**)&f_d, N*sizeof(float4));
    
    //float4 *f_h;
    //cudaMallocHost((void**)&f_h, N*sizeof(float4));

    fclose(ind);
//////////////END READING INPUT FILE//////


//Initialize Brownian Dynamics integrator parameters
	bd_h.kT=kT;
    bd_h.hoz=h/zeta;
    bd_h.Gamma=sqrt(2*(bd_h.hoz)*(bd_h.kT));
    cudaMemcpyToSymbol(bd_c, &bd_h, sizeof(BrDynPar), 0, cudaMemcpyHostToDevice);
    checkCUDAError("Brownian dynamics parameters init");
    
    
//Initialize Soft Sphere repulsion force field parameters;
    ss_h.Minus6eps=-6.0*ss_h.eps;
    ss_h.Rcut2=ss_h.Rcut*ss_h.Rcut;
    ss_h.CutOffFactor2inv=1.0f/ss_h.CutOffFactor/ss_h.CutOffFactor;
    ss_h.CutOffFactor6inv=ss_h.CutOffFactor2inv*ss_h.CutOffFactor2inv*ss_h.CutOffFactor2inv;
    ss_h.CutOffFactor8inv=ss_h.CutOffFactor6inv*ss_h.CutOffFactor2inv;
    cudaMemcpyToSymbol(ss_c, &ss_h, sizeof(SoftSphere), 0, cudaMemcpyHostToDevice);
    checkCUDAError("Soft sphere parameters init");
    
//Initialize FENE parameters
    fene_h.R02=fene_h.R0*fene_h.R0;
    fene_h.kR0=fene_h.R0*fene_h.k;
    cudaMemcpyToSymbol(fene_c, &fene_h, sizeof(FENE), 0, cudaMemcpyHostToDevice);
    checkCUDAError("FENE parameters init");
    
//Initialize electrostatic parameters
    cudaMemcpyToSymbol(els_c, &els_h, sizeof(ElStatPar), 0, cudaMemcpyHostToDevice);
    checkCUDAError("Electrostatic parameters init");
    
    
//Neighbor list allocate
    InteractionList<int> nl;
    nl.N=N;
    nl.Nmax=MaxNeighbors;
    nl.AllocateOnDevice("neighbor list");
    nl.AllocateOnHost();
    //cudaBindTexture(0, neibmap_t, nl.map_d, nl.N*nl.Nmax*sizeof(int));
    
    
    
//Simulation
    
    int THREADS=BLOCK_SIZE;
    int BLOCKS=N/THREADS+1;
    
    //Allocate and initialize random seeds
    curandStatePhilox4_32_10_t *RNGStates_d;
    cudaMalloc( (void **)&RNGStates_d, THREADS*BLOCKS*sizeof(curandStatePhilox4_32_10_t) );
    checkCUDAError("Brownian dynamics seeds allocation");
    rand_init<<<BLOCKS,THREADS>>>(seed,RNGStates_d);
    checkCUDAError("Random number initializion");
   
    printf("t\tTraj#\tE_TOTAL\t\tE_POTENTIAL\tE_SoftSpheres\tE_NatCont\tE_ElStat\tE_FENE\t\t~TEMP\t<v>*neighfreq/DeltaRcut\n");
    //float Delta=0.;
    int stride=neighfreq;
    for (int t=0;t<NumSteps;t+=stride) {
        
        bool CoordCopiedToHost=false;
        
        if ((t % neighfreq)==0) {
            SoftSphereNeighborListMultTraj<<<BLOCKS,THREADS>>>(r_d,nl,N/ntraj);
            checkCUDAError("Neighbor List");
        }
        
//        nl.CopyToHost("Neighbor List");
//        for (int i=0; i<N; i++) {
//            
//            int Nneib=nl.count_h[i];                                                  //Number of neighbors of the i-th bead
//            printf("%d, %d neibs: ",i,Nneib);
//            for (int ineib=0;ineib<Nneib;ineib++)                                    //Loop over neighbors of the i-th bead
//                printf("%d ",nl.map_h[ineib*nl.N+i]);
//            printf("\n");
//        }
        
        if ((t % outputfreq)==0) {
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Ekin");
            CoordCopiedToHost=true;
            
            float* Ekin;
            Ekin=(float*)calloc(ntraj,sizeof(float));
            for (int itraj=0; itraj<ntraj; itraj++) {
                for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
                    Ekin[itraj]+=r_h[i].w;
            }
            
            FENEEnergy<<<BLOCKS,THREADS>>>(r_d,bondlist);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Efene");
            
            float* Efene;
            Efene=(float*)calloc(ntraj,sizeof(float));
            for (int itraj=0; itraj<ntraj; itraj++) {
                for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
                    Efene[itraj]+=r_h[i].w;
            }
            
            SoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nl,sig_d);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Ess");
            
            float* Ess;
            Ess=(float*)calloc(ntraj,sizeof(float));
            for (int itraj=0; itraj<ntraj; itraj++) {
                for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
                    Ess[itraj]+=r_h[i].w;
            }
            
            
            NativeSubtractSoftSphereEnergy<<<BLOCKS,THREADS>>>(r_d,nclist,sig_d);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Enss");
            
            for (int itraj=0; itraj<ntraj; itraj++) {
                for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
                    Ess[itraj]+=r_h[i].w;
            }
            
            
            NativeEnergy<<<BLOCKS,THREADS>>>(r_d,nclist);
            cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            checkCUDAError("Copy coordinates back for Enat");
            
            float* Enat;
            Enat=(float*)calloc(ntraj,sizeof(float));
            for (int itraj=0; itraj<ntraj; itraj++) {
                for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
                    Enat[itraj]+=r_h[i].w;
            }
            
            //DebyeHuckelEnergy<<<BLOCKS,THREADS>>>(r_d,SaltBridgeList);
            //cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
            //checkCUDAError("Copy coordinates back for Eel");
            
            float* Eel;
            Eel=(float*)calloc(ntraj,sizeof(float));
            //for (int itraj=0; itraj<ntraj; itraj++) {
            //    for (int i=itraj*N/ntraj; i<(itraj+1)*N/ntraj; i++)
            //        Eel[itraj]+=r_h[i].w;
            //}
            float* Epot;
            Epot=(float*)malloc(ntraj*sizeof(float));
            float* Etot;
            Etot=(float*)malloc(ntraj*sizeof(float));
            
            for (int itraj=0; itraj<ntraj; itraj++) {
                Epot[itraj]=(Efene[itraj]+Ess[itraj]+Enat[itraj]+Eel[itraj])/2.;
                Etot[itraj]=Epot[itraj]+Ekin[itraj];
                printf("%d\t%d\t%e\t%e\t%e\t%e\t%e\t%e\t%f\t%f\n",t,itraj,Etot[itraj],Epot[itraj],Ess[itraj]/2.,Enat[itraj]/2.,Eel[itraj]/2.,Efene[itraj]/2.,Ekin[itraj]*ntraj/(N*6.*bd_h.hoz/503.6),sqrt(Ekin[itraj]*ntraj/N)*neighfreq/(ss_h.Rcut-ss_h.MaxSigma*ss_h.CutOffFactor));
            }
            
        }
        
        if ((t % trajfreq)==0) {
            if (!CoordCopiedToHost) {
                cudaMemcpy(r_h, r_d, N*sizeof(float4), cudaMemcpyDeviceToHost);
                checkCUDAError("Copy coordinates back");
                CoordCopiedToHost=true;
            }
            writexyz(traj,r_h,Naa,ntraj);
            
        }
        
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
            
            //DebyeHuckelForce<<<BLOCKS,THREADS>>>(r_d,f_d,SaltBridgeList);
            //checkCUDAError("DebyeHuckel");
            
            integrate<<<BLOCKS,THREADS>>>(r_d,f_d,N,RNGStates_d);
            checkCUDAError("Integrate");
            
        }
        
        /*if (t>StartingStateEquilSteps+SwitchingSteps)  {
            Delta=1.;
        } else if (t>StartingStateEquilSteps)  {
            Delta=(float)SwitchingStride/(float)SwitchingSteps*(float)((int)(t-StartingStateEquilSteps)/(int)SwitchingStride);
        } else {
            Delta=0.;
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
            
            printf("%d\t%e\t%e\t%e\t%e\t%e\t%e\t%f\t%f\n",t,Etot,Epot,Ess/2.,Enat/2.,Eel/2.,Efene/2.,Ekin/(N*6.*bd_h.hoz/503.6),sqrt(Ekin/N)*neighfreq/(ss_h.Rcut-ss_h.MaxSigma*ss_h.CutOffFactor));
            
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
            //writexyz(traj,r_h,Naa);
            writexyz(traj,r_h,Naa,ntraj);
            
        }
        
        if (t==StartingStateEquilSteps) {
            bd_h.hoz=.1*h/zeta;
            bd_h.Gamma=sqrt(2*(bd_h.hoz)*(bd_h.kT));
            cudaMemcpyToSymbol(bd_c, &bd_h, sizeof(BrDynPar), 0, cudaMemcpyHostToDevice);
        }
        
        if (t==StartingStateEquilSteps+SwitchingSteps) {
            bd_h.hoz=h/zeta;
            bd_h.Gamma=sqrt(2*(bd_h.hoz)*(bd_h.kT));
            cudaMemcpyToSymbol(bd_c, &bd_h, sizeof(BrDynPar), 0, cudaMemcpyHostToDevice);
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
        }*/
    }
    

    
    cudaFree(r_d);
    cudaFree(f_d);
    nclist.FreeOnDevice("native contacts");
    bondlist.FreeOnDevice("bonds");
    SaltBridgeList.FreeOnDevice("salt bridges");
    nl.FreeOnDevice("neighbor list");
    cudaDeviceReset();
}

