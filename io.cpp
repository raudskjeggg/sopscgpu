#include <stdio.h>
#include <vector_types.h>


//Output frame in XYZ format
void writexyz(FILE* traj, float4* r, int Naa) {
    fprintf(traj,"%d\nAtoms\n",2*Naa);
    for (int i=0;i<Naa;i++) {
        fprintf(traj,"CA %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
    for (int i=Naa;i<2*Naa;i++) {
        fprintf(traj,"CB %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
}

//Output frame in XYZ format multiple trajectiories
void writexyz(FILE** traj, float4* r, int Naa, int ntraj) {
    for (int itraj=0; itraj<ntraj; itraj++) {
        fprintf(traj[itraj],"%d\nAtoms\n",2*Naa);
        for (int i=2*Naa*itraj;i<(2*itraj+1)*Naa;i++) {
            fprintf(traj[itraj],"CA %f %f %f\n",r[i].x,r[i].y,r[i].z);
        }
        for (int i=(2*itraj+1)*Naa;i<2*Naa*(itraj+1);i++) {
            fprintf(traj[itraj],"CB %f %f %f\n",r[i].x,r[i].y,r[i].z);
        }
    }
}

//Output frame in XYZ format with translation on vector t (typically, center-of-mass)
void writexyz(FILE* traj, float4* r, float3 t,int Naa) {
    fprintf(traj,"%d\nAtoms\n",2*Naa);
    for (int i=0;i<Naa;i++) {
        fprintf(traj,"CA %f %f %f\n",r[i].x-t.x,r[i].y-t.y,r[i].z-t.z);
    }
    for (int i=Naa;i<2*Naa;i++) {
        fprintf(traj,"CB %f %f %f\n",r[i].x-t.x,r[i].y-t.y,r[i].z-t.z);
    }
}

//Write forces (needed only for debugging)
void writeforces(FILE* traj, float4* r, int Naa) {
    for (int i=0;i<Naa;i++) {
        fprintf(traj,"%d %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
    for (int i=Naa;i<2*Naa;i++) {
        fprintf(traj,"%d %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
}

//Read coordinates from input file
void readcoord(FILE* ind, float4* r, int N) {
    for (int i=0;i<N;i++) {
        if (fscanf(ind,"%f %f %f", &r[i].x,&r[i].y,&r[i].z)==EOF)
            printf("Premature end of file at line %d", i);
        r[i].w=0.;
    }
}

//Read coordinates from input file, multiple trajectories
void readcoord(FILE* ind, float4* r, int N, int ntraj) {
    for (int i=0;i<N;i++) {
        if (fscanf(ind,"%f %f %f", &r[i].x,&r[i].y,&r[i].z)==EOF)
            printf("Premature end of file at line %d", i);
        r[i].w=0.;
        for (int itraj=1; itraj<ntraj; itraj++)
            r[itraj*N+i]=r[i];
    }
}

//Read coordinates from XYZ file
void readxyz(FILE* ind, float4* r, int N) {
    char name[80];
    for (int i=0;i<N;i++) {
        if (fscanf(ind,"%s %f %f %f", name, &r[i].x,&r[i].y,&r[i].z)==EOF)
            printf("Premature end of file at line %d", i);
        r[i].w=0.;
    }
}

//Read coordinates from XYZ file, multiple trajectories
void readxyz(FILE* ind, float4* r, int N, int ntraj) {
    char name[80];
    for (int i=0;i<N;i++) {
        if (fscanf(ind,"%s %f %f %f", name, &r[i].x,&r[i].y,&r[i].z)==EOF)
            printf("Premature end of file at line %d", i);
        r[i].w=0.;
        for (int itraj=1; itraj<ntraj; itraj++)
            r[itraj*N+i]=r[i];
    }
}

//Read external forces from input file, multiple trajectories
void readextforce(FILE* ind, float4* f, int N, int ntraj) {
    
    for (int i=0; i<N*ntraj; i++) {
        f[i].x=0.;
        f[i].y=0.;
        f[i].z=0.;
        f[i].w=0.;
    }
    
    for (int i=0; i<2; i++) {
        int p1; //Indices of attachment beads
        fscanf(ind,"%d",&p1);
        fscanf(ind,"%f %f %f",&f[p1].x,&f[p1].y,&f[p1].z);
        for (int itraj=1; itraj<ntraj; itraj++)
            f[itraj*N+p1]=f[p1];
        printf("Force of {%f %f %f} applied to bead %d\n", f[p1].x,f[p1].y,f[p1].z,p1);
    
    }

}
