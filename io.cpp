#include <stdio.h>
#include <vector_types.h>

void writexyz(FILE* traj, float4* r, int Naa) {
    fprintf(traj,"%d\nAtoms\n",2*Naa);
    for (int i=0;i<Naa;i++) {
        fprintf(traj,"CA %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
    for (int i=Naa;i<2*Naa;i++) {
        fprintf(traj,"CB %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
}

void writexyz(FILE* traj, float4* r, float3 t,int Naa) {
    fprintf(traj,"%d\nAtoms\n",2*Naa);
    for (int i=0;i<Naa;i++) {
        fprintf(traj,"CA %f %f %f\n",r[i].x-t.x,r[i].y-t.y,r[i].z-t.z);
    }
    for (int i=Naa;i<2*Naa;i++) {
        fprintf(traj,"CB %f %f %f\n",r[i].x-t.x,r[i].y-t.y,r[i].z-t.z);
    }
}

void writeforces(FILE* traj, float4* r, int Naa) {
    for (int i=0;i<Naa;i++) {
        fprintf(traj,"%d %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
    for (int i=Naa;i<2*Naa;i++) {
        fprintf(traj,"%d %f %f %f\n",r[i].x,r[i].y,r[i].z);
    }
}
