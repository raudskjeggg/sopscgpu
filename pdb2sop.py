#!/usr/bin/python2

from Bio.PDB import *
from numpy import *
import re
import sys

# if len(sys.argv)<4:
# 	print "Usage: initialstructure.pdb finalstructure.pdb inputfile.sopscgpu"
# 	exit(1)
# outputname=sys.argv[3]
# PDBname2=sys.argv[2]
# PDBname=sys.argv[1]

if len(sys.argv)<2:
	print "Usage: structure.pdb inputfile.sopscgpu"
	exit(1)
outputname=sys.argv[2]
PDBname=sys.argv[1]

print "PDB 1 ",PDBname
#print "PDB 2 ",PDBname2
print "SOP-SC GPU input file ", outputname

#Protein residue names
resnames=["GLY", "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "SER", "THR", "ASN", "GLN", "TYR", "TRP", "ASP", "GLU", "HSE", "HSD", "HIS", "LYS", "ARG", "CYS"]

#Backbone atoms
BBA=["CA","N","C","O","HA","HN","1H","2H","3H","H","2HA","HA3","HT1","HT2","HT3","OT1","OT2","OXT"]

charged=["GLU","ASP","ARG","LYS","HIS"]
q=dict()
qlist=[-1,-1,-1,1,1] #respective charges in list charged
for i,res in enumerate(charged): 
	q[res]=qlist[i]

#dielectricepsilon=10.
#elstatprefactor=(4.8*4.8*6.02/4184.*1e+2)/dielectricepsilon #kcal/mol
 
#SOP-SC parameters. el is actually defined in the simulation code right now
ebb=0.55
ess=0.3
ebs=0.4
#el=1.
GoCut=8.
GoCutsq=GoCut**2


#Get CA and sidechain-center-of-mass positions (to lists cas and cbs) from the PDB structure. Get native contant and salt bridges lists
def pdb2sop(structure,cas,casv,cbs,cbsv,terres,seq,ncs,sbs):
	# for model in structure:
	for chain in structure[0]:
		for residue in chain:
			if residue.get_resname() in resnames:
				seq.append(residue.get_resname())
				ca=residue['CA']
				#cas.append(list(ca.get_vector()))
				cas.append(ca.get_coord())
				casv.append(ca.get_vector())
				#cm=Vector(0,0,0)
				m=0
				cm=zeros(3)
				for atom in residue:
					if not atom.get_name() in BBA:
						cm+=atom.get_coord()*atom.mass
						#cm+=atom.get_vector().left_multiply(atom.mass)
						m+=atom.mass
					if (atom.get_name()=='CB') or (atom.get_name()=='HA1'):
						cb=atom.get_coord()
				cm/=m
				#cbg=cm
				cbg=cm
				#print cb,cm
				#cbs.append(list(cbg))
				cbs.append(cbg)
				cbsv.append(Vector(cbg))
		terres.append(len(cas)-1);  #Terminal residues
	Naa=len(cas); #Number of residues
#Native contacts and salt-bridges
	for i in range(Naa):
	        print "Residue %d/%d\r" % (i,Naa)
		for j in range(i,Naa):
			if (j-i)>2:
				if ((casv[i]-casv[j]).normsq()<GoCutsq):
					ncs.append([i,j,(casv[i]-casv[j]).norm(),ebb])
				if ((cbsv[i]-cbsv[j]).normsq()<GoCutsq):
					ncs.append([i+Naa,j+Naa,(cbsv[i]-cbsv[j]).norm(),ess*fabs(BT[seq[i]][seq[j]]-.7)])
				if ((casv[i]-cbsv[j]).normsq()<GoCutsq):
					ncs.append([i,j+Naa,(casv[i]-cbsv[j]).norm(),ebs])
				if ((cbsv[i]-casv[j]).normsq()<GoCutsq):
					ncs.append([i+Naa,j,(cbsv[i]-casv[j]).norm(),ebs])
				if q.has_key(seq[i]) and q.has_key(seq[j]):
					sbs.append([i+Naa,j+Naa,q[seq[i]]*q[seq[j]]])
	return 1




# Read the Betancourt-Thirumalai matrix (see Betancourt M. R. & Thirumalai, D. (1999). Protein Sci. 8(2),361-369. doi:10.1110/ps.8.2.361)
BT=dict()
f=open('tb.dat')
aas=re.split(' ',f.readline().strip())[1:]
for i in range(len(aas)):
	if not BT.has_key(aas[i]):
		BT[aas[i]]=dict();
	l=re.split(' ',f.readline().strip())[1:]
	for j in range(i,len(aas)):
		BT[aas[i]][aas[j]]=double(l[j])
		if not BT.has_key(aas[j]):
			BT[aas[j]]=dict()
		BT[aas[j]][aas[i]]=double(l[j])
f.close

#Read the van der Waals diameters of the side chains
sbb=3.8
sss=dict()
f=open('aavdw.dat')
for l in f:
	s=re.split(' ',l)
	sss[s[0]]=2.*double(s[1])
print sss
	

parser=PDBParser()

#Get CAs, CBs, native contacts and salt bridges for initial structure
structure=parser.get_structure('Starting',PDBname)
cas=[];casv=[];cbs=[];cbsv=[];terres=[];seq=[];ncs=[];sbs=[]
pdb2sop(structure,cas,casv,cbs,cbsv,terres,seq,ncs,sbs);

# #Get CAs, CBs, native contacts and salt bridges for final structure
# structure=parser.get_structure('Final',PDBname2)
# cas2=[];casv2=[];cbs2=[];cbsv2=[];terres2=[];seq2=[];ncs2=[];sbs2=[]
# pdb2sop(structure,cas2,casv2,cbs2,cbsv2,terres2,seq2,ncs2,sbs2)	

print "Native Contacts in starting structure: ", len(ncs)
#print "Native Contacts in final structure: ", len(ncs2)
print "Salt bridges: ", len(sbs)

Naa=len(cas); #Number of residues
Nch=len(terres); #Number of chains
Nb=2*Naa-Nch; #Number of bonds in SOP-SC. Each residue has two bonds, except for Nch terminal residues
#Nb=Naa-Nch; #Number of bonds in SOP. Each residue has a bond, except for Nch terminal residues			


#Output everything to sopsc-gpu input file
f=open(outputname,'w')
f.write("NumSteps 3e+7\n")
f.write("Timestep(h) 0.05\n")
f.write("Friction(zeta) 50.\n")
f.write("Temperature 0.59\n")
f.write("NeighborListUpdateFrequency 10\n")
f.write("OutputFrequency 1000\n")
f.write("TrajectoryWriteFrequency 10000\n")
f.write("Trajectories 1\n")
f.write("RandomSeed 1234\n")
f.write("KernelBlockSize 512\n")
f.write("ExternalForces 1\n")

f.write("%d\n" % Naa) #Number of residues

f.write("%d\n" % Nch)  #Number of chains
#Chain starts
for ter in terres[:-1]:
	f.write("%d\n" % (ter+1))

f.write("%d\n" % Nb)  #Number of bonds

#Bonds
for i in range(Naa):
	f.write("%d %d %f\n" % (i,i+Naa,(casv[i]-cbsv[i]).norm()))
	if not i in terres:
		f.write("%d %d %f\n" % (i,i+1,(casv[i]-casv[i+1]).norm()))

#Native contacts of starting structure
f.write("%d\n" % len(ncs))
for nc in ncs:
	f.write("%d %d %f %f\n" % (nc[0],nc[1],nc[2],nc[3]))

# #Native contacts of final structure
# f.write("%d\n" % len(ncs2))
# for nc in ncs2:
# 	f.write("%d %d %f %f\n" % (nc[0],nc[1],nc[2],nc[3]))

#Sigmas for soft sphere repulsion				
for aa in seq:
	f.write('%f\n' % sbb)
for aa in seq:
	f.write('%f\n' % sss[aa])

# #Exclusions from soft shpere interactions (additional to bonded beads: ss of neighboring residues, bs to ss of neighboring residues)
# f.write("%d\n" % (2*(Naa-0*Nch)))
# for i in range(Naa):
# 	#f.write("%d %d\n" % (i,i+Naa))
# 	#if not i in terres:
# 		#f.write("%d %d\n" % (i,i+1))
# 	f.write("%d %d %f\n" % (i,i+Naa+1,0))
# 	f.write("%d %d %f\n" % (i+Naa,i+Naa+1,0))

#Salt bridges	
f.write("%d\n" % len(sbs))
for sb in sbs:
	f.write("%d %d %f\n" % (sb[0],sb[1],sb[2]))
	
#External forces
point1=Naa;
point2=106;#2*Naa-1;
force=15. #pN
f.write("%d\n" % point1) 
f.write("%f %f %f\n" % (force/70.,0,0))
f.write("%d\n" % point2) 
f.write("%f %f %f\n" % (-force/70.,0,0))

#Starting coordinates
for ac in vstack([array(cas),array(cbs)]):
	f.write("%f %f %f\n" % (ac[0],ac[1],ac[2]))
	
f.close
