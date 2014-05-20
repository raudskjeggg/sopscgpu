#!/usr/bin/python2

import sys
from Bio.PDB import *
from numpy import *
import re
import os

if len(sys.argv)<4:
	print "Usage: xyz2fullpdb input.xyz output.pdb structure.pdb [firstframe lastframe]"
	exit(1)

trajname=sys.argv[1]
outputname=sys.argv[2]
PDBname=sys.argv[3]
if len(sys.argv)>4:
	firstframe=int(sys.argv[4])
	lastframe=int(sys.argv[5])
else:
	firstframe=0
	lastframe=100

print "PDB ",PDBname
print "XYZ Trajectory", trajname
print "PDB Trajectory", outputname
print "Fisrt frame",firstframe
print "Last frame",lastframe


def readframexyz(f,atoms):
	f.readline();f.readline();	
	k=[double(array(re.findall(r'[-+]?[0-9]*\.?[0-9]+e?[+-]?[0-9]{0,3}',f.readline()))) for i in range(0,atoms)]
	a=array(k)
	return a;
	
def writeframexyz(f,atoms):
	f.write("%d\n" % len(atoms))
	f.write("Atoms\n")
	for a in atoms:
		f.write("%s %f %f %f\n" % ('C', a[0], a[1], a[2]))

#Protein residue names
resnames=["GLY", "ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "SER", "THR", "ASN", "GLN", "TYR", "TRP", "ASP", "GLU", "HSE", "HSD", "HIS", "LYS", "ARG", "CYS"]

#Backbone atoms
BBA=["CA","N","C","O","HA","HN","1H","2H","3H","H","2HA","HA3","HT1","HT2","HT3","OT1","OT2","OXT"]

parser=PDBParser()
structure=parser.get_structure('rnap',PDBname)

#Collect positions of CA and sidechain center-of-mass.
cas=[]
casv=[]
cbs=[]
cbsv=[]
terres=[]
seq=[]
# for model in structure:
for chain in structure[0]:
	for residue in chain:
		if residue.get_resname() in resnames:
			seq.append(residue.get_resname())
			ca=residue['CA']
			cas.append(list(ca.get_vector()))
			casv.append(ca.get_vector())
				
			cm=Vector(0,0,0);m=0;
			for atom in residue:
				if not atom.get_name() in BBA:
					cm+=atom.get_vector().left_multiply(atom.mass)
					m+=atom.mass
					
			cm/=m
			cbg=cm
			cbs.append(list(cbg))
			cbsv.append(cbg)
	terres.append(len(cas)-1);

#structurePDB=structure.copy()
	
Naa=len(cas); #Number of residues
Nch=len(terres); #Number of chains
Nb=2*Naa-Nch; #Number of bonds in SOP-SC. Each residue has two bonds, except for Nch terminal residues
#Nb=Naa-Nch; #Number of bonds in SOP. Each residue has a bond, except for Nch terminal residues

f=open(trajname)
io=PDBIO()
os.system('rm '+outputname)
#skip to first frame
for i in range((2*Naa+2)*firstframe):
	f.readline()
Nframes=lastframe-firstframe;
for i in range(Nframes):
	#parser=PDBParser()
	structure=parser.get_structure('rnap',PDBname)
	io.set_structure(structure)
	print "Frame %d/%d\r" % (i,Nframes)
	r=readframexyz(f,2*Naa)
	i=0
	for chain in structure[0]:
		for residue in chain:
			if residue.get_resname() in resnames:
				for atom in residue:
					atom.set_coord(atom.get_coord()-cas[i])
				rotation=rotmat(cbsv[i]-casv[i],Vector(r[Naa+i]-r[i]))
				for atom in residue:
					if not atom.get_name() in BBA:
						atom.set_coord(rotation.dot(atom.get_coord()))
				for atom in residue:
					atom.set_coord(atom.get_coord()+r[i])
				i+=1
	#io.set_structure(structure)
	io.save('1frame.pdb')
	os.system('cat 1frame.pdb>>'+outputname)
	#structure=structurePDB.copy()
	

