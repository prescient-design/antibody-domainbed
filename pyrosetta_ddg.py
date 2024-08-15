from pyrosetta import *
from rosetta import *
import pandas as pd
import numpy as np
import time
import os
import argparse

dict_1to3 = {
    "C": "CYS",
    "D": "ASP",
    "S": "SER",
    "Q": "GLN",
    "K": "LYS",
    "I": "ILE",
    "P": "PRO",
    "T": "THR",
    "F": "PHE",
    "N": "ASN",
    "G": "GLY",
    "H": "HIS",
    "L": "LEU",
    "R": "ARG",
    "W": "TRP",
    "A": "ALA",
    "V": "VAL",
    "E": "GLU",
    "Y": "TYR",
    "M": "MET",
}

init('-allow_jump_in_numbering')

sfxn = core.scoring.get_score_function() 

iam = protocols.analysis.InterfaceAnalyzerMover(
    interface_jump=2,
    tracer=False,
    sf=sfxn,
    compute_packstat=False,
    pack_input=True,
    pack_separated=False, 
    use_jobname=False,
    detect_disulfide_in_separated_pose=False
)



def ssm(fn, seq_rec):
    data = {}
    pose = pose_from_pdb(fn)

    iam.apply(pose)
    dG_initial = iam.get_interface_dG()
    print('dG_initial', dG_initial)
    print(pose.size())

    pose_seq = pose.sequence()

    seq_mut = pose_seq.replace(seq_rec.fv_heavy_aho_seed.replace('-',''), seq_rec.fv_heavy_aho.replace('-',''))
    seq_mut = seq_mut.replace(seq_rec.fv_light_aho_seed.replace('-',''), seq_rec.fv_light_aho.replace('-',''))
    
    diffs_positions = [i for i, (left, right) in enumerate(zip(pose.sequence(),seq_mut)) if left != right]
    print(diffs_positions)
    
    if len(diffs_positions) > 0:
        try: 
            if len(diffs_positions) < 160:
                for ii in range(0, len(diffs_positions)): 
                    mutated_pos = diffs_positions[ii]
                    chain, number = pose.pdb_info().chain(ii+1), pose.pdb_info().number(mutated_pos+1)

                    tlc = dict_1to3[seq_mut[mutated_pos]]
                    mut = protocols.simple_moves.MutateResidue(mutated_pos+1,tlc)
                    mut.apply(pose) 

                iam.apply(pose)
                dG_final = iam.get_interface_dG()
                print('dG_final', dG_final)
                print( dG_final - dG_initial)

                return dG_final - dG_initial
            else:
                f = open("incorrect_seq.txt", "a")
                f.write(f'pdb  {seq_rec.fv_heavy_aho} {seq_rec.fv_light_aho} not mutatable.  TOO long \n')
                f.close()


        except:

                f = open("incorrect_seq.txt \n", "a")
                f.write(f'pdb {fn} not mutatable.')
                f.close()

                return 10000000
    else:
        return 0
    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute ddG using pyrosetta",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file_input", type=str, 
                        default="wjs_sigma_0.5",
                        help="Location of file for computing ddG ")
    args = parser.parse_args()
    
    input_df = pd.read_csv(args.file_input + '.csv')
    ddgs = []
    for i in np.arange(input_df.shape[0]):
        path = 'your_path/'
        seq_rec = input_df.iloc[i]
        fn = path + seq_rec.pdb.upper() +'_1_0001_0001.pdb'
        if not os.path.exists(fn):
            print("path doesn't exist")
            fn =  path +  seq_rec.pdb.upper() +'_2_0001_0001.pdb'
        if not os.path.exists(fn):
            fn =  path +  seq_rec.pdb.upper() +'_3_0001_0001.pdb'    
            print(fn)
            pass
         
        ddG = ssm(fn, seq_rec)
        ddgs.append(ddG)
    
    input_df['ddG_rosetta'] = np.asarray(ddgs)
    
    print(input_df)
    input_df.to_csv(args.file_input + '_ddg.csv')
      
