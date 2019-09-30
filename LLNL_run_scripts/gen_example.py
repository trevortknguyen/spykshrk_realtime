import sys
import subprocess
import datetime
import os
import numpy as np

def main(path_base_rawdata, spykshrk_path, rat_name, day, epoch, path_out, num_shift):
    #
    #
    #
    #

    dir_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    os.mkdir(dir_name)
    os.mkdir(os.path.join(dir_name, 'slurm'))
    os.mkdir(os.path.join(dir_name, 'output'))
    
    with open('%s/slurm/slurm_sub.sh'%dir_name, 'w') as fid0:
        for i0 in range(num_shift):
            with open('%s/slurm/slurm_%s'%(dir_name, i0), 'w') as fid1:
                fid1.write('#!/bin/bash\n')
                fid1.write('#SBATCH -N 1\n')
                fid1.write('#SBATCH -t 5:00:00\n')
                fid1.write('#SBATCH -p pbatch\n')
                fid1.write('#SBATCH -A bioeng\n')
                #path_D = os.path.join(path_arm_nodes, 'python-scripts')
                path_D = os.path.join(spykshrk_path, 'LLNL_run_scripts')
                fid1.write('#SBATCH -D %s\n'%path_D)
                fid1.write('#SBATCH --license=lustre1\n')
                fid1.write('#SBATCH -o /p/lustre1/coulter5/runs/%s/slurm/slurm_out_%s\n'%(dir_name, i0))
                fid1.write('conda activate spykshrk-env\n')
                fid1.write('export OMP_NUM_THREADS=1\n')
                if i0 == 0:
                    shft = 0.0
                else:
                    shft = str(np.round(np.random.uniform(0.25, 0.75), 4))
                print(path_D)
                #fid1.write('python %s -p %s -n %s -a %s -l %s -s %s -o %s\n'%(os.path.join(path_D, '1d_decoder_classifier_functionalized_7_1_19.py'), path_base_rawdata, rat_name, path_arm_nodes, path_base_analysis, shft, os.path.join(path_out, dir_name + '/output')))
                fid1.write('python %s -p %s -n %s -d %s -e %s -s %s -o %s\n'%(os.path.join(path_D, '1d_decoder_classifier_LLNL.py'), path_base_rawdata, rat_name, day, epoch, shft, os.path.join(path_out, dir_name + '/output')))

                fid1.write('chmod -R g+rwx /p/lustre1/coulter5/runs/%s'%dir_name)
            fid0.write('msub slurm_%s\n'%i0)
    subprocess.run(['chmod', '-R', 'u+rwx,g+rwx', '/p/lustre1/coulter5/runs/%s'%dir_name])


if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store', dest='path_base_rawdata', help='Base path to raw data')
    parser.add_argument('-n', action='store', dest='rat_name', help='Rat Name')
    parser.add_argument('-d', action='store', dest='day', type=int, help='Day')
    parser.add_argument('-e', action='store', dest='epoch', type=int, help='Epoch')
    parser.add_argument('-a', action='store', dest='spykshrk_path', help='Path to spyk directory; also contains arm_nodes and simple_transition_matrix files')
    #parser.add_argument('-l', action='store', dest='path_base_linearization', help='Base path to linearization')
    parser.add_argument('-t', action='store', dest='num_shift', type=int, help='Number of random shifts in the range [0.25, 0.75]')
    parser.add_argument('-o', action='store', dest='path_out', help='Path to output')
    results = parser.parse_args()

    main(results.path_base_rawdata, results.spykshrk_path, results.rat_name, results.day, results.epoch, results.path_out, results.num_shift)
