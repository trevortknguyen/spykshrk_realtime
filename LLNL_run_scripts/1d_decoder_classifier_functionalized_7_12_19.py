# python script to run 1d clusterless decoder on sungod data using spykshrk
# written by MEC from notebooks written by AKG and JG
# 3-6-19
# this version includes support for linearizing the whole epoch (use pos_all_linear instead of pos_subset)
# on LLNL, this script runs from the folder '/usr/workspace/wsb/coulter5/spykshrk_realtime/LLNL_run_scripts'

#cell 1
# Setup and import packages
import sys
sys.path.append('/usr/workspace/wsb/coulter5/spykshrk_realtime')
import os
#import pdb
from datetime import datetime, date

import numpy as np
import scipy as sp

import trodes2SS
import sungod_util

from spykshrk.franklab.data_containers import RippleTimes, pos_col_format, Posteriors

#from spykshrk.franklab.data_containers import FlatLinearPosition, SpikeFeatures, Posteriors, EncodeSettings, pos_col_format, SpikeObservation, RippleTimes, DayEpochEvent, DayEpochTimeSeries
#from spykshrk.franklab.pp_decoder.util import normal_pdf_int_lookup, gaussian, apply_no_anim_boundary, normal2D
from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPEncoder, OfflinePPDecoder
#from spykshrk.franklab.pp_decoder.visualization import DecodeVisualizer
#from spykshrk.util import Groupby

#from replay_trajectory_classification.core import _causal_classify, _acausal_classify
#from replay_trajectory_classification.state_transition import strong_diagonal_discrete

#import numpy as np
#import scipy.io
#import scipy as sp
#import pandas as pd
#import loren_frank_data_processing as lfdp
#import scipy.io as sio # for saving .mat files 

def main(path_base, rat_name, path_arm_nodes, path_base_analysis, shift_amt, path_out):
    # set log file name
    #log_file = '/p/lustre1/coulter5/remy/1d_decoder_log.txt'
    print(datetime.now())
    today = str(date.today())
    #print(datetime.now(), file=open(log_file,"a"))

    #cell 2
    # Define parameters

    print(rat_name)
    print('Shift amount is: ',shift_amt)
    
    day_dictionary = {'remy':[20], 'gus':[28], 'bernard':[23], 'fievel':[19]}
    epoch_dictionary = {'remy':[2], 'gus':[4], 'bernard':[2], 'fievel':[4]} 
    tetrodes_dictionary = {'remy': [4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30], # 4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30
                           'gus': [6,7,8,9,10,11,12,17,18,19,20,21,24,25,26,27,30], # list(range(6,13)) + list(range(17,22)) + list(range(24,28)) + [30]
                           'bernard': [1,2,3,4,5,7,8,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
                           'fievel': [1,2,3,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,22,23,24,25,27,28,29]}
    #tetrodes_dictionary = {'remy': [4], # 4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30
    #                       'gus': [6], # list(range(6,13)) + list(range(17,22)) + list(range(24,28)) + [30]
    #                       'bernard': [1],
    #                       'fievel': [1]}
    
    pos_bin_size = 5
    velocity_thresh_for_enc_dec = 4
    velocity_buffer = 0

    shift_amt_for_shuffle = shift_amt

    discrete_tm_val=.99   # for classifier

    # version for LLNL
    # define data source filepaths
    path_base = path_base
    raw_directory = path_base + 'raw_data/' + rat_name + '/'
    linearization_path = path_base + 'maze_info/'
    day_ep = 'day' + str(day_dictionary[rat_name][0]) + '_epoch' + str(epoch_dictionary[rat_name][0])

    # IMPORT and process data

    #initialize data importer
    datasrc = trodes2SS.TrodesImport(raw_directory, rat_name, day_dictionary[rat_name], 
                           epoch_dictionary[rat_name], tetrodes_dictionary[rat_name])
    # Import marks
    marks = datasrc.import_marks()
    print('original length: '+str(marks.shape[0]))
    # OPTIONAL: to reduce mark number, can filter by size. Current detection threshold is 100  
    marks = trodes2SS.threshold_marks(marks, maxthresh=2000,minthresh=100)
    # remove any big negative events (artifacts?)
    marks = trodes2SS.threshold_marks_negative(marks, negthresh=-999)
    print('after filtering: '+str(marks.shape[0]))

    # Import trials
    trials = datasrc.import_trials()

    # Import raw position 
    linear_pos_raw = datasrc.import_pos(xy='x')   # pull in xpos and speed, x will be replaced by linear
    posY = datasrc.import_pos(xy='y')          #  OPTIONAL; useful for 2d visualization

    # Import ripples
    rips_tmp = datasrc.import_rips(linear_pos_raw, velthresh=4) 
    rips = RippleTimes.create_default(rips_tmp,1)  # cast to rippletimes obj
    print('Rips less than velocity thresh: '+str(len(rips)))

    # Position linearization
    # if linearization exists, load it. if not, run the linearization.
    lin_output1 = os.path.join(linearization_path, rat_name + '/' + rat_name + '_' + day_ep + '_' + 'linearized_distance.npy')

    if os.path.exists(lin_output1) == False:
        print('Linearization result doesnt exist. Doing linearization calculation!')
        sungod_util.run_linearization_routine(rat_name, day_dictionary[rat_name][0], epoch_dictionary[rat_name][0], linearization_path, raw_directory, gap_size=20)
    else: 
        print('Linearization found. Loading it!')
        lin_output2 = os.path.join(linearization_path, rat_name + '/' + rat_name + '_' + day_ep + '_' + 'linearized_track_segments.npy')

        linear_pos_raw['linpos_flat'] = np.load(lin_output1)   #replace x pos with linerized 
        track_segment_ids = np.load(lin_output2)

    # generate boundary definitions of each segment
    arm_coords, _ = sungod_util.define_segment_coordinates(linear_pos_raw, track_segment_ids)  # optional addition output of all occupied positions (not just bounds)

    #bin linear position 
    binned_linear_pos, binned_arm_coords, pos_bins = sungod_util.bin_position_data(linear_pos_raw, arm_coords, pos_bin_size)

    # calculate bin coverage based on determined binned arm bounds   TO DO: prevent the annnoying "copy of a slice" error [prob need .values rather than a whole column]
    pos_bin_delta = sungod_util.define_pos_bin_delta(binned_arm_coords, pos_bins, linear_pos_raw, pos_bin_size)

    max_pos = binned_arm_coords[-1][-1]+1

    # cell 8
    # decide what to use as encoding and decoding data
    marks, binned_linear_pos = sungod_util.assign_enc_dec_set_by_velocity(binned_linear_pos, marks, velocity_thresh_for_enc_dec, velocity_buffer)

    # rearrange data by trials 
    pos_reordered, marks_reordered, order = sungod_util.reorder_data_by_random_trial_order(trials, binned_linear_pos, marks)

    encoding_marks = marks_reordered.loc[marks_reordered['encoding_set']==1]
    decoding_marks = marks_reordered.loc[marks_reordered['encoding_set']==0]
    print('Encoding spikes: '+str(len(encoding_marks)))
    print('Decoding spikes: '+str(len(decoding_marks)))

    encoding_pos = pos_reordered.loc[pos_reordered['encoding_set']==1]

    # apply shift for shuffling 
    encoding_marks_shifted, shift_amount = sungod_util.shift_enc_marks_for_shuffle(encoding_marks, shift_amt_for_shuffle)
    # put marks back in chronological order for some reason
    encoding_marks_shifted.sort_index(level='time',inplace=True)
    print('Marks index shift: ',shift_amount)
    print('Shifted marks shape: ', encoding_marks_shifted.shape)

    # cell 9
    # populate enc/dec settings. any parameter settable should be defined in parameter cell above and used here as a variable

    encode_settings = trodes2SS.AttrDict({'sampling_rate': 3e4,
                                    'pos_bins': np.arange(0,max_pos,1), # actually indices of valid bins. different from pos_bins above 
                                    'pos_bin_edges': np.arange(0,max_pos + .1,1), # indices of valid bin edges
                                    'pos_bin_delta': pos_bin_delta, 
                                    # 'pos_kernel': sp.stats.norm.pdf(arm_coords_wewant, arm_coords_wewant[-1]/2, 1),
                                    'pos_kernel': sp.stats.norm.pdf(np.arange(0,max_pos,1), max_pos/2, 1), #note that the pos_kernel mean should be half of the range of positions (ie 180/90)     
                                    'pos_kernel_std': 0, # 0 for histogram encoding model, 1+ for smoothing
                                    'mark_kernel_std': int(20), 
                                    'pos_num_bins': max_pos, 
                                    'pos_col_names': [pos_col_format(ii, max_pos) for ii in range(max_pos)], # or range(0,max_pos,10)
                                    'arm_coordinates': binned_arm_coords,   
                                    'spk_amp': 60,
                                    'vel': 0}) 

    decode_settings = trodes2SS.AttrDict({'trans_smooth_std': 2,
                                    'trans_uniform_gain': 0.0001,
                                    'time_bin_size':60})
    print('Encode settings: ',encode_settings)
    print('Decode settings: ',decode_settings)

    #cell 10
    # Run encoder
    print('Starting encoder')
    time_started = datetime.now()

    encoder = OfflinePPEncoder(linflat=encoding_pos, dec_spk_amp=decoding_marks, encode_settings=encode_settings, 
                               decode_settings=decode_settings, enc_spk_amp=encoding_marks_shifted, dask_worker_memory=1e9,
                               dask_chunksize = None)

    #new output format from encoder: observ_obj
    observ_obj = encoder.run_encoder()

    time_finished =datetime.now()

    print('Enocder finished!')
    print('Encoder started at: %s'%str(time_started))
    print('Encoder finished at: %s'%str(time_finished))
    #print("Encoder finished!", file=open(log_file,"a"))

    #cell 15
    # Run PP decoding algorithm

    print('Starting decoder')
    decoder = OfflinePPDecoder(observ_obj=observ_obj, trans_mat=encoder.trans_mat['sungod'], 
                               prob_no_spike=encoder.prob_no_spike,
                               encode_settings=encode_settings, decode_settings=decode_settings, 
                               time_bin_size=decode_settings.time_bin_size, all_linear_position=binned_linear_pos)

    posteriors = decoder.run_decoder()
    print('Decoder finished!')
    print('Posteriors shape: '+ str(posteriors.shape))

    #cell 15.1
    # save posterior and linear position - netcdf
    posterior_file_name = os.path.join(path_out,  rat_name + '_' + str(day_dictionary[rat_name][0]) + '_' + str(epoch_dictionary[rat_name][0]) + '_shuffle_' + str(shift_amount) + '_posteriors_functionalized.nc')

    post1 = posteriors.apply_time_event(rips, event_mask_name='ripple_grp')
    post2 = post1.reset_index()
    post3 = trodes2SS.convert_dan_posterior_to_xarray(post2, tetrodes_dictionary[rat_name], 
                                            velocity_thresh_for_enc_dec, encode_settings, decode_settings, encoder.trans_mat['sungod'], order, shift_amount)
    post3.to_netcdf(posterior_file_name)
    print('Saved netcdf posteriors to '+posterior_file_name)

    # to export linearized position to MatLab: again convert to xarray and then save as netcdf

    position_file_name = os.path.join(path_out, rat_name + '_' + str(day_dictionary[rat_name][0]) + '_' + str(epoch_dictionary[rat_name][0]) + '_shuffle_' + str(shift_amount) + '_linearposition_functionalized.nc')

    linearized_pos1 = binned_linear_pos.apply_time_event(rips, event_mask_name='ripple_grp')
    linearized_pos2 = linearized_pos1.reset_index()
    linearized_pos3 = linearized_pos2.to_xarray()
    linearized_pos3.to_netcdf(position_file_name)
    print('Saved netcdf linearized position to '+position_file_name)

    #cell 15.2
    # save posterior as hdf5
    posterior_file_name_hdf5 = os.path.join(path_out,  rat_name + '_' + str(day_dictionary[rat_name][0]) + '_' + str(epoch_dictionary[rat_name][0]) + '_shuffle_' + str(shift_amount) + '_posteriors_functionalized.h5')
    posteriors._to_hdf_store(posterior_file_name_hdf5,'/analysis', 'decode/clusterless/offline/posterior', 'sungod_trans_mat')
    print('Saved hdf5 posteriors to '+posterior_file_name_hdf5)

    #cell 16
    # run replay classifier
    causal_state1, causal_state2, causal_state3, acausal_state1, acausal_state2, acausal_state3, trans_mat_dict = sungod_util.decode_with_classifier(decoder.likelihoods, 
                                                                                                                                 encoder.trans_mat['sungod'], 
    #cell 17
    # save classifier output
    base_name = os.path.join(path_out, + rat_name + '_' + day_ep + '_shuffle_' + str(shift_amount) + '_posterior_')

    fname = 'causal'
    trodes2SS.convert_save_classifier(base_name, fname, causal_state1, causal_state2, causal_state3, tetrodes_dictionary[rat_name], decoder.likelihoods,
                                      encode_settings, decode_settings, rips, velocity_thresh_for_enc_dec, velocity_buffer, encoder.trans_mat['sungod'], order, shift_amount)

    fname = 'acausal'
    trodes2SS.convert_save_classifier(base_name, fname, acausal_state1, acausal_state2, acausal_state3, tetrodes_dictionary[rat_name], decoder.likelihoods,
                                      encode_settings, decode_settings, rips, velocity_thresh_for_enc_dec, velocity_buffer, encoder.trans_mat['sungod'], order, shift_amount)
                                                                                                                                                                   encoder.occupancy, discrete_tm_val)

    # to calculate histogram of posterior max position in each time bin

    hist_bins = []
    post_hist1 = posteriors.drop(['num_spikes','dec_bin','ripple_grp'], axis=1)
    post_hist1.fillna(0,inplace=True)
    post_hist3 = post_hist1.idxmax(axis=1)
    post_hist3 = post_hist3.str.replace('x','')
    post_hist3 = post_hist3.astype(int)
    #print(post_hist3.shape)
    hist_bins = np.histogram(post_hist3,bins=np.arange(0,147))
    print(hist_bins)
    unique, counts = np.unique(post_hist3, return_counts=True)
    print(dict(zip(unique,counts)))

    print("End of script!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store', dest='path_base', help='Base path')
    parser.add_argument('-n', action='store', dest='rat_name', help='Rat Name')
    parser.add_argument('-a', action='store', dest='path_arm_nodes', help='Path to directory with arm_nodes and simple_transition_matrix files')
    parser.add_argument('-l', action='store', dest='path_base_linearization', help='Base path to linearization')
    parser.add_argument('-s', action='store', dest='shift_amt', type=float, help='Shift amount')
    parser.add_argument('-o', action='store', dest='path_out', help='Path to output')
    results = parser.parse_args()

main(results.path_base, results.rat_name, results.path_arm_nodes, results.path_base_linearization, results.shift_amt, results.path_out)
