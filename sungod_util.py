# based on sungod_linearization
# based on Loren Frank Data Processing by ED
# 
# Jun 2018 JAG, Jan 2019, MEC, AKG

import os
import numpy as np
import scipy as sp
import scipy.stats as ss
import scipy.io
import networkx as nx
import loren_frank_data_processing as lfdp
import scipy.io as sio # for saving .mat files 
import inspect # for inspecting files (e.g. finding file source)

import json
import functools
from replay_trajectory_classification.state_transition import strong_diagonal_discrete
from replay_trajectory_classification.core import _causal_classify, _acausal_classify


def createTrackGraph(maze_coordinates):
    #linearcoord = maze_coordinates['linearcoord_NEW'].squeeze()
    linearcoord = maze_coordinates['linearcoord_one_box'].squeeze()
    track_segments = [np.stack((arm[:-1], arm[1:]), axis=1) for arm in linearcoord]
    center_well_position = track_segments[0][0][0]
    track_segments, center_well_position = (np.unique(np.concatenate(track_segments), axis=0),
                center_well_position) # ## what does redefining center_well_position here do?
    nodes = np.unique(track_segments.reshape((-1, 2)), axis=0)
    edges = np.zeros(track_segments.shape[:2], dtype=int)
    for node_id, node in enumerate(nodes):
        edge_ind = np.nonzero(np.isin(track_segments, node).sum(axis=2) > 1)
        edges[edge_ind] = node_id
    edge_distances = np.linalg.norm(
        np.diff(track_segments, axis=-2).squeeze(), axis=1)
    track_graph = nx.Graph()
    for node_id, node_position in enumerate(nodes):
        track_graph.add_node(node_id, pos=tuple(node_position))
    for edge, distance in zip(edges, edge_distances):
        track_graph.add_edge(edge[0], edge[1], distance=distance)
    center_well_id = np.unique(
        np.nonzero(np.isin(nodes, center_well_position).sum(axis=1) > 1)[0])[0]
    return track_graph, track_segments, center_well_id

def hack_determinearmorder(track_segments): 
    # order arm segments based on y position. ASSUMES CERTAIN LAYOUT OF TRACK_SEGMENTS. 
    d_temp = [] 
    for track_segment in enumerate(track_segments):
        if track_segment[0] < 8:
            d_temp.append(track_segment[1][1,1])

    rank = ss.rankdata(d_temp) - 1 # - 1 to account for python indexing

    order1 = [None]*len(rank)
    for r in enumerate(rank):
        order1[int(r[1])] = int(r[0])
    return(order1)

def turn_array_into_ranges(array1):
    array1_diff = np.ediff1d(array1)

    start_temp = [] 
    end_temp = []
    start_temp.append(array1[0]) 

    some_end_indices = np.where(array1_diff > 1)

    for i in range(len(some_end_indices[0])):
        # This is always an end index
        end_temp.append(array1[some_end_indices[0][i]])
        if array1[some_end_indices[0][i]] == start_temp[i]: # if this is the same as the last start index, it was already added as a start index-- don't need to add it again
            start_temp.append(array1[some_end_indices[0][i] + 1]) # define next start index   
        elif array1_diff[some_end_indices[0][i] - 1] > 1: # if last value was more than 1 away, this is also a start index
            start_temp.append(array1[some_end_indices[0][i]])    
        else: # if last value was NOT more than 1 away, this is JUST an end index, and next start index is next index
            start_temp.append(array1[some_end_indices[0][i] + 1])   
    # The last entry in array is always the last end index
    end_temp.append(array1[-1])  

    return start_temp, end_temp

# Function to define chunked data 
def chunk_data(data,number_of_chunks):
    print('chunking data of length',len(data),'samples into',str(number_of_chunks),'chunk(s)')
    # Takes 1D data and splits into number of chunks (as equally as possible)
    
    # Calculate number of data points per chunk
    datapoints_per_chunk = math.ceil(len(data)/number_of_chunks)
    print('datapoints_per_chunk:',datapoints_per_chunk)
    
    # Initialize empty list for chunked data
    chunked_data = [] 
    
    # Define chunks
    for chunk_number in range(number_of_chunks): # for each chunk
        chunked_data.append(data[chunk_number*datapoints_per_chunk:(chunk_number + 1)*datapoints_per_chunk]) 
    
    return chunked_data

    # Toy example
    #hi = np.concatenate((np.ones(5),np.ones(5)*2),axis=0)
    #print(hi)
    #chunked_data = []
    #chunk_number = 2
    #for i in range(10): 
    #    x = hi[chunk_number*i:chunk_number*(i + 1)]
    #    print(x)
    #    chunked_data.append(x)
    #print(chunked_data)

def change_to_directory_make_if_nonexistent(directory_path):
    # Make directory if it doesn't exist
    if os.path.exists(directory_path) == False:
        print('making path ' + directory_path)
        os.chdir('/')        
        os.makedirs(directory_path)
    # Change to directory 
    os.chdir(directory_path)
    # Print working directory
    os.getcwd()
    
def define_segment_coordinates(pos_obj, track_segment_ids):
    
    ''' 
    Use segment assignments of position data to determine the ends of each segment.
     
    Important: do this per epoch in case maze coverage is incomplete - this will exclude areas
     where the animal never goes)

    Note: this implicity puts data into 1cm bins, since we are rounding position to integers
    
    Output: start and end bins for each segment (0-based, INCLUSIVE)
    '''

    track_segments = np.unique(track_segment_ids)
    seg_pos_range = []
    seg_pos_range_edges = [] 
    for seg in track_segments:
        seg_inds = track_segment_ids==seg      #get index of positions on that segment
        seg_pos = pos_obj['linpos_flat'].values[seg_inds]    # store those positions (pos only, not vel)
        seg_pos_range.append([seg_pos.min() ,seg_pos.max()])    # get min and max vals per segment
        seg_pos_range_edges.append([np.floor(seg_pos.min()), np.ceil(seg_pos.max())])

    binswewant_tmp = []
    for seg_range in np.floor(seg_pos_range_edges): # for each track segment
        binswewant_tmp.append(np.arange(seg_range[0],seg_range[1]+1)) # + 1 to account for np.arange not including last index

    occupied_bins = np.unique(np.concatenate(binswewant_tmp))   # concatenate and get rid of duplicate bins from box segs
    arm_coordinates = np.column_stack(turn_array_into_ranges(occupied_bins))   #  get start & end of each region

    return arm_coordinates, occupied_bins

def bin_position_data(pos_obj, arm_coordinates, pos_bin_size = 1):
    '''
    Bin linearized position 
    Bin size defaults to 1 

    In order to make arms all evenly divisible by bin_size, add extra values to arm bounds
    This could be problematic for extremely large bins, but should be accounted for by pos_bin_delta later

    Note that gap bins may differ in number and size but this shouldnt matter; dependent on arm bounds 

    Returns a new pos object with [linpos_flat] pos values replaced with indices of position bins
    '''

    armcoordinates_corrected = []
    pos_bins = []
    for arm in np.arange(0,arm_coordinates.shape[0],1):
        arm_bounds = arm_coordinates[arm]
        # add to arm edge value to make it divisible by posbin. each arm starts at the node, and now will end some 5cm bins later
        remainder = np.mod((arm_bounds[1]-arm_bounds[0]),pos_bin_size)   
        if remainder: 
            extra_needed = pos_bin_size-remainder     
        else:
            extra_needed = 0
        new_bounds = [arm_bounds[0], arm_bounds[1]+extra_needed]
        armcoordinates_corrected.append(new_bounds)
        # construct pos bin edges. this is not just a range, because the gaps are not strictly 5cm (thanks to the adjustments above)
        pos_bins.append(np.arange(new_bounds[0],new_bounds[1]+1,pos_bin_size))   
        if arm_bounds[1] < arm_coordinates[-1,-1]:  
            pos_bins.append(np.arange(new_bounds[1],arm_coordinates[arm+1][0]+1,pos_bin_size))  # add gap bins
        
    # there are lots of repeat bins in segments in the box - get rid of them         
    pos_bins = np.unique(np.concatenate(pos_bins))  # analog to mike's position_bins
    print(pos_bins)

    armcoordinates_binned = []     # find the indexes of the arm ends (mike's new_arm_coords)
    for end in armcoordinates_corrected:
        startind=np.where(pos_bins == end[0])
        endind = np.where(pos_bins == end[1])
        armcoordinates_binned.append([startind[0][0],endind[0][0]-1])
    armcoordinates_binned = np.array(armcoordinates_binned)

    print(armcoordinates_binned)

    # bin linearized position data with these bins
    # digitize returns the bin index for each value, converting position measure into binindex measure (1-based)
    digitized = np.digitize(pos_obj['linpos_flat'], pos_bins)
    binned_linear_pos = pos_obj.copy()    # make complete separate copy (deep)
    binned_linear_pos['linpos_flat'] = digitized-1  # -1  for python 0-based

    return binned_linear_pos, armcoordinates_binned, pos_bins

def define_pos_bin_delta(armcoordinates_binned, pos_bins, pos_raw, pos_bin_size):
    ''' 
    Determine what proportion of the bin is covered by original (unbinned) position data
    This will scale the decode likelihood down for end bins with only partial coverage 
    only define this for arm end bins, because everything else should necessarily be complete
    '''

    pos_delta = np.ones(armcoordinates_binned[-1][-1]+1)
    for i in np.arange(0,len(pos_bins),1):  #iterate through pos bins (excludes box end)
        if i in armcoordinates_binned[1:,1]:     # calc coverage if this is an arm end bin
            pos_in_bin = pos_raw['linpos_flat'][(pos_raw['linpos_flat']>pos_bins[i]) & (pos_raw['linpos_flat']<pos_bins[i+1])]        
            #if pos_in_bin.size > 0:
            pos_delta[i] = (max(pos_in_bin)-pos_bins[i])/pos_bin_size

    pos_delta = 1/pos_delta   # take inverse since we end up dividing by this instead of multiplying

    return pos_delta

def reorder_data_by_random_trial_order(trials, pos_obj, marks_obj):
    '''
    Randomize trial order so that chunks (for cross validation) don't contain clusters of certain arms
    '''

    trials.reset_index(level=['trial'], inplace=True)   
    trialsindex = trials['trial'].values
    print('Number of trials: ',len(trialsindex))
    np.random.shuffle(trialsindex)

    starttimes_shuffled = trials['starttime'].iloc[trialsindex]
    endtimes_shuffled = trials['endtime'].iloc[trialsindex]

    pos_reordered = pos_obj.head(0)   #initialize
    for i in range(len(starttimes_shuffled)):
        random_trial_pos = pos_obj.loc[(pos_obj.index.get_level_values('time') <= endtimes_shuffled.iloc[i]) 
                                    & (pos_obj.index.get_level_values('time') >= starttimes_shuffled.iloc[i])]
        pos_reordered = pos_reordered.append(random_trial_pos)
             
        #marks
    marks_reordered = marks_obj.head(0)
    for i in range(len(starttimes_shuffled)):
        random_trial_marks = marks_obj.loc[(marks_obj.index.get_level_values('time') <= endtimes_shuffled.iloc[i]) & (marks_obj.index.get_level_values('time') >= starttimes_shuffled.iloc[i])]
        marks_reordered = marks_reordered.append(random_trial_marks)

    trials.set_index(['trial'], drop=True, append=True, inplace=True)

    return pos_reordered, marks_reordered, trialsindex

def assign_enc_dec_set_by_velocity(pos_obj, marks_obj, velthresh, buffer = 0):
    '''
    Add column to marks object which designated whether mark is part of encoding set (1) or decoding set 0)
    Based on velocity threshold
    Optional buffer, which will include n seconds after the rat slows down in the encoding set
    '''

    posinfo_at_mark_times = pos_obj.get_irregular_resampled(marks_obj)
    marks_obj['encoding_set']=0   # add a new column to object, fill with zeros
    if buffer:
        raise Exception('not done yet!')
    else:
        assignments = (posinfo_at_mark_times['linvel_flat'].values>velthresh).astype(int)   # get boolean of above thresh, convert to int

    marks_obj['encoding_set'] = assignments  

    pos_obj['encoding_set'] = (pos_obj['linvel_flat'].values>velthresh).astype(int)

    return marks_obj, pos_obj

def shift_enc_marks_for_shuffle(marks_obj, shift=0):

    '''
    shift encoding marks by a fraction of total epoch time in order to break relationship between position and spikes
    returns a new marks object with marks shifted by a certain number of indices (shift_amount)

    '''

    encoding_marks_shifted = marks_obj.copy()   #deepcopy to create a new obj to fill with shifted values
    encoding_marks_shifted.reset_index(level=['time'],inplace=True)
    min_time =encoding_marks_shifted['time'].min()
    max_time = encoding_marks_shifted['time'].max()
    epoch_length = max_time - min_time
    print('Total epoch time (sec): ',epoch_length)

    shift_target_time = min_time + shift * epoch_length
    target_idx = np.abs(encoding_marks_shifted['time']-shift_target_time).idxmin()   # calculate multiindex of closest value to shift target
    shift_amount = encoding_marks_shifted.index.get_loc(target_idx)    # get the row number of the target 

    encoding_marks_shifted['c00'] = np.roll(encoding_marks_shifted['c00'],shift_amount)
    encoding_marks_shifted['c01'] = np.roll(encoding_marks_shifted['c01'],shift_amount)
    encoding_marks_shifted['c02'] = np.roll(encoding_marks_shifted['c02'],shift_amount)
    encoding_marks_shifted['c03'] = np.roll(encoding_marks_shifted['c03'],shift_amount)

    encoding_marks_shifted.set_index(['time'], drop=True, append=True, inplace=True)

    return encoding_marks_shifted, shift_amount

def decode_with_classifier(likelihoods_obj, sungod_transmat, occupancy, discrete_tm_val=.99):
        
    '''
    '''
    # set up ingredients for classifier
    num_bins = sungod_transmat.shape[0]
    gapmask = np.ceil(np.nan_to_num(occupancy))
    ic_tmp = gapmask * np.ones((3,num_bins)) # in all bins, 0s in all gaps, then repeat over 3 rows
    initial_conditions = ic_tmp[:,:,np.newaxis]   # add on the extra 3rd dimension 
    trans_mat_dict = {'continuous':sungod_transmat, 
                      'identity':gapmask*np.eye(num_bins),
                      'uniform':gapmask*gapmask[:,np.newaxis]*np.ones((num_bins,num_bins))/sum(gapmask)}

    all_tms = [[trans_mat_dict['continuous'],trans_mat_dict['uniform'], trans_mat_dict['identity']],
         [trans_mat_dict['uniform'],trans_mat_dict['uniform'],trans_mat_dict['uniform']],
         [trans_mat_dict['continuous'],trans_mat_dict['uniform'], trans_mat_dict['identity']]]

    continuous_state_transition = np.stack(all_tms,axis=0)
    discrete_state_transition = strong_diagonal_discrete(3,discrete_tm_val)  # controls how sticky the discrete state trans is

    likelihoods = likelihoods_obj.drop(['num_spikes','dec_bin'],axis=1)

    # initialize output structures
    causal_state1 = likelihoods.copy()   # continuous
    causal_state2 = likelihoods.copy()   # fragmented
    causal_state3 = likelihoods.copy()  # hover 
    acausal_state1 = likelihoods.copy()
    acausal_state2 = likelihoods.copy()
    acausal_state3 = likelihoods.copy() 
    posbin1 = np.isnan(likelihoods['x000'])  # use the first posbin to find nan time chunks
    blocks = (posbin1 != posbin1.shift()).cumsum()   # define each group of non-nan values as a block

    # run classifier
    for b in range(1,max(blocks)+1):    #  iterate through non-nan blocks
        print('running block '+ str(b) +' of ' + str(max(blocks)))
        chunk = likelihoods.loc[blocks==b].fillna(0).values   # get rid of the gap nans
        chunk_expanded = chunk[:,np.newaxis,:,np.newaxis] * np.ones((1,3,num_bins,1))   # put things in the right shape
        causal_posterior = _causal_classify(initial_conditions, continuous_state_transition,discrete_state_transition,chunk_expanded)
        causal_state1.loc[blocks==b]=causal_posterior[:,0,:,0]   # store posteriors for each state separately bc df is 2d only
        causal_state2.loc[blocks==b]=causal_posterior[:,1,:,0]
        causal_state3.loc[blocks==b]=causal_posterior[:,2,:,0]

        acausal_posterior = _acausal_classify(causal_posterior,continuous_state_transition,discrete_state_transition)
        acausal_state1.loc[blocks==b]=acausal_posterior[:,0,:,0]   # store posteriors for each state separately bc df is 2d only
        acausal_state2.loc[blocks==b]=acausal_posterior[:,1,:,0]
        acausal_state3.loc[blocks==b]=acausal_posterior[:,2,:,0]

    return causal_state1, causal_state2, causal_state3, acausal_state1, acausal_state2, acausal_state3
            
def convert_classifier_output_and_save(save_path, fname, classifier_output):

    raise Exception('not done yet!')
    causal_state1['num_spikes'] = decoder.likelihoods['num_spikes']
    causal_state1['dec_bin'] = decoder.likelihoods['dec_bin']
    causal_s1_obj = Posteriors.from_dataframe(causal_state1, enc_settings=encode_settings,
                                                            dec_settings=decode_settings,
                                                            user_key={'encode_settings': encode_settings,
                                                                      'decode_settings': decode_settings,
                                                                      'multi_index_keys': causal_state1.index.names})
    causal_s1_obj = causal_s1_obj.apply_time_event(rips_vel_filtered, event_mask_name='ripple_grp').reset_index()
    causal_s1_obj = convert_dan_posterior_to_xarray(causal_s1_obj, tetrodes_dictionary[rat_name], 
                                            velocity_filter, encode_settings, decode_settings, encoder.trans_mat['sungod'],
                                            trialsindex_shuffled, marks_index_shift)
    causal_s1_obj.to_netcdf(base_name+fname+'.nc')
