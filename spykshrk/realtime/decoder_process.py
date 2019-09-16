from mpi4py import MPI
import math
import numpy as np
from scipy.ndimage.interpolation import shift

from spykshrk.realtime import realtime_base, realtime_logging, binary_record, datatypes, encoder_process, ripple_process, main_process
from spykshrk.realtime.simulator import simulator_process
from spykshrk.realtime.camera_process import VelocityCalculator, LinearPositionAssignment

from spykshrk.franklab.pp_decoder.util import apply_no_anim_boundary


class DecoderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm :MPI.Comm, rank, config):
        super(DecoderMPISendInterface, self).__init__(comm=comm, rank=rank, config=config)

    def send_record_register_messages(self, record_register_messages):
        self.class_log.debug("Sending record register messages.")
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def send_time_sync_report(self, time):
        self.comm.send(obj=realtime_base.TimeSyncReport(time),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def all_barrier(self):
        self.comm.Barrier()


class SpikeDecodeRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(SpikeDecodeRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.msg_buffer = bytearray(50000)
        self.req = self.comm.Irecv(buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.SPIKE_DECODE_DATA)

    def __next__(self):
        rdy = self.req.Test()
        if rdy:

            msg = encoder_process.SpikeDecodeResultsMessage.unpack(self.msg_buffer)
            self.req = self.comm.Irecv(buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.SPIKE_DECODE_DATA)
            return msg

        else:
            return None

class RippleDecodeRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(RippleDecodeRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.mpi_status = MPI.Status()

        self.feedback_bytes = bytearray(12)

        self.mpi_reqs = []
        self.mpi_statuses = []

        req_feedback = self.comm.Irecv(buf=self.feedback_bytes,
                                       tag=realtime_base.MPIMessageTag.FEEDBACK_DATA.value)
        self.mpi_statuses.append(MPI.Status())
        self.mpi_reqs.append(req_feedback)

    def __next__(self):
        # get ripple threshold message (code from main process)
        ready = MPI.Request.Testall(requests=self.mpi_reqs, statuses=self.mpi_statuses)
        #print('ripple rec: ',self.mpi_statuses[0].source)
        if ready:
            if self.mpi_statuses[0].source in self.config['rank']['ripples']:
                message = ripple_process.RippleThresholdState.unpack(message_bytes=self.feedback_bytes)
                #if message.threshold_state > 0:
                #    print('ripple message to decoder: ',message)

                #self.num_above = self.pp_decoder.update_ripple_threshold_state(timestamp=message.timestamp,
                #                                        elec_grp_id=message.elec_grp_id,
                #                                        threshold_state=message.threshold_state)
                #print(self.num_above)

                self.mpi_reqs[0] = self.comm.Irecv(buf=self.feedback_bytes,tag=realtime_base.MPIMessageTag.FEEDBACK_DATA.value)
                return message

        else:
            return None

class PointProcessDecoder(realtime_logging.LoggingClass):

    def __init__(self, pos_range, pos_bins, time_bin_size, arm_coor, config, uniform_gain=0.01):
        self.pos_range = pos_range
        self.pos_bins = pos_bins
        self.time_bin_size = time_bin_size
        self.arm_coor = arm_coor
        self.config = config
        #self.uniform_gain = uniform_gain
        self.uniform_gain = self.config['pp_decoder']['trans_mat_uniform_gain']

        self.ntrode_list = []

        self.cur_pos_time = -1
        self.cur_pos = -1
        self.cur_pos_ind = 0
        self.pos_delta = (self.pos_range[1] - self.pos_range[0]) / self.pos_bins

        # Initialize major PP variables
        self.observation = np.ones(self.pos_bins)
        self.occ = np.ones(self.pos_bins)
        self.likelihood = np.ones(self.pos_bins)
        self.posterior = np.ones(self.pos_bins)
        self.prev_posterior = np.ones(self.pos_bins)
        self.firing_rate = {}
        #self.transition_mat = PointProcessDecoder._create_transition_matrix(self.pos_delta,
        #                                                                    self.pos_bins,
        #                                                                    self.arm_coor,
        #                                                                    self.uniform_gain)

        #create sungod transition matrix - should make transition matrix type an option in the config file and specify it there
        print(self.uniform_gain)
        self.transition_mat = PointProcessDecoder._sungod_transition_matrix(self.uniform_gain)        

        self.current_spike_count = 0
        self.pos_counter = 0
        self.current_vel = 0

        self._ripple_thresh_states = {}

        self.post_sum_bin_length = 20
        self.posterior_sum_time_bin = np.zeros((self.post_sum_bin_length,9))
        self.posterior_sum_result = np.zeros((1,9))

    @staticmethod
    def _create_transition_matrix(pos_delta, num_bins, arm_coor, uniform_gain=0.01):

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

            # Setup transition matrix
        x_bins = np.linspace(0, pos_delta*(num_bins-1), num_bins)

        transition_mat = np.ones([num_bins, num_bins])
        for bin_ii in range(num_bins):
            transition_mat[bin_ii, :] = gaussian(x_bins, x_bins[bin_ii], 3)

        # uniform offset
        uniform_dist = np.ones(transition_mat.shape)

        # apply no-animal boundary

        transition_mat = apply_no_anim_boundary(x_bins, arm_coor, transition_mat)
        uniform_dist = apply_no_anim_boundary(x_bins, arm_coor, uniform_dist)

        # normalize transition matrix
        transition_mat = transition_mat/(transition_mat.sum(axis=0)[None, :])
        transition_mat[np.isnan(transition_mat)] = 0

        # normalize uniform offset
        uniform_dist = uniform_dist/(uniform_dist.sum(axis=0)[None, :])
        uniform_dist[np.isnan(uniform_dist)] = 0

        # apply uniform offset
        transition_mat = transition_mat * (1 - uniform_gain) + uniform_dist * uniform_gain

        return transition_mat

    @staticmethod
    def _sungod_transition_matrix(uniform_gain):
        # updated for 2 pixels 8-14-19
        # arm_coords updated for 8 pixels 8-15-19
        # NOTE: by rounding up for binning position of outer arms, we get no position in first bin of each arm
        # we could just move the first position here in arm coords and then each arm will start 1 bin higher
        # based on looking at counts from position this should work, so each arm is 11 units

        arm_coords = np.array([[0,7],[12,23],[28,39],[44,55],[60,71],[76,87],[92,103],[108,119],[124,135]])
        #arm_coords = np.array([[0,7],[11,23],[27,39],[43,55],[59,71],[75,87],[91,103],[107,119],[123,135]])
        max_pos = arm_coords[-1][-1] + 1
        pos_bins = np.arange(0,max_pos,1)

        uniform_gain = uniform_gain

        from scipy.sparse import diags
        n = len(pos_bins)
        transition_mat = np.zeros([n,n])
        k = np.array([(1/3)*np.ones(n-1),(1/3)*np.ones(n),(1/3)*np.ones(n-1)])
        offset = [-1,0,1]
        transition_mat = diags(k,offset).toarray()
        box_end_bin = arm_coords[0,1]

        for x in arm_coords[:,0]:
            transition_mat[int(x),int(x)] = (5/9)
            transition_mat[box_end_bin,int(x)] = (1/9)
            transition_mat[int(x),box_end_bin] = (1/9)

        for y in arm_coords[:,1]:
            transition_mat[int(y),int(y)] = (2/3)

        transition_mat[box_end_bin,0] = 0
        transition_mat[0,box_end_bin] = 0
        transition_mat[box_end_bin,box_end_bin] = 0
        transition_mat[0,0] = (2/3)
        transition_mat[box_end_bin-1, box_end_bin-1] = (5/9)
        transition_mat[box_end_bin-1,box_end_bin] = (1/9)
        transition_mat[box_end_bin, box_end_bin-1] = (1/9)

        # uniform offset (gain, currently 0.0001)
        # 9-1-19 this is now taken from config file
        #uniform_gain = 0.0001
        uniform_dist = np.ones(transition_mat.shape)*uniform_gain

        # apply uniform offset
        transition_mat = transition_mat + uniform_dist

        # apply no animal boundary - make gaps between arms
        transition_mat = apply_no_anim_boundary(pos_bins, arm_coords, transition_mat)

        # to smooth: take the transition matrix to a power
        transition_mat = np.linalg.matrix_power(transition_mat,1)

        # normalize transition matrix
        transition_mat = transition_mat/(transition_mat.sum(axis=0)[None, :])

        transition_mat[np.isnan(transition_mat)] = 0

        return transition_mat

    def select_ntrodes(self, ntrode_list):
        self.ntrode_list = ntrode_list
        self.firing_rate = {elec_grp_id: np.ones(self.pos_bins)
                            for elec_grp_id in self.ntrode_list}

    def add_observation(self, spk_elec_grp_id, spk_pos_hist):
        self.firing_rate[spk_elec_grp_id][self.cur_pos_ind] += 1

        self.observation *= spk_pos_hist
        self.observation = self.observation / np.max(self.observation)
        self.current_spike_count += 1

    def update_position(self, pos_timestamp, pos_data, vel_data):
        # Convert position to bin index in histogram count
        self.cur_pos_time = pos_timestamp
        self.cur_pos = pos_data
        self.cur_vel = vel_data
        #print('update position result:',self.cur_pos)
        self.cur_pos_ind = int((self.cur_pos - self.pos_range[0]) /
                               self.pos_delta)

        if abs(self.cur_vel) >= self.config['encoder']['vel']:
            self.occ[self.cur_pos_ind] += 1

            self.pos_counter += 1
            if self.pos_counter % 10000 == 0:
                #print('prob_no_spike_occupancy: ',self.occ)
                print('number of position entries decode: ',self.pos_counter)

    def update_ripple_threshold_state(self, timestamp, elec_grp_id, threshold_state):
        self._ripple_thresh_states.setdefault(elec_grp_id, 0)
        # only write state if state changed

        self._ripple_thresh_states[elec_grp_id] = threshold_state
        num_above = 0
        for state in self._ripple_thresh_states.values():
            num_above += state

        return num_above

    def increment_no_spike_bin(self):

        prob_no_spike = {}
        global_prob_no = np.ones(self.pos_bins)
        for tet_id, tet_fr in self.firing_rate.items():
            # Normalize firing rate
            tet_fr_norm = tet_fr / tet_fr.sum()
            # MEC 9-3-19 to turn off prob_no_spike
            #prob_no_spike[tet_id] = np.ones(self.pos_bins)
            prob_no_spike[tet_id] = np.exp(-self.time_bin_size/30000 *
                                           tet_fr_norm / self.occ)

            global_prob_no *= prob_no_spike[tet_id]
        global_prob_no /= global_prob_no.sum()
        
        # MEC print statement added
        #if self.pos_counter % 10000 == 0:
        #    print('global prob no spike: ',global_prob_no)

        # Compute likelihood for all previous 0 spike bins
        # update last posterior
        self.prev_posterior = self.posterior

        # Compute no spike likelihood
        #for prob_no in prob_no_spike.values():
        #    self.likelihood *= prob_no
        self.likelihood = global_prob_no

        # Compute posterior for no spike
        self.posterior = self.likelihood * (self.transition_mat * self.prev_posterior).sum(axis=1)
        # Normalize
        self.posterior = self.posterior / self.posterior.sum()

        return self.posterior

    def increment_bin(self):

        # Compute conditional intensity function (probability of no spike)
        prob_no_spike = {}
        global_prob_no = np.ones(self.pos_bins)
        for tet_id, tet_fr in self.firing_rate.items():
            # Normalize firing rate
            tet_fr_norm = tet_fr / tet_fr.sum()
            # MEC 9-3-19 to turn off prob_no_spike
            #prob_no_spike[tet_id] = np.ones(self.pos_bins)
            prob_no_spike[tet_id] = np.exp(-self.time_bin_size/30000 *
                                           tet_fr_norm / self.occ)

            global_prob_no *= prob_no_spike[tet_id]
        global_prob_no /= global_prob_no.sum()

        # MEC print statement added
        #if self.pos_counter % 10000 == 0:
        #    print('global prob no spike: ',global_prob_no)

        # Update last posterior
        self.prev_posterior = self.posterior

        # Compute likelihood for previous bin with spikes
        self.likelihood = self.observation * global_prob_no

        # Compute posterior
        self.posterior = self.likelihood * (self.transition_mat * self.prev_posterior).sum(axis=1)
        # Normalize
        self.posterior = self.posterior / self.posterior.sum()

        # Save resulting posterior
        # self.record.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
        #                          self.current_time_bin * self.time_bin_size,
        #                          *self.posterior)

        self.current_spike_count = 0
        self.observation = np.ones(self.pos_bins)

        return self.posterior

    def calculate_posterior_arm_sum(self, posterior, ripple_time_bin):
        #this needs to hold on to the past 20 time bins so maybe we need to intailize like we do the velocity filter

        arm_coords_rt = [[0,7],[12,23],[28,39],[44,55],[60,71],[76,87],[92,103],[108,119],[124,135]]
        #post_sum_bin_length = 20
        posterior_for_sum = posterior
        ripple_time_bin = ripple_time_bin
        #posterior_sum_time_bin = np.zeros((post_sum_bin_length,9))
        #posterior_sum_result = np.zeros((1,9))

        # calculate the sum of the decode for each arm (box, then arms 1-8)
        # this currently only does the sum of 1 time bin, we need the cumulative sum
        # posterior is just an array 136 items long, so this should work

        # we need to replace this loop with saving the results from each of the last 20 time bins

        #reset posterior_sum_time_bin
        if ripple_time_bin == 0:
            print('posterior sum reset')
            self.posterior_sum_time_bin = np.zeros((self.post_sum_bin_length,9))
            self.posterior_sum_result = np.zeros((1,9))
            print(self.posterior_sum_result)
            print(self.posterior_sum_time_bin[0])

        #shift all values up one row to make room for next bin
        self.posterior_sum_time_bin = np.roll(self.posterior_sum_time_bin, -1, axis=0)
        self.posterior_sum_time_bin[self.post_sum_bin_length-1,:] = 0
        for j in np.arange(0,len(arm_coords_rt),1):
            self.posterior_sum_time_bin[self.post_sum_bin_length-1,j] = posterior_for_sum[arm_coords_rt[j][0]:(arm_coords_rt[j][1]+1)].sum()
        for m in np.arange(0,self.post_sum_bin_length):
            self.posterior_sum_result = self.posterior_sum_result + self.posterior_sum_time_bin[-(self.post_sum_bin_length-m)]
            if self.posterior_sum_result[0,0] > 0:
                self.posterior_sum_result = self.posterior_sum_result/self.posterior_sum_result.sum()
        return self.posterior_sum_result


class PPDecodeManager(realtime_base.BinaryRecordBaseWithTiming):
    def __init__(self, rank, config, local_rec_manager, send_interface: DecoderMPISendInterface,
                 spike_decode_interface: SpikeDecodeRecvInterface, ripple_decode_interface: RippleDecodeRecvInterface,
                 pos_interface: realtime_base.DataSourceReceiver):
        super(PPDecodeManager, self).__init__(rank=rank,
                                              local_rec_manager=local_rec_manager,
                                              send_interface=send_interface,
                                              rec_ids=[realtime_base.RecordIDs.DECODER_OUTPUT,
                                                       realtime_base.RecordIDs.DECODER_MISSED_SPIKES],
                                              rec_labels=[['timestamp', 'real_pos_time', 'real_pos','spike_count',
                                                            'ripple','ripple_number','ripple_length','shortcut_message',
                                                            'box','arm1','arm2','arm3','arm4','arm5','arm6','arm7','arm8'] +
                                                          ['x{:0{dig}d}'.
                                                           format(x, dig=len(str(config['encoder']
                                                                                 ['position']['bins'])))
                                                           for x in range(config['encoder']['position']['bins'])],
                                                          ['timestamp', 'elec_grp_id', 'real_bin', 'late_bin']],
                                              rec_formats=['qddqqqqqddddddddd'+'d'*config['encoder']['position']['bins'],
                                                           'qiii'])
                                                #i think if you change second q to d above, then you can replace real_pos_time
                                                # with velocity

        self.config = config
        self.mpi_send = send_interface
        self.spike_dec_interface = spike_decode_interface
        self.ripple_dec_interface = ripple_decode_interface
        self.pos_interface = pos_interface

        #initialize velocity calc and linear position assignment functions
        self.current_vel = 0
        self.velCalc = VelocityCalculator()
        self.linPosAssign = LinearPositionAssignment()

        # Send binary record register message
        # self.mpi_send.send_record_register_messages(self.get_record_register_messages())

        self.msg_counter = 0
        self.ntrode_list = []

        self.current_time_bin = 0
        self.time_bin_size = self.config['pp_decoder']['bin_size']
        self.pp_decoder = PointProcessDecoder(pos_range=[self.config['encoder']['position']['lower'],
                                                         self.config['encoder']['position']['upper']],
                                              pos_bins=self.config['encoder']['position']['bins'],
                                              time_bin_size=self.time_bin_size,
                                              arm_coor=self.config['encoder']['position']['arm_pos'],
                                              uniform_gain=config['pp_decoder']['trans_mat_uniform_gain'],
                                              config = self.config)
        # 7-2-19, added spike count for each decoding bin
        self.spike_count = 0
        self.ripple_thresh_decoder = False
        self.ripple_time_bin = 0
        self.no_ripple_time_bin = 0
        self.replay_target_arm = self.config['pp_decoder']['replay_target_arm']
        self.posterior_arm_sum = np.zeros((1,9))
        self.num_above = 0
        self.ripple_number = 0
        self.shortcut_message_sent = False

    def register_pos_interface(self):
        # Register position, right now only one position channel is supported
        self.pos_interface.register_datatype_channel(-1)
        if self.config['datasource'] == 'trodes':
            self.trodessource = True
            #done, MEC 6-30-19
            #self.class_log.warning("*****Position data subscribed, but update_position() needs to be changed to fit CameraModule position data. Delete this message when implemented*****")
        else:
            self.trodessource = False

    def turn_on_datastreams(self):
        self.pos_interface.start_all_streams()

    def select_ntrodes(self, ntrode_list):
        self.ntrode_list = ntrode_list
        self.pp_decoder.select_ntrodes(ntrode_list)

    def process_next_data(self):
        spike_dec_msg = self.spike_dec_interface.__next__()
        ripple_dec_message = self.ripple_dec_interface.__next__()
        
        if ripple_dec_message is not None:
            self.num_above = self.pp_decoder.update_ripple_threshold_state(timestamp=ripple_dec_message.timestamp,
                                                        elec_grp_id=ripple_dec_message.elec_grp_id,
                                                        threshold_state=ripple_dec_message.threshold_state)
            if self.num_above >= self.config['ripple']['RippleParameterMessage']['n_above_thresh']:
                self.ripple_thresh_decoder = True
                # it looks like there are dozens of entries for each time bin, when it crosses threshold
                # and each crossing time only seems to last 1 time bin - is that what we expect?
                #print('combined ripple threshold crossed',num_above,ripple_dec_message.timestamp,ripple_dec_message.elec_grp_id,self.current_time_bin)

            else:
                self.ripple_thresh_decoder = False
                #print(self.ripple_thresh_decoder)

            #if ripple_dec_message.threshold_state > 0:
            #    print('ripple message: ',ripple_dec_message)
            pass

        if spike_dec_msg is not None:
            # okay so the problem is that it is missing lots of lfp data because spike_dec_msg skips a lot - empty bins?
            # correct - only run when a spike comes in - so if we have few empty bins that will be okay

            self.record_timing(timestamp=spike_dec_msg.timestamp, elec_grp_id=spike_dec_msg.elec_grp_id,
                               datatype=datatypes.Datatypes.SPIKES, label='dec_recv')
            #print('message recieved by decoder:',spike_dec_msg.timestamp,spike_dec_msg.elec_grp_id)
            #print('ripple message recieved by decoder:',ripple_dec_message.timestamp,ripple_dec_message.elec_grp_id)


            # Update firing rate

            # Calculate which time bin spike belongs to
            if self.current_time_bin == 0:
                self.current_time_bin = int(math.floor(spike_dec_msg.timestamp/self.config['pp_decoder']['bin_size']))
                spike_time_bin = self.current_time_bin
            else:
                spike_time_bin = int(math.floor(spike_dec_msg.timestamp/self.config['pp_decoder']['bin_size']))

            
            if spike_time_bin == self.current_time_bin:
                # Spike is in current time bin
                self.pp_decoder.add_observation(spk_elec_grp_id=spike_dec_msg.elec_grp_id,
                                                spk_pos_hist=spike_dec_msg.pos_hist)
                self.spike_count += 1
                pass

            elif spike_time_bin > self.current_time_bin:
                # Spike is in next time bin, compute posterior based on observations, advance to tracking next time bin

                # increment last bin with spikes
                posterior = self.pp_decoder.increment_bin()
                #print(posterior)
                #print(posterior.shape)

                # add 1 to spike_count because it isnt added when starting a new bin, so 1st spike is missed
                self.spike_count += 1

                # TO DO call function to calculate sum of decode by arm
                # is ripple threshold crossed? - pull this in from above

                # check if sum for one particular arm is > 0.8
                # if so, then send shortcut message
                #print('decode time bin',self.current_time_bin)

                # hmmm reset of sum with ripple time bin isnt working currently
                # also seems like some ripples are being missed by the sum function
                if self.ripple_thresh_decoder == True:
                    print('ripple thresh crossed decode time bin',self.current_time_bin,' number tets: ', self.num_above)
                    if self.ripple_time_bin == 0:
                        self.ripple_number += 1
                        print('ripple number: ',self.ripple_number)
                    self.no_ripple_time_bin = 0
                    self.posterior_arm_sum = self.pp_decoder.calculate_posterior_arm_sum(posterior, self.ripple_time_bin)
                    self.ripple_time_bin += 1
                    print('posterior sum: ', self.posterior_arm_sum, self.current_time_bin, self.ripple_time_bin)
                    #print('arm 0 sum: ',posterior_arm_sum[0][1])
                    if (self.ripple_time_bin > 2) & (self.posterior_arm_sum[0][self.replay_target_arm] > 0.8):
                        # send shortcut message
                        # start lockout / reset function
                        self.shortcut_message_sent = True
                        print('arm', self.replay_target_arm, 'sum above 80 percent for time bins: ',self.ripple_time_bin)

                elif self.ripple_thresh_decoder == False:
                    #print('no ripple in decoder')
                    self.no_ripple_time_bin += 1
                    if self.no_ripple_time_bin > 2:
                        self.ripple_time_bin = 0


                #posterior = np.ones(130)
                # try replacing self.pp_decoder.cur_pos_time with self.cur_vel to get both position and velocity in the dataframe
                # and once more in the next paragraph
                self.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
                                  self.current_time_bin * self.time_bin_size,
                                  self.current_vel,
                                  self.pp_decoder.cur_pos,
                                  self.spike_count,
                                  self.ripple_thresh_decoder, self.ripple_number, self.ripple_time_bin,self.shortcut_message_sent,
                                  self.posterior_arm_sum[0][0],self.posterior_arm_sum[0][1],
                                  self.posterior_arm_sum[0][2],self.posterior_arm_sum[0][3],self.posterior_arm_sum[0][4],
                                  self.posterior_arm_sum[0][5],self.posterior_arm_sum[0][6],self.posterior_arm_sum[0][7],
                                  self.posterior_arm_sum[0][8],
                                  *posterior)

                self.current_time_bin += 1
                self.shortcut_message_sent = False

                for no_spk_ii in range(spike_time_bin - self.current_time_bin - 1):
                    #spike_count is set to 0 for no_spike_bins
                    posterior = self.pp_decoder.increment_no_spike_bin()
                    #posterior = np.ones(130)

                    # TO DO call function to calculate sum of decode by arm
                    # calculate the sum during ripples
                    # reset the matrix (name) to zeros if 3 or more time bins with no ripple

                    # check if sum for one particular arm is > 0.8
                    # if so, then send shortcut message
                    #print('decode time bin',self.current_time_bin)
                    if self.ripple_thresh_decoder == True:
                        print('ripple thresh crossed decode time bin',self.current_time_bin,' number tets: ', self.num_above)
                        if self.ripple_time_bin == 0:
                            self.ripple_number += 1
                            print('ripple number: ',self.ripple_number)
                        self.no_ripple_time_bin = 0
                        self.posterior_arm_sum = self.pp_decoder.calculate_posterior_arm_sum(posterior, self.ripple_time_bin)
                        self.ripple_time_bin += 1
                        print('posterior sum: ', self.posterior_arm_sum, self.current_time_bin, self.ripple_time_bin)
                        #print('arm 0 sum: ',posterior_arm_sum[0][1])
                        if (self.ripple_time_bin > 2) & (self.posterior_arm_sum[0][self.replay_target_arm] > 0.8):
                            # send shortcut message
                            # start lockout / reset function
                            self.shortcut_message_sent = True
                            print('arm', self.replay_target_arm, 'sum above 80 percent for time bins: ',self.ripple_time_bin)

                    elif self.ripple_thresh_decoder == False:
                        self.no_ripple_time_bin += 1
                        if self.no_ripple_time_bin > 2:
                            self.ripple_time_bin = 0

                    self.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
                                      self.current_time_bin * self.time_bin_size,
                                      self.current_vel,
                                      self.pp_decoder.cur_pos,
                                      0,
                                      self.ripple_thresh_decoder, self.ripple_number, self.ripple_time_bin,self.shortcut_message_sent,
                                      self.posterior_arm_sum[0][0],self.posterior_arm_sum[0][1],
                                      self.posterior_arm_sum[0][2],self.posterior_arm_sum[0][3],self.posterior_arm_sum[0][4],
                                      self.posterior_arm_sum[0][5],self.posterior_arm_sum[0][6],self.posterior_arm_sum[0][7],
                                      self.posterior_arm_sum[0][8],
                                      *posterior)
                    self.current_time_bin += 1
                    self.shortcut_message_sent = False

                self.pp_decoder.add_observation(spk_elec_grp_id=spike_dec_msg.elec_grp_id,
                                                spk_pos_hist=spike_dec_msg.pos_hist)

                # Increment current time bin to latest spike
                self.current_time_bin = spike_time_bin

                # reset spike count to 0
                self.spike_count = 0
                pass

            elif spike_time_bin < self.current_time_bin:
                self.write_record(realtime_base.RecordIDs.DECODER_MISSED_SPIKES,
                                  spike_dec_msg.timestamp, spike_dec_msg.elec_grp_id,
                                  spike_time_bin, self.current_time_bin)
                # Spike is in an old time bin, discard and mark as missed
                self.class_log.debug('Spike was excluded from PP decode calculation, arrived late.')
                pass

            self.msg_counter += 1
            if self.msg_counter % 1000 == 0:
                self.class_log.debug('Received {} decoded messages.'.format(self.msg_counter))

            self.record_timing(timestamp=spike_dec_msg.timestamp, elec_grp_id=spike_dec_msg.elec_grp_id,
                               datatype=datatypes.Datatypes.SPIKES, label='dec_proc')

            pass

        pos_msg = self.pos_interface.__next__()

        if pos_msg is not None:
            # self.class_log.debug("Pos msg received.")
            pos_data = pos_msg[0]
            #print(pos_data)
            if not self.trodessource:
                self.pp_decoder.update_position(pos_timestamp=pos_data.timestamp, pos_data=pos_data.x)
            else:
                #self.pp_decoder.update_position(pos_timestamp=pos_data.timestamp, pos_data=pos_data.x)
                # we want to use linearized position here
                self.current_vel = self.velCalc.calculator(pos_data.x, pos_data.y)
                current_pos = self.linPosAssign.assign_position(pos_data.segment, pos_data.position)

                self.pp_decoder.update_position(pos_timestamp=pos_data.timestamp, pos_data=current_pos, vel_data=self.current_vel)


                #print(pos_data.x, pos_data.segment)
                #TODO implement trodes cameramodule update position function
                #If data source is trodes, then pos_data is of class CameraModulePoint, in datatypes.py
                #   pos_data.timestamp: trodes timestamp
                #   pos_data.segment:   linear track segment animal is on (0 if none defined)
                #   pos_data.position:  position along segment (0 if no linear tracks defined)
                #   pos_data.x and y:   raw coordinates of animal, (0,0) is top left of image
                #                        bottom right is full resolution of image
                # Update position function implementation is left
                pass


class BayesianDecodeManager(realtime_base.BinaryRecordBaseWithTiming):
    def __init__(self, rank, config, local_rec_manager, send_interface: DecoderMPISendInterface,
                 spike_decode_interface: SpikeDecodeRecvInterface):
        super(BayesianDecodeManager, self).__init__(rank=rank,
                                                    local_rec_manager=local_rec_manager,
                                                    send_interface=send_interface,
                                                    rec_ids=[realtime_base.RecordIDs.DECODER_OUTPUT],
                                                    rec_labels=[['timestamp'] +
                                                                ['x'+str(x) for x in
                                                                range(config['encoder']['position']['bins'])]],
                                                    rec_formats=['q'+'d'*config['encoder']['position']['bins']])

        self.config = config
        self.mpi_send = send_interface
        self.spike_interface = spike_decode_interface

        # Send binary record register message
        # self.mpi_send.send_record_register_messages(self.get_record_register_messages())

        self.msg_counter = 0

        self.current_time_bin = 0
        self.current_est_pos_hist = np.ones(self.config['encoder']['position']['bins'])
        self.current_spike_count = 0
        self.ntrode_list = []

    def turn_on_datastreams(self):
        # Do nothing, no datastreams for this decoder
        pass

    def select_ntrodes(self, ntrode_list):
        self.ntrode_list = ntrode_list

    def process_next_data(self):
        spike_dec_msg = self.spike_interface.__next__()

        if spike_dec_msg is not None:

            self.record_timing(timestamp=spike_dec_msg.timestamp, elec_grp_id=spike_dec_msg.elec_grp_id,
                               datatype=datatypes.Datatypes.SPIKES, label='dec_recv')

            if self.current_time_bin == 0:
                self.current_time_bin = math.floor(spike_dec_msg.timestamp/self.config['bayesian_decoder']['bin_size'])
                spike_time_bin = self.current_time_bin
            else:
                spike_time_bin = math.floor(spike_dec_msg.timestamp/self.config['bayesian_decoder']['bin_size'])

            if spike_time_bin == self.current_time_bin:
                # Spike is in current time bin
                self.current_est_pos_hist *= spike_dec_msg.pos_hist
                self.current_est_pos_hist = self.current_est_pos_hist / np.max(self.current_est_pos_hist)
                self.current_spike_count += 1

            elif spike_time_bin > self.current_time_bin:
                # Spike is in next time bin, advance to tracking next time bin
                self.write_record(realtime_base.RecordIDs.DECODER_OUTPUT,
                                  self.current_time_bin*self.config['bayesian_decoder']['bin_size'],
                                  *self.current_est_pos_hist)
                self.current_spike_count = 1
                self.current_est_pos_hist = spike_dec_msg.pos_hist
                self.current_time_bin = spike_time_bin

            elif spike_time_bin < self.current_time_bin:
                # Spike is in an old time bin, discard and mark as missed
                self.class_log.debug('Spike was excluded from Bayesian decode calculation, arrived late.')
                pass

            self.msg_counter += 1
            if self.msg_counter % 1000 == 0:
                self.class_log.debug('Received {} decoded messages.'.format(self.msg_counter))

            pass
            # self.class_log.debug(spike_dec_msg)


class DecoderRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, decode_manager: BayesianDecodeManager):
        super(DecoderRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.dec_man = decode_manager

        self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def __next__(self):
        rdy, msg = self.req.test()
        if rdy:
            self.process_request_message(msg)

            self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def process_request_message(self, message):
        if isinstance(message, realtime_base.TerminateMessage):
            self.class_log.debug("Received TerminateMessage")
            raise StopIteration()

        elif isinstance(message, binary_record.BinaryRecordCreateMessage):
            self.dec_man.set_record_writer_from_message(message)

        elif isinstance(message, realtime_base.ChannelSelection):
            self.class_log.debug("Received NTrode channel selection {:}.".format(message.ntrode_list))
            self.dec_man.select_ntrodes(message.ntrode_list)

        elif isinstance(message, realtime_base.TimeSyncInit):
            self.class_log.debug("Received TimeSyncInit.")
            self.dec_man.sync_time()

        elif isinstance(message, realtime_base.TurnOnDataStream):
            self.class_log.debug("Turn on data stream")
            self.dec_man.turn_on_datastreams()

        elif isinstance(message, realtime_base.TimeSyncSetOffset):
            self.dec_man.update_offset(message.offset_time)

        elif isinstance(message, realtime_base.StartRecordMessage):
            self.dec_man.start_record_writing()

        elif isinstance(message, realtime_base.StopRecordMessage):
            self.dec_man.stop_record_writing()


class DecoderProcess(realtime_base.RealtimeProcess):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(DecoderProcess, self).__init__(comm=comm, rank=rank, config=config)

        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
                                                                          manager_rank=config['rank']['supervisor'])

        self.terminate = False

        self.mpi_send = DecoderMPISendInterface(comm=comm, rank=rank, config=config)
        self.spike_decode_interface = SpikeDecodeRecvInterface(comm=comm, rank=rank, config=config)
        self.ripple_decode_interface = RippleDecodeRecvInterface(comm=comm, rank=rank, config=config)

        if config['datasource'] == 'simulator':
            self.pos_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                       rank=self.rank,
                                                                       config=self.config,
                                                                       datatype=datatypes.Datatypes.LINEAR_POSITION)
        elif config['datasource'] == 'trodes':
            self.pos_interface = simulator_process.TrodesDataReceiver(comm=self.comm,
                                                                       rank=self.rank,
                                                                       config=self.config,
                                                                       datatype=datatypes.Datatypes.LINEAR_POSITION)

        if config['decoder'] == 'bayesian_decoder':
            self.dec_man = BayesianDecodeManager(rank=rank, config=config,
                                                 local_rec_manager=self.local_rec_manager,
                                                 send_interface=self.mpi_send,
                                                 spike_decode_interface=self.spike_decode_interface)
        elif config['decoder'] == 'pp_decoder':
            self.dec_man = PPDecodeManager(rank=rank, config=config,
                                           local_rec_manager=self.local_rec_manager,
                                           send_interface=self.mpi_send,
                                           spike_decode_interface=self.spike_decode_interface,
                                           ripple_decode_interface=self.ripple_decode_interface,
                                           pos_interface=self.pos_interface)

        self.mpi_recv = DecoderRecvInterface(comm=comm, rank=rank, config=config, decode_manager=self.dec_man)

        # First Barrier to finish setting up nodes

        self.class_log.debug("First Barrier")
        self.comm.Barrier()

    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):

        self.dec_man.setup_mpi()
        self.dec_man.register_pos_interface()

        try:
            while not self.terminate:
                self.dec_man.process_next_data()
                self.mpi_recv.__next__()

        except StopIteration as ex:
            self.class_log.info('Terminating DecoderProcess (rank: {:})'.format(self.rank))

        self.class_log.info("Decoding Process reached end, exiting.")
