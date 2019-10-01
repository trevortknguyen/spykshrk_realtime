import struct
import spykshrk.realtime.realtime_base
import spykshrk.realtime.realtime_logging as rt_logging
import spykshrk.realtime.realtime_base as realtime_base
import spykshrk.realtime.simulator.simulator_process as simulator_process
import spykshrk.realtime.ripple_process as ripple_process
import spykshrk.realtime.decoder_process as decoder_process
import spykshrk.realtime.binary_record as binary_record
import spykshrk.realtime.timing_system as timing_system
import spykshrk.realtime.datatypes as datatypes

from mpi4py import MPI
import numpy as np
import time

import sys

# try:
#     __IPYTHON__
#     from IPython.terminal.debugger import TerminalPdb
#     bp = TerminalPdb(color_scheme='linux').set_trace
# except NameError as err:
#     print('Warning: NameError ({}), not using ipython (__IPYTHON__ not set), disabling IPython TerminalPdb.'.
#           format(err))
#     bp = lambda: None
# except AttributeError as err:
#     print('Warning: Attribute Error ({}), disabling IPython TerminalPdb.'.format(err))
#     bp = lambda: None

from spikegadgets import trodesnetwork as tnp

class MainProcessClient(tnp.AbstractModuleClient):
    def __init__(self, name, addr, port, config):
        super().__init__(name, addr, port)
        # self.main_manager = main_manager
        self.config = config
        self.started = False
        self.ntrode_list_sent = False
        self.terminated = False
    
    def registerTerminationCallback(self, callback):
        self.terminate = callback

    def registerStartupCallback(self, callback):
        self.startup = callback

    #MEC added: to get ripple tetrode list
    def registerStartupCallbackRippleTetrodes(self, callback):
        self.startupRipple = callback
    
    def recv_acquisition(self, command, timestamp):
        if command == tnp.acq_PLAY:
            if not self.ntrode_list_sent:
                self.startup(self.config['trodes_network']['decoding_tetrodes'])
                #added MEC
                self.startupRipple(self.config['trodes_network']['ripple_tetrodes'])
                self.started = True
                self.ntrode_list_sent = True

        # if command == tnp.acq_STOP:
        #     if not self.terminated:
        #         # self.main_manager.trigger_termination()
        #         self.terminate()
        #         self.terminated = True
        #         self.started = False

    def recv_quit(self):
        self.terminate()

class MainProcess(realtime_base.RealtimeProcess):

    def __init__(self, comm: MPI.Comm, rank, config):

        self.comm = comm    # type: MPI.Comm
        self.rank = rank
        self.config = config

        super().__init__(comm=comm, rank=rank, config=config)

        # MEC added
        self.stim_decider_send_interface = StimDeciderMPISendInterface(comm=comm, rank=rank, config=config)

        self.stim_decider = StimDecider(rank=rank, config=config,
                                        send_interface=self.stim_decider_send_interface)

        self.posterior_recv_interface = PosteriorSumRecvInterface(comm=comm, rank=rank, config=config,
                                                                  stim_decider=self.stim_decider)

        #self.stim_decider = StimDecider(rank=rank, config=config,
        #                                send_interface=StimDeciderMPISendInterface(comm=comm,
        #                                                                           rank=rank,
        #                                                                           config=config))

        self.data_recv = StimDeciderMPIRecvInterface(comm=comm, rank=rank, config=config,
                                                     stim_decider=self.stim_decider)

        self.send_interface = MainMPISendInterface(comm=comm, rank=rank, config=config)

        self.manager = MainSimulatorManager(rank=rank, config=config, parent=self, send_interface=self.send_interface,
                                            stim_decider=self.stim_decider)
        
        if config['datasource'] == 'trodes':
            self.networkclient = MainProcessClient("SpykshrkMainProc", config['trodes_network']['address'],config['trodes_network']['port'], self.config)
            if self.networkclient.initialize() != 0:
                print("Network could not successfully initialize")
                del self.networkclient
                quit()
            self.networkclient.registerStartupCallback(self.manager.handle_ntrode_list)
            #added MEC
            self.networkclient.registerStartupCallbackRippleTetrodes(self.manager.handle_ripple_ntrode_list)
            self.networkclient.registerTerminationCallback(self.manager.trigger_termination)

        self.recv_interface = MainSimulatorMPIRecvInterface(comm=comm, rank=rank,
                                                            config=config, main_manager=self.manager)

        self.terminate = False

        self.mpi_status = MPI.Status()

        # First Barrier to finish setting up nodes, waiting for Simulator to send ntrode list.
        # The main loop must be active to receive binary record registration messages, so the
        # first Barrier is placed here.
        self.class_log.debug("First Barrier")
        self.send_interface.all_barrier()
        self.class_log.debug("Past First Barrier")


    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):
        # self.thread.start()

        # Synchronize rank times immediately
        last_time_bin = int(time.time())

        while not self.terminate:

            # Synchronize rank times
            if self.manager.time_sync_on:
                current_time_bin = int(time.time())
                if current_time_bin >= last_time_bin+10:
                    self.manager.synchronize_time()
                    last_time_bin = current_time_bin

            self.recv_interface.__next__()
            self.data_recv.__next__()
            self.posterior_recv_interface.__next__()

        self.class_log.info("Main Process Main reached end, exiting.")

class StimulationDecision(rt_logging.PrintableMessage):
    """"Message containing whether or not at a given timestamp a ntrode's ripple filter threshold is crossed.

    This message has helper serializer/deserializer functions to be used to speed transmission.
    """
    _byte_format = 'Ii'

    def __init__(self, timestamp, stim_decision):
        self.timestamp = timestamp
        self.stim_decision = stim_decision

    def pack(self):
        return struct.pack(self._byte_format, self.timestamp, self.stim_decision)

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, stim_decision = struct.unpack(cls._byte_format, message_bytes)
        return cls(timestamp=timestamp, stim_decision=stim_decision)


class StimDeciderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(StimDeciderMPISendInterface, self).__init__(comm=comm, rank=rank, config=config)
        self.comm = comm
        self.rank = rank
        self.config = config

    def start_stimulation(self):
        pass

    def send_record_register_messages(self, record_register_messages):
        self.class_log.debug("Sending binary record registration messages.")
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)


class StimDecider(realtime_base.BinaryRecordBaseWithTiming):
    def __init__(self, rank, config,
                 send_interface: StimDeciderMPISendInterface, ripple_n_above_thresh=sys.maxsize,
                 lockout_time=0):

        super().__init__(rank=rank,
                         local_rec_manager=binary_record.RemoteBinaryRecordsManager(manager_label='state',
                                                                                    local_rank=rank,
                                                                                    manager_rank=
                                                                                    config['rank']['supervisor']),
                         send_interface=send_interface,
                         rec_ids=[realtime_base.RecordIDs.STIM_STATE,
                                  realtime_base.RecordIDs.STIM_LOCKOUT,
                                  realtime_base.RecordIDs.STIM_MESSAGE],
                         rec_labels=[['timestamp', 'elec_grp_id', 'threshold_state'],
                                     ['timestamp', 'time', 'lockout_num', 'lockout_state','tets_above_thresh'],
                                     ['bin_timestamp', 'spike_timestamp','time', 'stim_sent', 'ripple_number', 'ripple_time_bin']],
                         rec_formats=['Iii',
                                      'Idiiq',
                                      'IIdiii'])
        self.rank = rank
        self._send_interface = send_interface
        self._ripple_n_above_thresh = ripple_n_above_thresh
        self._lockout_time = lockout_time
        self._ripple_thresh_states = {}
        self._enabled = False
        self.config = config

        self._last_lockout_timestamp = 0
        self._lockout_count = 0
        self._in_lockout = False
        self.stim_thresh = False

        self.ripple_time_bin = 0
        self.no_ripple_time_bin = 0
        self.replay_target_arm = self.config['pp_decoder']['replay_target_arm']
        self.posterior_arm_sum = np.zeros((1,9))
        self.num_above = 0
        self.ripple_number = 0
        self.shortcut_message_sent = False

        time = MPI.Wtime()

        # Setup bin rec file
        # main_manager.rec_manager.register_rec_type_message(rec_type_message=self.get_record_register_message())

    def reset(self):
        self._ripple_thresh_states = {}

    def enable(self):
        self.class_log.info('Enabled stim decider.')
        self._enabled = True
        self.reset()

    def disable(self):
        self.class_log.info('Disable stim decider.')
        self._enabled = False
        self.reset()

    def update_n_threshold(self, ripple_n_above_thresh):
        self._ripple_n_above_thresh = ripple_n_above_thresh

    def update_lockout_time(self, lockout_time):
        self._lockout_time = lockout_time

    def update_ripple_threshold_state(self, timestamp, elec_grp_id, threshold_state):
        # Log timing
        self.record_timing(timestamp=timestamp, elec_grp_id=elec_grp_id,
                           datatype=datatypes.Datatypes.LFP, label='stim_rip_state')
        time = MPI.Wtime()

        if self._enabled:

            self._ripple_thresh_states.setdefault(elec_grp_id, 0)
            # only write state if state changed
            if self._ripple_thresh_states[elec_grp_id] != threshold_state:
                self.write_record(realtime_base.RecordIDs.STIM_STATE, timestamp, elec_grp_id, threshold_state)

            self._ripple_thresh_states[elec_grp_id] = threshold_state
            num_above = 0
            for state in self._ripple_thresh_states.values():
                num_above += state

            if self._in_lockout and (timestamp > self._last_lockout_timestamp + self._lockout_time):
                # End lockout
                self._in_lockout = False
                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, time, self._lockout_count, self._in_lockout, num_above)
                self._lockout_count += 1

            if (num_above >= self._ripple_n_above_thresh) and not self._in_lockout:
                print('tets above ripple thresh: ',num_above,timestamp)
                self._in_lockout = True
                self.stim_thresh = True
                self._last_lockout_timestamp = timestamp
                self.class_log.debug("Ripple threshold detected {}.".format(self._ripple_thresh_states))
                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, time, self._lockout_count, self._in_lockout, num_above)

                self._send_interface.start_stimulation()
                # here we want to send the stim_decider message to the decoder
                # this should be similiar to the replay threshold message from ripple_process
                # i think _in_lockout will be 1 when stim threshold is crossed
                #self._send_interface.send_stim_decision(timestamp, self.stim_thresh)
                #print('stim message function called',timestamp,time*1000)

            elif num_above < self._ripple_n_above_thresh and not self._in_lockout:
                self.stim_thresh = False
                #self._send_interface.send_stim_decision(timestamp, self.stim_thresh)

            return num_above

    def posterior_sum(self, bin_timestamp, spike_timestamp, box,arm1,arm2,arm3,arm4):
        time = MPI.Wtime()
        # caclulate running sum
        # keep track of time ripple time
        #print('posterior sum at function: ',timestamp,time*1000,self._last_lockout_timestamp)
        # try to put in the actual check for whether posterior is above 0.8 and last for 2 bins etc
        # we may need to move this function
        if self.stim_thresh == True:
            #print('posterior sum is: ',timestamp,box,time*1000)
            #print('stim threshold: ',self.stim_thresh,self._last_lockout_timestamp)
            #print('ripple time bin: ',self.ripple_time_bin)
            if self.ripple_time_bin == 0:
                self.ripple_number += 1
                print('ripple number: ',self.ripple_number)
            self.no_ripple_time_bin = 0
            self.ripple_time_bin += 1

            self.record_timing(timestamp=spike_timestamp, elec_grp_id=0,
                               datatype=datatypes.Datatypes.LFP, label='postsum_in')
            self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                              bin_timestamp, spike_timestamp, time, self.shortcut_message_sent, 
                              self.ripple_number, self.ripple_time_bin)            
            
            #this is where it decides whether or not to send shortcut message
            # move this to below so that it sends message at end of ripple not during
            # currently this is set to any arm > 0.8, next line is for 1 specific arm
            
            # need to replace this with box,arm1,arm2,etc
            #if (self.ripple_time_bin > 2) & (box > 0.8):
            #if (self.ripple_time_bin > 2) & (self.posterior_arm_sum[self.posterior_arm_sum > 0.8].shape[0] > 0):
            #if (self.ripple_time_bin > 2) & (self.posterior_arm_sum[0][self.replay_target_arm] > 0.8):
            # send shortcut message - dont sent here send after ripple is over
            # start lockout / reset function
                #self.shortcut_message_sent = True
                #print('arm', self.replay_target_arm, 'sum above 80 percent for time bins: ',self.ripple_time_bin)
                # here we should make a new dataframe saving the posterior and stim decision
                #self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                #                  timestamp, self.shortcut_message_sent, self.ripple_number, self.ripple_time_bin)

        if self.stim_thresh == False:
            #print('no ripple in decoder')

            self.no_ripple_time_bin += 1

            #send a shrotcut message if it has been a ripple (>3 bins) and now is not a ripple (>2 no ripple bins)
            if (self.no_ripple_time_bin == 3) & (self.ripple_time_bin > 3):
                self.shortcut_message_sent = True
                # return arm with max posterior 
                # send shortcut message at end of ripple
                print("end of ripple message sent",self.ripple_time_bin,self.ripple_number)
                self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                                  bin_timestamp, spike_timestamp, time, self.shortcut_message_sent, 
                                  self.ripple_number, self.ripple_time_bin)                
            if self.no_ripple_time_bin > 3:
                self.ripple_time_bin = 0
                self.shortcut_message_sent = False                

class StimDeciderMPIRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider: StimDecider):
        super(StimDeciderMPIRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.stim = stim_decider

        self.mpi_status = MPI.Status()

        self.feedback_bytes = bytearray(12)
        self.timing_bytes = bytearray(100)

        self.mpi_reqs = []
        self.mpi_statuses = []

        req_feedback = self.comm.Irecv(buf=self.feedback_bytes,
                                       tag=realtime_base.MPIMessageTag.FEEDBACK_DATA.value)
        self.mpi_statuses.append(MPI.Status())
        self.mpi_reqs.append(req_feedback)

    def __iter__(self):
        return self

    def __next__(self):
        rdy = MPI.Request.Testall(requests=self.mpi_reqs, statuses=self.mpi_statuses)

        if rdy:
            if self.mpi_statuses[0].source in self.config['rank']['ripples']:
                message = ripple_process.RippleThresholdState.unpack(message_bytes=self.feedback_bytes)
                self.stim.update_ripple_threshold_state(timestamp=message.timestamp,
                                                        elec_grp_id=message.elec_grp_id,
                                                        threshold_state=message.threshold_state)

                self.mpi_reqs[0] = self.comm.Irecv(buf=self.feedback_bytes,
                                                   tag=realtime_base.MPIMessageTag.FEEDBACK_DATA.value)
                

class PosteriorSumRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider: StimDecider):
        super(PosteriorSumRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.stim = stim_decider
        self.msg_buffer = bytearray(48)
        self.req = self.comm.Irecv(buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.POSTERIOR.value)

    def __next__(self):
        rdy = self.req.Test()
        time = MPI.Wtime()
        if rdy:

            message = decoder_process.PosteriorSum.unpack(self.msg_buffer)
            self.req = self.comm.Irecv(buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.POSTERIOR.value)

            #need to activate record_timing in this class if we want to use this here
            #self.record_timing(timestamp=timestamp, elec_grp_id=elec_grp_id,
            #                   datatype=datatypes.Datatypes.SPIKES, label='post_sum_recv')

            # okay so we are receiving the message! but now it needs to get into the stim decider
            self.stim.posterior_sum(bin_timestamp=message.bin_timestamp,spike_timestamp=message.spike_timestamp,
                                    box=message.box,arm1=message.arm1,
                                    arm2=message.arm2,arm3=message.arm3,arm4=message.arm4)             
            #print('posterior sum message supervisor: ',message.timestamp,time*1000)
            #return posterior_sum

        else:
            return None

class MainMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):

        super().__init__(comm=comm, rank=rank, config=config)

    def send_num_ntrode(self, rank, num_ntrodes):
        self.class_log.debug("Sending number of ntrodes to rank {:}".format(rank))
        self.comm.send(realtime_base.NumTrodesMessage(num_ntrodes), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_channel_selection(self, rank, channel_selects):
        self.comm.send(obj=spykshrk.realtime.realtime_base.ChannelSelection(channel_selects), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    #MEC added
    def send_ripple_channel_selection(self, rank, channel_selects):
        self.comm.send(obj=spykshrk.realtime.realtime_base.RippleChannelSelection(channel_selects), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_new_writer_message(self, rank, new_writer_message):
        self.comm.send(obj=new_writer_message, dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_start_rec_message(self, rank):
        self.comm.send(obj=realtime_base.StartRecordMessage(), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_turn_on_datastreams(self, rank):
        self.comm.send(obj=spykshrk.realtime.realtime_base.TurnOnDataStream(), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_ripple_parameter(self, rank, param_message):
        self.comm.send(obj=param_message, dest=rank, tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_ripple_baseline_mean(self, rank, mean_dict):
        self.comm.send(obj=ripple_process.CustomRippleBaselineMeanMessage(mean_dict=mean_dict), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def send_ripple_baseline_std(self, rank, std_dict):
        self.comm.send(obj=ripple_process.CustomRippleBaselineStdMessage(std_dict=std_dict), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)



    def send_time_sync_simulator(self):
        if self.config['datasource'] == 'trodes':
            ranks = list(range(self.comm.size))
            ranks.remove(self.rank)
            for rank in ranks:
                self.comm.send(obj=realtime_base.TimeSyncInit(), dest=rank,
                        tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)
        else:
            self.comm.send(obj=realtime_base.TimeSyncInit(), dest=self.config['rank']['simulator'],
                        tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def all_barrier(self):
        self.comm.Barrier()

    def send_time_sync_offset(self, rank, offset_time):
        self.comm.send(obj=realtime_base.TimeSyncSetOffset(offset_time), dest=rank,
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def terminate_all(self):
        terminate_ranks = list(range(self.comm.size))
        terminate_ranks.remove(self.rank)
        for rank in terminate_ranks:
            self.comm.send(obj=realtime_base.TerminateMessage(), dest=rank,
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)


class MainSimulatorManager(rt_logging.LoggingClass):

    def __init__(self, rank, config, parent: MainProcess, send_interface: MainMPISendInterface,
                 stim_decider: StimDecider):

        self.rank = rank
        self.config = config
        self.parent = parent
        self.send_interface = send_interface
        self.stim_decider = stim_decider

        self.time_sync_on = False

        self.rec_manager = binary_record.BinaryRecordsManager(manager_label='state',
                                                              save_dir=self.config['files']['output_dir'],
                                                              file_prefix=self.config['files']['prefix'],
                                                              file_postfix=self.config['files']['rec_postfix'])

        self.local_timing_file = \
            timing_system.TimingFileWriter(save_dir=self.config['files']['output_dir'],
                                           file_prefix=self.config['files']['prefix'],
                                           mpi_rank=self.rank,
                                           file_postfix=self.config['files']['timing_postfix'])

        # stim decider bypass the normal record registration message sending
        for message in stim_decider.get_record_register_messages():
            self.rec_manager.register_rec_type_message(message)

        self.master_time = MPI.Wtime()

        super().__init__()

    def synchronize_time(self):
        self.class_log.debug("Sending time sync messages to simulator node.")
        self.send_interface.send_time_sync_simulator()
        self.send_interface.all_barrier()
        self.master_time = MPI.Wtime()
        self.class_log.debug("Post barrier time set as master.")

    def send_calc_offset_time(self, rank, remote_time):
        offset_time = self.master_time - remote_time
        self.send_interface.send_time_sync_offset(rank, offset_time)

    #MEC edited this function to take in list of ripple tetrodes only
    def _ripple_ranks_startup(self, ripple_trode_list):
        for rip_rank in self.config['rank']['ripples']:
            self.send_interface.send_num_ntrode(rank=rip_rank, num_ntrodes=len(ripple_trode_list))

        # Round robin allocation of channels to ripple
        enable_count = 0
        all_ripple_process_enable = [[] for _ in self.config['rank']['ripples']]
        #MEC changed trode_liist to ripple_trode_list
        for chan_ind, chan_id in enumerate(ripple_trode_list):
            all_ripple_process_enable[enable_count % len(self.config['rank']['ripples'])].append(chan_id)
            enable_count += 1

        # Set channel assignments for all ripple ranks
        #MEC changed send_channel_selection to sned_ripple_channel_selection
        for rank_ind, rank in enumerate(self.config['rank']['ripples']):
            self.send_interface.send_ripple_channel_selection(rank, all_ripple_process_enable[rank_ind])

        for rip_rank in self.config['rank']['ripples']:

            # Map json RippleParameterMessage onto python object and then send
            rip_param_message = ripple_process.RippleParameterMessage(**self.config['ripple']['RippleParameterMessage'])
            self.send_interface.send_ripple_parameter(rank=rip_rank, param_message=rip_param_message)

            # Convert json string keys into int (ntrode_id) and send
            rip_mean_base_dict = dict(map(lambda x: (int(x[0]), x[1]),
                                          self.config['ripple']['CustomRippleBaselineMeanMessage'].items()))
            print('ripple mean: ',rip_mean_base_dict)
            self.send_interface.send_ripple_baseline_mean(rank=rip_rank, mean_dict=rip_mean_base_dict)

            # Convert json string keys into int (ntrode_id) and send
            rip_std_base_dict = dict(map(lambda x: (int(x[0]), x[1]),
                                         self.config['ripple']['CustomRippleBaselineStdMessage'].items()))
            print('ripple std: ',rip_std_base_dict)
            self.send_interface.send_ripple_baseline_std(rank=rip_rank, std_dict=rip_std_base_dict)

    def _stim_decider_startup(self):
        # Convert JSON Ripple Parameter config into message
        rip_param_message = ripple_process.RippleParameterMessage(**self.config['ripple']['RippleParameterMessage'])

        # Update stim decider's ripple parameters
        self.stim_decider.update_n_threshold(rip_param_message.n_above_thresh)
        self.stim_decider.update_lockout_time(rip_param_message.lockout_time)
        if rip_param_message.enabled:
            self.stim_decider.enable()
        else:
            self.stim_decider.disable()

    def _encoder_rank_startup(self, trode_list):

        for enc_rank in self.config['rank']['encoders']:
            self.send_interface.send_num_ntrode(rank=enc_rank, num_ntrodes=len(trode_list))

        # Round robin allocation of channels to encoders
        enable_count = 0
        all_encoder_process_enable = [[] for _ in self.config['rank']['encoders']]
        for chan_ind, chan_id in enumerate(trode_list):
            all_encoder_process_enable[enable_count % len(self.config['rank']['encoders'])].append(chan_id)
            enable_count += 1

        # Set channel assignments for all encoder ranks
        for rank_ind, rank in enumerate(self.config['rank']['encoders']):
            self.send_interface.send_channel_selection(rank, all_encoder_process_enable[rank_ind])

    def _decoder_rank_startup(self, trode_list):
        rank = self.config['rank']['decoder']
        self.send_interface.send_channel_selection(rank, trode_list)

    def _writer_startup(self):
        # Update binary_record file writers before starting datastream
        for rec_rank in self.config['rank_settings']['enable_rec']:
            if rec_rank is not self.rank:
                self.send_interface.send_new_writer_message(rank=rec_rank,
                                                            new_writer_message=self.rec_manager.new_writer_message())

                self.send_interface.send_start_rec_message(rank=rec_rank)

        # Update and start bin rec for StimDecider.  Registration is done through MPI but setting and starting
        # the writer must be done locally because StimDecider does not have a MPI command message receiver
        self.stim_decider.set_record_writer_from_message(self.rec_manager.new_writer_message())
        self.stim_decider.start_record_writing()

    def _turn_on_datastreams(self):
        # Then turn on data streaming to ripple ranks
        for rank in self.config['rank']['ripples']:
            self.send_interface.send_turn_on_datastreams(rank)

        # Turn on data streaming to encoder
        for rank in self.config['rank']['encoders']:
            self.send_interface.send_turn_on_datastreams(rank)

        # Turn on data streaming to decoder
        self.send_interface.send_turn_on_datastreams(self.config['rank']['decoder'])

        self.time_sync_on = True

    #MEC edited
    def handle_ntrode_list(self, trode_list):

        self.class_log.debug("Received decoding ntrode list {:}.".format(trode_list))

        #self._ripple_ranks_startup(trode_list)
        self._encoder_rank_startup(trode_list)
        self._decoder_rank_startup(trode_list)
        self._stim_decider_startup()

        #self._writer_startup()
        #self._turn_on_datastreams()

    #MEC added
    def handle_ripple_ntrode_list(self, ripple_trode_list):

        self.class_log.debug("Received ripple ntrode list {:}.".format(ripple_trode_list))

        self._ripple_ranks_startup(ripple_trode_list)

        self._writer_startup()
        self._turn_on_datastreams()

    def register_rec_type_message(self, message):
        self.rec_manager.register_rec_type_message(message)

    def trigger_termination(self):
        self.send_interface.terminate_all()

        self.parent.trigger_termination()


class MainSimulatorMPIRecvInterface(realtime_base.RealtimeMPIClass):

    def __init__(self, comm: MPI.Comm, rank, config, main_manager: MainSimulatorManager):
        super().__init__(comm=comm, rank=rank, config=config)
        self.main_manager = main_manager

        self.mpi_status = MPI.Status()

        self.req_cmd = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def __iter__(self):
        return self

    def __next__(self):

        (req_rdy, msg) = self.req_cmd.test(status=self.mpi_status)

        if req_rdy:
            self.process_request_message(msg)

            self.req_cmd = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def process_request_message(self, message):

        if isinstance(message, simulator_process.SimTrodeListMessage):
            self.main_manager.handle_ntrode_list(message.trode_list)
            print('decoding tetrodes message',message.trode_list)

        #MEC added
        if isinstance(message, simulator_process.RippleTrodeListMessage):
            self.main_manager.handle_ripple_ntrode_list(message.ripple_trode_list)
            print('ripple tetrodes message',message.ripple_trode_list)

        elif isinstance(message, binary_record.BinaryRecordTypeMessage):
            self.class_log.debug("BinaryRecordTypeMessage received for rec id {} from rank {}".
                                 format(message.rec_id, self.mpi_status.source))
            self.main_manager.register_rec_type_message(message)

        elif isinstance(message, realtime_base.TimeSyncReport):
            self.main_manager.send_calc_offset_time(self.mpi_status.source, message.time)

        elif isinstance(message, realtime_base.TerminateMessage):
            self.class_log.info('Received TerminateMessage from rank {:}, now terminating all.'.
                                format(self.mpi_status.source))

            self.main_manager.trigger_termination()


