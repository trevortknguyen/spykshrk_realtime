import struct
import fcntl
import os

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

        if command == tnp.acq_STOP:
            if not self.terminated:
                # self.main_manager.trigger_termination()
                self.terminate()
                self.terminated = True
                self.started = False

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

        #self.posterior_recv_interface = PosteriorSumRecvInterface(comm=comm, rank=rank, config=config,
        #                                                          stim_decider=self.stim_decider)

        #self.stim_decider = StimDecider(rank=rank, config=config,
        #                                send_interface=StimDeciderMPISendInterface(comm=comm,
        #                                                                           rank=rank,
        #                                                                           config=config))

        #self.data_recv = StimDeciderMPIRecvInterface(comm=comm, rank=rank, config=config,
        #                                             stim_decider=self.stim_decider)

        self.send_interface = MainMPISendInterface(comm=comm, rank=rank, config=config)

        self.manager = MainSimulatorManager(rank=rank, config=config, parent=self, send_interface=self.send_interface,
                                            stim_decider=self.stim_decider)
        print('===============================')
        print('In MainProcess: datasource = ', config['datasource'])
        print('===============================')
        if config['datasource'] == 'trodes':
            print('about to configure trdoes network for tetrode: ',self.manager.handle_ntrode_list,self.rank)
            time.sleep(0.5*self.rank)

            self.networkclient = MainProcessClient("SpykshrkMainProc", config['trodes_network']['address'],config['trodes_network']['port'], self.config)
            if self.networkclient.initialize() != 0:
                print("Network could not successfully initialize")
                del self.networkclient
                quit()
            #added MEC
            self.networkclient.initializeHardwareConnection()
            self.networkclient.registerStartupCallback(self.manager.handle_ntrode_list)
            #added MEC
            self.networkclient.registerStartupCallbackRippleTetrodes(self.manager.handle_ripple_ntrode_list)
            self.networkclient.registerTerminationCallback(self.manager.trigger_termination)
            print('completed trodes setup')


        self.vel_pos_recv_interface = VelocityPositionRecvInterface(comm=comm, rank=rank, config=config,
                                                                  stim_decider=self.stim_decider,
                                                                  networkclient=self.networkclient)        

        self.posterior_recv_interface = PosteriorSumRecvInterface(comm=comm, rank=rank, config=config,
                                                                  stim_decider=self.stim_decider,
                                                                  networkclient=self.networkclient)

        self.data_recv = StimDeciderMPIRecvInterface(comm=comm, rank=rank, config=config,
                                                     stim_decider=self.stim_decider,networkclient=self.networkclient)

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
            self.vel_pos_recv_interface.__next__()
            # should make posterior receiver conditional - only on if in replay conditioning mode
            if self.config['ripple_conditioning']['replay_conditioning'] == True:
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
                                     ['timestamp', 'time', 'lockout_num', 'lockout_state','tets_above_thresh','ripple_cond_message_sent'],
                                     ['bin_timestamp', 'spike_timestamp','time', 'stim_sent', 'ripple_number',
                                      'ripple_time_bin','posterior_max_arm','box','arm1','arm2','arm3',
                                      'arm4','arm5','arm6','arm7','arm8']],
                         rec_formats=['Iii',
                                      'Idiiqi',
                                      'IIdiiiiddddddddd'])
        self.rank = rank
        self._send_interface = send_interface
        self._ripple_n_above_thresh = ripple_n_above_thresh
        self._lockout_time = lockout_time
        self._ripple_thresh_states = {}
        self._conditioning_ripple_thresh_states = {}
        self._enabled = False
        self.config = config

        self._last_lockout_timestamp = 0
        self._lockout_count = 0
        self._in_lockout = False
        self.stim_thresh = False

        # separate lockout for conditioning
        self._conditioning_last_lockout_timestamp = 0
        self._conditioning_in_lockout = False

        self.ripple_time_bin = 0
        self.no_ripple_time_bin = 0
        self.replay_target_arm = self.config['pp_decoder']['replay_target_arm']
        #self.posterior_arm_sum = np.zeros((1,9))
        self.posterior_arm_sum = np.asarray([0,0,0,0,0,0,0,0,0])
        self.num_above = 0
        self.ripple_number = 0
        self.shortcut_message_sent = False
        self.shortcut_message_arm = 10

        self.velocity = 0
        self.linearized_position = 0
        self.thresh_counter = 0
        self.postsum_counter = 0
        self.stim_message_sent = 0
        self.ripple_cond_message_sent = 0
        self.arm1_replay_counter = 0
        self.arm2_replay_counter = 0
        self.arm3_replay_counter = 0
        self.arm4_replay_counter = 0
        self.posterior_time_bin = 0
        self.posterior_arm_threshold = 0.8

        #if self.config['datasource'] == 'trodes':
        #    self.networkclient = MainProcessClient("SpykshrkMainProc", config['trodes_network']['address'],config['trodes_network']['port'], self.config)
        #self.networkclient.initializeHardwareConnection()
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
        self._conditioning_lockout_time = lockout_time

    def update_ripple_threshold_state(self, timestamp, elec_grp_id, threshold_state, conditioning_thresh_state, networkclient):
        # Log timing
        if self.thresh_counter % 1000 == 0:
            self.record_timing(timestamp=timestamp, elec_grp_id=elec_grp_id,
                               datatype=datatypes.Datatypes.LFP, label='stim_rip_state')
        time = MPI.Wtime()
        #print('received thresh states: ',threshold_state,conditioning_thresh_state)

        if self._enabled:
            self.thresh_counter += 1

            if self.thresh_counter % 1500 == 0 and self._in_lockout:
                #print('in normal lockout for ripple detection - one line per tetrode')
                pass
            
            if self.thresh_counter % 1500 == 0 and self._conditioning_in_lockout:
                #print('in conditoning lockout for ripple detection - one line per tetrode')
                pass

            self._ripple_thresh_states.setdefault(elec_grp_id, 0)
            self._conditioning_ripple_thresh_states.setdefault(elec_grp_id, 0)
            
            # only write state if state changed
            if self._ripple_thresh_states[elec_grp_id] != threshold_state:
                self.write_record(realtime_base.RecordIDs.STIM_STATE, timestamp, elec_grp_id, threshold_state)

            self._ripple_thresh_states[elec_grp_id] = threshold_state
            num_above = 0
            for state in self._ripple_thresh_states.values():
                num_above += state

            #need another copy of this for conditioning threshold
            self._conditioning_ripple_thresh_states[elec_grp_id] = conditioning_thresh_state
            conditioning_num_above = 0
            for conditioning_state in self._conditioning_ripple_thresh_states.values():
                conditioning_num_above += conditioning_state

            if self._in_lockout and (timestamp > self._last_lockout_timestamp + self._lockout_time):
                # End lockout
                self._in_lockout = False
                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, time, self._lockout_count, self._in_lockout,
                                  num_above, self.ripple_cond_message_sent)
                self._lockout_count += 1

            #need another copy of this for conditioning lockout
            # note: currently only one variable for counting both lockouts
            if self._conditioning_in_lockout and (timestamp > self._conditioning_last_lockout_timestamp + 
                                                  self._conditioning_lockout_time):
                # End lockout
                self._conditioning_in_lockout = False
                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, time, self._lockout_count, self._conditioning_in_lockout,
                                  num_above, self.ripple_cond_message_sent)
                self._lockout_count += 1

            if (conditioning_num_above >= self._ripple_n_above_thresh) and not self._conditioning_in_lockout:            
            #if (num_above >= self._ripple_n_above_thresh) and not self._in_lockout:
                if self.velocity < self.config['encoder']['vel']:
                    self.ripple_cond_message_sent = 0
                    print('tets above cond ripple thresh: ',conditioning_num_above,timestamp,
                          self._conditioning_ripple_thresh_states, np.around(self.velocity,decimals=2))
                    #print('lockout time: ',self._lockout_time)
                    # this will flash light every time a ripple is detected
                    #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(16);\n'])
                    
                    # this is the easiest place to send a statescript message for ripple conditioning
                    # this will send a message for 1st threshold crossing - no time delay
                    # need to re-introduce lockout so that only one message is sent per ripple (7500 = 5 sec)
                    # lockout is in timestamps I think - so 2 seconds = 60000
                    # for a dynamic filter, we need to get ripple size from the threshold message and send to statescript
                    if self.config['ripple_conditioning']['ripple_condition_status'] == True:
                        print('sent behavior message based on ripple thresh',time,timestamp)
                        # no light flash no, removed trigger16
                        networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(7);\n'])
                        
                        # this doesnt appear to give compile errors with variable set first
                        #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 1;\ntrigger(7);\n'])

                        #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(16);\n'])
                        #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['homeCount = 100;\n'])

                        # to test statescript for replay content detection use this function - should set reward in arm 1
                        #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])
                        #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 1;\n'])
                        #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])
                        
                        # this starts the lockout
                        self._conditioning_in_lockout = True
                        self._conditioning_last_lockout_timestamp = timestamp
                        self.ripple_cond_message_sent = 1               

                #MEC this will get rid of lockout
                #self._in_lockout = True
                #self._last_lockout_timestamp = timestamp
                
                self.stim_thresh = True
                #self.class_log.debug("Ripple threshold detected {}.".format(self._ripple_thresh_states))
                
                 

                # want to send the test shortcut message here
                #networkclient.sendStateScriptShortcutMessage(1)

                #print('sent shortcut message based on ripple thresh',time,timestamp)
                #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])
                #networkclient.sendStateScriptShortcutMessage(1)

                #print('sent regular message to MCU on ripple thresh',time,timestamp)
                
                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, self.velocity, self._lockout_count, self._conditioning_in_lockout,
                                  conditioning_num_above, self.ripple_cond_message_sent)

                self._send_interface.start_stimulation()
                # here we want to send the stim_decider message to the decoder
                # this should be similiar to the replay threshold message from ripple_process
                # i think _in_lockout will be 1 when stim threshold is crossed
                #self._send_interface.send_stim_decision(timestamp, self.stim_thresh)
                #print('stim message function called',timestamp,time*1000)

            elif (num_above >= self._ripple_n_above_thresh) and not self._in_lockout:
                if self.velocity < self.config['encoder']['vel']:
                    #print('tets above normal ripple thresh: ',num_above,timestamp,
                    #      self._ripple_thresh_states, np.around(self.velocity,decimals=2))
                    pass

                # this needs to just turn on light
                # i think this needs to use lockout too, otherwise too many messages
                # but this could interfere with detection of larger ripples
                # i think we need no lockout here because it will mask large ripples
                # so need to come up with a better way to trigger light - turn off for now
                if self.config['ripple_conditioning']['ripple_condition_status'] == True:
                        #print('sent light message based on ripple thresh',time,timestamp)
                        #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])
                        # this starts the lockout
                        self._in_lockout = True
                        self._last_lockout_timestamp = timestamp
                        # this should allow us to tell difference between ripples that trigger conditioning
                        self.ripple_cond_message_sent = 0
                
                self.stim_thresh = True

                self.write_record(realtime_base.RecordIDs.STIM_LOCKOUT,
                                  timestamp, self.velocity, self._lockout_count, self._in_lockout,
                                  num_above, self.ripple_cond_message_sent)

            # i think this should care about both lockouts - but im not sure
            elif num_above < self._ripple_n_above_thresh and not self._in_lockout and not self._conditioning_in_lockout:
                self.stim_thresh = False
                #self._send_interface.send_stim_decision(timestamp, self.stim_thresh)

            return num_above

    # MEC: this function brings in velocity and linearized position from decoder process
    # it is working as expected - both velocity filter and position filter!
    # to do: incorporate these two variables below in posterior sum
    def velocity_position(self, bin_timestamp, vel, pos):
        self.velocity = vel
        self.linearized_position = pos

        if self.velocity < self.config['encoder']['vel']:
            #print('immobile, vel = ',self.velocity)
            pass
        if self.linearized_position >= 3 and self.linearized_position <= 5:
            #print('position at rip/wait!')
            pass

    # MEC: this function calculates the sum of the posterior over the course of each ripple, then sends shortcut message
    # need to add velocity filter so that it only calculates sum during immobility ripples
    # need to add location filter so it only sends message when rat is at rip/wait well
    def posterior_sum(self, bin_timestamp, spike_timestamp, box,arm1,arm2,arm3,arm4,arm5,arm6,arm7,arm8,networkclient):
        time = MPI.Wtime()
        # i think you can put a counter here for the lockout after a shortcut message
        # have it reset to 0 when a message is sent
        self.stim_message_sent += 1
        # 200 msec = 300 time bins at 1500 Hz, no this function is triggered
        # by posterior message so its only sent every 5 msec

        # reset posterior arm threshold based on the new_ripple_threshold text file
        # this should run every 10 sec, using thresh_counter which refers to each message from ripple node
        if self.thresh_counter % 1000 == 0:
            print('thresh_counter: ',self.thresh_counter)
            with open('config/new_ripple_threshold.txt') as posterior_threshold_file:
                fd = posterior_threshold_file.fileno()
                fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
                # read file
                for post_thresh_file_line in posterior_threshold_file:
                    pass
                new_posterior_threshold = post_thresh_file_line
            # this allows half SD increase in ripple threshold (looks for three digits, eg 065 = 6.5 SD)
            # final 2 characters in line are new threshold (eg 08 > 0.8)
            # this seesm to work, but in the sum function posterior_arm_threshold is seen as 0, why??
            self.posterior_arm_threshold = np.int(new_posterior_threshold[8:10])/10
            print('posterior arm threshold = ',self.posterior_arm_threshold)

        if self.stim_thresh == True and self.velocity < self.config['encoder']['vel']:
            # is posterior arm sum getting set back to 0 every time?
            #print('starting posterior arm sum',self.posterior_arm_sum, self.posterior_arm_sum.sum(),new_posterior_sum.sum())
            #print('posterior sum is: ',timestamp,box,time*1000)
            #print('stim threshold: ',self.stim_thresh,self._last_lockout_timestamp)

            # this generates too many output lines with few tetrodes - because huge gaps from decoder
            #print('ripple number: ',self.ripple_number,' time bin: ',self.ripple_time_bin)

            #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])
            if self.ripple_time_bin == 0:
                self.ripple_number += 1
                #print('ripple number: ',self.ripple_number)
            self.no_ripple_time_bin = 0
            self.ripple_time_bin += 1
            self.postsum_counter += 1

            if self.postsum_counter % 1000 == 0:
                self.record_timing(timestamp=spike_timestamp, elec_grp_id=0,
                                   datatype=datatypes.Datatypes.LFP, label='postsum_in')
            
            #while the ripple is progressing we need to add the current posterior sum to the sum of all earlier ones
            new_posterior_sum = np.asarray([box,arm1,arm2,arm3,arm4,arm5,arm6,arm7,arm8])
            # this look good, but normalization might fail if entries for arm5-arm8 are 0
            #print('incoming posterior sum', new_posterior_sum, new_posterior_sum.sum())
            # okay so i can see the posterior problem with new_posterior_arm_sum, so i think its the data from the decoder
            #print('starting posterior arm sum',self.posterior_arm_sum, self.posterior_arm_sum.sum(),new_posterior_sum.sum())
            self.posterior_arm_sum = self.posterior_arm_sum + new_posterior_sum
            # counter for number of posterior values during ripple
            self.posterior_time_bin +=1
            #print('total posterior sum', self.posterior_arm_sum)

            # MEC: 10-27-19: try turning off stim_message record to see if that helps record saving problem
            self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                              bin_timestamp, spike_timestamp, time, self.shortcut_message_sent, 
                              self.ripple_number, self.ripple_time_bin, self.shortcut_message_arm,
                              new_posterior_sum[0],new_posterior_sum[1],new_posterior_sum[2],
                              new_posterior_sum[3],new_posterior_sum[4],new_posterior_sum[5],
                              new_posterior_sum[6],new_posterior_sum[7],new_posterior_sum[8])    

        if self.stim_thresh == False and self.velocity < self.config['encoder']['vel']:
            #print('no ripple in decoder')
            self.no_ripple_time_bin += 1

            #send a shrotcut message if it has been a ripple (>3 bins) and now is not a ripple (no ripple bins = 3)
            # includes velocity filter now
            # includes repeated stim filter: self.stim_message_sent, set at 250 msec
            # now this is updated at 1500 Hz, so the number of bins should be 22 to give 15 msec
            # no, these are still 5 msec bins

            # why is there a check for stim_message_sent > 50 - creates a delay after last stim (counter above)
            # 250/5 should be 50*5 msec, but may not be accurate if lots of empty bins, we can try 125
            # this may be unnecessary because of the ripple lockout actually
            #print('lockout counter for posterior sum',self.stim_message_sent)
            if (self.no_ripple_time_bin == 3) & (self.ripple_time_bin > 3) & (self.stim_message_sent>(125/5)):
                # comment this out because its making false lights for ripple detection
                #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])
                pass
                
                # normalize posterior_arm_sum
                # this is a problem: the normalized posterior doesnt often add to 1, so all values are lower than
                # they should be - this might be because ripple time bin count is counting wrong
                # try to print number of posteriors included in this sum - posterior bins seems to match ripple bins
                # try shortening posterior_arm_sum to get rid of 0s in arms 5-8
                # this doesnt fix the problem
                #self.posterior_arm_sum = self.posterior_arm_sum[0:5]
                
                self.posterior_arm_sum = self.posterior_arm_sum/self.ripple_time_bin
                #print('normalized posterior arm sum in main',self.posterior_arm_sum,
                #      self.posterior_arm_sum.sum(),self.ripple_time_bin,self.posterior_time_bin)
                # reset posterior time bin counter
                self.posterior_time_bin = 0

                # return arm with max posterior: first check if only 1 arm above 0.8, then make that arm a variable
                # then check if each arm is the max, and if so, send that shortcut message
                # shortcut message fn_num is an interger for function number in statescript

                # alternative: only require that any outer arm is above 0.3 - then put reward in that arm

                print('in posterior sum function, posterior threshold = ',self.posterior_arm_threshold)
                if len(np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold) == 1):
                    
                    # replay detection of box
                    if np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold)[0][0] == 0:
                        print('max posterior in box',self.posterior_arm_sum[0],'interval',self.stim_message_sent,
                              'posterior sum: ',self.posterior_arm_sum.sum())
                        self.shortcut_message_arm = np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold)[0][0]
                        self.stim_message_sent = 0
                        # note for testing this seems to be triggered most frequently
                        #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 1;\ntrigger(15);\n'])
                    
                    # replay detection of arm 1
                    elif np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold)[0][0] == 1:
                        print('max posterior in arm 1',self.posterior_arm_sum[1],'interval',self.stim_message_sent,
                              'posterior sum: ',self.posterior_arm_sum.sum())
                        self.shortcut_message_arm = np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold)[0][0]
                        self.stim_message_sent = 0
                        # only send message for arm 1 replay if less than replays 3 in a row
                        if self.arm1_replay_counter < 3:
                            # note: statescript can only execute one function at a time, so trigger function 15 and set replay_arm variable
                            networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 1;\ntrigger(15);\n'])
                            print('sent StateScript message for arm 1 replay')
                            # arm replay counters, only active at wait well and adds to current counter and sets other arms to 0
                            if self.linearized_position >= 3 and self.linearized_position <= 5:
                                self.arm1_replay_counter += 1
                                self.arm2_replay_counter = 0
                                self.arm3_replay_counter = 0
                                self.arm4_replay_counter = 0
                                print('arm counters: ',self.arm1_replay_counter,self.arm2_replay_counter,
                                      self.arm3_replay_counter,self.arm4_replay_counter)
                        else:
                            print('more than 2 replays of arm 1 in a row!')

                    # replay detection of arm 2
                    elif np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold)[0][0] == 2:
                        print('max posterior in arm 2',self.posterior_arm_sum[2],'interval',self.stim_message_sent,
                              'posterior sum: ',self.posterior_arm_sum.sum())
                        self.shortcut_message_arm = np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold)[0][0]
                        self.stim_message_sent = 0
                        if self.arm2_replay_counter < 3:
                            networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 2;\ntrigger(15);\n'])
                            print('sent StateScript message for arm 2 replay')
                            if self.linearized_position >= 3 and self.linearized_position <= 5:
                                self.arm1_replay_counter = 0
                                self.arm2_replay_counter += 1
                                self.arm3_replay_counter = 0
                                self.arm4_replay_counter = 0
                                print('arm counters: ',self.arm1_replay_counter,self.arm2_replay_counter,
                                      self.arm3_replay_counter,self.arm4_replay_counter)
                        else:
                            print('more than 2 replays of arm 2 in a row!')

                    # replay detection of arm 3
                    elif np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold)[0][0] == 3:
                        print('max posterior in arm 3',self.posterior_arm_sum[3],'interval',self.stim_message_sent,
                              'posterior sum: ',self.posterior_arm_sum.sum())
                        self.shortcut_message_arm = np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold)[0][0]
                        self.stim_message_sent = 0
                        if self.arm3_replay_counter < 3:
                            networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 3;\ntrigger(15);\n'])
                            print('sent StateScript message for arm 3 replay')
                            if self.linearized_position >= 3 and self.linearized_position <= 5:
                                self.arm1_replay_counter = 0
                                self.arm2_replay_counter = 0
                                self.arm3_replay_counter += 1
                                self.arm4_replay_counter = 0
                                print('arm counters: ',self.arm1_replay_counter,self.arm2_replay_counter,
                                      self.arm3_replay_counter,self.arm4_replay_counter)
                        else:
                            print('more than 2 replays of arm 3 in a row!')

                    # replay detection of arm 4
                    elif np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold)[0][0] == 4:
                        print('max posterior in arm 4',self.posterior_arm_sum[4],'interval',self.stim_message_sent,
                              'posterior sum: ',self.posterior_arm_sum.sum())
                        self.shortcut_message_arm = np.argwhere(self.posterior_arm_sum>self.posterior_arm_threshold)[0][0]
                        self.stim_message_sent = 0
                        if self.arm4_replay_counter < 3:
                            networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 4;\ntrigger(15);\n'])
                            print('sent StateScript message for arm 4 replay')
                            if self.linearized_position >= 3 and self.linearized_position <= 5:
                                self.arm1_replay_counter = 0
                                self.arm2_replay_counter = 0
                                self.arm3_replay_counter = 0
                                self.arm4_replay_counter += 1
                                print('arm counters: ',self.arm1_replay_counter,self.arm2_replay_counter,
                                      self.arm3_replay_counter,self.arm4_replay_counter)
                        else:
                            print('more than 2 replays of arm 4 in a row!')
                    
                    # replay detection of arm 5
                    #elif np.argwhere(self.posterior_arm_sum>0.8)[0][0] == 5:
                    #    print('max posterior in arm 5',self.posterior_arm_sum[5],'interval',self.stim_message_sent)
                    #    self.shortcut_message_arm = np.argwhere(self.posterior_arm_sum>0.8)[0][0]
                    #    self.stim_message_sent = 0
                    #    #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])
                    # replay detection of arm 6
                    #elif np.argwhere(self.posterior_arm_sum>0.8)[0][0] == 6:
                    #    print('max posterior in arm 6',self.posterior_arm_sum[6],'interval',self.stim_message_sent)
                    #    self.shortcut_message_arm = np.argwhere(self.posterior_arm_sum>0.8)[0][0]
                    #    self.stim_message_sent = 0
                    #    #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])
                    # replay detection of arm 7
                    #elif np.argwhere(self.posterior_arm_sum>0.8)[0][0] == 7:
                    #    print('max posterior in arm 7',self.posterior_arm_sum[7],'interval',self.stim_message_sent)
                    #    self.shortcut_message_arm = np.argwhere(self.posterior_arm_sum>0.8)[0][0]
                    #    self.stim_message_sent = 0
                    #    #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])
                    # replay detection of arm 8
                    #elif np.argwhere(self.posterior_arm_sum>0.8)[0][0] == 8:
                    #    print('max posterior in arm 8',self.posterior_arm_sum[8],'interval',self.stim_message_sent)
                    #    self.shortcut_message_arm = np.argwhere(self.posterior_arm_sum>0.8)[0][0]
                    #    self.stim_message_sent = 0
                    #    #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['trigger(15);\n'])
                
                # message if no location > posterior threshold (0.8) in posterior sum
                else:
                    print('no arm posterior above ',self.posterior_arm_threshold,' ',self.posterior_arm_sum,'interval',self.stim_message_sent,
                          'ripple: ',self.ripple_number,'posterior sum: ',self.posterior_arm_sum.sum())
                    #networkclient.sendMsgToModule('StateScript', 'StatescriptCommand', 's', ['replay_arm = 3;\ntrigger(15);\n'])

                ## send shortcut message based on posterior max arm, each arm has own function number
                ## fn_num is an interger for function number in statescript
                #self.networkclient.sendStateScriptShortcutMessage(22)
                self.shortcut_message_sent = True
                print("end of ripple message sent",self.ripple_time_bin,self.ripple_number)

                self.write_record(realtime_base.RecordIDs.STIM_MESSAGE,
                                  bin_timestamp, spike_timestamp, time, self.shortcut_message_sent, 
                                  self.ripple_number, self.ripple_time_bin, self.shortcut_message_arm,
                                  self.posterior_arm_sum[0],self.posterior_arm_sum[1],self.posterior_arm_sum[2],
                                  self.posterior_arm_sum[3],self.posterior_arm_sum[4],self.posterior_arm_sum[5],
                                  self.posterior_arm_sum[6],self.posterior_arm_sum[7],self.posterior_arm_sum[8])

            # switch to 22 bins to get 15 msec - no, these are still 5msec bins
            if self.no_ripple_time_bin > 3:
                self.ripple_time_bin = 0
                self.posterior_arm_sum = np.asarray([0,0,0,0,0,0,0,0,0])
                self.shortcut_message_arm = 10
                self.shortcut_message_sent = False                

class StimDeciderMPIRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider: StimDecider, networkclient):
        super(StimDeciderMPIRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.stim = stim_decider
        self.networkclient = networkclient

        self.mpi_status = MPI.Status()

        self.feedback_bytes = bytearray(16)
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
                #MEC: we need to add ripple size to this messsage
                message = ripple_process.RippleThresholdState.unpack(message_bytes=self.feedback_bytes)
                self.stim.update_ripple_threshold_state(timestamp=message.timestamp,
                                                        elec_grp_id=message.elec_grp_id,
                                                        threshold_state=message.threshold_state,
                                                        conditioning_thresh_state=message.conditioning_thresh_state,
                                                        networkclient=self.networkclient)

                self.mpi_reqs[0] = self.comm.Irecv(buf=self.feedback_bytes,
                                                   tag=realtime_base.MPIMessageTag.FEEDBACK_DATA.value)
                

class PosteriorSumRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider: StimDecider, networkclient):
        super(PosteriorSumRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.stim = stim_decider
        self.networkclient = networkclient
        #NOTE: if you dont know how large the buffer should be, set it to a large number
        # then you will get an error saying what it should be set to
        self.msg_buffer = bytearray(80)
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
                                    arm2=message.arm2,arm3=message.arm3,arm4=message.arm4,arm5=message.arm5,
                                    arm6=message.arm6,arm7=message.arm7,arm8=message.arm8,networkclient=self.networkclient)             
            #print('posterior sum message supervisor: ',message.timestamp,time*1000)
            #return posterior_sum

        else:
            return None

class VelocityPositionRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, stim_decider: StimDecider, networkclient):
        super(VelocityPositionRecvInterface, self).__init__(comm=comm, rank=rank, config=config)

        self.stim = stim_decider
        self.networkclient = networkclient
        #NOTE: if you dont know how large the buffer should be, set it to a large number
        # then you will get an error saying what it should be set to
        self.msg_buffer = bytearray(16)
        self.req = self.comm.Irecv(buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.VEL_POS.value)

    def __next__(self):
        rdy = self.req.Test()
        time = MPI.Wtime()
        if rdy:

            message = decoder_process.VelocityPosition.unpack(self.msg_buffer)
            self.req = self.comm.Irecv(buf=self.msg_buffer, tag=realtime_base.MPIMessageTag.VEL_POS.value)

            # okay so we are receiving the message! but now it needs to get into the stim decider
            self.stim.velocity_position(bin_timestamp=message.bin_timestamp, pos=message.pos, vel=message.vel)             
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
            #print('ripple mean: ',rip_mean_base_dict)
            self.send_interface.send_ripple_baseline_mean(rank=rip_rank, mean_dict=rip_mean_base_dict)

            # Convert json string keys into int (ntrode_id) and send
            rip_std_base_dict = dict(map(lambda x: (int(x[0]), x[1]),
                                         self.config['ripple']['CustomRippleBaselineStdMessage'].items()))
            #print('ripple std: ',rip_std_base_dict)
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


