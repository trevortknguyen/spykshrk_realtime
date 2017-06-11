import sys

from mpi4py import MPI
import threading
import struct
import time
import numpy as np

import spykshrk.realtime.realtime_process as realtime_process
import spykshrk.realtime.datatypes as datatypes
import spykshrk.realtime.simulator.nspike_data as nspike_data
import spykshrk.realtime.simulator.sim_databuffer as sim_databuffer
import spykshrk.realtime.timing_system as timing_system


class SimulatorError(RuntimeError):
    pass


class ReqDatatypeChannelDataMessage(realtime_process.RealtimeMessage):
    def __init__(self, datatype, channel):
        self. datatype = datatype
        self.channel = channel


class StartAllStreamMessage(realtime_process.RealtimeMessage):
    def __init__(self):
        pass


class StopAllStreamMessage(realtime_process.RealtimeMessage):
    def __init__(self):
        pass


class PauseAllStreamMessages(realtime_process.RealtimeMessage):
    def __init__(self):
        pass


class SimTrodeListMessage(realtime_process.RealtimeMessage):
    def __init__(self, trode_list):
        self.trode_list = trode_list


class SimulatorRemoteReceiver(realtime_process.DataSourceReceiver):
    """ A Class to be created and used by ranks that need to communicate with the Simulator Process/Rank.
    
    Goal is to provide an abstraction layer for interacting with other sources.
    """
    def __init__(self, comm: MPI.Comm, rank, config):
        super().__init__()
        self.comm = comm
        self.rank = rank
        self.config = config
        self.stop = False

    def register_datatype_channel(self, datatype, channel):
        self.comm.send(ReqDatatypeChannelDataMessage(datatype=datatype, channel=channel),
                       dest=self.config['rank']['simulator'],
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def start_all_streams(self):
        self.comm.send(StartAllStreamMessage(), dest=self.config['rank']['simulator'],
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def stop_all_streams(self):
        self.comm.send(StopAllStreamMessage(), dest=self.config['rank']['simulator'],
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

    def stop_iterator(self):
        self.stop = True

    def __iter__(self):
        return self

    def __next__(self):

        data = bytearray(20)
        time = bytearray(100)
        mpi_reqs = []
        if self.config['timing']['enable_lfp']:
            mpi_reqs.append(self.comm.Irecv(buf=time, tag=realtime_process.MPIMessageTag.TIMING_MESSAGE.value))

        mpi_reqs.append(self.comm.Irecv(buf=data, tag=realtime_process.MPIMessageTag.SIMULATOR_DATA.value))

        while not MPI.Request.Testall(requests=mpi_reqs) and not self.stop:
            # Loop waiting for next message
            # time.sleep(0.000001)
            pass

        if self.stop:
            raise StopIteration()

        if self.config['timing']['enable_lfp']:
            timing_message = timing_system.TimingMessage.unpack(time)

        data_message = datatypes.LFPPoint.unpack(data)

        return data_message


class SimulatorProcess(realtime_process.RealtimeProcess):
    def __init__(self, comm: MPI.Comm, rank, config):
        super().__init__(comm=comm, rank=rank, config=config, ThreadClass=SimulatorThread)
        self.terminate = False

        self.sim = self.thread.sim

    def trigger_termination(self):
        self.terminate = True

    def main_loop(self):
        self.thread.start()

        mpi_status = MPI.Status()
        while not self.terminate:
            message = self.comm.recv(status=mpi_status, tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)
            if isinstance(message, ReqDatatypeChannelDataMessage):
                if message.datatype is datatypes.Datatypes.CONTINUOUS:
                    self.sim.update_cont_chan_req(mpi_status.source, message.channel)
                elif message.datatype is datatypes.Datatypes.SPIKES:
                    raise NotImplementedError("The Spike datatype is not implemented yet for the simulator.")
                elif message.datatype is datatypes.Datatypes.POSITION:
                    self.sim.update_pos_chan_req(mpi_status.source)

            elif isinstance(message, StartAllStreamMessage):
                self.sim.start_datastream()

            elif isinstance(message, PauseAllStreamMessages):
                self.sim.pause_datastream()

            elif isinstance(message, realtime_process.TerminateMessage):
                self.thread.trigger_termination()
                self.trigger_termination()

        self.class_log.info("Simulator Process Main reached end, exiting.")


class Simulator(realtime_process.RealtimeClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super().__init__()
        self.comm = comm
        self.rank = rank
        self.config = config

        self._stop_next = False

        try:
            self.nspike_anim = nspike_data.AnimalInfo(**config['simulator']['nspike_animal_info'])
            lfp_stream = nspike_data.EEGDataStream(self.nspike_anim)
            #pos_stream = nspike_data.PosMatDataStream(self.nspike_anim, 1000)
            self.databuffer = sim_databuffer.SimDataBuffer([lfp_stream()])

            self.lfp_chan_req_dict = {}
            self.pos_chan_req = []
            self.data_itr = self.databuffer()

        except TypeError as err:
            self.class_log.exception("TypeError: nspike_animal_info does not match nspike_data.AnimalInfo arguments.",
                                     exc_info=err)
            comm.send(realtime_process.TerminateErrorMessage("For SimulatorThread, nspike_animal_info config did"
                                                             "not match nspike_data.AnimalInfo arguments."),
                      dest=config['rank']['supervisor'],
                      tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

        self.comm.send(obj=SimTrodeListMessage(self.config['simulator']['nspike_animal_info']['tetrodes']),
                       dest=config['rank']['supervisor'],
                       tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)

        self.running = False

    def update_cont_chan_req(self, dest_rank, lfp_chan):
        if lfp_chan not in self.nspike_anim.tetrodes:
            raise SimulatorError("Rank {:} tried to request channel ({:}) not available in animal info.".
                                 format(dest_rank, lfp_chan))
        if lfp_chan in self.lfp_chan_req_dict:
            self.class_log.error(("LFP channels cannot be requested by more than one rank. Channel ({:}) requested by "
                                  "rank ({:}) but is already owned by rank ({:}). "
                                  "Overwriting previous assignment.").format(lfp_chan, dest_rank,
                                                                             self.lfp_chan_req_dict[lfp_chan]))
        self.lfp_chan_req_dict[lfp_chan] = dest_rank
        self.class_log.debug("Continuous channel/ntrode {:} registered by rank {:}".format(lfp_chan, dest_rank))

    def update_pos_chan_req(self, dest_rank):
        self.pos_chan_req.append(dest_rank)

    def start_datastream(self):
        self.class_log.debug("Start datastream.")
        self.running = True

    def pause_datastream(self):
        self.running = False

    def send_next_data(self):
        if not self.running:
            return None

        try:
            data_to_send = self.data_itr.__next__()
            if isinstance(data_to_send, datatypes.LFPPoint):
                try:
                    bytes_to_send = data_to_send.pack()

                    if self.config['timing']['enable_lfp']:
                        timing_msg = timing_system.TimingMessage(label='lfp',
                                                                 timestamp=data_to_send.timestamp,
                                                                 start_rank=self.rank)
                        self.comm.Send(buf=timing_msg.pack(), dest=self.lfp_chan_req_dict[data_to_send.ntrode_id],
                                       tag=realtime_process.MPIMessageTag.TIMING_MESSAGE.value)

                    self.comm.Send(buf=bytes_to_send, dest=self.lfp_chan_req_dict[data_to_send.ntrode_id],
                                   tag=realtime_process.MPIMessageTag.SIMULATOR_DATA.value)

                except KeyError as err:
                    self.class_log.exception(("KeyError: Tetrode id ({:}) not in lfp channel request dict {:}, "
                                              "was likely never requested by a receiving/computing ranks.").
                                             format(data_to_send.ntrode_index, self.lfp_chan_req_dict), exc_info=err)

        except StopIteration as err:
            # Simulation is done, send terminate message
            self.comm.send(obj=realtime_process.TerminateMessage(), dest=self.config['rank']['supervisor'],
                           tag=realtime_process.MPIMessageTag.COMMAND_MESSAGE.value)



class SimulatorThread(realtime_process.RealtimeThread):

    def __init__(self, comm: MPI.Comm, rank, config, parent):
        super().__init__(comm=comm, rank=rank, config=config, parent=parent)

        self.sim = Simulator(comm=comm, rank=rank, config=config)

        self._stop_next = False

    def trigger_termination(self):
        self._stop_next = True

    def run(self):
        while not self._stop_next:
            self.sim.send_next_data()

        self.class_log.info("Simulator Process Thread reached end, exiting.")
