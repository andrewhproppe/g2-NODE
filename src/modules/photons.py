'''
Python for analysis of photon stream data from the Picoqunt GmbH Hydraharp and the Swabian TimeTagger.

Adapted from photons.m V5.0 @ HENDRIK UTZAT, KATIE SHULENBERGER, BORIS SPOKOYNY, TIMOTHY SINCLAIR (10/29/2017)

Weiwei Sun, July, 2019
'''

import numpy as np
import time as timing
import seaborn as sns
import os, struct, scipy, warnings
from tqdm import tqdm
from numba import jit, prange
from scipy.special import erf as erf
import pandas as pd
from matplotlib import pyplot as plt
from src.modules.MLE_MAP import MLE, lossP
warnings.filterwarnings('ignore')

def function_clean(theta, x, function): # function output (clean)
    w = np.array(theta)
    f = eval(function)
    return f
'''
For HH data, after reading the metadata with the constructor methods (.photons), the function get_photon_records() parses the raw hoton-arrival time data recorded with an absolute experimental clock (T2 data, .ht2) or a relative experimental clock (T3 data, .ht3) into the true photon-arrival time records. These photon-records are written to a .photons file in uint64 representation.

For Swabian data (t3 mode), there's already a .photons file in int64 and a .ht3 file with header info.
'''
class photons:
    """
    Called when creating photon class. It reads the header from the input photon stream file.
    file_path is the path to the photon stream file. NOTE: '\' needs to be replaced to '\\' or '/'.
    memory_limit is the maximum memory to read metadata at once, set to default 1 MB.

    For HH data, skipheader is set to default as 0.
    For Swabian data, file_path is the .header file, skipheader = 1.
    """
    def __init__(self, file_path, memory_limit=1, skipheader=0):
        # properties
        self.lifetime = None
        self.file_path = file_path
        self.header_size = None # size of headers in bytes
        self.header = None # dictionary for header information
        self.memory_limit = memory_limit
        self.buffer_size = int(self.memory_limit * 1024 * 1024 / 8) # how many uint64 can be read at once
        self.cross_corr = None
        self.auto_corr = None # dictionary to store the correlations

        # extract photon stream file info OLD
        # self.path_str = os.path.split(file_path)[0]
        # self.file_name = os.path.split(file_path)[1].split('.')[0]
        # self.file_ext = os.path.split(file_path)[1].split('.')[1]

        self.path_str = os.path.split(file_path)[0] # + "/"
        self.file_ext = os.path.split(file_path)[1].split('.')[-1]

        # Hacky way of accounting for other periods in the filename that don't separate the name from the extension
        split = os.path.split(file_path)[1].split('.')
        self.file_name = split[0]
        for i in range(1, len(split)-1):
            self.file_name = self.file_name + '.' + split[i]

        # read the photon stream file header info
        if skipheader == 0:
            self.get_photon_stream_header(print_on=False)
            self.datatype = np.uint64
        else:
            self.get_swabian_header()
            self.datatype = np.int64

        print('Photon class created', end='\r')

    def get_photon_stream_header(self, print_on):
        """
        ============================================================================================
        Parse the photon stream and meta data
        """

        '''
        This function reads the ASCII and binary headers of the photon stream file.
        Code adapted from Picoquant GmBH, Germany.
        '''
        fid = open(self.file_path, 'rb') # read the file as bianry

        # ASCII file headers
        ########################################################################################
        Ident = str(fid.read(16).decode('ascii')).strip('\x00')
        if print_on:
            print('Ident: %s \n'  % (Ident) )
        FormatVersion = str(fid.read(6).decode('ascii')).strip('\x00')
        if print_on:
            print('FormatVersion: %s \n'  % (FormatVersion) )

        if FormatVersion != '2.0':
            print('\n\n Warning: This program is for version 2.0 only. Aborted.')
            return 1

        CreatorName = str(fid.read(18).decode('ascii')).strip('\x00')
        if print_on:
            print('Creator name: %s\n'  % (CreatorName))

        CreatorVersion = str(fid.read(12).decode('ascii')).strip('\x00')
        if print_on:
            print('Creator version: %s\n'  % (CreatorVersion))

        FileTime = str(fid.read(18).decode('ascii')).strip('\x00')
        if print_on:
            print('File Time: %s\n'  % (FileTime))

        CRLF = str(fid.read(2).decode('ascii')).strip('\x00')

        Comment = str(fid.read(256).decode('ascii')).strip('\x00')
        if print_on:
            print('Comment: %s\n'  % (Comment))


        # Binary file header
        # It is the same as HHD files, but some of the info is not meaningful in the time-tagged mode, thus not output.
        ########################################################################################
        NumberOfCurves = struct.unpack('i',fid.read(4))[0]

        BitsPerRecord = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Bits per record: %d\n'  % (BitsPerRecord))

        ActiveCurve = struct.unpack('i',fid.read(4))[0]

        MeasurementMode = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Measurement mode: %d\n'  % (MeasurementMode))

        SubMode = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Submode: %d\n'  % (SubMode))

        Binning = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Binning: %d\n'  % (Binning))

        Resolution = struct.unpack('d',fid.read(8))[0]
        if print_on:
            print('Resolution: %f ps\n'  % (Resolution))

        Offset = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Offset: %d\n'  % (Offset))

        Tacq = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Acquisition time: %d ms\n'  % (Tacq))

        StopAt = struct.unpack('I',fid.read(4))[0]
        StopOnOvfl = struct.unpack('i',fid.read(4))[0]
        Restart = struct.unpack('i',fid.read(4))[0]
        DispLinLog = struct.unpack('i',fid.read(4))[0]
        DispTimeAxisFrom = struct.unpack('i',fid.read(4))[0]
        DispTimeAxisTo = struct.unpack('i',fid.read(4))[0]
        DispCountAxisFrom = struct.unpack('i',fid.read(4))[0]
        DispCountAxisTo = struct.unpack('i',fid.read(4))[0]

        DispCurveMapTo, DispCurveShow, ParamStart, ParamStep, ParamEnd = [],[],[],[],[]

        for i in range(8):
            DispCurveMapTo.append(struct.unpack('i',fid.read(4))[0])
            DispCurveShow.append(struct.unpack('i',fid.read(4))[0])

        for i in range(3):
            ParamStart.append(struct.unpack('f',fid.read(4))[0])
            ParamStep.append(struct.unpack('f',fid.read(4))[0])
            ParamEnd.append(struct.unpack('f',fid.read(4))[0])

        RepeatMode = struct.unpack('i',fid.read(4))[0]
        RepeatsPerCurve = struct.unpack('i',fid.read(4))[0]
        Repaobjime = struct.unpack('i',fid.read(4))[0]
        RepeatWaiobjime = struct.unpack('i',fid.read(4))[0]
        ScriptName = str(fid.read(20).decode('ascii')).strip('\x00')

        if print_on:
            print('-------------------------------------------')

        # Hardware information header
        ########################################################################################
        HardwareIdent = str(fid.read(16).decode('ascii')).strip('\x00')
        if print_on:
            print('Hardware Identifier: %s\n'  % HardwareIdent)

        HardwarePartNo = str(fid.read(8).decode('ascii')).strip('\x00')
        if print_on:
            print('Hardware Part Number: %s\n'  % HardwarePartNo)

        HardwareSerial = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('HW Serial Number: %d\n'  % HardwareSerial)

        nModulesPresent = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Modules present: %d\n'  % nModulesPresent)

        ModelCode, VersionCode = [], []
        for i in range(10):
            ModelCode.append(struct.unpack('i',fid.read(4))[0])
            VersionCode.append(struct.unpack('i',fid.read(4))[0])

        if print_on:
            for i in range(nModulesPresent):
                print('ModuleInfo[%02d]: %08x %08x\n' % (i-1, ModelCode[i], VersionCode[i]))


        BaseResolution = struct.unpack('d',fid.read(8))[0]
        if print_on:
            print('BaseResolution: %f\n'  % BaseResolution)


        InputsEnabled = struct.unpack('Q', fid.read(8))[0] # uint64
        if print_on:
            print('Inputs Enabled: %d\n' % InputsEnabled)

        InpChansPresent  = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Input Chan. Present: %d\n' % InpChansPresent)

        RefClockSource  = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('RefClockSource: %d\n' % RefClockSource)

        ExtDevices  = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('External Devices: %x\n' % ExtDevices)

        MarkerSeobjings  = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Marker Seobjings: %x\n' % MarkerSeobjings)

        SyncDivider = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Sync divider: %d \n' % SyncDivider)

        SyncCFDLevel = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Sync CFD Level: %d mV\n' % SyncCFDLevel)

        SyncCFDZeroCross = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Sync CFD ZeroCross: %d mV\n' % SyncCFDZeroCross)

        SyncOffset = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Sync Offset: %d\n' % SyncOffset)

        # Channels' information header
        ########################################################################################
        InputModuleIndex, InputCFDLevel, InputCFDZeroCross, InputOffset =[], [], [], []
        for i in range(InpChansPresent):
            InputModuleIndex.append(struct.unpack('i',fid.read(4))[0])
            InputCFDLevel.append(struct.unpack('i',fid.read(4))[0])
            InputCFDZeroCross.append(struct.unpack('i',fid.read(4))[0])
            InputOffset.append(struct.unpack('i',fid.read(4))[0])

            if print_on:
                print('\n-------------------------------------\n')
                print('Input Channel No. %d\n' % i)
                print('-------------------------------------\n')
                print('Input Module Index: %d\n' % InputModuleIndex[i])
                print('Input CFD Level: %d mV\n' % InputCFDLevel[i])
                print('Input CFD ZeroCross: %d mV\n' % InputCFDZeroCross[i])
                print('Input Offset: %d\n' % InputOffset[i])



        # Time tagging mode specific header
        ########################################################################################
        if print_on:
            print('\n-------------------------------------\n')
        InputRate = []
        for i in range(InpChansPresent):
            InputRate.append(struct.unpack('i',fid.read(4))[0])
            if print_on:
                print('Input Rate [%02d]: %d\n' % (i, InputRate[i]))

        if print_on:
            print('------------------------------------\n')

        SyncRate = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Sync Rate: %d Hz\n' % SyncRate)

        StopAfter = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Stop After: %d ms \n' % StopAfter)

        StopReason = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Stop Reason: %d\n' % StopReason)

        ImgHdrSize = struct.unpack('i',fid.read(4))[0]
        if print_on:
            print('Imaging Header Size: %d bytes\n' % ImgHdrSize)

        nRecords = struct.unpack('Q',fid.read(8))[0] #uint64
        if print_on:
            print('Number of Records: %d\n' % nRecords)

        # Special header for imaging. How many of the following ImgHdr array elements
        # are actually present in the file is indicated by ImgHdrSize above.
        # Storage must be allocated dynamically if ImgHdrSize other than 0 is found.

        ImgHdr = fid.read(4*ImgHdrSize)

        # The header section ends after ImgHdr. Following in the file are only event records.
        # How many of them actually are in the file is indicated by nRecords in above.

        self.header_size = fid.tell() # header size is how many lines read so far.
        fid.close()

        # wrapping the header info into a dictionary.

        header = {}
        header['Ident']=Ident
        header['FormatVersion'] = FormatVersion
        header['CreatorVersion'] = CreatorVersion
        header['Comment'] = Comment
        header['BitsPerRecord'] = BitsPerRecord
        header['FileTime'] = FileTime
        header['CRLF'] = CRLF
        header['NumberOfCurves'] = NumberOfCurves
        header['MeasurementMode'] = MeasurementMode
        header['SubMode'] = SubMode
        header['Binning'] = Binning
        header['Resolution'] = Resolution
        header['Offset'] = Offset
        header['Tacq'] = Tacq
        header['StopAt'] = StopAt
        header['StopOnOvfl'] = StopOnOvfl
        header['Restart'] = Restart
        header['DispLinLog'] = DispLinLog
        header['DispTimeAxisFrom'] = DispTimeAxisFrom
        header['DispTimeAxisTo'] = DispTimeAxisTo
        header['DispCountAxisFrom'] = DispCountAxisFrom
        header['HardwareIdent'] = HardwareIdent
        header['HardwarePartNo'] = HardwarePartNo
        header['HardwareSerial'] = HardwareSerial
        header['nModulesPresent'] = nModulesPresent
        header['BaseResolution'] = BaseResolution
        header['InputsEnabled'] = InputsEnabled
        header['InpChansPresent'] = InpChansPresent
        header['ExtDevices'] = ExtDevices
        header['RefClockSource'] = RefClockSource
        header['SyncDivider'] = SyncDivider
        header['SyncDivider'] = SyncDivider
        header['SyncCFDLevel'] = SyncCFDLevel
        header['SyncCFDZeroCross'] = SyncCFDZeroCross
        header['SyncOffset'] = SyncOffset
        header['SyncDivider'] = SyncDivider
        header['SyncDivider'] = SyncDivider
        header['SyncDivider'] = SyncDivider
        header['SyncRate'] = SyncRate
        header['nRecords'] = nRecords



        # Channels information header.
        for i in range(InpChansPresent):
            header[('channel_'+str(i))]={'InputModuleIndex': InputModuleIndex[i],\
            'InputCFDLevel': InputCFDLevel[i], 'InputCFDZeroCross': \
            InputCFDZeroCross[i],'InputOffset': InputOffset[i]}

        self.header = header

        return 0

    def get_swabian_header(self):
        """
        This function reads the header info (Comment, AcqTime, SyncRate, and nRecords) from Swabian ht3 file.
        """
        with open(self.file_path) as f:
                    data = f.read()
        data = data.split()
        header = {}
        for d in data:
            temp = d.split(':')
            header[temp[0]] = temp[1]
        print('Comment: %s\n'  % (header['Comment']))
        header['AcqTime'] = int(header['AcqTime'])
        print('Acquisition Time: %d s\n' % header['AcqTime'])
        header['SyncRate'] = int(header['SyncRate'])
        print('Sync Rate: %d Hz\n' % header['SyncRate'])
        header['nRecords'] = np.int64(header['nRecords'])
        print('Number of Records: %d\n' % header['nRecords'])
        header['MeasurementMode'] = int(header['MeasurementMode'])
        header['Resolution'] = int(header['Resolution']) # in ps
        print('Resolution: %d ps\n' % header['Resolution'])
        self.header = header

    def get_photon_records(self, memory_limit=1, overwrite=True):
        """
          This function reads the photon stream and return to the true photon records to a binary .photons file.
              t3: [channel, n_sync, dtime]
              t2: [channel, dtime]

          COMMENT: It is important to understand the difference between 'records' and 'photons' according to the convention used by Picoquat GmBH.

          Each record in the .ht3 or .ht2 file has 32 bit which encode a bset of paramters:
              .ht2:  channel, absolute time, special character
              .ht3:  channel, number_of_sync_pulse, time after the syncpulse, special character

          The special cahracter is important to distinguish whether a record is a true photon. If the special bit is 0, the 32 bit record corresponds to a true photon arrival event, from channel 0 to 3. If the record is 1, the record is a special overflow record and not a true photon arrival event, and the channel number will be 63. The absolute time or number_of_sync_pulse will record the overflow.

          Use this overflow , we can encode time (or syncpulses) to infinity, although we only have a finite bit-depth. Specifically, to recover the true arrival time of a photon, we add the number of overflows*the resolution*2^(number of bits) to the actual recorded arrival time.

          In this function, we parse the binary photon-stream data, correct for the overflow, and write a binary file containing the true photon arrival times.
        """
        time_start = timing.time()
        self.memory_limit = memory_limit
        self.buffer_size = int(self.memory_limit * 1024 * 1024 / 8) # how many uint64 can be stored at once
        self.file_name_photons = self.path_str + "/" + self.file_name + '.photons'

        if not os.path.isfile(self.file_name_photons) or overwrite:
            fout = open(self.path_str + self.file_name + '.photons', 'wb') # create an output file
            # fout = open(self.file_name_photons, 'wb') # FFFF
            fin = open(self.file_path, 'rb')
            fin.seek(self.header_size) # skip the headers

            # for t3 mode
            if self.header['MeasurementMode'] == 3:

                over_flow_correction = 0
                t3_wrap_around = 1024 # if overflow happens, the true n_sync is n_sync + 1024

                over_flow_correction_last_batch  = 0 # for ocerflow correction

                while 1:
                    batch = np.fromfile(fin, dtype=np.uint32, count=self.buffer_size) # a bit array
                    lbatch = len(batch) # number of photon records, each record has 32 bits
                    special = np.zeros(lbatch)
                    channel = np.zeros(lbatch)
                    dtime = np.zeros(lbatch)
                    n_sync = np.zeros(lbatch)

                    special[:] = np.right_shift(batch[:], 31) # the first bit in the 32 bit is the special character for overflow
                    channel[:] = np.right_shift(np.left_shift(batch[:],1), 26) # the 2-7 bits
                    channel = channel.astype(int)
                    channel = np.bitwise_and(channel, 63)
                    dtime[:] = np.right_shift(np.left_shift(batch[:],7), 17) # the 8-22 bits
                    dtime = dtime.astype(int)
                    dtime = np.bitwise_and(dtime, 32767)
                    n_sync = np.bitwise_and(batch, 1023) # the 23-32 bits
                    n_sync = n_sync.astype(int)

                    num_arr = int(lbatch - sum(special)) # if special == 0, there is a true photon arrival
                    records = np.zeros((num_arr,3)) # array to store channel, n_sync, and dtime

                    ind_rec = special == 0 # index where special = 0
                    ind_channel_63 = np.invert(ind_rec) # index where special == 1, channel == 63
                    ind_n_sync_0 = n_sync == 0 # index where n_sync == 0

                    records[:, 0] = channel[ind_rec] # record channel
                    records[:, 2] = dtime[ind_rec] * self.header['Resolution'] # record true time
                    records[:, 1] = n_sync[ind_rec].copy()  # adding overflow from previous batch, but not counting overflow for this batch
                    n_sync[ind_rec & ind_n_sync_0] = 1 # for correction
                    n_sync[ind_rec] = 0
                    over_flow_correction = t3_wrap_around * np.cumsum(n_sync, dtype=self.datatype) + over_flow_correction_last_batch # oveflow correction from this batch starting from batch[0] + over_flow_correction from the last batch
                    records[:, 1] += over_flow_correction[ind_rec] # true n_sync
                    over_flow_correction_last_batch = over_flow_correction[-1] # record the last over_flow_correction to add onto the fist of next batch
                    records.astype(self.datatype).tofile(fout) # write to the output file

                    if lbatch < self.buffer_size:
                        break

            # for t2 mode
            elif self.header['MeasurementMode'] == 2:

                over_flow_correction = 0
                t2_wrap_around = 33554432

                over_flow_correction_last_batch = 0
                while 1:
                    batch = np.fromfile(fin, dtype=np.uint32, count=self.buffer_size) # a bit array
                    lbatch = len(batch) # number of photon records, each record has 32 bits
                    special = np.zeros(lbatch)
                    channel = np.zeros(lbatch)
                    dtime = np.zeros(lbatch)

                    special[:] = np.right_shift(batch[:], 31) # the first bit in the 32 bit is the special character for overflow
                    channel[:] = np.right_shift(np.left_shift(batch[:],1), 26) # the 2-7 bits
                    channel = channel.astype(int)
                    channel= np.bitwise_and(channel, 63)
                    dtime = np.bitwise_and(batch, 33554431 ) # the 8-32 bits

                    num_arr = int(lbatch - sum(special)) # if special == 0, there is a true photon arrival
                    records = np.zeros((num_arr,2)) # array to store channel and true_time

                    ind_rec = special == 0 # index where special = 0
                    ind_channel_63 = np.invert(ind_rec) # index where special == 1, channel == 63
                    ind_dtime_0 = dtime == 0 # index where dtime == 0

                    records[:,0] = channel[ind_rec] # record channel
                    records[:,1] = dtime[ind_rec].copy() # for correction
                    dtime[ind_rec] = 0
                    dtime[ind_channel_63 * ind_dtime_0] = 1
                    over_flow_correction = t2_wrap_around * np.cumsum(dtime, dtype = self.datatype) + over_flow_correction_last_batch # oveflow correction from this batch starting from batch[0] + over_flow_correction from the last batch
                    over_flow_correction_last_batch = over_flow_correction[-1] # record the last over_flow_correction to add onto the fist of next batch
                    records[:, 1] += over_flow_correction[ind_rec] # true time


                    records.astype(self.datatype).tofile(fout) # write to the output file

                    if lbatch < self.buffer_size:
                        break

            fin.close()
            fout.close()
            time_end = timing.time()
            total_time = time_end - time_start
            print('========================================')
            print('Photon records written to %s.photons' % self.file_name,)
            print('Time elapsed to get single photons record is %4f s' % total_time)
            print('========================================')

        else:
            print('%s already exists' % self.file_name_photons)

    def write_photons_to_one_channel(self, file_in, file_out):
        """
        This function writes the photons detected on different channels to one channel (set as channel 0). Useful to get the autocorrelation of the sum signal for PCFS analysis.
        """
        time_start = timing.time()
        counts = self.buffer_size * self.header['MeasurementMode']
        fout_file = self.path_str + file_out + '.photons'
        fin_file = self.path_str + file_in + '.photons'
        fout = open(fout_file, 'wb')
        fin = open(fin_file, 'rb')

        while 1:
           batch = np.fromfile(fin, dtype=self.datatype, count = counts)
           batch[::self.header['MeasurementMode']] = 0 # set the channel number to be 0
           batch.tofile(fout)

           if len(batch) < counts:
               break

        fout.close()
        fin.close()
        time_end = timing.time()
        total_time = time_end - time_start
        print('========================================')
        print('Photon records written to %s.photons' % file_out,)
        print('Time elapsed is %4f s' % total_time)
        print('========================================')

    def photons_to_channel(self, file_in, file_out, n_channel = 4):
        """
        This function creates four .photons output files containing the photon arrival data of each channel.

        file_in is the filename of the .photons file without extension.
        file_out is the filename (_ch0, _ch1, _ch2, _ch3) of the new photons files.
        n_channels is the number of channels used (2 for PCFS), set default to 4.
        """
        time_start = timing.time()
        counts = self.buffer_size * self.header['MeasurementMode']
        fin_file = self.path_str + file_in + '.photons'
        fin = open(fin_file, 'rb')
        fout_file = [self.path_str + file_out + '_ch' + str(i) + '.photons' for i in range(n_channel)]
        fout = [open(file, 'wb') for file in fout_file]

        while 1:
            batch = np.fromfile(fin, dtype=self.datatype, count = counts)
            lbatch = len(batch)//self.header['MeasurementMode']
            batch.shape = lbatch, self.header['MeasurementMode']
            for i in range(n_channel):
                batch[batch[:, 0] == i].tofile(fout[i])

            if lbatch < self.buffer_size:
                break

        fin.close()
        for i in range(n_channel):
            fout[i].close()
        time_end = timing.time()
        total_time = time_end - time_start
        print('Total time elapsed is %4f s' % total_time)

    def arrival_time_sorting(self, file_in, file_out, tau_window):
        """
        This function sorts photon data according to photon arrival time.
        For t2 data the time in ps is the absolute arrival time of the photons.
        For t3 data the time is relative to the sync pulse.
        A new .photons file is written containing the photons detected within tau_window: [lower_tau, upper_tau] in ps.
        """
        time_start = timing.time()
        # counts = self.buffer_size * self.header['MeasurementMode']
        fin_file = self.path_str +file_in + '.photons'
        fin = open(fin_file, 'rb')
        fout_file = self.path_str + file_out + '.photons'
        fout = open(fout_file, 'wb')

        while 1:
            batch = np.fromfile(fin, dtype=self.datatype, count = counts)
            lbatch = len(batch)//self.header['MeasurementMode']
            batch.shape = lbatch, self.header['MeasurementMode']
            ind_lower = batch[:, -1] > tau_window[0]
            ind_upper = batch[:, -1] <= tau_window[1]
            batch[ind_lower * ind_upper].tofile(fout)

            if lbatch < self.buffer_size:
                break

        fin.close()
        fout.close()
        time_end = timing.time()
        total_time = time_end - time_start
        print('Total time elapsed is %4f s' % total_time)

    def get_intensity_trace(self, bin_width, intensity_correction=False, threshold=150):
        """
        This function compiles and stores the intensity trace as a property of the photons class: self.intensity_counts.

        Two *args must be given, in the order of:
            file_in: filename of the .photons file without ending
            bin_width: width of the bin for intensity compilation
        """
        # time_start = timing.time()
        # fin_file = self.path_str + file_in + '.photons'
        # fin = open(fin_file, 'rb')
        fin = open(self.file_name_photons, 'rb')
        photons_records = np.fromfile(fin, dtype=self.datatype)
        fin.close()

        self.intensity_counts = {}
        length_photons = len(photons_records) // self.header['MeasurementMode']
        photons_records.shape = length_photons, self.header['MeasurementMode']

        if self.header['MeasurementMode'] == 3:
            pulse_spacing = (1/self.header['SyncRate'])*1e12 # pulse spacing in ps
            n_bins = int(photons_records[-1, 1]*pulse_spacing // bin_width)
            bins = np.arange(0.5, n_bins + 1.5) * bin_width
            time_vec = np.arange(1, n_bins + 1) * bin_width
            photon_trace = np.zeros((n_bins, 4))  # store the histogram
            for i in range(4):
                temp = photons_records[photons_records[:, 0] == i, 1]*pulse_spacing
                photon_trace[:, i] = np.histogram(temp, bins=bins)[0]
            #
            # if intensity_correction:
            #     trace_total = np.sum(photon_trace, axis=1)
            #     on_times = time_vec[trace_total > threshold]
            #     on_times_lower = on_times - (bin_width / 2)
            #     on_times_upper = on_times + (bin_width / 2)
            #     logic = np.full(len(photons_records[:, 1]), False)
            #
            #     # for idx in tqdm(range(len(trace_total))):
            #     #     if trace_total[idx] > threshold:
            #     #         # if idx == 0:
            #     #         #     tlower = 0
            #     #         # else:
            #     #         tlower = time_vec[idx]-(bin_width/2)
            #     #         tupper = time_vec[idx]+(bin_width/2)
            #     #         logic = logic + ((photons_records[:, 1]*pulse_spacing > tlower) & (photons_records[:, 1]*pulse_spacing < tupper))
            #     #         # del_idxs = np.where((photons_records[:, 1]*pulse_spacing > tlower) & (photons_records[:, 1]*pulse_spacing < tupper))
            #
            #     for idx in tqdm(range(len(on_times))):
            #         logic = logic + ((photons_records[:, 1] * pulse_spacing > on_times_lower[idx]) & (photons_records[:, 1] * pulse_spacing < on_times_upper[idx]))
            #
            #     photons_records = photons_records[logic]
            #     for i in range(2):
            #         temp = photons_records[photons_records[:, 0] == i, 1] * pulse_spacing
            #         photon_trace[:, i] = np.histogram(temp, bins=bins)[0]

        else:
            n_bins = int(photons_records[-1, 1] // bin_width)
            bins = np.arange(0.5, n_bins+1.5) * bin_width
            time_vec = np.arange(1, n_bins+1) * bin_width
            photon_trace = np.zeros((n_bins, 4)) # store the histogram

            for i in range(4):
                temp = photons_records[photons_records[:, 0] == i, 1]
                photon_trace[:, i] = np.histogram(temp, bins=bins)[0]

        class trace:
            def __init__(self):
                self.time = time_vec
                self.trace = photon_trace
                self.trace_total = np.sum(photon_trace, axis=1)
                self.bin_width = bin_width

            def plot(self, channels=[0, 1], plot_channels=False, plot_total=True, figsize=(4, 4), dpi=150, spacer=0., label=None, color=None, fig=None, plot_fit=False):
                if fig is None:
                    fig = plt.figure(figsize=figsize, dpi=dpi)

                if color is None:
                    color = np.array([0, 0, 0])

                if label is None:
                    label = ''

                if plot_channels:
                    for channel in channels:
                        plt.plot(self.time, self.trace[:, channel])

                if plot_total:
                    plt.plot(self.time, self.trace_total)

                return fig

        self.trace = trace()


    def intensity_trace_correction(self, photons_records, bin_width, threshold):
        self.intensity_counts = {}
        length_photons = len(photons_records) // self.header['MeasurementMode']
        photons_records.shape = length_photons, self.header['MeasurementMode']

        if self.header['MeasurementMode'] == 3:
            pulse_spacing = (1/self.header['SyncRate'])*1e12 # pulse spacing in ps
            n_bins = int(photons_records[-1, 1]*pulse_spacing // bin_width)
            bins = np.arange(0.5, n_bins + 1.5) * bin_width
            time_vec = np.arange(1, n_bins + 1) * bin_width
            photon_trace = np.zeros((n_bins, 2))  # store the histogram
            for i in range(2):
                temp = photons_records[photons_records[:, 0] == i, 1]*pulse_spacing
                photon_trace[:, i] = np.histogram(temp, bins=bins)[0]

            trace_total = np.sum(photon_trace, axis=1)
            on_times = time_vec[trace_total > threshold]
            on_times_lower = on_times - (bin_width/2)
            on_times_upper = on_times + (bin_width/2)

            logic = np.full(len(photons_records[:, 1]), False)
            for idx in tqdm(range(len(on_times))):
                logic = logic + ((photons_records[:, 1] * pulse_spacing > on_times_lower[idx]) & (photons_records[:, 1] * pulse_spacing < on_times_upper[idx]))

            photons_records = photons_records[logic]
            for i in range(2):
                temp = photons_records[photons_records[:, 0] == i, 1] * pulse_spacing
                photon_trace[:, i] = np.histogram(temp, bins=bins)[0]


        else:
            n_bins = int(photons_records[-1, 1] // bin_width)
            bins = np.arange(0.5, n_bins+1.5) * bin_width
            time_vec = np.arange(1, n_bins+1) * bin_width
            photon_trace = np.zeros((n_bins, 4)) # store the histogram

            for i in range(4):
                temp = photons_records[photons_records[:, 0] == i, 1]
                photon_trace[:, i] = np.histogram(temp, bins=bins)[0]

        return photons_records

    def get_lifetime_histogram_multichannel(self, ch0_offset=0, ch1_offset=0, resolution=1, normalize=True, bckgrnd_sub=0, auto_offset=True, units='ps'):
        """
        This function histograms the lifetime of a .ht3 file with a given resolution.The histogram is stored as a property of the photons class: self.lifetime.
        The given resolution should be a multiple of the original resolution used for the measurement. For instance, if the measurement resolution was 64 ps, then the resolution to form the histogram of the photon-arrival records could be 128, 256, 384, or 512 ps ...
        """
        resolution = self.header['Resolution']*resolution
        if self.header['MeasurementMode'] == 2:
            print('Only fot t3 data!')
            return False
        if resolution % int(self.header['Resolution']) != 0:
            print('The given resolution must be a multiple of the original resolution!\n')
            print('Check obj.header[\'Resolution\'].')
            return False

        time_start = timing.time()

        # fin_file = self.path_str + file_in + '.photons'
        # fin = open(fin_file, 'rb')
        fin = open(self.file_name_photons, 'rb')

        # initializations
        rep_time = 1e12/self.header['SyncRate'] # in ps
        n_bins = int(rep_time//resolution)
        bins = np.arange(0.5, n_bins+1.5) * resolution
        time = np.arange(1, n_bins+1) * resolution
        ch_counts = np.zeros((2, n_bins))

        counts = self.buffer_size * self.header['MeasurementMode']
        while 1:
            batch = np.fromfile(fin, dtype=self.datatype, count=counts)
            time_stamps = batch[2::3]
            ch0_inds = np.where(batch[::3]==0)[0]
            ch1_inds = np.where(batch[::3]==1)[0]
            ch0_stamps = time_stamps[ch0_inds] - ch0_offset
            ch1_stamps = time_stamps[ch1_inds] - ch1_offset
            ch0_hist = np.histogram(ch0_stamps, bins=bins)
            ch1_hist = np.histogram(ch1_stamps, bins=bins)
            # histo = np.histogram(batch[2::3], bins=bins)
            ch_counts[0, :] += ch0_hist[0]
            ch_counts[1, :] += ch1_hist[0]

            if len(batch) < counts:
                break

        fin.close()

        temp = np.sum(ch_counts, axis=0)
        hist_counts = temp

        if bckgrnd_sub > 0:
            bckgrnd = sum(hist_counts[0:bckgrnd_sub])/bckgrnd_sub
            hist_counts = hist_counts - bckgrnd

        if normalize:
            hist_counts = hist_counts/max(hist_counts)

        if self.datatype == np.int64:
            time = rep_time - time

        if auto_offset:
            offset = time[np.where(hist_counts==max(hist_counts))]
            time = time - offset

        time_end = timing.time()
        total_time = time_end - time_start
        print('Total time elapsed is %4f s' % total_time)
        file_path = self.file_path

        if units == 'µs':
            time = time/1e6
        elif units == 'ns':
            time = time/1e3

        class lifetime:
            def __init__(self):
                self.time = time
                self.counts = hist_counts
                self.ch_counts = ch_counts
                self.resolution = resolution
                self.file_path = file_path
                self.mle = None

            def save_csv(self):
                x = time
                y = hist_counts
                d = np.array([x, y]).T
                np.savetxt(file_path.split('.ht3')[0] + '.csv', d, delimiter=',')

            def _make_fit(self, lifetime_function, theta):
                x = self.time
                return function_clean(theta, x, lifetime_function)

            def get_fit(self, lifetime_function, theta, nruns=1, IRF=True):
                x = self.time
                y = self.counts

                if not IRF:
                    max_idx = int(np.where(y==max(y))[0])
                    x = x[max_idx:]
                    y = y[max_idx:]

                self.x_fit = x
                self.y_fit = y

                mle = MLE(
                    theta_in=theta,
                    args=(x, y, lifetime_function),
                    nruns=nruns,
                    obj_function=lossP,
                    guess=theta,
                    randomizer='bounded',
                    bounds=([theta['lb'], theta['ub']]),
                    printing=False
                )

                self.mle = mle

                self.fit = g2_clean(mle.theta, self.x_fit, lifetime_function)

                with np.printoptions(precision=5, suppress=True):
                    print(mle.theta)
                    print(mle.theta_diff)

            def plot(self, figsize=(4, 4), dpi=150, spacer=0., label=None, color=None, fig=None, plot_fit=False, fit_color_scale=0.3):
                if fig is None:
                    fig = plt.figure(figsize=figsize, dpi=dpi)
                    plt.yscale('log')

                if color is None:
                    color = np.array([0, 0, 0])

                if label is None:
                    label = ''

                if plot_fit:
                    plt.scatter(self.time, self.counts + spacer, marker='s', s=2, color=color, label=label, alpha=0.75)
                    plt.plot(self.x_fit, self.fit + spacer, color=color*fit_color_scale, label=label+' (fit)')

                else:
                    plt.plot(self.time, self.counts + spacer, color=color, label=label)

                return fig

        self.lifetime = lifetime()

    def get_lifetime_histogram(self, resolution=1, normalize=True, bckgrnd_sub=0, auto_offset=True, units='ps'):
        """
        This function histograms the lifetime of a .ht3 file with a given resolution.The histogram is stored as a property of the photons class: self.lifetime.
        The given resolution should be a multiple of the original resolution used for the measurement. For instance, if the measurement resolution was 64 ps, then the resolution to form the histogram of the photon-arrival records could be 128, 256, 384, or 512 ps ...
        """
        resolution = self.header['Resolution']*resolution
        if self.header['MeasurementMode'] == 2:
            print('Only fot t3 data!')
            return False
        if resolution % int(self.header['Resolution']) != 0:
            print('The given resolution must be a multiple of the original resolution!\n')
            print('Check obj.header[\'Resolution\'].')
            return False

        time_start = timing.time()

        # fin_file = self.path_str + file_in + '.photons'
        # fin = open(fin_file, 'rb')
        fin = open(self.file_name_photons, 'rb')

        # initializations
        rep_time = 1e12/self.header['SyncRate'] # in ps
        n_bins = int(rep_time//resolution)
        bins = np.arange(0.5, n_bins+1.5) * resolution
        time = np.arange(1, n_bins+1) * resolution
        hist_counts = np.zeros(n_bins)

        counts = self.buffer_size * self.header['MeasurementMode']
        while 1:
            batch = np.fromfile(fin, dtype=self.datatype, count=counts)
            histo = np.histogram(batch[2::3], bins=bins)
            hist_counts += histo[0]

            if len(batch) < counts:
                break

        fin.close()

        if bckgrnd_sub > 0:
            bckgrnd = sum(hist_counts[0:bckgrnd_sub])/bckgrnd_sub
            hist_counts = hist_counts - bckgrnd

        if normalize:
            hist_counts = hist_counts/max(hist_counts)

        if self.datatype == np.int64:
            time = rep_time - time

        if auto_offset:
            offset = time[np.where(hist_counts==max(hist_counts))]
            time = time - offset

        time_end = timing.time()
        total_time = time_end - time_start
        print('Total time elapsed is %4f s' % total_time)
        file_path = self.file_path

        if units == 'µs':
            time = time/1e6
        elif units == 'ns':
            time = time/1e3

        class lifetime:
            def __init__(self):
                self.time = time
                self.counts = hist_counts
                self.resolution = resolution
                self.file_path = file_path
                self.mle = None

            def save_csv(self):
                x = time
                y = hist_counts
                d = np.array([x, y]).T
                np.savetxt(file_path.split('.ht3')[0] + '.csv', d, delimiter=',')

            def _make_fit(self, lifetime_function, theta):
                x = self.time
                return function_clean(theta, x, lifetime_function)

            def get_fit(self, lifetime_function, theta, nruns=1):
                x = self.time
                y = self.counts
                mle = MLE(
                    theta_in=theta,
                    args=(x, y, lifetime_function),
                    nruns=nruns,
                    obj_function=lossP,
                    guess=theta,
                    randomizer='bounded',
                    bounds=([theta['lb'], theta['ub']]),
                    printing=False
                )

                self.mle = mle

                self.fit = g2_clean(mle.theta, self.time, lifetime_function)

                with np.printoptions(precision=5, suppress=True):
                    print(mle.theta)

            def plot(self, figsize=(4, 4), dpi=150, spacer=0., label=None, color=None, fig=None, plot_fit=False, fit_color_scale=0.3):
                if fig is None:
                    fig = plt.figure(figsize=figsize, dpi=dpi)
                    plt.yscale('log')

                if color is None:
                    color = np.array([0, 0, 0])

                if label is None:
                    label = ''

                if plot_fit:
                    plt.scatter(self.time, self.counts + spacer, marker='s', s=2, color=color, label=label, alpha=0.75)
                    plt.plot(self.time, self.fit + spacer, color=color*fit_color_scale, label=label+' (fit)')

                else:
                    plt.plot(self.time, self.counts + spacer, color=color, label=label)

                return fig

        self.lifetime = lifetime()


    @jit
    def photons_in_bins(self, ch0, ch1, lag_bin_edges, offset_lag):
        """
        Adapted from Boris Spokoyny's code.
        This function allows to correlate the photon-stream on a log timescale. The photon correlation is stored as a property of the photons class: self.cross_corr or self.auto_corr.

        file_in: file ID of the photons-file to correlate
        correlations: 'cross_corr' or 'auto_corr'.
        channels: Hydraharp channels to be correlated. e.g. [0,1] for cross-correlation of channels 0 and 1.
        time_bounds: upper and lower limit for the correlation. In ps for T2, in pulses for T3.
        lag_precision: Number of correlation points between time-spacings of log(2). Must be integers larger than 1.
        lag_offset: offset in ps or pulses between the channels.

        This algorithm computes the cross-correlation between ch0 and ch1 variables.
        For T2 data, ch0 and ch1 correspond to absolute photon arrival times.
        For T3 data, ch0 and ch1 should correspond to the photon arrival sync pulse number.
        start_time and stop_time for T2 data should be in time units of the photon arrival times, and for T3 data should be in units of sync pulses.

        The correlation lags are log(2) spaced with coarseness # of lags per cascade, i.e. if start_time = 1; stop_time = 50; coarseness = 4; the lag bins will be [1, 2, 3, 4;  6, 8, 10, 12;  16, 20, 24, 28;  36, 44 ]. If coarseness = 2, the lag bins become [1, 2;  4, 6;  10, 14;  22,30;  46].
        The algorithm works by taking each photon arrival time and counting the number of photons that are lag bins away. For example say our lag bin edges are [1, 2, 4, 6, 10, 14]
                Time Slot: 1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
                Photon     1   1   0   0   1   1   0   1   1   0   1   0   0   0   1
                1st Photon ^
                Lag bins   |   |       |       |               |               |
                # photons    1     0       2           2                1
                2nd Photon     ^
                Lag bins       |   |       |       |               |               |
                # photons        0     1       1           3               1
                3rd Photon                 ^
                Lag bins                   |   |       |       |               |               |
                # photons                    1     2       1           1               1

                etc..
        The cross-correlation is the sum of all the photons for each bin, i.e. for the three runs above we get [2, 3, 4, 6, 3].

        This function counts the number of photons in the photon stream bins according to a prescription from Ted Laurence: Fast flexible algorithm for calculating photon correlation, Optics Letters, 31, 829, 2006
        """
        num_ch0 = len(ch0)
        num_ch1 = len(ch1)
        n_edges = len(lag_bin_edges)
        low_inds = np.zeros(n_edges, dtype=int)
        max_inds = np.zeros(n_edges, dtype=int)
        acf = np.zeros(n_edges-1)

        low_inds[0] = 1
        for phot_ind in range(num_ch0):
            bin_edges = ch0[phot_ind] + lag_bin_edges + offset_lag

            for k in range(n_edges-1):
                while low_inds[k] < num_ch1 and ch1[low_inds[k]] < bin_edges[k]:
                    low_inds[k] += 1

                while max_inds[k] < num_ch1 and ch1[max_inds[k]] <= bin_edges[k+1]:
                    max_inds[k] += 1

                low_inds[k+1] = max_inds[k]
                acf[k] += max_inds[k] - low_inds[k]

        return acf

    def photon_corr(self, file_in, correlations, channels, time_bounds, lag_precision, lag_offset=0):

        time_start = timing.time()

        fin_file = self.path_str +file_in + '.photons'
        fin = open(fin_file, 'rb')
        photons_records = np.fromfile(fin, dtype=self.datatype)
        length_photons = len(photons_records) // self.header['MeasurementMode']
        photons_records.shape = length_photons, self.header['MeasurementMode']
        fin.close()

        # split into channels
        ch0 = photons_records[photons_records[:,0] == channels[0], 1] # ch0 syncs
        ch1 = photons_records[photons_records[:,0] == channels[1], 1] # ch1 syncs

        start_time, stop_time = time_bounds

        '''create log 2 spaced lags'''
        cascade_end = int(np.log2(stop_time)) # cascades are collections of lags  with equal bin spacing 2^cascade
        nper_cascade =  lag_precision # number of equal
        a = np.array([2**i for i in range(1,cascade_end+1)])
        b = np.ones(nper_cascade)
        division_factor = np.kron(a,b)
        lag_bin_edges = np.cumsum(division_factor/2)
        lags = (lag_bin_edges[:-1] + lag_bin_edges[1:]) * 0.5

        # find the bin region
        start_bin = np.argmin(np.abs(lag_bin_edges - start_time))
        stop_bin = np.argmin(np.abs(lag_bin_edges - stop_time))
        lag_bin_edges = lag_bin_edges[start_bin:stop_bin+1] # bins
        lags = lags[start_bin+1:stop_bin+1] # center of the bins
        division_factor = division_factor[start_bin+1:stop_bin+1] # normalization factor


        # counters etc for normalization
        ch0_min = np.inf
        ch1_min = np.inf # minimum time tag
        ch0_count = len(ch0)
        ch1_count = len(ch1) # photon numbers in each channel
        ch0_min = min(ch0_min, min(ch0))
        ch1_min = min(ch1_min, min(ch1))

        '''correlating '''
        tic = timing.time()
        print('Correlating data...\n')

        corr = self.photons_in_bins(ch0, ch1, lag_bin_edges, lag_offset)

        # normalization
        ch0_max = max(ch0)
        ch1_max = max(ch1)
        tag_range = max(ch1_max, ch0_max) - min(ch1_min, ch0_min) # range of tags in the entire dataset

        corr_div = corr/division_factor
        corr_norm = 2 * corr_div * tag_range / (tag_range - lags) * ch0_max / (ch0_count * ch1_count)

        print('Done\n')
        toc = timing.time()
        print('Time elapsed during correlating is %4f s' % (toc-tic))

        # store in property
        if self.header['MeasurementMode'] == 3:
            sync_period = 1e12/self.header['SyncRate']
            lags = lags * sync_period

        if 'cross' in correlations:
            self.cross_corr = {}
            self.cross_corr['lags'] = lags
            self.cross_corr['corr_norm'] = corr_norm
        elif 'auto' in correlations:
            self.auto_corr = {}
            self.auto_corr['lags'] = lags
            self.auto_corr['corr_norm'] = corr_norm


        time_end = timing.time()
        print('Total time elapsed is %4f s' % (time_end - time_start))

    def get_g2(self, channels, time_range, n_bins, normalize=False, trace_correction=False, threshold=0, binwidth=50e9):
        """
        This function calculates g2 for t3 data.
        file_in and channels are the same as photon_corr.
        time_range is the maximum time we're interested in, in ps.
        n_bins are the number of bins for correlation.
        """
        if self.header['MeasurementMode'] == 2:
            print('Only for t3 data!')
            return False

        time_start = timing.time()

        # fin_file = self.path_str + file_in + '.photons'
        # fin = open(fin_file, 'rb')

        fin = open(self.file_name_photons, 'rb')
        photons_records = np.fromfile(fin, dtype=self.datatype)
        fin.close()

        if trace_correction:
            photons_records = self.intensity_trace_correction(photons_records=photons_records, bin_width=binwidth, threshold=threshold)
        else:

            length_photons = len(photons_records) // 3
            photons_records.shape = length_photons, 3

        # split into channels
        pulse = 1e12 / self.header['SyncRate']
        ch0 = photons_records[photons_records[:, 0] == channels[0], 2] + photons_records[photons_records[:, 0] == channels[0], 1] * pulse# ch0 time
        ch1 = photons_records[photons_records[:, 0] == channels[1], 2] + photons_records[photons_records[:, 0] == channels[1], 1] * pulse# ch1 time

        # use linear spaced bins for g2 calculation
        bin_width = time_range // n_bins
        lag_bin_edges = np.arange(0, time_range+2*bin_width, bin_width)
        lags = np.hstack((-lag_bin_edges[-2::-1], lag_bin_edges[1:]))
        lag_bin_edges = lags - bin_width/2

        '''correlating '''
        tic = timing.time()
        print('Correlating data...\n')

        corr = self.photons_in_bins(ch0, ch1, lag_bin_edges, 0)
        if normalize:
            corr = corr/max(corr)

        # correct for offset
        n_ind = pulse // bin_width
        ind_pulse_1 = np.argmax(corr[n_bins:int(n_bins+1.5*n_ind)]) # find the index of the first pulse
        offset = pulse - lags[ind_pulse_1+n_bins] # correct for offset


        print('Done\n')
        toc = timing.time()
        print('Time elapsed during correlating is %4f s' % (toc-tic))


        # self.g2 = {}
        # self.g2['lags'] = (lags[:-1] + offset) / 1e3 # in ns
        # self.g2['g2'] = corr
        #
        # plt.plot(self.g2['lags'], self.g2['g2'], '-o', markersize=1)
        # plt.ylabel('Event counts')
        # plt.xlabel('Pulse separation [ns]')
        # plt.show()

        time_end = timing.time()
        print('Total time elapsed is %4f s' % (time_end - time_start))

        class g2:
            def __init__(self):
                self.time = (lags[:-1] + offset) / 1e6 # in µs
                self.counts = corr
                self.mle = None

            def get_fit(self, g2_function, theta, nruns=1):
                x = self.time
                y = self.counts
                # y = y / max(y)
                mle = MLE(
                    theta_in=theta,
                    args=(x, y, g2_function),
                    nruns=nruns,
                    obj_function=lossP,
                    guess='init',
                    randomizer='bounded',
                    bounds=([theta['lb'], theta['ub']]),
                    printing=False
                )

                self.mle = mle

                self.fit = g2_clean(mle.theta, self.time, g2_function)


            def plot(self, figsize=(4, 4), dpi=150, spacer=0., label=None, color=None, fig=None, plot_fit=False, fixframe=False, fit_color_scale=0.3):
                if fig is None:
                    if fixframe:
                        fig = plt.figure(figsize=figsize, dpi=dpi)
                    else:
                        from AP_figs_funcs import make_fig
                        fig = make_fig(figsize[0], figsize[1], dpi=dpi)

                if color is None:
                    color = np.array([0, 0, 0])

                if label is None:
                    label = ''

                if plot_fit:
                    # plt.scatter(self.time, self.counts + spacer, marker='s', s=2, color=color, label=label)
                    # plt.plot(self.time, self.counts + spacer, '-s', markersize=2, color=color, label=label)
                    plt.plot(self.time, self.counts + spacer, color=color, label=label, alpha=0.75)
                    plt.plot(self.time, self.fit + spacer, color=color*fit_color_scale, label=label+' (fit)')
                else:
                    plt.plot(self.time, self.counts + spacer, color=color, label=label)

                return fig


        self.g2 = g2()

    def get_ht3_header(self):
        """
        This function reads the header info (Comment, AcqTime, SyncRate, and nRecords) from Swabian ht3 file. Depreciated
        """
        header = {}
        with open(self.file_path,'rb') as f:
            temp = f.read(72)
            temp = f.read(256)
            if temp[:4] == b'none':
                header['Comment'] = 'none'
            else:
                header['Comment'] = temp
            print('Comment: %s\n'  % (header['Comment']))
            temp = f.read(36)
            header['AcqTime'] = struct.unpack('i',f.read(4))[0]
            print('Acquisition Time: %d s\n' % header['AcqTime'])
            temp = f.read(488)
            # temp = f.read(528)
            header['SyncRate'] = struct.unpack('i',f.read(4))[0]
            print('Sync Rate: %d Hz\n' % header['SyncRate'])
            temp = f.read(12)
            header['nRecords'] = struct.unpack('Q',f.read(8))[0]
            print('Number of Records: %d\n' % header['nRecords'])
            header['MeasurementMode'] = 3
            header['Resolution'] = 34 # in ps
            print('Resolution: %d ps\n' % header['Resolution'])
        self.header = header

    def rmv_after_pulse(self, file_in, file_out):
        """
        This function removes afterpulsing from t3 data and writes to a new binary file, accomplished by removing any second photons detected after the same sync pulse for each detection channel.
        """
        time_start = timing.time()
        if self.header['MeasurementMode'] == 2:
            print('Only for t3 data!')
            return False
        counts = self.buffer_size * 3
        fin_file = self.path_str +file_in + '.photons'
        fin = open(fin_file, 'rb')
        fout_file = self.path_str + file_out + '.photons'
        fout = open(fout_file, 'wb')


        temp1 = np.array([-3,-2,-1]) # Assuming there are no more than three afterpulse photons
        temp2 = [-1]*3
        while 1:
            batch = np.fromfile(fin, dtype=self.datatype, count = counts)
            lbatch = len(batch)//3
            batch.shape = lbatch, 3
            channel_col = batch[:,0].copy()
            n_sync_col = np.zeros((lbatch+3))
            n_sync_col[3:] = batch[:,1].copy()
            n_sync_col[:3] = temp1[:]
            temp1 = batch[-3:, 1]


            u, ind_n_sync = np.unique(n_sync_col, return_index=True) # record the indice of the first time a n_sync appears
            ind_channel_same = np.zeros((lbatch, 3))
            num_after_pulse = np.zeros(len(ind_n_sync))
            num_after_pulse[1:] = np.diff(ind_n_sync) - 1 # when the two unique index has lager than one difference, which means there are photons between them. This is the number of afterpulse photons at each pulse.
            ind_after_pulse_sync = np.full((lbatch+3,3),False)
            ind_num_after_pulse = []
            for i in range(1,4):
                ind_num_after_pulse.append((num_after_pulse == i)[:]) # index in the ind_n_sync array where the sync pulse has i+1 afterpulse photons
                for j in range(1,i+1):
                    ind_after_pulse_sync[ind_n_sync[ind_num_after_pulse[i-1]]-j, i-1] = True

            channel_move = [np.roll(channel_col,i) for i in range(1,4)] # shift the array by 1, 2, 3 to compare whether the channel number is the same as the one/two/three before
            for i in range(1,4):
                channel_move[i-1][:i] = temp2[-i:] # compare with the first one
                ind_channel_same[:, i-1] = channel_col == channel_move[i-1] # index of the records which have the same channel number as the one/two/three before
            temp2 = channel_col[-3:]

            ind_rmv = np.zeros(lbatch)
            for i in range(3):
                ind_rmv += ind_channel_same[:, i] * ind_after_pulse_sync[3:,i] # index of the records which have both the same channel number and n_sync as the one/two/three before, these should be removed
            ind_remain = ind_rmv == 0
            batch[ind_remain, :].tofile(fout)
             # record the last three to compare with the first one in next batch

            if lbatch < self.buffer_size:
                break

        fin.close()
        fout.close()
        time_end = timing.time()
        total_time = time_end - time_start
        print('Total time elapsed is %4f s' % total_time)



monoexp_lifetime =  "(w[0]/2)*np.exp(((1/w[1])**2*(w[2]**2)/2)-((x-w[3])*(1/w[1])))*(1+erf(((x-w[3])/(w[2]*np.sqrt(2)))-(w[2]*(1/w[1])/np.sqrt(2))))+w[4]"

def make_theta_monoexp_lifetime(p):
    var_names = ['A1', 'T1', 'IRF', 't0', 'z0']
    return {                     #  A1,           T1,          IRF,           t0,           z0
        'init': np.array([p.iloc[0][0], p.iloc[0][1], p.iloc[0][2], p.iloc[0][3], p.iloc[0][4]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][1], p.iloc[1][2], p.iloc[1][3], p.iloc[1][4]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][1], p.iloc[2][2], p.iloc[2][3], p.iloc[2][4]]),
    }, var_names


biexp_lifetime   = '(w[0]/2)*np.exp(((1/w[1])**2*(w[2]**2)/2)-((x-w[3])*(1/w[1])))*(1+erf(((x-w[3])/(w[2]*np.sqrt(2)))-(w[2]*(1/w[1])/np.sqrt(2)))) \
                   +(w[5]/2)*np.exp(((1/w[6])**2*(w[2]**2)/2)-((x-w[3])*(1/w[6])))*(1+erf(((x-w[3])/(w[2]*np.sqrt(2)))-(w[2]*(1/w[6])/(np.sqrt(2))))) \
                   +w[4]'

biexp_lifetime_no_IRF   = 'w[0]*np.exp((-x+w[-2])/w[1]) + w[2]*np.exp((-x+w[-2])/w[3])+w[-1]'

def make_theta_biexp_lifetime_no_IRF(p):
    var_names = ['A1', 'T1', 'A2', 'T2', 't0', 'z0']
    return {                     #  A1,           T1,          A1,           T2,            t0,            z0,
        'init': np.array([p.iloc[0][0], p.iloc[0][1], p.iloc[0][2], p.iloc[0][3], p.iloc[0][6], p.iloc[0][7]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][1], p.iloc[1][2], p.iloc[1][3], p.iloc[1][6], p.iloc[1][7]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][1], p.iloc[2][2], p.iloc[2][3], p.iloc[2][6], p.iloc[2][7]]),
    }, var_names


triexp_lifetime_no_IRF   = 'w[0]*np.exp((-x+w[-2])/w[1]) + w[2]*np.exp((-x+w[-2])/w[3]) + w[4]*np.exp((-x+w[-2])/w[5]) + w[-1]'

def make_theta_triexp_lifetime_no_IRF(p):
    var_names = ['A1', 'T1', 'A2', 'T2', 'A3', 'T3', 't0', 'z0']
    return {                     #  A1,           T1,          A2,           T2,           A3,           T3,            t0,            z0,
        'init': np.array([p.iloc[0][0], p.iloc[0][1], p.iloc[0][2], p.iloc[0][3], p.iloc[0][4], p.iloc[0][5], p.iloc[0][6], p.iloc[0][7]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][1], p.iloc[1][2], p.iloc[1][3], p.iloc[1][4], p.iloc[1][5], p.iloc[1][6], p.iloc[1][7]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][1], p.iloc[2][2], p.iloc[2][3], p.iloc[2][4], p.iloc[2][5], p.iloc[2][6], p.iloc[2][7]]),
    }, var_names

# + np.exp(-np.abs(x+w[2])/w[4]) + np.exp(-np.abs(x-w[2])/w[4])" \

def make_theta_biexp_lifetime(p):
    var_names = ['A1', 'T1', 'IRF', 't0', 'z0', 'A2', 'T2']
    return {                     #  A1,           T1,          IRF,           t0,           z0,          A2,             T2
        'init': np.array([p.iloc[0][0], p.iloc[0][1], p.iloc[0][2], p.iloc[0][3], p.iloc[0][4], p.iloc[0][5],  p.iloc[0][6]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][1], p.iloc[1][2], p.iloc[1][3], p.iloc[1][4], p.iloc[1][5],  p.iloc[1][6]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][1], p.iloc[2][2], p.iloc[2][3], p.iloc[2][4], p.iloc[2][5],  p.iloc[2][6]]),
    }, var_names


triexp_lifetime  = '(w[0]/2)*np.exp(((1/w[1])**2*(w[2]**2)/2)-((x-w[3])*(1/w[1])))*(1+erf(((x-w[3])/(w[2]*np.sqrt(2)))-(w[2]*(1/w[1])/np.sqrt(2)))) \
                   +(w[5]/2)*np.exp(((1/w[6])**2*(w[2]**2)/2)-((x-w[3])*(1/w[6])))*(1+erf(((x-w[3])/(w[2]*np.sqrt(2)))-(w[2]*(1/w[6])/(np.sqrt(2))))) \
                   +(w[7]/2)*np.exp(((1/w[8])**2*(w[2]**2)/2)-((x-w[3])*(1/w[8])))*(1+erf(((x-w[3])/(w[2]*np.sqrt(2)))-(w[2]*(1/w[8])/(np.sqrt(2))))) \
                   +w[4]'


def make_theta_triexp_lifetime(p):
    var_names = ['A1', 'T1', 'IRF', 't0', 'z0', 'A2', 'T2', 'A3', 'T3']
    return {                     #  A1,           T1,          IRF,           t0,           z0,          A2,             T2,            A3,            T3,
        'init': np.array([p.iloc[0][0], p.iloc[0][1], p.iloc[0][2], p.iloc[0][3], p.iloc[0][4], p.iloc[0][5],  p.iloc[0][6],  p.iloc[0][7],  p.iloc[0][8]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][1], p.iloc[1][2], p.iloc[1][3], p.iloc[1][4], p.iloc[1][5],  p.iloc[1][6],  p.iloc[1][7],  p.iloc[1][8]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][1], p.iloc[2][2], p.iloc[2][3], p.iloc[2][4], p.iloc[2][5],  p.iloc[2][6],  p.iloc[2][7],  p.iloc[2][8]]),
    }, var_names


# Monoexponential (two lifetimes) g2 function with up to 3 side peaks, variable reprate
g2_monoexp  = "w[1] + w[3] * (w[0]*np.exp(-np.abs(x)/w[4]) " \
              "                  + np.exp(-np.abs(x+w[2])/w[4]) + np.exp(-np.abs(x-w[2])/w[4])" \
              "                  + np.exp(-np.abs(x+2*w[2])/w[4]) + np.exp(-np.abs(x-2*w[2])/w[4]) " \
              "                  + np.exp(-np.abs(x+3*w[2])/w[4]) + np.exp(-np.abs(x-3*w[2])/w[4]) )" \

def make_theta_g2_monoexp(p):
    var_names = ['r', 'y0', 'rr', 'a1', 't1']
    return {                      #  r,           y0,           rr,           a1,           t1,
        'init': np.array([p.iloc[0][0], p.iloc[0][1], p.iloc[0][2], p.iloc[0][3], p.iloc[0][4]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][1], p.iloc[1][2], p.iloc[1][3], p.iloc[1][4]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][1], p.iloc[2][2], p.iloc[2][3], p.iloc[2][4]]),
    }, var_names


# Biexponential (two lifetimes) g2 function with up to 3 side peaks, variable reprate
g2_biexp   = "w[1] + w[3] * (w[0]*np.exp(-np.abs(x)/w[4]) " \
             "                  + np.exp(-np.abs(x+w[2])/w[4]) + np.exp(-np.abs(x-w[2])/w[4])" \
             "                  + np.exp(-np.abs(x+2*w[2])/w[4]) + np.exp(-np.abs(x-2*w[2])/w[4]) " \
             "                  + np.exp(-np.abs(x+3*w[2])/w[4]) + np.exp(-np.abs(x-3*w[2])/w[4]) )" \
             "     + w[5] * (w[0]*np.exp(-np.abs(x)/w[6])" \
             "                  + np.exp(-np.abs(x+w[2])/w[6]) + np.exp(-np.abs(x-w[2])/w[6])" \
             "                  + np.exp(-np.abs(x+2*w[2])/w[6]) + np.exp(-np.abs(x-2*w[2])/w[6])" \
             "                  + np.exp(-np.abs(x+3*w[2])/w[6]) + np.exp(-np.abs(x-3*w[2])/w[6]) )"



def make_theta_g2_biexp(p):
    var_names = ['r', 'y0', 'rr', 'a1', 't1', 'a2', 't2']
    return {                      #  r,           y0,           rr,           a1,           t1,          a2,             t2,
        'init': np.array([p.iloc[0][0], p.iloc[0][1], p.iloc[0][2], p.iloc[0][3], p.iloc[0][4], p.iloc[0][5],  p.iloc[0][6]]),
          'lb': np.array([p.iloc[1][0], p.iloc[1][1], p.iloc[1][2], p.iloc[1][3], p.iloc[1][4], p.iloc[1][5],  p.iloc[1][6]]),
          'ub': np.array([p.iloc[2][0], p.iloc[2][1], p.iloc[2][2], p.iloc[2][3], p.iloc[2][4], p.iloc[2][5],  p.iloc[2][6]]),
    }, var_names


def g2_clean(theta, x, function): # function output (clean)
    w = np.array(theta)
    f = eval(function)
    return f

# from tkinter import filedialog
# from AP_figs_funcs import *
# ## Debugging / testing
# # fnames = filedialog.askopenfilenames(title="Select .ht3 file")
# # fnames = ('/Users/andrewproppe/Dropbox (MIT)/Bawendi Lab Data/Streams/Montana_HydraHarp/Winter2022/20220306 InP 5R/dot24_0.2uW_0.25MHz.ht3',) # pulsed
# fnames = ('/Users/andrewproppe/Dropbox (MIT)/Bawendi Lab Data/Streams/Montana_HydraHarp/Winter2022/20220314 CK155DDACl/dot20_1uW.ht3',) # pulsed
# # fnames = ('/Users/andrewproppe/Dropbox (MIT)/Bawendi Lab Data/PCFS/Montana_PCFS/Winter2022/20220307_5R/dot37/dot37_000.stream',) # cw
#
#
# save_csv = False
# colors = np.array(sns.color_palette("icefire", 4))
# for i, f in enumerate(fnames):
#     name = f.split('/')[-1].split('.ht3')[0]
#     data = photons(f)
#     data.get_photon_records(overwrite=True)
#     # data.get_intensity_trace(
#     #     bin_width=200e9,
#     #     intensity_correction=True
#     #     # bin_width=100000
#     # )
#
#     data.get_g2(channels=[0, 1], time_range=2e7, n_bins=1000)
#
# # fig = plt.figure(dpi=150, figsize=(4, 4))
# # plt.plot(data.intensity_counts['time'], data.intensity_counts['trace'][:, 0]+data.intensity_counts['trace'][:, 1])
# # dress_fig(tight=True, xlabel='Time (ps)', ylabel='Counts (norm.)')