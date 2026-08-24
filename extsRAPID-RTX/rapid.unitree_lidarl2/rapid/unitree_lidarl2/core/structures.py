# ============================================================
# Unitree L2 Python SDK
#
# structures.py
#
# Python equivalent of:
# unitree_lidar_protocol.h
#
# SDK version:
# 2.0.9
#
# ============================================================


import struct
import numpy as np



# ============================================================
# Constants
# ============================================================

FRAME_HEADER = bytes(
    [
        0x55,
        0xAA,
        0x05,
        0x0A
    ]
)



# ============================================================
# FrameHeader
#
# C++:
#
# typedef struct
# {
#     uint8_t header[4];
#     uint32_t packet_type;
#     uint32_t packet_size;
# }FrameHeader;
#
# size = 12
#
# ============================================================


class FrameHeader:


    SIZE = 12


    def __init__(
        self,
        header,
        packet_type,
        packet_size
    ):

        self.header = header

        self.packet_type = packet_type

        self.packet_size = packet_size



    @classmethod
    def parse(
        cls,
        data,
        offset=0
    ):

        header,ptype,psize = struct.unpack_from(
            "<4sII",
            data,
            offset
        )


        return cls(
            header,
            ptype,
            psize
        )





# ============================================================
# TimeStamp
#
# uint32 sec
# uint32 nsec
#
# size=8
#
# ============================================================


class TimeStamp:


    SIZE=8


    def __init__(
        self,
        sec,
        nsec
    ):

        self.sec=sec

        self.nsec=nsec



    @classmethod
    def parse(
        cls,
        data,
        offset
    ):


        sec,nsec=struct.unpack_from(
            "<II",
            data,
            offset
        )


        return cls(
            sec,
            nsec
        )





# ============================================================
# DataInfo
#
# C++
#
# uint32 seq
# uint32 payload_size
# TimeStamp stamp
#
# size=16
#
# ============================================================


class DataInfo:


    SIZE=16


    def __init__(
        self,
        seq,
        payload_size,
        stamp
    ):

        self.seq=seq

        self.payload_size=payload_size

        self.stamp=stamp



    @classmethod
    def parse(
        cls,
        data,
        offset
    ):


        seq,payload_size=struct.unpack_from(
            "<II",
            data,
            offset
        )


        stamp=TimeStamp.parse(
            data,
            offset+8
        )


        return cls(
            seq,
            payload_size,
            stamp
        )





# ============================================================
# LidarInsideState
#
# size=36
#
# ============================================================


class LidarInsideState:


    SIZE=36



    def __init__(self):

        self.sys_rotation_period=0

        self.com_rotation_period=0

        self.dirty_index=0

        self.packet_lost_up=0

        self.packet_lost_down=0

        self.apd_temperature=0

        self.apd_voltage=0

        self.laser_voltage=0

        self.imu_temperature=0



    @classmethod
    def parse(
        cls,
        data,
        offset
    ):


        obj=cls()


        values=struct.unpack_from(
            "<IIfffffff",
            data,
            offset
        )


        (
            obj.sys_rotation_period,
            obj.com_rotation_period,
            obj.dirty_index,
            obj.packet_lost_up,
            obj.packet_lost_down,
            obj.apd_temperature,
            obj.apd_voltage,
            obj.laser_voltage,
            obj.imu_temperature

        )=values



        return obj





# ============================================================
# LidarCalibParam
#
# C++ order:
#
# float a_axis_dist;
# float b_axis_dist;
# float theta_angle_bias;
# float alpha_angle_bias;
# float beta_angle;
# float xi_angle;
# float range_bias;
# float range_scale;
#
# size=32
#
# ============================================================


class LidarCalibParam:


    SIZE=32



    def __init__(self):

        self.a_axis_dist=0

        self.b_axis_dist=0

        self.theta_angle_bias=0

        self.alpha_angle_bias=0

        self.beta_angle=0

        self.xi_angle=0

        self.range_bias=0

        self.range_scale=0



    @classmethod
    def parse(
        cls,
        data,
        offset
    ):


        obj=cls()


        values=struct.unpack_from(
            "<ffffffff",
            data,
            offset
        )


        (
            obj.a_axis_dist,
            obj.b_axis_dist,
            obj.theta_angle_bias,
            obj.alpha_angle_bias,
            obj.beta_angle,
            obj.xi_angle,
            obj.range_bias,
            obj.range_scale

        )=values



        return obj





# ============================================================
# LidarPointData
#
# size:
#
# DataInfo          16
# InsideState       36
# Calib             32
# LineInfo          32
# point_num          4
# ranges           600
# intensity        300
#
# total = 1020
#
# ============================================================


class LidarPointData:


    SIZE=1020



    def __init__(self):

        self.info=None

        self.state=None

        self.param=None


        self.com_horizontal_angle_start=0

        self.com_horizontal_angle_step=0

        self.scan_period=0

        self.range_min=0

        self.range_max=0

        self.angle_min=0

        self.angle_increment=0

        self.time_increment=0


        self.point_num=0


        self.ranges=None

        self.intensities=None




    @classmethod
    def parse(
        cls,
        data,
        offset=12
    ):


        obj=cls()



        # -------------------------
        # DataInfo
        # -------------------------

        obj.info=DataInfo.parse(
            data,
            offset
        )


        offset += DataInfo.SIZE



        # -------------------------
        # Inside state
        # -------------------------

        obj.state=LidarInsideState.parse(
            data,
            offset
        )


        offset += LidarInsideState.SIZE



        # -------------------------
        # Calibration
        # -------------------------

        obj.param=LidarCalibParam.parse(
            data,
            offset
        )


        offset += LidarCalibParam.SIZE



        # -------------------------
        # Line info
        # -------------------------

        (
            obj.com_horizontal_angle_start,
            obj.com_horizontal_angle_step,
            obj.scan_period,
            obj.range_min,
            obj.range_max,
            obj.angle_min,
            obj.angle_increment,
            obj.time_increment

        )=struct.unpack_from(
            "<ffffffff",
            data,
            offset
        )


        offset += 32



        # -------------------------
        # Point number
        # -------------------------

        obj.point_num=struct.unpack_from(
            "<I",
            data,
            offset
        )[0]


        offset +=4



        # -------------------------
        # ranges
        # -------------------------

        obj.ranges=np.frombuffer(
            data[
                offset:
                offset+600
            ],
            dtype="<u2"
        ).copy()


        offset +=600



        # -------------------------
        # intensities
        # -------------------------

        obj.intensities=np.frombuffer(
            data[
                offset:
                offset+300
            ],
            dtype=np.uint8
        ).copy()



        return obj





# ============================================================
# FrameTail
#
# C++:
#
# uint32 crc32
# uint32 msg_type_check
# uint8 reserve[2]
# uint8 tail[2]
#
# size=12
#
# ============================================================


class FrameTail:


    SIZE = 12


    def __init__(self):

        self.crc32 = 0

        self.msg_type_check = 0

        self.reserve = bytes(2)

        self.tail = bytes(2)



    @classmethod
    def parse(
        cls,
        data,
        offset
    ):


        obj = cls()


        (
            obj.crc32,
            obj.msg_type_check

        ) = struct.unpack_from(
            "<II",
            data,
            offset
        )


        obj.reserve = data[
            offset+8:
            offset+10
        ]


        obj.tail = data[
            offset+10:
            offset+12
        ]


        return obj





# ============================================================
# Complete Point Packet
#
# FrameHeader
# LidarPointData
# FrameTail
#
# ============================================================


class LidarPointDataPacket:



    SIZE = 1044



    def __init__(self):


        self.header=None


        self.data=None


        self.tail=None


        self.raw=None



    @classmethod
    def parse(
        cls,
        packet
    ):


        obj=cls()


        obj.raw=packet



        # ------------------------
        # Header
        # ------------------------

        obj.header=FrameHeader.parse(
            packet,
            0
        )



        # ------------------------
        # Point data
        # ------------------------

        obj.data=LidarPointData.parse(
            packet,
            12
        )



        # ------------------------
        # Tail
        #
        # official:
        # last 12 bytes
        #
        # ------------------------

        if len(packet)>=12:


            obj.tail=FrameTail.parse(
                packet,
                len(packet)-12
            )



        return obj