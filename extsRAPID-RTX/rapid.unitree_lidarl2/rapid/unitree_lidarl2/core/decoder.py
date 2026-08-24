# ============================================================
# Unitree L2 Python SDK
#
# decoder.py
#
# UDP packet decoder
#
# Convert:
#
# UDP bytes
#      |
#      v
# LidarPointDataPacket
#
# SDK version:
# 2.0.10
#
# ============================================================


import struct


from .structures import (
    FRAME_HEADER,
    LidarPointDataPacket
)



# ============================================================
# Packet Types
#
# unitree_lidar_protocol.h
# ============================================================


LIDAR_USER_CMD_PACKET_TYPE = 100

LIDAR_ACK_DATA_PACKET_TYPE = 101

LIDAR_POINT_DATA_PACKET_TYPE = 102

LIDAR_2D_POINT_DATA_PACKET_TYPE = 103

LIDAR_IMU_DATA_PACKET_TYPE = 104





# ============================================================
# Frame Header
#
# uint8 header[4]
# uint32 packet_type
# uint32 packet_size
#
# total 12 bytes
#
# ============================================================


def parse_header(data):


    if len(data) < 12:

        return None



    header,ptype,psize = struct.unpack_from(
        "<4sII",
        data,
        0
    )


    return {


        "header":header,


        "type":ptype,


        "size":psize

    }





# ============================================================
# Decode UDP Packet
#
# Input:
#
# data : bytes
#
#
# Output:
#
# {
#   "type":"POINT",
#
#   "packet": LidarPointDataPacket,
#
#   "raw": bytes
# }
#
# ============================================================


def decode_packet(data):


    if len(data)<12:

        return None



    header=parse_header(data)


    if header is None:

        return None



    # -------------------------
    # Check magic header
    # -------------------------

    if header["header"] != FRAME_HEADER:


        return None




    packet_type=header["type"]



    # ========================================================
    # Point Cloud Packet
    # ========================================================


    if packet_type == LIDAR_POINT_DATA_PACKET_TYPE:


        packet=LidarPointDataPacket.parse(
            data
        )



        return {


            "type":"POINT",


            "packet":packet,


            "raw":data,


            "header":header

        }




    # ========================================================
    # IMU Packet
    # ========================================================


    elif packet_type == LIDAR_IMU_DATA_PACKET_TYPE:


        return {


            "type":"IMU",


            "raw":data,


            "header":header

        }





    # ========================================================
    # Other Packet
    # ========================================================


    else:


        return {


            "type":"OTHER",


            "raw":data,


            "header":header

        }





# ============================================================
# Debug Point Packet
#
# Print structure information
#
# ============================================================


def debug_point_packet(data):


    print("\n==============================")
    print(" Unitree L2 Point Packet Debug")
    print("==============================")



    print(
        "Packet size:",
        len(data)
    )



    header=parse_header(data)



    print("\n[FrameHeader]")


    print(
        "header:",
        header["header"].hex()
    )


    print(
        "packet_type:",
        header["type"]
    )


    print(
        "packet_size:",
        header["size"]
    )





    packet=LidarPointDataPacket.parse(
        data
    )



    lidar=packet.data



    print("\n[DataInfo]")


    print(
        "seq:",
        lidar.info.seq
    )


    print(
        "stamp:",
        lidar.info.stamp.sec,
        lidar.info.stamp.nsec
    )




    print("\n[LidarCalibParam]")


    print(
        "a_axis_dist:",
        lidar.param.a_axis_dist
    )


    print(
        "b_axis_dist:",
        lidar.param.b_axis_dist
    )


    print(
        "theta_angle_bias:",
        lidar.param.theta_angle_bias
    )


    print(
        "alpha_angle_bias:",
        lidar.param.alpha_angle_bias
    )


    print(
        "beta_angle:",
        lidar.param.beta_angle
    )


    print(
        "xi_angle:",
        lidar.param.xi_angle
    )


    print(
        "range_bias:",
        lidar.param.range_bias
    )


    print(
        "range_scale:",
        lidar.param.range_scale
    )





    print("\n[Scan Parameter]")


    print(
        "angle_min:",
        lidar.angle_min
    )


    print(
        "angle_increment:",
        lidar.angle_increment
    )


    print(
        "theta_start:",
        lidar.com_horizontal_angle_start
    )


    print(
        "theta_step:",
        lidar.com_horizontal_angle_step
    )


    print(
        "time_increment:",
        lidar.time_increment
    )





    print("\n[Raw measurement]")


    print(
        "point_num:",
        lidar.point_num
    )


    print(
        "ranges first 10:"
    )


    print(
        lidar.ranges[:10]
    )



    print(
        "intensities first 10:"
    )


    print(
        lidar.intensities[:10]
    )



    print("==============================\n")