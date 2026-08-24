'''
Packet 1044 bytes


0
│
├── FrameHeader
│
│   offset 0
│   size 12
│
│   uint8[4] header
│   uint32 packet_type
│   uint32 packet_size
│
├────────────────────
│
├── DataInfo
│
│   offset 12
│   size 16
│
│   uint32 seq
│   uint32 payload_size
│   uint32 stamp_sec
│   uint32 stamp_nsec
│
├────────────────────
│
├── LidarPointPayload
│
│   offset 28
│
│
│   ├── LidarInsideState
│   │
│   │ offset:
│   │ 28 ~ ?
│   │
│   │
│   ├── LidarCalibParam
│   │
│   │
│   ├── Scan Parameter
│   │
│   │
│   ├── point_num
│   │
│   │ offset 128 (payload)
│   │ offset 156(packet)
│   │
│   │ uint32 point_num
│   │
│   │ value = 300
│   │
│   │
│   ├── ranges
│   │
│   │ 300 float
│   │
│   │
│   └── intensities
│       300 float
│
└── FrameTail'''
import struct


# ============================
# Packet Type
# ============================

LIDAR_USER_CMD_PACKET_TYPE = 100
LIDAR_ACK_DATA_PACKET_TYPE = 101
LIDAR_POINT_DATA_PACKET_TYPE = 102
LIDAR_2D_POINT_DATA_PACKET_TYPE = 103
LIDAR_IMU_DATA_PACKET_TYPE = 104
LIDAR_VERSION_PACKET_TYPE = 105


# ============================
# Frame
# ============================

FRAME_HEADER = b'\x55\xAA\x05\x0A'

FRAME_HEADER_SIZE = 12
FRAME_TAIL_SIZE = 12


HEADER_FORMAT = "<4sII"


def parse_header(data):

    header, packet_type, packet_size = struct.unpack(
        HEADER_FORMAT,
        data[:12]
    )

    return {
        "header": header,
        "type": packet_type,
        "size": packet_size
    }