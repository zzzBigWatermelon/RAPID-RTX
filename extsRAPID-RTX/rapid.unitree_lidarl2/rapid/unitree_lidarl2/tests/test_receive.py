'''
现在数据流：

receive()
    |
    |
bytes
    |
    |
decode_packet()
    |
    |
{
 type:"POINT",
 packet:LidarPointDataPacket
}
    |
    |
packet_to_pointcloud()
    |
    |
numpy:
[
 x,
 y,
 z,
 intensity,
 time,
 ring
]
'''
import time
import numpy as np

from unilidar.lidar import UnitreeL2
from unilidar.decoder import decode_packet
from unilidar.pointcloud import packet_to_pointcloud



lidar = UnitreeL2()


print("==============================")
print(" Unitree L2 Full Scan Capture")
print("==============================")


cloud_buffer=[]


start=time.time()


packet_count=0



while True:


    data,addr = lidar.receive()


    msg=decode_packet(data)


    if msg is None:
        continue



    if msg["type"]=="POINT":


        cloud = packet_to_pointcloud(
            msg["packet"]
        )


        if cloud.shape[0]>0:


            cloud_buffer.append(
                cloud
            )


        packet_count +=1



    # --------------------------------
    # capture duration
    # --------------------------------

    if time.time()-start > 5:

        break



print()
print("==============================")
print("Capture finished")
print("==============================")


print(
    "Packets:",
    packet_count
)



# merge

if len(cloud_buffer)>0:


    full_cloud=np.vstack(
        cloud_buffer
    )


else:

    full_cloud=np.empty(
        (0,6)
    )



print(
    "Total points:",
    full_cloud.shape
)



np.save(
    "unitree_l2_scan_5s.npy",
    full_cloud
)


print(
    "Saved:",
    "unitree_l2_scan_5s.npy"
)