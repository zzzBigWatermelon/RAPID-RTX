import socket


class UnitreeL2:

    def __init__(
        self,
        lidar_ip="192.168.1.62",
        lidar_port=6101,
        local_ip="192.168.1.2",
        local_port=6201
    ):

        self.lidar_ip = lidar_ip
        self.lidar_port = lidar_port

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        self.sock.bind(
            (
                local_ip,
                local_port
            )
        )

        print(
            "UDP bind:",
            local_ip,
            local_port
        )


    def receive(self):

        data,addr = self.sock.recvfrom(
            4096
        )

        return data,addr