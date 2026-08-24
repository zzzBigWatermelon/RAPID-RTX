# ============================================================
# Unitree L2 Python SDK
#
# pointcloud.py
#
# Python implementation of:
#
# parseFromPacketToPointCloud()
#
# from unitree_lidar_utilities.h
#
# ============================================================


import numpy as np
import math



DEGREE_TO_RADIAN = math.pi / 180.0



# ============================================================
# Convert LidarPointData to XYZITR point cloud
#
# Input:
#
# packet.data
#
#
# Output:
#
# numpy array:
#
# [
#   x,
#   y,
#   z,
#   intensity,
#   time,
#   ring
# ]
#
# ============================================================


def parse_pointcloud(
        lidar_data,
        range_min=0,
        range_max=100
):


    # --------------------------------------------------------
    # Scan information
    # --------------------------------------------------------

    num_points = lidar_data.point_num


    time_step = lidar_data.time_increment


    scan_period = lidar_data.scan_period



    # --------------------------------------------------------
    # Calibration parameters
    # --------------------------------------------------------


    param = lidar_data.param



    sin_beta = math.sin(
        param.beta_angle
    )

    cos_beta = math.cos(
        param.beta_angle
    )


    sin_xi = math.sin(
        param.xi_angle
    )

    cos_xi = math.cos(
        param.xi_angle
    )



    cos_beta_sin_xi = (
        cos_beta *
        sin_xi
    )


    sin_beta_cos_xi = (
        sin_beta *
        cos_xi
    )


    sin_beta_sin_xi = (
        sin_beta *
        sin_xi
    )


    cos_beta_cos_xi = (
        cos_beta *
        cos_xi
    )



    # --------------------------------------------------------
    # Scan angle
    # --------------------------------------------------------


    alpha_cur = (
        lidar_data.angle_min
        +
        param.alpha_angle_bias
    )


    alpha_step = (
        lidar_data.angle_increment
    )



    theta_cur = (
        lidar_data.com_horizontal_angle_start
        +
        param.theta_angle_bias
    )


    theta_step = (
        lidar_data.com_horizontal_angle_step
    )



    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    points=[]



    time_relative = 0.0



    ranges = lidar_data.ranges

    intensities = lidar_data.intensities



    # --------------------------------------------------------
    # Main loop
    #
    # identical to C++:
    #
    # for(j=0;j<num_of_points;j++)
    #
    # --------------------------------------------------------


    for j in range(num_points):


        raw_range = ranges[j]



        # ----------------------------------------------------
        # invalid point
        #
        # C++:
        #
        # if(ranges[j]<1)
        # continue;
        #
        # ----------------------------------------------------

        if raw_range < 1:

            alpha_cur += alpha_step

            theta_cur += theta_step

            time_relative += time_step

            continue



        # ----------------------------------------------------
        # range conversion
        #
        # range_float =
        #
        # range_scale *
        # (range + range_bias)
        #
        # ----------------------------------------------------


        range_float = (

            param.range_scale *

            (
                float(raw_range)
                +
                param.range_bias
            )

        )



        # ----------------------------------------------------
        # lidar internal limit
        # ----------------------------------------------------


        if (
            range_float < lidar_data.range_min
            or
            range_float > lidar_data.range_max
        ):

            alpha_cur += alpha_step

            theta_cur += theta_step

            time_relative += time_step

            continue



        # ----------------------------------------------------
        # user limit
        # ----------------------------------------------------

        if (
            range_float < range_min
            or
            range_float > range_max
        ):

            alpha_cur += alpha_step

            theta_cur += theta_step

            time_relative += time_step

            continue




        # ----------------------------------------------------
        # Calculate trigonometry
        # ----------------------------------------------------


        sin_alpha = math.sin(
            alpha_cur
        )

        cos_alpha = math.cos(
            alpha_cur
        )


        sin_theta = math.sin(
            theta_cur
        )

        cos_theta = math.cos(
            theta_cur
        )



        # ----------------------------------------------------
        # Official Unitree formula
        #
        # A
        #
        # ----------------------------------------------------


        A = (

            (
                -cos_beta_sin_xi

                +

                sin_beta_cos_xi
                *
                sin_alpha
            )

            *
            range_float

            +

            param.b_axis_dist

        )



        # ----------------------------------------------------
        # B
        # ----------------------------------------------------


        B = (

            cos_alpha

            *

            cos_xi

            *

            range_float

        )



        # ----------------------------------------------------
        # C
        # ----------------------------------------------------


        C = (

            (
                sin_beta_sin_xi

                +

                cos_beta_cos_xi
                *
                sin_alpha
            )

            *

            range_float

            +

            param.a_axis_dist

        )



        # ----------------------------------------------------
        # Coordinate transform
        #
        # x
        # y
        # z
        #
        # ----------------------------------------------------


        x = (

            cos_theta * A

            -

            sin_theta * B

        )


        y = (

            sin_theta * A

            +

            cos_theta * B

        )


        z = C



        intensity = (

            float(
                intensities[j]
            )

        )



        ring = 1



        points.append(
            [
                x,
                y,
                z,
                intensity,
                time_relative,
                ring
            ]
        )



        # ----------------------------------------------------
        # update scan state
        # ----------------------------------------------------

        alpha_cur += alpha_step

        theta_cur += theta_step

        time_relative += time_step




    if len(points)==0:

        return np.empty(
            (0,6),
            dtype=np.float32
        )



    return np.asarray(
        points,
        dtype=np.float32
    )





# ============================================================
# Helper
#
# packet -> point cloud
#
# ============================================================


def packet_to_pointcloud(
        packet,
        range_min=0,
        range_max=100
):


    return parse_pointcloud(
        packet.data,
        range_min,
        range_max
    )