import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import launch

################### user configure parameters for ros2 start ###################
xfer_format   = 0    # 0-Pointcloud2(PointXYZRTL), 1-customized pointcloud format
multi_topic   = 0    # 0-All LiDARs share the same topic, 1-One LiDAR one topic
data_src      = 0    # 0-lidar,1-hub
publish_freq  = 10.0 # freqency of publish,1.0,2.0,5.0,10.0,etc
output_type   = 0
frame_id      = 'livox_frame'
lvx_file_path = get_package_share_directory('radar_livox_driver')+'/livox/livox_test.lvx'
cmdline_bd_code = 'livox0000000001'


# cur_path = os.path.split(os.path.realpath(__file__))[0] + '/'
# cur_config_path = cur_path + '../config'
cur_config_path = get_package_share_directory('radar_livox_driver') + '/config'
rviz_config_path = os.path.join(cur_config_path, 'livox_lidar.rviz')
user_config_path = os.path.join(cur_config_path, 'livox_lidar_config.json')
################### user configure parameters for ros2 end #####################

livox_ros2_params = [
    {"xfer_format": xfer_format},
    {"multi_topic": multi_topic},
    {"data_src": data_src},
    {"publish_freq": publish_freq},
    {"output_data_type": output_type},
    {"frame_id": frame_id},
    {"lvx_file_path": lvx_file_path},
    {"user_config_path": user_config_path},
    {"cmdline_input_bd_code": cmdline_bd_code}
]


def generate_launch_description():
    livox_driver = Node(
        package='radar_livox_driver',
        executable='radar_livox_driver_node',
        name='radar_livox_publisher',
        output='screen',
        parameters=livox_ros2_params
        )

    livox_rviz = Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['--display-config', rviz_config_path]
        )

    return LaunchDescription([
        livox_driver,
        livox_rviz,
        # launch.actions.RegisterEventHandler(
        #     event_handler=launch.event_handlers.OnProcessExit(
        #         target_action=livox_rviz,
        #         on_exit=[
        #             launch.actions.EmitEvent(event=launch.events.Shutdown()),
        #         ]
        #     )
        # )
    ])