from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the package share directory
    pkg_share = get_package_share_directory('identification')
    
    # Path to the URDF file
    urdf_file = os.path.join(pkg_share, 'resource', 'robot', 'urdf', 'serial_pm_v2_identify.urdf')
    
    # Read URDF content
    with open(urdf_file, 'r') as f:
        urdf_content = f.read()
    
    # RViz config file path
    rviz_config = os.path.join(pkg_share, 'config', 'rviz.rviz')
    
    # Create launch description
    ld = LaunchDescription()
    
    # Robot State Publisher Node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': urdf_content}],
        output='screen'
    )
    
    # Joint State Publisher GUI Node
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen'
    )
    
    # RViz Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
        output='screen'
    )
    
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher_gui)
    ld.add_action(rviz_node)
    
    return ld
