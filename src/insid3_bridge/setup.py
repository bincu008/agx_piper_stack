from setuptools import setup
package_name = 'insid3_bridge'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Muse',
    maintainer_email='muse@agent.local',
    description='ROS2 bridge node exposing INSID3 grounding as /insid3/* services.',
    license='MIT',
    entry_points={'console_scripts': [
        'insid3_bridge_node = insid3_bridge.insid3_bridge_node:main',
    ]},
)
