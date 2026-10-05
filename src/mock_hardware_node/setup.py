from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'mock_hardware_node'
snapshot_files = [
    path for path in glob('snapshots/current/*') if os.path.isfile(path)
]


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/snapshots/current', snapshot_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='parallels',
    maintainer_email='parallels@localhost',
    description='Replay recorded RGB-D and Piper feedback topics.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'capture_snapshot = mock_hardware_node.capture_snapshot:main',
            'mock_hardware = mock_hardware_node.mock_hardware:main',
        ],
    },
)

