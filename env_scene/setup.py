from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'env_scene'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['tests', 'tests.*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vision',
    maintainer_email='vision@example.com',
    description='環境相機：A/B 區物件座標、C 區 8 格座標與佔用',
    license='Apache-2.0',
    entry_points={'console_scripts': ['env_scene_node = env_scene.env_scene_node:main',
                                      'env_scene_offline = env_scene.offline:main',
                                      'edge_check = env_scene.edge_check:main',
                                      'calibrate_points = env_scene.calibrate_points:main']},
)
