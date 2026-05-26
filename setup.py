import os
import glob

from setuptools import find_packages, setup

package_name = 'identification'

resource_files = []
resource_root = 'resource'
for path in glob.glob(os.path.join(resource_root, '**', '*'), recursive=True):
    if not os.path.isfile(path):
        continue
    rel_dir = os.path.dirname(os.path.relpath(path, resource_root))
    install_dir = os.path.join('share', package_name, resource_root, rel_dir)
    resource_files.append((install_dir, [path]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=(
        [
            (
                'share/ament_index/resource_index/packages',
                ['resource/' + package_name],
            ),
            (
                'share/' + package_name, ['package.xml']
            ),
            (
                os.path.join('share', package_name, 'launch'),
                glob.glob('launch/*.launch.py')
            ),
            (
                os.path.join('share', package_name, 'config'),
                glob.glob('config/*')
            ),
        ]
        + resource_files
    ),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xiaoran',
    maintainer_email='xiaoran.yang@tum.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
