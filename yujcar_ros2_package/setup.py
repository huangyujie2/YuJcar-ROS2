from setuptools import find_packages, setup

package_name = 'YuJcar_Ros2_Package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    py_modules=[
        'YuJcar_Ros2_Package.yujcar_node',
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='2639029140@qq.com',
    description='Description of your package',
    license='License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yujcar_node = YuJcar_Ros2_Package.yujcar_node:main',
        ],
    },
)
