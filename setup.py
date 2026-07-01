# from setuptools import setup, find_packages

# setup(
#     name='yolo-obj-detector',
#     version='0.1',
#     packages=find_packages(),
#     install_requires=[
#         # List your project dependencies here
#         'numpy',
#         'requests',
#         # Add other dependencies as needed
#     ],
#     entry_points={
#         'console_scripts': [
#             'your_script_name = your_package.module:main_function',
#             # Add other console scripts if needed
#         ],
#     },
#     author='Your Name',
#     author_email='your.email@example.com',
#     description='A short description of your project',
#     url='https://github.com/your_username/your_project',
#     classifiers=[
#         'Programming Language :: Python :: 3',
#         'License :: OSI Approved :: MIT License',
#         'Operating System :: OS Independent',
#     ],
# )


from setuptools import setup, find_packages
from pathlib import Path


HERE = Path(__file__).parent
REQUIRED = [i.strip() for i in open(HERE / 'requirements.txt') if not i.startswith('#')]


setup(
    name='yolo-obj-detector',
    version='0.1.1',
    description='A package to train a custom YOLOv8 model',
    author='Jacob Rivera',
    author_email='jakemichaelrivera32@gmail.com',
    url='https://github.com/jacobmrivera/YOLO-Object-Detection-Project',
    packages=find_packages(include=['obj_detector', 'obj_detector.*']),
    install_requires = REQUIRED,
)