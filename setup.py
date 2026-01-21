from setuptools import setup, find_packages

setup(name='RIFE',
      version='0.1.0',
      author="Denis Moshensky",
      author_email="loven7doo@gmail.com",
      description="SDK version of RIFE",
      url="https://github.com/loven-doo/RIFE-lib",
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      packages=find_packages(exclude=find_packages("Research") +
                                     ["eagle.tests", "eaglib.tests", "eagledb.tests", "Research"]),
      package_data={'rife': ['model/*']},
      install_requires=[
          "numpy == 1.26.4",
          "tqdm >= 4.66.1",
          "torch >= 2.1",
          "opencv-python >= 4.8",
          "torchvision == 0.16",
      ])
