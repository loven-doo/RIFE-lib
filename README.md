# SDK version of RIFE (Real-Time Intermediate Flow Estimation) for Video Frame Interpolation
## Install the package
```commandline
pip install git+https://github.com/lovem-doo/RIFE-lib
```
## Use as SDK
```python
# see rife/__main__.py as example
from rife import Interpolation

your_frames_array = ...  # numpy.ndarray

interpolation = Interpolation.load()  # see rife.interpolation.Interpolation() for options
interpolation(your_frames_array)
```
## Use as executable
```commandline
python3 -m rife -h
```
