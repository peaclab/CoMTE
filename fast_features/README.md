## What does it do?

Generate the 11 statistical features from a 2d numpy array which has metrics as
columns and timeseries as rows. Approximately 50 times faster than python.

## Usage

```
from fast_features import generate_features
# df is the standard timeseries dataframe
features = generate_features(df.values)
```

## Installation
Two binary packages are provided with this repository, and can be installed in the following way:

`For python 3.6.x: ${PIP} install ./dist/fast_features-0.1.0-cp36-cp36m-linux_x86_64.whl`

`For python 3.7.x: ${PIP} install ./dist/fast_features-0.1.0-cp37-cp37m-linux_x86_64.whl`

If there are any incompatibilities with libc version, then the package needs to
be compiled from source:

## Compilation
Compiling this python package requires Rust, which requires a working C
compiler.

First insall Rust nightly (necessary because of PyO3)
1. `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. `rustup toolchain install nightly`
3. `rustup override set nightly`
Then install necessary python packages and compile using setup.py
4. `pip3 install --user -r requirements.txt`
5. `python3 setup.py sdist bdist_wheel`
After this, install the fast_features package using the above pip install
command.
