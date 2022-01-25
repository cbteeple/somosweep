# SoMoSweep
A utility for running parameter sweeps in SoMo or SoMoGym. Perform multidimensional parameter sweeps with true parallel processing and easy data handling.


## Installation
### Requirements
- [Python 3.6](https://www.python.org/downloads/release/python-360/)+
- Tested on:
	- Ubuntu 16.04 and Ubuntu 18.04 with Python 3.6.9
	- Ubuntu 20.04 with Python 3.6.9, 3.7.9 and 3.8.2
	- Windows 10 with Python 3.7 and Python 3.8 through [Anaconda](https://www.anaconda.com/products/individual#Downloads)
- Recommended: pip (`sudo apt-get install python3-pip`) 
- Recommended (for Ubuntu): [venv](https://docs.python.org/3/library/venv.html) (`sudo apt-get install python3-venv`)

### Setup
0. Make sure your system meets the requirements
1. Clone this repository
2. Set up a dedicated virtual environment using `venv`
3. Activate virtual environment
4. Install this module
	- Install the most recent release version using pip:
		- `pip install somosweep`
    - OR Install the development version:
        - Clone this repo to your machine and navitage to the root folder
        - Install requirements `pip install -r requirements.txt`
        - Install the module `pip install -e .`


### Explore the examples
- Run any of the files in the examples folder.

### Contributing
- only through a new branch and reviewed PR (no pushes to master!)
- always use [Black](https://pypi.org/project/black/) for code formatting
- always bump the version of your branch by increasing the version number listed in somo/_version.py

### Testing
SoMoSweep uses pytest for testing. You can run all tests with `pytest` from the repository's root.


## Using this framework
Check out the [documentation](https://github.com/cbteeple/somosweep/tree/main/docs). _Read-the-docs site coming soon._


## License

MIT open source

Copyright (c) 2022 Clark B. Teeple