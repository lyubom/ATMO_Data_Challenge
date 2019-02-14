## README

You will find in this repositery three files : 

- predict.py contains the predict function that you need to modify. It contains also a local version of the main loop over a year that will execute your code on the platform. You can run it locally for testing. On the platform only your predict function will be called.
- utils.py : utilities for loading data.
- scoring.py : a local version of the scoring program that is used to evaluate your code.

You will have to upload in a zip the predict.py file and the metadata file, and if needed, a saved trained model (in that case, add the relative path to this file in the model parameter of your predict function).
The submission must be a zip file with all files at the root of the zip file (no folder).

### How to run the example
If you want to use the notebook example, you need to work in a python 3.6+ environment and install the packages described in requirements.txt.
You can install then for example with the command `python3 -m pip install -r requirements.txt`