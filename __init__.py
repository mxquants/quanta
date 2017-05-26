#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 23:05:30 2017
import quanta.data 
@author: rhdzmota
"""

# %% setup 

import subprocess
bashCommand = """\
bash quanta/setup.sh
"""

try:
    output = subprocess.check_output(['bash','-c', bashCommand])
except: 
    print("Not to ruin your day... but there is something with the imports. Setup your environment correctly to use quanta.")
    output = None 

# %% define main imports

import quanta.data 

# %% 
print("\nCongrats, you are using quanta beta-version by mxquants.\nContact developer @rhdzmota => rhdzmota@mxquants.com for any consern or further info.")

# %% main function 
def main():
    return True 

# %% 
if __name__ == "__main__":
    main()
    