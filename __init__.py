#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 23:05:30 2017
import quanta.data 
@author: rhdzmota
"""

# %% define imports

try:
    import quanta.data 
except:
    import subprocess
    bashCommand = """\
    bash quanta/setup.sh
    """
    output = subprocess.check_output(['bash','-c', bashCommand])
    import quanta.data 

# %% 
print("\nCongrats, you are using quanta mxquants package.\n")

# %% main function 
def main():
    return True 

# %% 
if __name__ == "__main__":
    main()
    