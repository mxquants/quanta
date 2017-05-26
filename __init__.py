#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 23:05:30 2017
import quanta.data 
@author: rhdzmota
"""

# %% setup 

def installReq():
    import subprocess
    bashCommand = """\
    pip install -r quanta/requirements.txt
    """
    
    try:
        install_req = False
        if install_req:
            subprocess.check_output(['bash','-c', bashCommand])
    except: 
        print("Not to ruin your day... but there is something with the imports. Setup your environment correctly to use quanta.")

# %% define main imports

try:
    import quanta.data 
except: 
    installReq()
    import quanta.data 

del installReq
# %% 
print("\nCongrats, you are using quanta beta-version by mxquants.\nContact developer @rhdzmota => rhdzmota@mxquants.com for any consern or further info.")

# %% main function 
def main():
    return True 

# %% 
if __name__ == "__main__":
    main()
    