#!/usr/bin/env python

# splash.py - :)
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford 
# SHBASECOPYRIGHT



def splash(logo='mrs'):
    """
       Display splash screen
       These logos were generated using the tool from http://patorjk.com
       Font Name: Standard
    """
    logo_mrs = """
     _____ ____  _          __  __ ____  ____  
    |  ___/ ___|| |        |  \/  |  _ \/ ___| 
    | |_  \___ \| |   _____| |\/| | |_) \___ \ 
    |  _|  ___) | |__|_____| |  | |  _ < ___) |
    |_|   |____/|_____|    |_|  |_|_| \_\____/                                             

    """
    logo_mrsi = """
     _____ ____  _          __  __ ____  ____ ___ 
    |  ___/ ___|| |        |  \/  |  _ \/ ___|_ _|
    | |_  \___ \| |   _____| |\/| | |_) \___ \| | 
    |  _|  ___) | |__|_____| |  | |  _ < ___) | | 
    |_|   |____/|_____|    |_|  |_|_| \_\____/___|
                    
    """                           

    print('\n\n\n-----------------------------------------------------\n\n\n')
    if logo == 'mrsi':
        print('{}'.format(logo_mrsi))
    else:
        print('{}'.format(logo_mrs))
    print('\n\n\n-----------------------------------------------------\n\n\n')
