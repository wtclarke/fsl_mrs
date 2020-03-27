import numpy as np

def apodize(FID,dwelltime,broadening,filter='exp'):
    """ Apodize FID
    
    Args:
        FID (ndarray): Time domain data
        dwelltime (float): dwelltime in seconds
        broadening (tuple,float): shift in Hz
        filter (str,optional):'exp','l2g'

    Returns:
        FID (ndarray): Apodised FID
    """
    taxis = np.linspace(0,dwelltime*(FID.size-1),FID.size)
    if filter=='exp':
        Tl = 1/broadening[0]
        window = np.exp(-taxis/Tl)
    elif filter=='l2g':
        Tl = 1/broadening[0]
        Tg = 1/broadening[1]
        window = np.exp(taxis/Tl)*np.exp(taxis**2/Tg**2)
    return window*FID