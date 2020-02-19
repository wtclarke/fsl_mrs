import numpy as np
import json

# Get the built in MM basis set
def getMMBasis(bw,points,cf,lw=5,addShift = 0.0):
    positions,widths,names = loadMMDescriptions()
    taxis = np.arange(0,points*(1/bw),1/bw)
    basisFIDs = []
    for p,w in zip(positions,widths):
        curFID = genMMBasisFIDS(taxis,cf,p,w,lw,addShift=addShift)
        basisFIDs.append(curFID)
    basisFIDs = np.array(basisFIDs)
    return basisFIDs, names

# Load MM description file
def loadMMDescriptions(mmBasisFile = 'mmbasis.json'):
    
    with open(mmBasisFile,'r') as mmfile:
        jsonString = mmfile.read()
        residues = json.loads(jsonString)
    
    names = []
    positions = []
    widths = []
    for res in residues:
        names.append('MM_' + res)
        positions.append(residues[res]['shifts'])
        widths.append(residues[res]['widths'])

    return positions,widths,names

# Generate fids from the positions and widths
def genMMBasisFIDS(taxis,cf,pos,width,lw,addShift = 0.0):
    combinedFID = np.zeros(taxis.shape,dtype=np.complex64)
    damping = 1/(lw*np.pi)    
    for cs,w in zip(pos,width):
        w = np.array(w)
        cs = np.array(cs)
        peakDamp = damping/w
        combinedFID += np.exp((-taxis/peakDamp))*np.exp((-1j*2*np.pi*(cs+addShift)*cf*taxis))
                        
    return combinedFID


## Functions for creation of the basis set
import requests
def createMMBasis(residues,sd=5,ambiguity='1'):
    positions = []
    widths = []
    for res in residues:
        p,w = fetchCSFromDatabase(res,FilterSD=sd,ambiguity=ambiguity)
        positions.append(p)
        widths.append(w)

    return positions,widths

def saveMMBasis(filename,names,positions,widths):
    outputDict = {}
    for n,p,w in zip(names,positions,widths):
        outputDict.update({n:{'shifts':p,'widths':w}})

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(outputDict, f, ensure_ascii=False, indent='\t')


def fetchCSFromDatabase(res,FilterSD=10,ambiguity='*'):
    baseURL = 'http://webapi.bmrb.wisc.edu/v2/search/chemical_shifts'
    db = 'macromolecules'
    atm = 'H'
    payload = {'comp_id': res, 'atom_type': atm, 'database': db}
    headers = {"Application":"FSL_MRS"}
    with requests.get(baseURL, params=payload, headers=headers) as r:
            if(r.ok):
                print(r.url)
                print('OK!')
            else:
                print(r.url)
                # If response code is not ok (200), print the resulting http error code with description
                r.raise_for_status()
                
    out = r.json()    
    currData = out['data']
    colNames = out['columns']

    namelist = set([f'{i[colNames.index("Atom_chem_shift.Comp_ID")]}'\
                f'-{i[colNames.index("Atom_chem_shift.Atom_ID")]}' for i in currData])
    dataDict = {key: [] for key in namelist}                
        
    for i in currData:
        currKey = f'{i[colNames.index("Atom_chem_shift.Comp_ID")]}-{i[colNames.index("Atom_chem_shift.Atom_ID")]}'
        if (ambiguity == '*')\
        or (i[colNames.index('Atom_chem_shift.Ambiguity_code')] == str(ambiguity)):         
            dataDict[currKey].append(i[colNames.index('Atom_chem_shift.Val')])

    peakPos = []
    peakWidths = []
    for key in dataDict:
        mean = np.mean(dataDict[key])
        sd = np.std(dataDict[key])
        lb = mean - (FilterSD * sd)
        ub = mean + (FilterSD * sd)
        filtered = [i for i in dataDict[key] if lb < i < ub]
        bins = np.arange(0,15,0.1)
        histObj,binEdges = np.histogram(filtered,bins=bins)
        idx = np.argmax(histObj)
        mode = binEdges[idx] + 0.05 
        peakPos.append(mode)
        # peakPos.append(np.mean(filtered))
        peakWidths.append(np.std(filtered))
                        
    return peakPos,peakWidths