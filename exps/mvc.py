#############
# Mean Value Coordinates
#
# Input :
#      1. polyCoords :  Coordinates of closed polygon in the Counter
#                       clockwise direction. The input is not tested inside.
#
#       2. queryCoord:   the xyCoords of the query Point
# Output:
#       1:  baryCoords: baryCentric Coords of the query Point.
#
# Reference: Mean Value Coordinates for Arbitrary Planar Polygons:
#            Kai Hormann and Michael Floater;
# Written by:
#            Chaman Singh Verma
#            University of Wisconsin at Madison.
#            18th March, 2011.
# Ported to python by
#        Rodrigo Baravalle, Universidad Nacional de Rosario - 
#        CIFASIS, 21/05/2014
#############

import numpy as np

def mvc(query,cage):
    nSize = len(cage)

    dx = 0.0
    dy = 0.0

    s = np.zeros((nSize,2)).astype(np.float32)
    for i in range(nSize):
        dx  =   cage[i][0] - query[0]
        dy  =   cage[i][1] - query[1]
        s[i][0]  =   dx
        s[i][1]  =   dy

    baryCoords = np.zeros((nSize)).astype(np.float32)

    ip = im = np.int32(0)
    ri = rp = Ai = Di = dl = mu = np.float32(0.0)
    eps = 10.0*np.nextafter(0,1)



    # First check if any coordinates close to the cage point or
    # lie on the cage boundary. These are special cases.
    for i in range(nSize):
        ip = (i+1)%nSize
        ri = np.sqrt( s[i][0]*s[i][0] + s[i][1]*s[i][1] )
        Ai = 0.5*(s[i][0]*s[ip][1] - s[ip][0]*s[i][1])
        Di = s[ip][0]*s[i][0] + s[ip][1]*s[i][1]
        if( ri <= eps):
            baryCoords[i] = 1.0
            return baryCoords
        else:
            if( np.abs(Ai) <= 0.0 and Di < 0.0):
                dx = cage[ip][0] - cage[i][0]
                dy = cage[ip][1] - cage[i][1]
                dl = np.sqrt(dx*dx + dy*dy)
                #assert(dl > eps);
                dx = query[0] - cage[i][0]
                dy = query[1] - cage[i][1]
                mu = np.sqrt(dx*dx + dy*dy)/dl
                #assert( mu >= 0.0 && mu <= 1.0);
                baryCoords[i]  = 1.0-mu
                baryCoords[ip] = mu
                return baryCoords

    # Page #12, from the paper
    tanalpha = np.zeros((nSize)).astype(np.float32) # tan(alpha/2)
    for i in range(0,nSize):
        ip = (i+1)%nSize
        im = (nSize-1+i)%nSize
        ri = np.sqrt( s[i][0]*s[i][0] + s[i][1]*s[i][1] )
        rp = np.sqrt( s[ip][0]*s[ip][0] + s[ip][1]*s[ip][1] )
        Ai = 0.5*(s[i][0]*s[ip][1] - s[ip][0]*s[i][1])
        Di = s[ip][0]*s[i][0] + s[ip][1]*s[i][1]
        tanalpha[i] = (ri*rp - Di)/(2.0*Ai)

    # Equation #11, from the paper
    wi = wsum = np.float32(0.0)
    for i in range(0,nSize):
        im = (nSize-1+i)%nSize
        ri = np.sqrt( s[i][0]*s[i][0] + s[i][1]*s[i][1] )
        wi = 2.0*( tanalpha[i] + tanalpha[im] )/ri
        wsum += wi
        baryCoords[i] = wi

    if( np.abs(wsum ) > 0.0):
        for i in range(0,nSize):
            baryCoords[i] /= wsum

    return baryCoords


def test():
    # COUNTER CLOCK WISE
    cageOrig = np.array([[0.0,0.0],[50.0,0.0],[100.0,100.0],[0.0,100.0]]).astype(np.float32)
    cage = np.array([[0.0,0.0],[50.0,0.0],[100.0,100.0],[0.0,100.0]]).astype(np.float32)
    query = np.array([50,10]).astype(np.float32)
    print mvc(query,cageOrig)


