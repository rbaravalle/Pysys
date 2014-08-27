import os
from subprocess import call
import time
import sys


output = 'Ogre/output/media/fields/warped.field'
outputO = 'Ogre/output/media/fields/warpedO.field'
outputC = 'Ogre/output/media/fields/warpedC.field'

try:
    filename = sys.argv[1]    
    print "Using ", filename
except:
    filename = 'textures/imagenSystem2.png'
    print "No filename specified, using default filename: ", filename


t = time.clock()

command1 = "python loadTex.py "+filename+" > "+output
print command1
os.system(command1)
command2 = "python loadTexO.py "+filename+" > "+outputO
print command2
os.system(command2)
command3 = "python loadTexC.py "+filename+" > "+outputC
print command3
os.system(command3)
