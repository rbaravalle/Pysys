import os
from subprocess import call
import time
import sys


output = 'Ogre/output/media/fields/warped.field'
outputO = 'Ogre/output/media/fields/warpedO.field'

try:
    filename = sys.argv[1]    
    print "Using ", filename
except:
    filename = 'textures/imagenSystem2.png'
    print "No filename specified, using default filename: ", filename

command1 = "python loadTex.py "+filename+" > "+output
command2 = "python loadTexO.py "+filename+" > "+outputO

print command1
print command2

t = time.clock()
os.system(command1)
os.system(command2)
print "Time: ", time.clock()-t
