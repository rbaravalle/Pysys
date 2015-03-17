import os
from subprocess import call
import time
import sys

command1 = "cython cprint.pyx "
command2 = "gcc -c -fPIC -I/usr/include/python2.7/ cprint.c"
command3 = "gcc -shared cprint.o -o cprint.so"

print command1
os.system(command1)
print command2
os.system(command2)
print command3
os.system(command3)

try:
    filename = sys.argv[1]
except:
    filename = 'textures/imagenpp.png'

command4 = "python texs.py "+filename
# compute textures
os.system(command4)

