import os
from subprocess import call
import time
import sys

command1 = "python compileCython.py"
command2 = "python main.py"

print command1
os.system(command1)
print command2
os.system(command2)

