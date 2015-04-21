import os
from subprocess import call
import time
import sys

command1 = "python pipelineCython.py"
command2 = "python pipeline.py"

print command1
os.system(command1)
print command2
os.system(command2)

