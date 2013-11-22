import Image
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sys

print 'Argument List:', sys.argv[0]

threshold = np.float32(sys.argv[1])

def main():
    I = Image.open('MengelMetaballStar4.png')
    arr = np.asarray(I)
    print arr.shape
    print arr
    print threshold
    a = (arr>threshold)*1#+(arr<threshold+40)*1

    I = Image.frombuffer('L',arr.shape, np.array(a).astype(np.uint8),'raw','L',0,1)

    plt.imshow(I, cmap=matplotlib.cm.gray)
    plt.show()

main()
