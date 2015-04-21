import numpy as np
import Image

file1 = 'images/image2.png'
file2 = 'images/image52.png'

def main():
    I = Image.open(file1)
    I2 = Image.open(file2)
    i1 = np.array(I)
    i2 = np.array(I2)
    diff = np.float64(i1)-i2
    I = Image.frombuffer('L',(diff.shape),  (np.asarray(diff)).astype(np.uint8) ,'raw','L',0,1)
    #I = np.array
    print diff.sum()
    I.save('file3.png')

    print np.asarray(diff<np.float64(10)).astype(np.uint8)#.sum()

    I2 = Image.frombuffer('L',(diff.shape),  255*np.asarray(diff<np.float64(5)).astype(np.uint8) ,'raw','L',0,1)
    I2.save('file3t.png')
main()
