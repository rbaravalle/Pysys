## @brief Gives a Poisson sample of points of a rectangle.
##
# @param width
#        The width of the rectangle to sample
# @param height
#        The height of the rectangle to sample
# @param r
#        The mimum distance between points, in terms of 
#        rectangle units. For example, in a 10 by 10 grid, a mimum distance of 
#        10 will probably only give you one sample point.
# @param k
#        The algorithm generates k points around points already 
#        in the sample, and then check if they are not too close
#        to other points. Typically, k = 30 is sufficient. The larger 
#        k is, the slower th algorithm, but the more sample points
#        are produced.
# @return A list of tuples representing x, y coordinates of
#        of the sample points. The coordinates are not necesarily
#        integers, so that the can be more accurately scaled to be
#        used on larger rectangles.

import numpy as np
import Image
from math import log
from math import sin
from math import cos
from math import sqrt
from math import floor
from math import ceil
from math import pi

from random import random
from random import randint
from random import uniform

## The square of the distance between the given points
def sqr_dist((x0, y0), (x1, y1)):
	return (x1 - x0)*(x1 - x0) + (y1 - y0)*(y1 - y0)

def sample_poisson_uniform(width, height, r, k,fs):

    #fs: functions to check from previous layers
    def free(p):
        return all(map(lambda (f): not f(p), fs))

    #Convert rectangle (the one to be sampled) coordinates to 
    # coordinates in the grid.
    def grid_coordinates((x, y)):
        return int(x*inv_cell_size), int(y*inv_cell_size)
    
    # Puts a sample point in all the algorithm's relevant containers.
    def put_point(p):
        process_list.append(p)
        sample_points.append(p)  
        a,b = grid_coordinates(p)
        grid[a, b] = p

    # Generates a point randomly selected around
    # the given point, between r and 2*r units away.
    def generate_random_around((x, y), r):
        rr = uniform(r, w)
        rt = uniform(0, 2*pi)
        
        return rr*sin(rt) + x, rr*cos(rt) + y
        
    # Is the given point in the rectangle to be sampled?
    def in_rectangle((x, y)):
        return 0 <= x < width and 0 <= y < height
        
    def in_neighbourhood(p,aux):
        gx, gy = grid_coordinates(p)
        
        if grid[gx,gy][0]>=0: return True
        
        for x in range(gx-1,gx+2):
            for y in range(gy-1,gy+2):
                try:
                    if grid[x,y][0]>=0  and sqr_dist(grid[x,y], p) <= r_sqr*aux:
                        return True
                except: pass
        return False

    #Create the grid
    cell_size = r/sqrt(2)
    inv_cell_size = 1.0 / cell_size    
    r_sqr = r*r
        
    grid = np.zeros((int(ceil(width/cell_size)),int(ceil(height/cell_size)),2)).astype(np.int32)-1

    process_list = []
    sample_points = []    
    
    #generate the first point
    put_point((randint(0,width), randint(0,height)))
    
    #generate other points from points in queue.
    #while not process_list == []:
    maxx = 0.5*w*h/((r**3.2))
    print maxx
    for i in range(int(maxx)):
        if(process_list != []):
            p = process_list[0]
            process_list = process_list[1:]
            
            for i in range(k):
                q = generate_random_around(p, r)
                if in_rectangle(q) and not in_neighbourhood(q,1.0):# and free(q):
                        put_point(q)
        else: break;
    return sample_points,grid,grid_coordinates,in_neighbourhood


w = 512
h = 512

def shape(arr,p,r):
    x = p[0]
    y = p[1]
    for i in range(int(x-r),int(x+r+1)):
        for j in range(int(y-r),int(y+r+1)):
            if((x-i)*(x-i)+(y-j)*(y-j) < r*r):
                if(i < w and i >= 0 and j < w and j >= 0 ):
                    arr[i,j] = 255
    return arr

# closures
fs = []
grids = []
fgrids = []
sizes = [55,45,35,25,18,12,10,8]
arr = np.zeros((w,h)).astype(np.uint8)
for f in range(len(sizes)):
    points,grid,fgrid,in_neigh = sample_poisson_uniform(w, h, sizes[f], 30,fs)
    #grids.append(grid)
    #fgrids.append(fgrid)
    fs.append(in_neigh)
    for p in points:
        if(f>0):
            if(all(map (lambda f: not f(p,0.5),fs[:len(fs)-1]))):
                arr = shape(arr,p,sizes[f]/2-2)#arr[p[0],p[1]] = 255-50*f
        else: arr = shape(arr,p,sizes[f]/2-2)#arr[p[0],p[1]] = 255-50*f
    
#points2,grid2,in_neigh2 = sample_poisson_uniform(w, h, 20, 30,[in_neigh1])
#points3,grid3,in_neigh3 = sample_poisson_uniform(w, h, 10, 30,[in_neigh1,in_neigh2])


I = Image.frombuffer('L',(w,h),arr,'raw','L',0,1)
I.save('poisson.png',cmap='hot')
