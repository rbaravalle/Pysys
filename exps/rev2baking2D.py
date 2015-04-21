########################
#MAIN PROGRAMME
#######################
from rev2bakingFunctions2D import *
import numpy as np
import Image
import matplotlib
matplotlib.use('GTKAgg') # Change this as desired.
import gobject
from pylab import *
import lsystem

# Nx is number of spacial nodes (in x)
# Ny is number of spacial nodes (in y)
# M is number of temporal nodes
#**********************************
#% inputting the initial values
#**********************************

T = np.zeros((Nx+1,Ny+1)).astype(np.float32)
V = np.zeros((Nx+1,Ny+1)).astype(np.float32)
W = np.zeros((Nx+1,Ny+1)).astype(np.float32)

#T1 = np.zeros((M+2,Nx+2)).astype(np.float32)
#V1 = np.zeros((M+2,Nx+2)).astype(np.float32)
#W1 = np.zeros((M+2,Nx+2)).astype(np.float32)


# initial conditions
#cx = 30
#cy = 25
for i in range(0,Nx+1):
    for j in range(0,Ny+1):
        T[i,j]=25#+0.2*np.random.random()
        V[i,j]=0.0#+0.2*np.random.random()
        W[i,j] = 0.4061+0.1*np.random.random()
        #if(np.abs(i-cx)+(j-cy)**2 < 4):
        #    W[i,j] = 0
        #else:
        #    W[i,j]=0.4061+0.1*np.random.random()
        #T1[0,i]=T[i]
        #V1[0,i]=V[i]
        #W1[0,i]=W[i]
lsys = 1-lsystem.lin()/255
W*=lsys

def updatefig():
    global T,V,W,m
    m+=1

    T_new=Tnew(T,V,W,Nx,Ny,dt,dx,dy,theta1,theta2)

    #b = T_new.reshape((Nx+1,Ny+1))


    V_temp,W_temp,V_s,P=correction(T_new,V,W,Nx,Ny)
    T_new = T_new.reshape((Nx+1,Ny+1))
    print "T:",T_new[Nx/2]

    V_new=Vnew(T_new,V_temp,W_temp,dx,dy,dt,Nx,Ny,theta1, theta2)
    V_new = V_new.reshape((Nx+1,Ny+1))
    W_temp = W_temp.reshape((Nx+1,Ny+1))

    V_new,W_temp=Correction2(T_new,V_new,W_temp,V_s,Nx,Ny,P,W)
    W_new=Wnew(W_temp,V_new,W_temp,dx,dy,dt,Nx,Ny,theta1,theta2)

    W_new = W_new.reshape((Nx+1,Ny+1))

    T=T_new
    V=V_new
    W=W_new


    print "T:",T[1]
    print "V:",V[Nx/2]
    print "W:",W[Nx/2]
    print "------------ "
    print "iteracion: ", m
    print "iteracion: ", m
    print "------------ "

    imT.set_array(W)
    #T.astype(np.int)*60)
    #fig.colorbar( imT )
    #imV.set_array(V)
    #imW.set_array(W)
    manager.canvas.draw()

    if(dt*m / 60 > 40):
        print "Coccion terminada!"
        return False
    return True


m = 0
fig = plt.figure(1)

imgT = subplot(111)
imgT.set_title("Temperature")

temp = np.zeros((Nx+1,Ny+1))
temp = map(lambda i:i+1.1*np.random.random(), temp)
temp = np.array(temp).astype(np.int)
print temp

imT = imgT.imshow( temp, cmap=cm.hot)

#imgV = subplot(132)
#imV = imgV.imshow( V, cmap=cm.hot)
#imgV.set_title("Vapour content")

#imgW = subplot(133)
#imgW.set_title("Water content")
#imW = imgW.imshow( W, cmap=cm.hot)

manager = get_current_fig_manager()

#m=1
fig.colorbar( imT ) # Show the colorbar along the side
#fig.colorbar( im ) # Show the colorbar along the side
#fig.colorbar( im ) # Show the colorbar along the side

# once idle, call updatefig until it returns false.
gobject.idle_add(updatefig)

show()

