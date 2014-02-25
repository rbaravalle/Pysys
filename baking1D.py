########################
#MAIN PROGRAMME
#######################
from bakingFunctions import *
import numpy as np

N=32;
theta=0;
dx=0.01/np.float32(N); 
dt=30;
Time=5400;
M=Time/np.float32(dt);
# N is number of spacial nodes
# M is number of temporal nodes
#**********************************
#% inputting the initial values
#**********************************

T = np.zeros((N+2)).astype(np.float32)
V = np.zeros((N+2)).astype(np.float32)
W = np.zeros((N+2)).astype(np.float32)

T1 = np.zeros((M+2,N+2)).astype(np.float32)
V1 = np.zeros((M+2,N+2)).astype(np.float32)
W1 = np.zeros((M+2,N+2)).astype(np.float32)


# initial conditions
for i in range(0,N+1):
    T[i]=25
    V[i]=0
    W[i]=0.4061
    T1[0,i]=T[i]
    V1[0,i]=V[i]
    W1[0,i]=W[i]

for t in range(0,np.int(M)+1):
    T_new=Tnew(T,V,W,N,dt,dx,theta)
    V_temp,W_temp,V_s,P=correction(T_new,V,W,N) #,P) ;
    V_new=Vnew(T_new,V_temp,W_temp,dx,dt,N,theta)
    V_new,W_temp=Correction2(T_new,V_new,W_temp,V_s,N,P,W)
    W_new=Wnew(T_new,V_new,W_temp,dx,dt,N,theta)
    T[0:N+1]=T_new[0:N+1]
    V[0:N+1]=V_new[0:N+1]
    W[0:N+1]=W_new[0:N+1]
    for i in range(0,N+1):
        T1[t+1,i]=T[i]
        V1[t+1,i]=V[i]
        W1[t+1,i]=W[i]

    import os
    clear = lambda: os.system('clear')
    clear()
    print T
    print V
    print W

Times = np.zeros((M+1,N+1)).astype(np.float32)
T = np.zeros((M+1,N+1)).astype(np.float32)
V = np.zeros((M+1,N+1)).astype(np.float32)
W = np.zeros((M+1,N+1)).astype(np.float32)

for t in range(0,np.int(M)+1):
    for i in range(0,N+1):
        l=(t)*dt/np.float32(60)
        x=(i)*dx
        Times[t,i]=l
        T[t,i]=T1[t,i]
        V[t,i]=V1[t,i]
        W[t,i]=W1[t,i]

from matplotlib import pyplot as plt
print Times.shape
print T.shape
print Times
plt.plot(Times,T)
plt.show()

plt.plot(Times,V)
plt.show()

plt.plot(Times,W)
plt.show()
