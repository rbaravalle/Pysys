######################################
# Function to calculate New Temperature.
######################################
import numpy as np
from math import atan
from scipy import interpolate


Nx=32
Ny=32
theta1=0
theta2=0
dx=0.01/np.float32(Nx) 
dy=0.01/np.float32(Ny)
dt=30
Time=5400
M=Time/np.float32(dt)

sig=5.670*10**(-8)
T_r=210
k=0.07
cp=3500
Dw=1.35*10**(-10)
T_air=210
esp_p=0.9
esp_r=0.9
lam=2.261*10**(6)
W_air=0
hc=0.5

a1=(12/np.float32(5.6))
b1= (12/np.float32(5.6))
a2=1+a1*a1
b2=1+b1*b1
F_sp=(2./(np.pi*a1*b1))*(np.log(np.sqrt(a2*b2/(1+a1*a1+b1*b1)))+a1*np.sqrt(b2)*atan(a1/np.sqrt(b2)) +b1*np.sqrt(a2)*atan(b1/np.sqrt(a2))-a1*atan(a1)-b1*atan(b1))

def hr(T,x,y):
    return sig*((T_r+273.5)**(2)+(T[x,y]+273.5)**(2))*((T_r+273.5)+(T[x,y]+273.5))/(1/esp_p+1/esp_r-2+1/F_sp)

def hw(T,W,x,y):
    return 1.4*10**(-3)*T[x,y]+0.27*W[x,y]-4.0*10**(-4)*T[x,y]*W[x,y]-0.77*W[x,y]**(2)

# Border conditions

def T256(T,W,i,j): # equation (2.56) # i always -1, only for clarity below
    temp=lam*(170+284*W[0,j])*Dw*hw(T,W,0,j) # REVISAR W[0,j] ????
    right = hr(T,0,j)*(T_r-T[0,j])+hc*(T_air-T[0,j])-temp*(W[0,j]-W_air)
    return T[1,j] + (2*dy/k)*right


def T257(T,W,i,j): # equation (2.57) j always -1
    temp=lam*(170+284*W[i,0])*Dw*hw(T,W,i,0);
    right = hr(T,i,0)*(T_r-T[i,0])+hc*(T_air-T[i,0])-temp*(W[i,0]-W_air)
    return T[i,1] + (2*dx/k)*right

def T258(T,i,j):
    return T[Nx-1,j]

def T259(T,i,j): 
    return T[i,Ny-1]

def W260(T,W,i,j): # equation (2.60) # i always -1, only for clarity below
    right = hw(T,W,0,j)*(W[0,j]-W_air)
    return W[1,j] + 2*dy*right

def W261(T,W,i,j): # j = -1
    right = hw(T,W,i,0)*(W[i,0]-W_air)
    return W[i,1] + 2*dx*right

def W262(W,i,j):
    return W[Nx-1,j]

def W263(W,i,j): 
    return W[i,Ny-1]

def tf(T,V,W,Nx,Ny,dt,dx,dy,theta1,theta2,i,j,a,b):
    alpha5 = lam*Dw*dt/(cp*dx*dx)
    alpha6 = lam*Dw*dt/(cp*dy*dy)
    actual = i*(Ny+1)+j # actual index

    rx=k*dt/((170+284*W[i,j])*cp*dx*dx)
    ry=k*dt/((170+284*W[i,j])*cp*dy*dy)
    alpha1 = ry*(1-theta2)
    alpha2 = rx*(1-theta1)
    alpha3 = ry*theta2
    alpha4 = rx*theta1


    a[actual,actual] = 1+2*alpha1+2*alpha2
    if(j!=0):
        a[actual,actual-1] = -alpha2
    if(j!=Ny):
        a[actual,actual+1] = -alpha2
    if(i!=0):
        a[actual,actual-(Ny+1)] = -alpha1
    if(i!=Nx):
        a[actual,actual+(Ny+1)] = -alpha1

    # Ghost points handling
    temp = 0
    if(i!=0):
        temp  += alpha3*T[i-1,j]
    else:
        temp += alpha3*T256(T,W,-1,j)

    temp+= (1-2*alpha3-2*alpha4)*T[i,j]
    
    if(i!=Nx):
        temp  += alpha3*T[i+1,j]
    else:
        temp += alpha3*T258(T,Nx+1,j)

    
    if(j!=0):
        temp += alpha4*T[i,j-1]
    else:
        temp += alpha4*T257(T,W,i,-1)

    if(j!=Ny):
        temp += alpha4*T[i,j+1]
    else:
        temp += alpha4*T259(T,i,Ny+1)

    if(i!=0):
        temp+= alpha5*W[i-1,j]
    else:
        temp+= alpha5*W260(T,W,-1,j)

    temp+= -2*alpha5*W[i,j]

    if(i!=Nx):
        temp += alpha5*W[i+1,j]
    else:
        temp += alpha5*W262(W,Nx+1,j)

    if(j!=0):
        temp += alpha6*W[i,j-1]
    else:
        temp += alpha6*W261(T,W,i,-1)

    temp += -2*alpha6*W[i,j]

    if(j!=Ny):
        temp += alpha6*W[i,j+1]
    else:
        temp += alpha6*W263(W,i,Ny+1)

    b[actual] = temp

    return a,b


def Tnew(T,V,W,Nx,Ny,dt,dx,dy,theta1,theta2):
    #*********************************
    #interior nodes
    #********************************
    a=np.zeros(((Nx+1)*(Ny+1),(Nx+1)*(Ny+1)))
    b=np.zeros((Nx+1)*(Ny+1))
    alpha5 = lam*Dw*dt/(cp*dx*dx)
    alpha6 = lam*Dw*dt/(cp*dy*dy)
    for i in range(0,Nx+1):
        for j in range(0,Ny+1):
            a,b = tf(T,V,W,Nx,Ny,dt,dx,dy,theta1,theta2,i,j,a,b)

    #for i in range(a.shape[0]):
    #    v = np.count_nonzero(a[i])
    #    if(v!=5): print v, i
    #    if(b[i]==0): print "B:",i


    return np.linalg.solve(a,b)


################################################
#Function to correct vapour and water content.
################################################
def correction(T_new,V,W,Nx,Ny):
    R=8.314
    #********************************
    # data points for interploation
    #********************************
    x=range(0,100+1,2)
    y=[.611, .705, .813, .934, 1.072, 1.226, 1.401, 1.597, 1.817, 2.062, 2.337, 2.642, 2.983, 3.360, 3.779, 4.242, 4.755, 5.319, 5.941, 6.625, 7.377, 8.201, 9.102, 10.087, 11.164, 12.34, 13.61, 15., 16.5, 18.14, 19.92, 21.83, 23.9, 26.14, 28.55, 31.15, 33.94, 36.95, 40.18, 43.63, 47.33, 51.31, 55.56, 60.11, 64.93, 70.09, 75.58, 81.43, 87.66, 94.28, 101.31]
    x=np.hstack((x,range(105,180+1,5) ))
    y=np.hstack((y, [120.82, 143.27, 169.06, 198.53, 232.1, 270.1, 313., 361.2, 415.4, 475.8, 543.1, 617.8, 700.5, 791.7, 892.0, 1002.1]))
    x=np.hstack((x, [190, 200, 225, 250, 275, 300]))
    y=np.hstack((y, [1254.4, 1553.8, 2548, 3973, 5942, 8581]))
    #************************************************************
    # interpolation and calculation of saturated amount of vapor
    #************************************************************

    P = np.zeros((Nx+1,Ny+1))
    V_s = np.zeros((Nx+1,Ny+1))
    V_temp = np.zeros((Nx+1,Ny+1))
    W_temp = np.zeros((Nx+1,Ny+1))

    f=interpolate.interp1d(x,y) #interp1(x,y,T_new[i],'spline')*1000
    for i in range(0,Nx+1):
        for j in range(0,Ny+1):
            actual = i*(Ny+1)+j
            if(T_new[actual] < 0): print "TEMPERATURA NEGATIVA: ",i,j
            P[i,j] = f(T_new[actual])*1000
            V_s[i,j]=18.*10**(-3)*P[i,j]/(R*(T_new[actual]+273.5)*(170+281*W[i,j]))*0.7*3.8
    #****************************************
    # correction in vapour and water content
    #****************************************
    for i in range(0,Nx+1):
        for j in range(0,Ny+1):
            if (W[i,j]+V[i,j]<V_s[i,j]):
                V_temp[i,j]=W[i,j]+V[i,j]
                W_temp[i,j]=0
            else:
                V_temp[i,j]=V_s[i,j]
                W_temp[i,j]=W[i,j]+V[i,j]-V_s[i,j]

    return V_temp,W_temp,V_s,P

####################################
# Function to find new Vapour
####################################
def Dv(T_new,i,j):
    return 9.0*10**(-12)*(T_new[i,j]+273.5)**(2) 

def hv(T_new,i,j):
    return 3.2*10**(9)/((T_new[i,j]+273.5)**(3))

def V2671(T_new,V_temp,i,j):
    Vair = 0
    return V_temp[1,j]+2*dy*hv(T_new,0,j)*(V_temp[0,j]-Vair)

def V2672(T_new,V_temp,i,j):
    Vair = 0
    return V_temp[i,1]+2*dx*hv(T_new,i,0)*(V_temp[i,0]-Vair)

def V2673(V_temp,i,j):
    return V_temp[Nx-1,j]

def V2674(V_temp,i,j):
    return V_temp[i,Ny-1]


def vf(T_new,V_temp,W_temp,dx,dy,dt,Nx,Ny,theta1, theta2,i,j,a,b):
    actual = i*(Ny+1)+j
    Dvij = Dv(T_new,i,j)
    eta1 = 2*Dvij # author's implementation did it wrong?
    eta2 = 2*Dvij
    alpha1 = (1-theta1)*dt/(dx*dx)
    alpha2 = (1-theta2)*dt/(dy*dy)
    alpha3 = (theta1)*dt/(dx*dx)
    alpha4 = (theta2)*dt/(dy*dy)
    r=dt*Dvij/(dx*dx)

    a[actual,actual] = 1+alpha1*eta1+alpha2*eta2
    if(j!=0):
        a[actual,actual-1] = -alpha1*Dvij
    if(j!=Ny):
        a[actual,actual+1] = -alpha1*Dvij
    if(i!=0):
        a[actual,actual-(Ny+1)] = -alpha2*Dvij
    if(i!=Nx):
        a[actual,actual+(Ny+1)] = -alpha2*Dvij

    # Ghost points
    temp = 0

    if(i!=0):
        temp  += alpha4*Dvij*V_temp[i-1,j]
    else:
        temp += alpha4*Dvij*V2671(T_new,V_temp,-1,j)

    temp+= (1+alpha3*eta1+alpha4*eta2)*V_temp[i,j]
    
    if(j!=0):
        temp += -alpha3*Dvij*V_temp[i,j-1]
    else:
        temp += alpha4*V2672(T_new,V_temp,i,-1)
    
    if(i!=Nx):
        temp  += alpha4*Dvij*V_temp[i+1,j]
    else:
        temp += alpha4*Dvij*V2673(V_temp,Nx+1,j)

    if(j!=Ny):
        temp += alpha3*Dvij*V_temp[i,j+1] 
    else:
        temp += alpha3*Dvij*V2674(V_temp,i,Ny+1 )

    b[actual]=temp

    return a,b

def Vnew(T_new,V_temp,W_temp,dx,dy,dt,Nx,Ny,theta1, theta2):
    a=np.zeros(((Nx+1)*(Ny+1),(Nx+1)*(Ny+1)))
    b=np.zeros((Nx+1)*(Ny+1))
    V_air=0
    #**********************************
    # V at internal points
    #**********************************

    for i in range(0,Nx+1):
        for j in range(0,Ny+1):
            a,b = vf(T_new,V_temp,W_temp,dx,dy,dt,Nx,Ny,theta1, theta2,i,j,a,b)
    #********************
    #solving
    #*********************
    return np.linalg.solve(a,b)

##############################################
#second correction of vapour and water content.
##############################################
def Correction2(T_new,V_new,W_temp,V_s,Nx, Ny,P,W):
    R=8.314
    for i in range(0,Nx+1):
        for j in range(0,Ny+1):
            actual = i*(Ny+1)+j
            V_s[i,j]=18.*10**(-3)*P[i,j]/(R*(T_new[i,j]+273.5)*(170+281*W[i,j]))*0.7*3.8

    for i in range(0,Nx+1):
        for i in range(0,Ny+1):
            actual = i*(Ny+1)+j
            if (W_temp[i,j]+V_new[i,j]<V_s[i,j]):
                V_new[i,j]=W_temp[i,j]+V_new[i,j]
                W_temp[i,j]=0
            else:
                W_temp[i,j]=W_temp[i,j]+V_new[i,j]-V_s[i,j]
                V_new[i,j]=V_s[i,j]

    return V_new,W_temp

##########################################
#Function to calculate new water content.
##########################################
def W2692(T_new,W,i,j):
    Wair = 0
    return W[1,j]+2*dy*hw(T_new,W,0,j)*(W[0,j]-Wair)

def W2691(T_new,W,i,j):
    Wair = 0
    return W[i,1]+2*dx*hw(T_new,W,i,0)*(W[i,0]-Wair)


def W2693(W,i,j):
    return W[Ny-1,j]

def W2694(W,i,j):
    return W[i,Nx-1]


def hw(T_new,W,i,j):
    return (1.4*10**(-3)*T_new[i,j]+0.27*W[i,j]-4.0*10**(-4)*T_new[i,j]*W[i,j]-0.77*W[i,j]*W[i,j])


def fW(T_new,V_new,W,dx,dy,dt,Nx,Ny,theta1,theta2,i,j,a,b):
    alpha1 = Dw*(1-theta1)*dt/(dx*dx)
    alpha2 = Dw*(1-theta2)*dt/(dy*dy)
    alpha3 = Dw*(theta1)*dt/(dx*dx)
    alpha4 = Dw*(theta2)*dt/(dy*dy)
    actual = i*(Ny+1)+j

    a[actual,actual] = 1+2*alpha1+2*alpha2
    if(j!=0):
        a[actual,actual-1] = -alpha1
    if(j!=Ny):
        a[actual,actual+1] = -alpha1
    if(i!=0):
        a[actual,actual-(Ny+1)] = -alpha2
    if(i!=Nx):
        a[actual,actual+(Ny+1)] = -alpha2


    temp = 0
    temp+=(1-2*alpha3-2*alpha4)*W[i,j]

    if(j!=0):
        temp += alpha3*W[i,j-1]
    else:
        temp += alpha3*W2692(T_new,W,i,-1)

    if(j!=Ny):
        temp += alpha3*W[i,j+1]
    else:
        temp += alpha3*W2694(W,i,Ny+1)

    if(i!=0):
        temp +=  alpha4*W[i-1,j]
    else:
        temp +=  alpha4*W2691(T_new,W,-1,j)

    if(i!=Nx):
        temp +=  alpha4*W[i+1,j]
    else:
        temp +=  alpha4*W2693(W,Nx+1,j)

    b[actual]=temp
    return a,b

def Wnew(T_new,V_new,W,dx,dy,dt,Nx,Ny,theta1,theta2):
    a=np.zeros(((Nx+1)*(Ny+1),(Nx+1)*(Ny+1)))
    b=np.zeros((Nx+1)*(Ny+1))
    W_air=0
    Dw=1.35*10**(-10)
    #*******************************
    #Internal nodes
    #*******************************

    for i in range(0,Nx+1):
        for j in range(0,Ny+1):
            a,b = fW(T_new,V_new,W,dx,dy,dt,Nx,Ny,theta1,theta2,i,j,a,b)   

    return np.linalg.solve(a,b)
 
    ########### END ##########


