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
    hw=1.4*10**(-3)*T[x,y]+0.27*W[x,y]-4.0*10**(-4)*T[x,y]*W[x,y]-0.77*W[x,y]**(2)
    return hw

# Border conditions

def T256(T,W,i,j): # equation (2.56) # i always -1, only for clarity below
    temp=lam*(170+284*W[0,j])*Dw*hw(T,W,0,0) # REVISAR W[0,j] ????
    right = hr(T,0,j)*(T_r-T[0,j])+hc*(T_air-T[0,j])-temp*(W[0,j]-W_air)
    return T[1,j] + (2*dy/k)*right


def T257(T,W,i,j): # equation (2.57) j always -1
    temp=lam*(170+284*W[0,j])*Dw*hw(T,W,0,0);
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



def Tnew(T,V,W,Nx,Ny,dt,dx,dy,theta1,theta2):
#**********************************
#constants below
#**********************************

    #*********************************
    #interior nodes
    #********************************
    a=np.zeros(((Nx+1)*(Ny+1),(Nx+1)*(Ny+1)))
    b=np.zeros((Nx+1)*(Ny+1))
    alpha5 = lam*Dw*dt/(cp*dx*dx)
    alpha6 = lam*Dw*dt/(cp*dy*dy)
    for i in range(1,Nx):
        for j in range(1,Ny):
            actual = i*(Nx+1)+j # actual index

            rx=k*dt/((170+284*W[i,j])*cp*dx*dx)
            ry=k*dt/((170+284*W[i,j])*cp*dy*dy)
            alpha1 = -ry*(1-theta2)
            alpha2 = -rx*(1-theta1)
            alpha3 = ry*theta2
            alpha4 = rx*theta1


            a[actual,actual] = 1+2*alpha1+2*alpha2
            a[actual,actual-1] = -alpha2
            a[actual,actual+1] = -alpha2
            a[actual,actual+Ny] = -alpha1
            a[actual,actual-Ny] = -alpha1

            b[actual] = alpha3*T[i-1,j]+(1-2*alpha3-2*alpha4)*T[i,j]+alpha3*T[i+1,j]+alpha4*T[i,j-1]+alpha4*T[i,j+1]+alpha5*(W[i-1,j]-2*W[i,j]+W[i+1,j])+alpha6*(W[i,j-1]-2*W[i,j]+W[i,j+1])
    #**********************************************
    #Border conditions. for temp at 1st node where T_f is fictious node
    #**********************************************

    i = 0
    j = 0
    actual = 0 # actual index

    rx=k*dt/((170+284*W[0,0])*cp*dx*dx)
    ry=k*dt/((170+284*W[0,0])*cp*dy*dy)
    alpha1 = -ry*(1-theta2)
    alpha2 = -rx*(1-theta1)
    alpha3 = ry*theta2
    alpha4 = rx*theta1

    a[0,0] = 1+2*alpha1+2*alpha2
    #a[0,actual-1] = -alpha2
    a[0,1] = -alpha2
    a[0,Ny] = -alpha1
    #a[actual,actual-Ny] = -alpha1

    b[actual] = alpha3*T256(T,W,-1,0)+(1-2*alpha3-2*alpha4)*T[0,0] + alpha3*T[1,0] + alpha4*T257(T,W,0,-1) + alpha4*T[0,1] + alpha5*(W260(T,W,-1,0)-2*W[0,0] + W[1,0]) + alpha6*(W261(T,W,0,-1)-2*W[0,0]+W[0,1])

    i = 0
    for j in range(1,Ny):
        actual = i*(Nx+1)+j # actual index

        rx=k*dt/((170+284*W[i,j])*cp*dx*dx)
        ry=k*dt/((170+284*W[i,j])*cp*dy*dy)
        alpha1 = -ry*(1-theta2)
        alpha2 = -rx*(1-theta1)
        alpha3 = ry*theta2
        alpha4 = rx*theta1

        a[actual,actual] = 1+2*alpha1+2*alpha2
        a[actual,actual-1] = -alpha2
        a[actual,actual+1] = -alpha2
        a[actual,actual+Ny] = -alpha1
        a[actual,actual-Ny] = -alpha1

        b[actual] = alpha3*T256(T,W,-1,j)+(1-2*alpha3-2*alpha4)*T[0,j]+alpha3*T[1,j]+alpha4*T[0,j-1]+alpha4*T[0,j+1]+alpha5*(W260(T,W,-1,j)-2*W[0,j]+W[1,j])+alpha6*(W[0,j-1]-2*W[0,j]+W[0,j+1])

    j = Ny
    i = 0
    actual = i*(Nx+1)+j # actual index

    rx=k*dt/((170+284*W[0,Ny])*cp*dx*dx)
    ry=k*dt/((170+284*W[0,Ny])*cp*dy*dy)
    alpha1 = -ry*(1-theta2)
    alpha2 = -rx*(1-theta1)
    alpha3 = ry*theta2
    alpha4 = rx*theta1

    a[0,0] = 1+2*alpha1+2*alpha2
    a[0,actual-1] = -alpha2
    a[0,1] = -alpha2
    a[0,Ny] = -alpha1
    a[actual,actual-Ny] = -alpha1

    b[actual] = alpha3*T256(T,W,-1,Ny)+(1-2*alpha3-2*alpha4)*T[0,Ny]+alpha3*T[1,Ny]+alpha4*T[0,Ny-1]+alpha4*T259(T,0,Ny+1)+alpha5*(W260(T,W,-1,Ny)-2*W[0,Ny]+W[1,Ny])+alpha6*(W[0,Ny-1]-2*W[0,Ny]+W263(W,0,Ny+1))

    j = 0
    for i in range(1,Nx):
        actual = i*(Nx+1)+j # actual index

        # ghost points # hr(x) == hr(i,0) ??

        rx=k*dt/((170+284*W[i,j])*cp*dx*dx)
        ry=k*dt/((170+284*W[i,j])*cp*dy*dy)
        alpha1 = -ry*(1-theta2)
        alpha2 = -rx*(1-theta1)
        alpha3 = ry*theta2
        alpha4 = rx*theta1

        a[actual,actual] = 1+2*alpha1+2*alpha2
        a[actual,actual-1] = -alpha2
        a[actual,actual+1] = -alpha2
        a[actual,actual+Ny] = -alpha1
        a[actual,actual-Ny] = -alpha1

        b[actual] = alpha3*T[i-1,j]+(1-2*alpha3-2*alpha4)*T[i,j]+alpha3*T[i+1,j]+alpha4*T[i-1,0]+alpha4*T[i,j+1]+alpha5*(W[i-1,j]-2*W[i,j]+W[i+1,j])+alpha6*(W261(T,W,i,-1)-2*W[i,j]+W[i,j+1])

    j = Ny
    for i in range(1,Nx):
        actual = i*(Nx+1)+j # actual index

        # ghost points # hr(x) == hr(i,0) ??

        rx=k*dt/((170+284*W[i,Nx])*cp*dx*dx)
        ry=k*dt/((170+284*W[i,Nx])*cp*dy*dy)
        alpha1 = -ry*(1-theta2)
        alpha2 = -rx*(1-theta1)
        alpha3 = ry*theta2
        alpha4 = rx*theta1

        a[actual,actual] = 1+2*alpha1+2*alpha2
        a[actual,actual-1] = -alpha2
        a[actual,actual+1] = -alpha2
        a[actual,actual+Ny] = -alpha1
        a[actual,actual-Ny] = -alpha1

        b[actual] = alpha3*T[i-1,Ny]+(1-2*alpha3-2*alpha4)*T[i,Ny]+alpha3*T[i+1,Ny]+alpha4*T[i,Ny-1]+alpha4*T259(T,i,Ny+1)+alpha5*(W[i-1,Ny]-2*W[i,Ny]+W[i+1,Ny])+alpha6*(W[i,Ny-1]-2*W[i,Ny]+W263(W,i,Ny+1))

    j = 0
    i = Nx
    actual = i*(Nx+1)+j # actual index

    rx=k*dt/((170+284*W[Nx,0])*cp*dx*dx)
    ry=k*dt/((170+284*W[Nx,0])*cp*dy*dy)
    alpha1 = -ry*(1-theta2)
    alpha2 = -rx*(1-theta1)
    alpha3 = ry*theta2
    alpha4 = rx*theta1


    a[actual,actual] = 1+2*alpha1+2*alpha2
    a[actual,actual-1] = -alpha2
    a[actual,actual+1] = -alpha2
    a[actual,actual+Ny] = -alpha1
    a[actual,actual-Ny] = -alpha1

    b[actual] = alpha3*T[Nx-1,0]+(1-2*alpha3-2*alpha4)*T[Nx,0]+alpha3*T258(T,Nx+1,0)+alpha4*T257(T,W,Nx,-1)+alpha4*T[Nx,1]+alpha5*(W[Nx-1,0]-2*W[Nx,0]+W262(W,Nx+1,0))+alpha6*(W261(T,W,Nx,-1)-2*W[Nx,0]+W[Nx,1])

    i = Nx
    for j in range(1,Ny):
        actual = i*(Nx+1)+j # actual index

        rx=k*dt/((170+284*W[Nx,j])*cp*dx*dx)
        ry=k*dt/((170+284*W[Nx,j])*cp*dy*dy)
        alpha1 = -ry*(1-theta2)
        alpha2 = -rx*(1-theta1)
        alpha3 = ry*theta2
        alpha4 = rx*theta1

        a[actual,actual] = 1+2*alpha1+2*alpha2
        a[actual,actual-1] = -alpha2
        a[actual,actual+1] = -alpha2
        #a[actual,actual+Ny] = -alpha1
        a[actual,actual-Ny] = -alpha1

        b[actual] = alpha3*T[Nx-1,j]+(1-2*alpha3-2*alpha4)*T[Nx,j]+alpha3*T258(T,Nx+1,j)+alpha4*T[Nx,j-1]+alpha4*T[Nx,j+1]+alpha5*(W[Nx-1,j]-2*W[Nx,j]+W262(W,Nx+1,j))+alpha6*(W[Nx,j-1]-2*W[Nx,j]+W[Nx,j+1])


    j = Ny
    i = Nx
    actual = i*(Nx+1)+j # actual index

    rx=k*dt/((170+284*W[Nx,Ny])*cp*dx*dx)
    ry=k*dt/((170+284*W[Nx,Ny])*cp*dy*dy)
    alpha1 = -ry*(1-theta2)
    alpha2 = -rx*(1-theta1)
    alpha3 = ry*theta2
    alpha4 = rx*theta1


    a[actual,actual] = 1+2*alpha1+2*alpha2
    a[actual,actual-1] = -alpha2
    #a[actual,actual+1] = -alpha2
    #a[actual,actual+Ny] = -alpha1
    a[actual,actual-Ny] = -alpha1

    print actual
    b[actual] = alpha3*T[Nx-1,Ny]+(1-2*alpha3-2*alpha4)*T[Nx,Ny]+alpha3*T258(T,Nx+1,Ny)+alpha4*T[Nx,Ny-1]+alpha4*T259(T,Nx,Ny+1)+alpha5*(W[Nx-1,Ny]-2*W[Nx,Ny]+W262(W,Nx+1,Ny))+alpha6*(W[Nx,Ny-1]-2*W[Nx,Ny]+W263(W,Nx,Ny+1))

    for i in range(a.shape[0]):
        v = np.count_nonzero(a[i])
        if(v!=5): print v, i
        if(b[i]==0): print "B:",i


    return np.linalg.solve(a,b)


################################################
#Function to correct vapour and water content.
################################################
def correction(T_new,V,W,N):
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

    P = np.zeros((N+1))
    V_s = np.zeros((N+1))
    V_temp = np.zeros((N+1))
    W_temp = np.zeros((N+1))

    for i in range(0,N+1):
        f=interpolate.interp1d(x,y) #interp1(x,y,T_new[i],'spline')*1000
        P[i] = f(T_new[i])*1000
        V_s[i]=18.*10**(-3)*P[i]/(R*(T_new[i]+273.5)*(170+281*W[i]))*0.7*3.8
    #****************************************
    # correction in vapour and water content
    #****************************************
    for i in range(0,N+1):
        if (W[i]+V[i]<V_s[i]):
            V_temp[i]=W[i]+V[i]
            W_temp[i]=0
        else:
            V_temp[i]=V_s[i]
            W_temp[i]=W[i]+V[i]-V_s[i]

    return V_temp,W_temp,V_s,P

####################################
# Function to find new Vapour
####################################

def Vnew(T_new,V_temp,W_temp,dx,dt,N,theta):
    a=np.zeros((N+1,N+1))
    b=np.zeros((N+1))
    V_air=0
    #**********************************
    # V at internal points
    #**********************************
    for i in range(1,N):# i=2:1:N
        r=dt*9.0*10**(-12)*(T_new[i]+273.5)**(2)/(dx*dx)
        a[i,i-1]=-r*(1-theta)
        a[i,i]=1+2*r*(1-theta)
        a[i,i+1]=-r*(1-theta)
        b[i]=r*theta*V_temp[i-1]+(1-2*r*theta)*V_temp[i]+r*theta*V_temp[i+1]
    #*************************
    # V at 1st boundary
    #*************************
    temp=2*dx*3.2*10**(9)/((T_new[0]+273.5)**(3))
    r=dt*9.0*10**(-12)*(T_new[0]+273.5)**(2)/(dx*dx)
    V_f=V_temp[1]-temp*(V_temp[0]-V_air)
    a[0,0]=1+r*(1-theta)*(2+temp)
    a[0,1]=-2*r*(1-theta)
    b[0]=r*theta*V_f+(1-2*r*theta)*V_temp[0]+r*theta*V_temp[1]+temp*r*(1-theta)*V_air
    #**************************
    #V at last boundary
    #**************************
    V_temp[N]=V_temp[N-2]
    r=dt*9.0*10**(-12)*(T_new[N]+273.5)**(2)/(dx*dx)
    a[N,N-1]=-2*r*(1-theta)
    a[N,N]=1+2*r*(1-theta)
    b[N]=r*theta*V_temp[N-2]+(1-2*r*theta)*V_temp[N]+r*theta*V_temp[N]
    #********************
    #solving
    #*********************
    return np.linalg.solve(a,b)

##############################################
#second correction of vapour and water content.
##############################################
def Correction2(T_new,V_new,W_temp,V_s,N,P,W):
    R=8.314
    for i in range(0,N+1):#i=1:1:N+1
        V_s[i]=18.*10**(-3)*P[i]/(R*(T_new[i]+273.5)*(170+281*W[i]))*0.7*3.8

    for i in range(0,N+1):
        if (W_temp[i]+V_new[i]<V_s[i]):
            V_new[i]=W_temp[i]+V_new[i]
            W_temp[i]=0
        else:
            W_temp[i]=W_temp[i]+V_new[i]-V_s[i]
            V_new[i]=V_s[i]

    return V_new,W_temp

##########################################
#Function to calculate new water content.
##########################################
def Wnew(T_new,V_new,W_temp,dx,dt,N,theta):
    a=np.zeros((N+1,N+1))
    b=np.zeros((N+1))
    W_air=0
    Dw=1.35*10**(-10)
    #*******************************
    #Internal nodes
    #*******************************
    for i in range(1,N):#i=2:1:N:
        r=dt*Dw/(dx*dx)
        a[i,i-1]=-r*(1-theta)
        a[i,i]=1+2*r*(1-theta)
        a[i,i+1]=-r*(1-theta)
        b[i]=r*theta*W_temp[i-1]+(1-2*r*theta)*W_temp[i]+r*theta*W_temp[i+1]
    #******************************
    # W at 1st boundary
    #******************************
    temp=2*dx*(1.4*10**(-3)*T_new[0]+0.27*W_temp[0]-4.0*10**(-4)*T_new[0]*W_temp[0]-0.77*W_temp[0]*W_temp[0])
    w_f=W_temp[1]-temp*(W_temp[0]-W_air)
    r=dt*Dw/(dx*dx)
    a[0,0]=1+r*(1-theta)*(2+temp)
    a[0,1]=-2*r*(1-theta)
    b[0]=r*theta*w_f+(1-2*r*theta)*W_temp[0]+r*theta*W_temp[1]+r*(1-theta)*temp*W_air
    #*****************************
    #W at last boundary
    #*****************************
    W_temp[N]=W_temp[N-2]
    r=dt*Dw/(dx*dx)
    a[N,N-1]=-2*r*(1-theta)
    a[N,N]=1+2*r*(1-theta)
    b[N]=r*theta*W_temp[N-2]+(1-2*r*theta)*W_temp[N-1]+r*theta*W_temp[N]

    return np.linalg.solve(a,b)
 
    ########### END ##########


