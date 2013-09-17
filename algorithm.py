from particle import *

TIEMPO = 4


def init_particles():
    particles = np.zeros(N*sum(cp)).astype(np.float32)

# una iteracion del algoritmo
def mover() :
    largoCont = 0
    m = []
    for i in range(0,len(particles)):
        pi = particles[i]
        if(pi.tActual > pi.tiempoDeVida) :
            pi.morir()
            m.append(i) 
        else : 
            pi.grow()
            largoCont = largoCont + len(pi.contorno)


    for i in range(0,len(m)):
        del particles[m[i]]
  



def dibujarParticulas() :

    print "we'll draw soon!"
    #print occupied[10:20]
    # we only print those pixels that lie in the visZ plane
    #for(var j = maxcoord2*visZ j < maxcoord2*(visZ+1) j++) :
    #    if(occupied[j].particle > 0) :
    #        var p = occupied[j]
    #        j2 = j - maxcoord2*visZ
    #        var x = j2%maxcoord
    #        var y = Math.floor(j2*m1)
    #        
    #        vertices.push(x*m1,y*m1,0.0)
    #        colors.push(p.r,p.g,p.b,1.0)
    #        cant++
    

def alg() :  

    for t in range(0,TIEMPO-1):
        print "IT"
        mover()
        t = t+1
        if(len(particles) == 0) : break

    dibujarParticulas()
    print "good bye!"
    exit()


def ocupada(i) :
    o = occupied[i]
    return (o >= 0 and sparticles[o])


