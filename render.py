#! /usr/bin/env python
# -*- coding: utf8 -*-
"""Port of NeHe Lesson 16 by Ivan Izuver <izuver@users.sourceforge.net>"""
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import Image
from OpenGL.GL.shaders import *
import random
import numpy as np

# Some api in the chain is translating the keystrokes to this octal string
# so instead of saying: ESCAPE = 27, we use the following.
ESCAPE = '\033'

# Number of the glut window.
window = 0

# Rotations for cube.
xrot = yrot = zrot = 0.0

fogColor=(0.5, 0.5, 0.5, 1.0) # fog color

texture = 0

# vertex shader
strVS = open('shaderv.glsl').read()

# fragment shader
strFS = open('shaderf.glsl').read()

strTex = 'imagen3.png'

global BaseProgram
global pos
global posCam

sizeOfFloat = 4
dim = 0
angle = 0.0
posCam = [-2.0,0.0,0.0]
uOffset = np.array([0.0,0.0,0.0])
uLightP = np.array([0.0,0.0,0.0])
alpha = 0.0
uTMK = 6.0
uTMK2 = 64.0

class renderParam( object ):
	"""Class holding current parameters for rendering.

	Parameters are modified by user interaction"""
	def __init__( self ):
		self.initialColor = [1, 1, 1]
		self.drawColor = self.initialColor
		self.tVec = [0, 0, 0]
		self.mouseButton = None

	def reset( self ):
		self.drawColor = self.initialColor
		setList( self.tVec, 0 )
		self.mouseButton = None

def set_uniforms():
    global BaseProgram
    global uLightP
    global alpha

    alpha = alpha+0.01
    uLightP = uLightP + np.array([0.0,4*np.cos(alpha),4*np.sin(alpha)]);
    print uLightP

    uniforms1 = ["uTMK", "uTMK2", "uShininess","uShin2"]
    uniforms3 = [ "uCamPos", "uLightP","uLightC","uTexDim","uColor", "uOffset"]
    textures = ["uTex"]
    values1 = [uTMK,uTMK2,1.0,8.0]
    values3 = [posCam,uLightP,[1.0,1.0,1.0],[dim,dim,dim],[0.8,0.7,0.65],uOffset]

    temp = glGetUniformLocation(BaseProgram,"uTex")

    glUniform1i(temp, 0) # !! pay attention to 0

    for i in range(len(uniforms1)):
        temp =  glGetUniformLocation(BaseProgram, uniforms1[i])
        print uniforms1[i], temp
        if not temp in (None,-1):
            glUniform1f(temp,values1[i])

    for i in range(len(uniforms3)):
        temp =  glGetUniformLocation(BaseProgram, uniforms3[i])
        print uniforms3[i], temp
        if not temp in (None,-1):
            glUniform3f(temp,values3[i][0],values3[i][1],values3[i][2])


def LoadTextures():
    image = Image.open(strTex)


    image = image.convert("L") #np.uint8(1) #tostring("raw", "L", 0, -1)
    #im = np.array([random.randint(0,255) for _ in range(dim*dim*dim)]).astype(np.uint8)
    dim = image.size[0]
    im = np.zeros(dim*dim*dim).astype(np.uint8)
    for i in range(dim):
        im[i*dim*dim:(i+1)*dim*dim] = np.array(image.crop((0,i*dim,dim,(i+1)*dim)).getdata())

#    for i in range(dim):
#        for j in range(dim):
#            for k in range(dim):
#                if(i*i+j*j+k*k>80000-random.randint(0,20000)):
#                    im[i+j*dim+k*dim*dim] = im[i+j*dim+k*dim*dim]-random.randint(0,100)

    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    glBindTexture(GL_TEXTURE_3D, glGenTextures(1))
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
    glTexImage3D(GL_TEXTURE_3D, 0,GL_LUMINANCE, dim, dim,dim,0, GL_LUMINANCE, GL_UNSIGNED_BYTE,im)

# A general OpenGL initialization function.  Sets all of the initial parameters.
def InitGL(Width, Height):                # We call this right after our OpenGL window is created.
    global BaseProgram
    global pos

    LoadTextures()
    glClearColor(0.0, 0.0, 0.0, 0.0)    # This Will Clear The Background Color To Black

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()                    # Reset The Projection Matrix
                                        # Calculate The Aspect Ratio Of The Window
    gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)

        
    BaseProgram = compileProgram(compileShader(strVS,
                                                GL_VERTEX_SHADER),
                                 compileShader(strFS,
                                                GL_FRAGMENT_SHADER))


# The function called when our window is resized (which shouldn't happen if you enable fullscreen, below)
def ReSizeGLScene(Width, Height):
    if Height == 0:                        # Prevent A Divide By Zero If The Window Is Too Small
        Height = 1

    glViewport(0, 0, Width, Height)        # Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


# The main drawing function.
def DrawGLScene():
    global xrot, yrot, zrot, texture, posCam

    glClear(GL_COLOR_BUFFER_BIT);# | GL_DEPTH_BUFFER_BIT)    # Clear The Screen And The Depth Buffer
    glLoadIdentity()                    # Reset The View

    glClearColor(0.0,0.0,0.0,0.0)

    glEnable(GL_CULL_FACE)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

    gluLookAt(posCam[0],posCam[1],posCam[2],0,0,0,0,0,1);

    glUseProgram(BaseProgram)
    set_uniforms()
    glBegin(GL_QUADS)		# Draw The Cube Using quads
    glVertex3f( 1.0, 1.0,-1.0);	# Top Right Of The Quad (Top)
    glVertex3f(-1.0, 1.0,-1.0);	# Top Let O The Quad (Top)
    glVertex3f(-1.0, 1.0, 1.0);	# Bottom Let O The Quad (Top)
    glVertex3f( 1.0, 1.0, 1.0);	# Bottom Right O The Quad (Top)
    glVertex3f( 1.0,-1.0, 1.0);	# Top Right O The Quad (Bottom)
    glVertex3f(-1.0,-1.0, 1.0);	# Top Let O The Quad (Bottom)
    glVertex3f(-1.0,-1.0,-1.0);	# Bottom Let O The Quad (Bottom)
    glVertex3f( 1.0,-1.0,-1.0);	# Bottom Right Of The Quad (Bottom)
    glVertex3f( 1.0, 1.0, 1.0);	# Top Right Of The Quad (Front)
    glVertex3f(-1.0, 1.0, 1.0);	# Top Left Of The Quad (Front)
    glVertex3f(-1.0,-1.0, 1.0);	# Bottom Left Of The Quad (Front)
    glVertex3f( 1.0,-1.0, 1.0);	# Bottom Right Of The Quad (Front)
    glVertex3f( 1.0,-1.0,-1.0);	# Top Right Of The Quad (Back)
    glVertex3f(-1.0,-1.0,-1.0);	# Top Left Of The Quad (Back)
    glVertex3f(-1.0, 1.0,-1.0);	# Bottom Left Of The Quad (Back)
    glVertex3f( 1.0, 1.0,-1.0);	# Bottom Right Of The Quad (Back)
    glVertex3f(-1.0, 1.0, 1.0);	# Top Right Of The Quad (Left)
    glVertex3f(-1.0, 1.0,-1.0);	# Top Left Of The Quad (Left)
    glVertex3f(-1.0,-1.0,-1.0);	# Bottom Left Of The Quad (Left)
    glVertex3f(-1.0,-1.0, 1.0);	# Bottom Right Of The Quad (Left)
    glVertex3f( 1.0, 1.0,-1.0);	# Top Right Of The Quad (Right)
    glVertex3f( 1.0, 1.0, 1.0);	# Top Left Of The Quad (Right)
    glVertex3f( 1.0,-1.0, 1.0);	# Bottom Left Of The Quad (Right)
    glVertex3f( 1.0,-1.0,-1.0);	# Bottom Right Of The Quad (Right)   
    glEnd();                # Done Drawing The Cube

    xrot  = xrot + 0.0                # X rotation
    yrot = yrot + 0.0                # Y rotation
    zrot = zrot + 0.0                 # Z rotation

    #  since this is double buffered, swap the buffers to display what just got drawn.
    glutSwapBuffers()

# The function called whenever a key is pressed. Note the use of Python tuples to pass in: (key, x, y)
def keyPressed(*args):
    global uOffset
    global angle
    global posCam, uTMK,uTMK2
    # If escape is pressed, kill everything.
    if args[0] == ESCAPE:
        sys.exit()
    if args[0] == "a":
        posCam[0] = posCam[0]+0.1;
    if args[0] == "s":
        posCam[0] = posCam[0]-0.1;
    if args[0] == "d":
        posCam[1] = posCam[1]+0.1;
    if args[0] == "f":
        posCam[1] = posCam[1]-0.1;
    if args[0] == "g":
        posCam[2] = posCam[2]+0.1;
    if args[0] == "h":
        posCam[2] = posCam[2]-0.1;

    if args[0] == "o":
        uTMK = uTMK+1;
    if args[0] == "p":
        uTMK = uTMK-1;

    if args[0] == "k":
        uTMK2 = uTMK2-1;
    if args[0] == "l":
        uTMK2 = uTMK2+1;

    if args[0] == "c":
        angle = angle+0.05
        r = -2        
        posCam = [-2.0,r*np.cos(angle),r*np.sin(angle)]
    if args[0] == "v":
        angle = angle-0.05
        r = -2        
        posCam = [-2.0,r*np.cos(angle),r*np.sin(angle)]




    #print posCam
    

rP = renderParam( )

oldMousePos = [ 0, 0 ]
def mouseButton( button, mode, x, y ):
	"""Callback function (mouse button pressed or released).

	The current and old mouse positions are stored in
	a	global renderParam and a global list respectively"""

	global rP, oldMousePos
	if mode == GLUT_DOWN:
		rP.mouseButton = button
	else:
		rP.mouseButton = None
	oldMousePos[0], oldMousePos[1] = x, y
	glutPostRedisplay( )

def mouseMotion( x, y ):
	"""Callback function (mouse moved while button is pressed).

	The current and old mouse positions are stored in
	a	global renderParam and a global list respectively.
	The global translation vector is updated according to
	the movement of the mouse pointer."""

	global rP, oldMousePos
	deltaX = x - oldMousePos[ 0 ]
	deltaY = y - oldMousePos[ 1 ]
	if rP.mouseButton == GLUT_LEFT_BUTTON:
		factor = 0.01
		rP.tVec[0] += deltaX * factor
		rP.tVec[1] -= deltaY * factor
		oldMousePos[0], oldMousePos[1] = x, y
	glutPostRedisplay( )

def main():
    global window
    glutInit(sys.argv)

    # Select type of Display mode:
    #  Double buffer
    #  RGBA color
    # Alpha components supported
    # Depth buffer
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    # get a 800x600 window
    glutInitWindowSize(800, 600)

    # the window starts at the upper left corner of the screen
    glutInitWindowPosition(0, 0)

    # Okay, like the C version we retain the window id to use when closing, but for those of you new
    # to Python (like myself), remember this assignment would make the variable local and not global
    # if it weren't for the global declaration at the start of main.
    window = glutCreateWindow("Fog by RISC")

       # Register the drawing function with glut, BUT in Python land, at least using PyOpenGL, we need to
    # set the function pointer and invoke a function to actually register the callback, otherwise it
    # would be very much like the C version of the code.
    glutDisplayFunc(DrawGLScene)

    # Uncomment this line to get full screen.
    # glutFullScreen()

    # When we are doing nothing, redraw the scene.
    glutIdleFunc(DrawGLScene)

    # Register the function called when our window is resized.
    glutReshapeFunc(ReSizeGLScene)

    # Register the function called when the keyboard is pressed.
    glutKeyboardFunc(keyPressed)

    # Initialize our window.
    InitGL(800,600)

    # Start Event Processing Engine
    glutMainLoop()

# Print message to console, and kick off the main to get it rolling.
if __name__ == "__main__":
    print "Hit ESC key to quit."
    main()
