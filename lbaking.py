#!/usr/bin/python
from pyparsing import Word, alphas, Regex, Literal, OneOrMore, ZeroOrMore, \
  ParseException, ParserElement, LineStart, LineEnd, SkipTo, Optional, Empty, \
    StringEnd, Suppress
from Tkinter import *
from turtle import *
import turtle
import random
import getopt
import sys
import re
import ImageDraw
import numpy as np
from random import randint
from multifractal import *
from lparams import *
import baking1D as bak

bubbles = np.zeros((N+1,N+1)).astype(np.int32) # posicion y tamanio burbujas
delta = N/16 # radius to check for intersections with other bubbles



def fdist(a): # distance depends on size
    #if(a > 40): return 20*a
    #if(a > 30): return 20*a
    #if(a > 20): return 20*a
    #if(a > 10): return 20*a
    return 11*a

#def ffrac(r):
    #if(r > 40): return 0.75
    #if(r > 30): return 0.75-random.random()*0.05
    #if(r > 20): return 0.75-random.random()*0.05
    #if(r > 10): return 0.75-random.random()*0.05
    #return 0.75

def drawShape(draw,x,y,r,c):
    if(c == 0): return
    r = int(r)
    #print str()
    rr = 255#int(r+10)
    draw.ellipse((x-r, y-r, x+r, y+r), fill=rr)
    return
    r2 = r
    x2 = x
    y2 = y
    #for h in range(maxx):
    r2 = int(r2/(2))
    dd = int(r*1.0)
    for i in range(2):
        x3 = x2+randint(-dd,dd)
        y3 = y2+randint(-dd,dd)
        drawShape(draw,x3,y3,r2,c-1)

class Lindenmayer(object):
    def __init__(self, stream):
        # Set the default image dimensions ...
        self.width = N
        self.height = N
        
        # ... and the number of iterations.
        self.iterations = 5
        
        # Set the default rotation angle in degrees.
        self.alpha = 2*np.pi/8
        self.angle = 0
        
        # Initialize the branch stack, ...
        self.stack = []
        
        # ... the constants, the rules, the variables and the axiom ...
        self.const = {'+':'+', '-':'-', '[':'[', ']':']'}
        self.rules = {}
        self.vars = []
        self.axiom = None
        
        # ... and drawing settings.
        self.bgcolor = (1.0, 1.0, 1.0)
        self.lineLength = 20
        self.lineWidth = 5
        self.lineColor = (0, 0, 0)
        
        # Calculate the starting position.
        self.offset = (0, -self.height*0.5)
        print 'Offset :', self.offset
        
        # Finally store the stream ...
        self.stream = stream

        self.arr = bak.calc()
        print "Baking Bread..."

        # state
        self.x = int(self.width/2)#int(self.width/2)
        self.y = int(self.height/2)
        self.r = 16

        self.xparent = self.x
        self.yparent = self.y
        
        # ... and initialize the parser.
        self.initialize()

    # calculates new radius based on a poisson distribution (?)
    def poisson(self):
        global bubbles, delta
        x1 = min(max(self.x-delta,0),N)
        y1 = min(max(self.y-delta,0),N)

        suma = 0.0
        x0 = max(x1-delta,0)
        y0 = max(y1-delta,0)
        x2 = min(x1+delta,N)
        y2 = min(y1+delta,N)
        for i in range(x0,x2):
            for j in range(y0,y2):
                #d = np.sqrt((x1-i)*(x1-i) + (y1-j)*(y1-j)).astype(np.float32) # distance
                #if(d==0): suma+=bubbles[i,j]
                #else:
                suma += bubbles[i,j]#*delta*delta
                #suma += np.pi*bubbles[i,j]*bubbles[i,j]
                #if(bubbles[i,j]): print "RADIO:" , bubbles[i,j]

        factor = delta # (?)
        #print "Suma:", suma, "Delta: ", delta

        #print factor*(1/suma)

        #1/ sum_D (cant*radios*D^2) 

        #D = np.sqrt(np.power((self.x-self.height/2),2)+np.power((self.y-self.width/2),2))
        G = 1+4*self.arr[min(self.width,max(self.x-1,0)),min(self.width,max(self.y-1,0))]#D/300 #gelatinization
        print "G: ", G

        if(suma > 0.0):
            #print "NUEVO RADIO: ", np.floor(factor*(1/suma))/2, self.x, self.y
            m = 1/suma
            return np.random.randint(delta*m/2,delta*m+1)/G
        else: # free
            #print "NUEVO RADIO: ", delta/4-1, self.x, self.y
            return np.random.randint(delta/2.6,delta/2-1)/G

    def rotate(self):
        #self.x += self.r
        #self.y += self.r
        #d = 2*np.pi*random.random()
        ang = self.angle#+random.random()/10
        self.x = self.xparent + np.int32(fdist(self.r)*np.cos(ang))+randint(-int(self.r),int(self.r))
        self.y = self.yparent + np.int32(fdist(self.r)*np.sin(ang))+randint(-int(self.r),int(self.r))
        #pass

    def moveX(self):
        self.x += self.r
    def mmoveX(self):
        self.x -= self.r
    def moveY(self):
        self.y += self.r
    def mmoveY(self):
        self.y -= self.r

    def initialize(self):
        ParserElement.setDefaultWhitespaceChars(' \t\r')
    
        integer = Regex(r"[+-]?\d+") \
            .setParseAction(lambda s,l,t: [ int(t[0]) ])
        number = Regex(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?") \
            .setParseAction(lambda s,l,t: [ float(t[0]) ])
        color = Regex(r"#([0-9a-fA-F]{6})")
        angle = "'" + Regex(r"(360|3[0-5][0-9]|[12][0-9]{2}|[0-9]{1,2})") \
            .setParseAction(lambda s,l,t: [ int(t[0]) ])
        alpha = "'" + Regex(r"(360|3[0-5][0-9]|[12][0-9]{2}|[0-9]{1,2})") \
            .setParseAction(lambda s,l,t: [ int(t[0]) ])
        variable = Word(alphas, exact=1).setParseAction(self.addVar)
        colon = Literal(":").suppress()
        comma = Literal(",")
        lBrace = Literal("(")
        rBrace = Literal(")")
        lBracket = Literal("[")
        rBracket = Literal("]")
        lAngle = Literal("<")
        rAngle = Literal(">")
        plus = Literal("+")
        minus = Literal("-")
        FTerm = Literal("F")
        fTerm = Literal("f")
        ZTerm = Literal("Z")
        zTerm = Literal("z")
        xTerm = Literal("x")
        cTerm = Literal("c")
        
        eol = OneOrMore(LineEnd()).suppress()
        param = ( angle | color | "!" + number | "|" + number )
        self.pList = lBrace + param + ZeroOrMore(comma + param) + rBrace
        literal = ((lBracket + ( variable + Optional(self.pList) 
                | plus + Optional(self.pList) | minus + Optional(self.pList) ) + rBracket)
            | (variable + Optional(self.pList) | plus + Optional(self.pList) 
                | minus + Optional(self.pList)))
        terminal = (ZTerm | zTerm | FTerm | fTerm | xTerm | cTerm
            | plus | minus | lBracket | rBracket)
        lprod = ( 
            (OneOrMore(terminal) + lAngle + variable + rAngle + OneOrMore(terminal))
            | (OneOrMore(terminal) + lAngle + variable) 
            | (variable + rAngle + OneOrMore(terminal)) 
            | variable )
        rProd = OneOrMore(literal | terminal)
        comment = Suppress((LineStart() + "#" + SkipTo(eol, include=True)))
        rules = ( 
            (lprod + Literal("=") + rProd + eol).setParseAction(self.addRule) \
            | comment )
        defaults = ( ( ("Dimensions" + colon + integer + comma + integer) 
            | ("Position" + colon + integer + comma + integer)
            | ("Iterations" + colon + integer)
            | ("Angle" + colon + angle)
            | ("Linelength" + colon + number)
            | ("Linewidth" + colon + number)
            | ("Linecolor" + colon + color)
            | ("Background" + colon + color)
            | ("Axiom" + colon + rProd) ) + eol ).setParseAction(self.setAttribute)
        header = ( defaults | comment )
        self.grammar = Suppress(ZeroOrMore(LineEnd())) \
            + ZeroOrMore(header) \
            + OneOrMore(rules)
            
        try:
            L = self.grammar.parseString( self.stream )
        except ParseException, err:
            print err.line
            print " "*(err.column-1) + "^"
            print err
            
        print 'Rules:', self.rules
        
    def setAttribute(self, stream, loc, toks):
        if toks[0] == 'Dimensions':
            self.width = toks[1]
            self.height = toks[3]
        if toks[0] == 'Position':
            self.offset = (toks[1], toks[3])
        elif toks[0] == 'Iterations':
            self.iterations = toks[1]
        elif toks[0] == 'Alpha':
            self.alpha = toks[2]
        elif toks[0] == 'Angle':
            self.angle = toks[2]
        elif toks[0] == 'Linelength':
            self.lineLength = toks[1]
        elif toks[0] == 'Linewidth':
            self.lineWidth = toks[1]
        elif toks[0] == 'Linecolor':
            self.lineColor = toks[1]
        elif toks[0] == 'Axiom':
            self.axiom = ''.join(toks[1:])
        elif toks[0] == 'Background':
            self.bgcolor = toks[1]
        
    def addVar(self, stream, loc, toks):
        self.vars.append(toks[0])
        
    def addRule(self, stream, loc, toks):
        toks = list(toks)
        
        if "<" in toks and ">" in toks:
            key = (
                ''.join(toks[:toks.index('<')]), 
                toks[toks.index('<')+1],
                ''.join(toks[toks.index('>')+1:toks.index('=')])
            )
        elif "<" in toks:
            key = (
                ''.join(toks[:toks.index('<')]), 
                toks[toks.index('<')+1],
                None
            )
        elif ">" in toks:
            key = (
                None,
                toks[toks.index('>')],
                ''.join(toks[toks.index('>')+1:toks.index('=')])
            )
        else:
            key = (
                None,
                toks[0],
                None
            )
        
        self.rules[key] = toks[toks.index('=')+1:]
                
    def iterate(self):
        if self.axiom == None or not self.rules:
            return
            
        result = self.axiom
        for repeat in range(0, self.iterations):
            result = self.translate(result)

        return result

    def translate(self, axiom):
        result = ''
        for i in range(len(axiom)):
            if axiom[i] in self.const:
                result += axiom[i]
                continue
            if i > 0 and i < len(axiom)-1:
                key = (axiom[i-1], axiom[i], axiom[i+1])

                if key in self.rules:
                    result += ''.join(map(str, self.rules.get(key)))
                    continue
            if i > 0:
                key = (axiom[i-1], axiom[i], None)
                
                if key in self.rules:
                    result += ''.join(map(str, self.rules.get(key)))
                    continue
            if i < len(axiom)-1:
                key = (None, axiom[i], axiom[i+1])
                
                if key in self.rules:
                    result += ''.join(map(str, self.rules.get(key)))
                    continue
            key = (None, axiom[i], None)
            result += ''.join(map(str, self.rules.get(key, axiom[i])))
        return result

    def delta(self,x,y):
        temp = (np.max(self.arr)-self.arr[x,y])
        #if(temp == 0):
        #return 40
        return 30*(temp)

    def rad(self):
        #x1 = min(self.width,max(self.x-1,0))
        #y1 = min(self.height,max(self.y-1,0))
        #return self.r+self.delta(x1,y1)
        return self.poisson()

    def draw2(self,stream):
        maxX = self.width
        maxY = self.height
        import Image
        print self.width,self.height
        I = Image.new('L',(self.width,self.height),(0))
        draw = ImageDraw.Draw(I)

        # Process the result stream symbol by symbol.
        for i in range(len(stream)):
            c = stream[i]
            
            if i+1 < len(stream)-1 and stream[i+1] == '(':
                end = stream.find(')', i+1)
                if end > 0:
                    params = stream[i+1:end+1]
                    #print params
            
                    try:
                        L = self.pList.parseString( params )
                    except ParseException, err:
                        L = 'ERROR'
                        print err.line
                        print " "*(err.column-1) + "^"
                        print err
            
                    #print 'Params:', L
                    
            if c == 'F':
                # draw
                #raphael.forward(self.lineLength)
    
                #draw.ellipse((self.x-self.r, self.y-self.r, self.x+self.r, self.y+self.r), fill=255)
                r = self.rad()
                x1 = min(self.width,max(self.x-1,0))
                y1 = min(self.height,max(self.y-1,0))
                drawShape(draw,self.x,self.y,r,2)
                bubbles[x1,y1] = r # set actual bubble

            if c == 'G':
                # draw
                #raphael.forward(self.lineLength)
    
                #draw.ellipse((self.x-self.r, self.y-self.r, self.x+self.r, self.y+self.r), fill=255)
                r = self.rad()
                x1 = min(self.width,max(self.x-1,0))
                y1 = min(self.height,max(self.y-1,0))
                drawShape(draw,self.x,self.y,r,2)
                bubbles[x1,y1] = r # set actual bubble
            if c == 'H':
                # draw
                #raphael.forward(self.lineLength)
    
                #draw.ellipse((self.x-self.r, self.y-self.r, self.x+self.r, self.y+self.r), fill=255)
                r = self.rad()
                x1 = min(self.width,max(self.x-1,0))
                y1 = min(self.height,max(self.y-1,0))
                drawShape(draw,self.x,self.y,r,2)
                bubbles[x1,y1] = r # set actual bubble
                #self.forward()
            if c == 'I':
                # draw
                #raphael.forward(self.lineLength)
    
                #draw.ellipse((self.x-self.r, self.y-self.r, self.x+self.r, self.y+self.r), fill=255)
                r = self.rad()
                x1 = min(self.width,max(self.x-1,0))
                y1 = min(self.height,max(self.y-1,0))
                drawShape(draw,self.x,self.y,r,2)
                bubbles[x1,y1] = r # set actual bubble
                #self.forward()
            if c == 'X':
                # Move forward
                #raphael.forward(self.lineLength)
                self.moveX()
            if c == 'x':
                # Move forward
                #raphael.forward(self.lineLength)
                self.mmoveX()
            if c == 'Y':
                # Move forward
                #raphael.forward(self.lineLength)
                self.moveY()
            if c == 'y':
                # Move forward
                #raphael.forward(self.lineLength)
                self.mmoveY()
            if c == 'f':
                #self.forward()
                self.r = self.r*ffrac(self.r)
                # Move forward without drawing
                #raphael.penup()
                #raphael.forward(self.lineLength)
                #raphael.pendown()
            if c == '+':
                # rotate clockwise
                #raphael.right(self.alpha)
                self.angle+=self.alpha
                self.rotate()
            if c == '-':
                # rotate anti-clockwise
                #raphael.left(self.alpha)
                self.angle-=self.alpha
                self.rotate()
            if c == '[':
                # Push the current turtle state to the stack
                self.stack.append((
                    #self.lineColor, self.lineWidth, raphael.heading(), raphael.pos()))
                    self.x, self.y, self.r,self.angle,self.alpha))
                self.xparent = self.x
                self.yparent = self.y
            if c == ']':
                # restore the transform and orientation from the stack
                self.x, self.y, self.r,self.angle,self.alpha = self.stack.pop()
                self.xparent = self.x
                self.yparent = self.y
                #raphael.penup()
                #raphael.pencolor(self.lineColor)
                #raphael.pensize(self.lineWidth)
                #raphael.goto(p)
                #raphael.setheading(d)
                #raphael.pendown()

        # now save the image
        I.save('lbread.png')
        print np.asarray(I)/255
        return np.asarray(I)#/255

def lin():
    with open('bread.txt', 'r') as fp:
        stream = fp.read()
    lindenmayer = Lindenmayer(stream)
    stream = lindenmayer.iterate()
    #lindenmayer.draw(stream, outputFile)
    return lindenmayer.draw2(stream)
     
        

         
        
def usage():
    print "Usage: python lsystem.py [-i <inputfile>] [-o <outputfile>] [-h]\n\n" \
        "-i\tPath to the input file.\n\n" \
        "-o\tPath to the eps output file.\n" \
        "\tDefault is Lindenmayer.eps.\n\n" \
        "-h\tShows this help";

def main(argv):
    print argv
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["help", "input=", "output="])
    except getopt.GetoptError as err:
        # Print usage information and exit.
        print(err)
        usage()
        sys.exit(2)
    
    inputFile = None
    outputFile = None
    
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit(0)
        elif o in ("-i", "--input"):
            inputFile = a;
        elif o in ("-o", "--output"):
            outputFile = a;
        else:
            assert False, "unhandled option"
    
    if inputFile == None:
        print "Input file missing ..."
        usage()
        sys.exit(2)
    
    # Read the whole input file.
    with open(inputFile, 'r') as fp:
        stream = fp.read()
    
    # Do something!
    lindenmayer = Lindenmayer(stream)
    stream = lindenmayer.iterate()
    #lindenmayer.draw(stream, outputFile)
    lindenmayer.draw2(stream)
    
if __name__ == "__main__":
    main(sys.argv[1:])

#def calc(x):
#    main(x,['-i','bread.txt'])
