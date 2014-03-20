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

def fdist(a): # distance depends on size
    if(a > 40): return 8*a
    if(a > 30): return 8*a
    if(a > 20): return 8*a
    if(a > 10): return 8*a
    return 9*a

def ffrac(r):
    if(r > 40): return 0.8
    if(r > 30): return 0.8-random.random()*0.05
    if(r > 20): return 0.8-random.random()*0.05
    if(r > 10): return 0.8-random.random()*0.05
    return 0.74

class Lindenmayer(object):
    def __init__(self, stream):
        # Set the default image dimensions ...
        self.width = 128
        self.height = 128
        self.maxZ = 128
        
        # ... and the number of iterations.
        self.iterations = 5
        
        # Set the default rotation angle in degrees.
        self.alpha = 2*np.pi/5
        self.alpha2 = 2*np.pi/5
        self.angle = 0
        self.angle2 = 0
        
        # Initialize th
        # e branch stack, ...
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
        #print 'Offset :', self.offset
        
        # Finally store the stream ...
        self.stream = stream

        # state
        self.x = int(self.width/2)#int(self.width/2)
        self.y = int(self.height/2)
        self.z = int(self.maxZ/2)

        self.r = int(self.width/6)
        #print self.r

        self.xparent = self.x
        self.yparent = self.y
        self.zparent = self.z

        
        # ... and initialize the parser.
        self.initialize()

    def drawShape(self,field,x,y,z,r,c):
        if(c == 0): return
        r = int(r)

        for i in range(x-r-1,x+r+1):
            for j in range(y-r-1,y+r+1):
                for k in range(z-r-1,z+r+1):
                     if(i >= 0 and i < self.width and j >= 0 and j < self.height and k >= 0 and k<self.maxZ):
                         i2 = i-x
                         j2 = j-y
                         k2 = k-z
                         if(i2*i2+j2*j2+k2*k2 < r*r):
                             #print float(i), ",", float(j),",", float(k)
                             field[i][j][k] = np.uint8(0)

        #print "ellipse!"
        #    draw.ellipse((x-r+randint(-r/2,r/2), y-r+randint(-r/2,r/2), x+r, y+r), fill=255)
        return
        r2 = r
        x2 = x
        y2 = y
        z2 = z
        #for h in range(maxx):
        r2 = int(r2/(2))
        dd = int(r*1.0)
        for i in range(2):
            x3 = x2+randint(-dd,dd)
            y3 = y2+randint(-dd,dd)
            z3 = z2+randint(-dd,dd)
            self.drawShape(field,x3,y3,z3,r2,c-1)

    def rotate(self):
        ang = self.angle#+random.random()/10
        self.x = self.xparent + np.int32(fdist(self.r)*np.cos(ang))+randint(-int(self.r),int(self.r))
        self.y = self.yparent + np.int32(fdist(self.r)*np.sin(ang))+randint(-int(self.r),int(self.r))
        self.z = self.zparent

    def rotate2(self):
        ang = self.angle2#+random.random()/10
        self.x = self.xparent
        self.y = self.yparent + np.int32(fdist(self.r)*np.cos(ang))+randint(-int(self.r),int(self.r))
        self.z = self.zparent + np.int32(fdist(self.r)*np.sin(ang))+randint(-int(self.r),int(self.r))

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
            
        #print 'Rules:', self.rules
        
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

    def draw2(self,stream):
        maxX = self.width
        maxY = self.height
        import Image
        #print self.width,self.height, self.maxZ
        I = Image.new('L',(self.maxZ,self.width*self.height),0.0)
        field = np.zeros((self.maxZ, self.width, self.height)).astype(np.uint8) + np.uint8(255)
        #draw = ImageDraw.Draw(I)

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
                self.drawShape(field,self.x,self.y,self.z,self.r,2)
                self.r = self.r*ffrac(self.r)

            if c == 'G':
                # draw
                #raphael.forward(self.lineLength)
    
                #draw.ellipse((self.x-self.r, self.y-self.r, self.x+self.r, self.y+self.r), fill=255)
                self.drawShape(field,self.x,self.y,self.z,self.r,2)
                self.r = self.r*ffrac(self.r)
            if c == 'H':
                # draw
                #raphael.forward(self.lineLength)
    
                #draw.ellipse((self.x-self.r, self.y-self.r, self.x+self.r, self.y+self.r), fill=255)
                self.drawShape(field,self.x,self.y,self.z,self.r,2)
                self.r = self.r*ffrac(self.r)
                #self.forward()
            if c == 'I':
                # draw
                #raphael.forward(self.lineLength)
    
                #draw.ellipse((self.x-self.r, self.y-self.r, self.x+self.r, self.y+self.r), fill=255)
                self.drawShape(field,self.x,self.y,self.z,self.r,2)
                self.r = self.r*ffrac(self.r)
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
                self.forward()
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

            if c == 'x':
                self.angle2+=self.alpha
                self.rotate2()
            if c == 'y':
                # rotate anti-clockwise
                #raphael.left(self.alpha)
                self.angle2-=self.alpha
                self.rotate2()
            if c == '[':
                # Push the current turtle state to the stack
                self.stack.append((
                    #self.lineColor, self.lineWidth, raphael.heading(), raphael.pos()))
                    self.x, self.y, self.z, self.r,self.angle,self.angle2,self.alpha))
                self.xparent = self.x
                self.yparent = self.y
                self.zparent = self.z
            if c == ']':
                # restore the transform and orientation from the stack
                self.x, self.y, self.z, self.r,self.angle,self.angle2, self.alpha = self.stack.pop()
                self.xparent = self.x
                self.yparent = self.y
                self.zparent = self.z
                #raphael.penup()
                #raphael.pencolor(self.lineColor)
                #raphael.pensize(self.lineWidth)
                #raphael.goto(p)
                #raphael.setheading(d)
                #raphael.pendown()

        # now save the image
        rowsPerSlice = self.width

        for i in range(self.maxZ):
            I2 = Image.frombuffer('L',(self.width,self.height), np.array(field[:,:,i]).astype(np.uint8),'raw','L',0,1)
            #print np.array(field[:,:,i]).astype(np.uint8)
            #print np.array(field[:,:,i]).astype(np.uint8).sum()
            I.paste(I2,(0,rowsPerSlice*i))


        I.save('lbread3D.png')
        
        
def usage():
    print "Usage: python lsystem.py [-i <inputfile>] [-o <outputfile>] [-h]\n\n" \
        "-i\tPath to the input file.\n\n" \
        "-o\tPath to the eps output file.\n" \
        "\tDefault is Lindenmayer.eps.\n\n" \
        "-h\tShows this help";

def main(argv):
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
