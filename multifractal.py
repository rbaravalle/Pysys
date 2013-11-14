import random

maxSize = 25

def f(a): # amount of sons depend in size (multifractality)
    if(a > 40): return 5
    if(a > 30): return 5
    if(a > 20): return 8
    if(a > 10): return 10
    return 11

def fdist(a): # distance depends on size
    if(a > 40): return 20*a
    if(a > 30): return 18*a
    if(a > 20): return 16*a
    if(a > 10): return 14*a
    return 12*a

def ffrac(r):
    if(r > 40): return 0.6-random.random()*0.07
    if(r > 30): return 0.65-random.random()*0.07
    if(r > 20): return 0.60-random.random()*0.07
    if(r > 10): return 0.55-random.random()*0.07
    return 0.4-random.random()*0.07

