#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:13:29 2018

@author: jonathan
"""

from scipy import linspace
from scipy.stats import binom
from scipy.stats import geom
from scipy.stats import poisson
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# linspace creates probability space
# pmf / cdf builds corresponding distro
# plt graphs
# simple

N=100
p = .7
outcomes = linspace(0,N,N+1)

def binomial_pmf (N,p):
    outcomes = linspace(0,N,N+1)
    probs = binom.pmf(outcomes,N,p)
    return probs

test = binomial_pmf(N,p)
plt.bar(outcomes,test)

def binomial_cmf (N,p):
    outcomes = linspace(0,N,N+1)
    probs = binom.cdf(outcomes,N,p)
    return probs

test = binomial_cmf(N,p)
plt.bar(outcomes,test)

#k = number of failures before first success
#outcomes now represent k
# memoryless
N=10
p=.5
outcomes = linspace(0,N,N+1)

def geometric_pmf(K,p):
    outcomes = linspace(0,K,K+1)
    probs = geom.pmf(outcomes,p)
    return probs

test = geometric_pmf(N,p)
plt.bar(outcomes,test)

def geometric_cmf(K,p):
    outcomes = linspace(0,K,K+1)
    probs = geom.cdf(outcomes,p)
    return probs

test = geometric_cmf(N,p)
plt.bar(outcomes,test)

#how many events happen in a given period of time, mu
#helpful in predicting the probability of number of events occuring 
#-->when you know how many times it has occured for a related process
X=10 # X = number of events occuring in fixed time, fixed time not listed (assign any time wanted theoretically)
mu=5 # mu = average number drawn from data or sample
outcomes = linspace(0,X,X+1)

def poisson_pmf(X,mu):
    outcomes = linspace(0,X,X+1)
    probs = poisson.pmf(outcomes,mu)
    return probs

test = poisson_pmf(X,mu)
plt.bar(outcomes,test)

def poisson_cdf(X,mu):
    outcomes = linspace(0,X,X+1)
    probs = poisson.cdf(outcomes,mu)
    return probs

test = poisson_cdf(X,mu)
plt.bar(outcomes,test)


#describes how much time elapses between consecutive events, homogeneous
# makes sense, expected waiting time continually decreases
#when you get into continuous, these X values become arbitrary, good to visualize from transformations sense 
X=100 #make big enough proportionally to visualize limit
mu=10#lambda
start = 0
outcomes = linspace(0,X,X+1)
def exponential_pdf(X,mu):
    outcomes = linspace(0,X,X+1)
    probs = expon.pdf(outcomes,loc=start,scale = mu)
    return probs

test = exponential_pdf(X,mu)
plt.bar(outcomes,test)

def exponential_cdf(X,mu):
    outcomes = linspace(0,X,X+1)
    probs = expon.cdf(outcomes,loc=start,scale=mu)
    return probs

test = exponential_cdf(X,mu)
plt.bar(outcomes,test)

#sum of exponentials
#describes how long I have to wait to see N events derived from 1/mu
# makes sense, as I increase N, the mass in my distribution shifts right
X=100
N = 7
mu = 2#lambda
start = 0
outcomes = linspace(0,X,X+1)

def gamma_pdf(X,N,mu):
    outcomes = linspace(0,X,X+1)
    probs = gamma.pdf(outcomes,N,loc=start,scale=mu)
    return probs

test = gamma_pdf(X,N,mu)
plt.plot(outcomes,test)

def gamma_cdf(X,N,mu):
    outcomes = linspace(0,X,X+1)
    probs = gamma.cdf(outcomes,N,loc=start,scale=mu)
    return probs

test = gamma_cdf(X,N,mu)
plt.plot(outcomes,test)


## normal just for comparison
X = 100
mu = 50
sd = 15

def normal_pdf(X,mu,sd):
    outcomes = linspace(0,X,X+1)
    probs = norm.pdf(outcomes, loc = mu, scale = sd)
    return probs

test = normal_pdf(X,mu,sd)
plt.bar(outcomes,test)

def normal_cdf(X,mu,sd):
    outcomes = linspace(0,X,X+1)
    probs = norm.cdf(outcomes, loc = mu, scale = sd)
    return probs

test = normal_cdf(X,mu,sd)
plt.bar(outcomes,test)


##beta, probability of probabilities
#cant impose values because it is inherently on 0,1
# just toggling these numbers to get values on blog
a, b = 152, 226 #2 out of 20 successes
x = np.linspace(beta.ppf(0.001, a, b), beta.ppf(0.999, a, b), 1000)

def beta_pdf(a,b):
    x = np.linspace(beta.ppf(0.001, a, b), beta.ppf(0.999, a, b), 1000)
    probs = beta.pdf(x,a,b)
    return probs

test = beta_pdf(a,b)    
plt.plot(x,test,"-g")
plt.title("Joe's Posterior Winning Average")
plt.savefig('Joe.png', bbox_inches='tight')

def beta_cdf(a,b):
    x = np.linspace(beta.ppf(0.001, a, b), beta.ppf(0.999, a, b), 1000)
    probs = beta.cdf(x,a,b)
    return probs
 
test = beta_cdf(a,b)      
plt.plot(x,test)

##Visualizing Bayes
matplotlib.rcParams.update({'font.size': 12})
from matplotlib import pyplot as plt
from matplotlib_venn import venn2, venn2_circles
set1 = set([1,2])
set2 = set([2,3])
out = venn2([set1, set2], ('P(A)', 'P(B)'))
plt.text(-.1,.18,'P(AB)')
plt.text(-.24,.05,'= P(A|B)*P(B)')
plt.text(-.24,-.08,'= P(B|A)*P(A)')
for text in out.set_labels:
    text.set_fontsize(14)
for text in out.subset_labels:
    text.set_fontsize(0)

plt.savefig('Bayes_pic.png', bbox_inches='tight')


#### brief example of using bayesian approach
#### baseball
#  posterior = (likelihood  / marginal likelihood) * prior
#  likelihood = p(data)
#  marginal likelihood --> used to scale --> use proper conjugate prior
#  prior --> p(prior)
    

































