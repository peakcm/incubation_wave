#simulate the model of Corey for epidemics with different latent times

    #a) agent-based model

        #agents are in four compartments: SEIR
        #S are susceptible, E are infected but in the latency state, I are infected and infectious and R recovered.
        #S+E+R move freely on a spatial network (2d lattice)
        #I do not move
        
        #S+I->I+E with prob beta
        #E->I after a certain latent time T_l (deterministic or from a distribution)
        #I->R after a certain infectious time T_i (deterministic or from a distribution)
        
    #b) network minimal model for comparison of basic mechanism
        
        #metapopulation-like model
        #interaction network is a sum of powers of the mobility network in a) up to power T_l
        #Time is in generations, where a generation is takes a time of T_l
        
import networkx as nx
import numpy as np
from numpy.random import binomial,seed
import random
import matplotlib.pyplot as plt
import os
from scipy.stats import erlang

random.seed(1235)
seed(seed=1235)
#a) ABM

#initialize

## space and demographics

l=10 # side of the square lattice
n_agloc=100 # number of agents in each location initially

N_loc=l*l # number of locations
N_agents=l*l*n_agloc # total number of agents
G=nx.grid_2d_graph(l,l) # network for the spatial substrate

## epidemic model parameters

m=1 # number of steps per day

alpha=0.1
beta=1.0/n_agloc
mu=0.1

T_l=9 # latent time
alpha_step=1-np.power((1-alpha),1.0/float(m)) # probability of moving
beta_step=beta/(m) # probability of disease transmission
mu_step=1-np.power((1-mu),1.0/float(m)) # probability of recovery
gamma=0.0 # increase/decrease on moving probability for infectious individuals

# other parameters

end_time=10

## give locations to all agents

loc=dict() # location of each agent
agents=dict() # agents in each location
S=dict() # S agents in each location
E=dict() # E agents in each location
I=dict() # I agents in each location
R=dict() # R agents in each location
k=0
for i in range(l):
    for j in range(l):
        iloc=(i,j)
        agents[iloc]=set()
        S[iloc]=set()
        E[iloc]=set()
        I[iloc]=set()
        R[iloc]=set()
        for j in range(n_agloc):
            loc[k]=iloc
            agents[iloc].add(k)
            S[iloc].add(k)
            k+=1

state=np.zeros(N_agents) # epidemic state of each agent
inf_time=np.zeros(N_agents) # time at which agent became E

## initial condition for the epidemics

iloc=(l/2,l/2) # the place where we start having a percentage perc of E individuals 
perc=0.05 # percentage of E agents initially in location iloc; the rest of agents are S

for agent in random.sample(agents[iloc],int(perc*n_agloc)):
    state[agent]=1
    #tt=erlang(T_l, loc=0, scale=1)
    tt=np.random.uniform(0,10)
    #print (tt)
    inf_time[agent]=tt
    #inf_time[agent]=T_l # change here the time T_l for one from a distribution to get variation on it
    S[iloc].remove(agent)
    E[iloc].add(agent)
    
###SAVE INITIAL CONDITION/PRINT WHATEVER/PLOT WHATEVER

i_plot=np.zeros((l,l))
for ix in range(l):
    for iy in range(l):
        i_plot[ix][iy]=float(len(I[(ix,iy)]))/float(len(agents[(ix,iy)]))
fig=plt.figure()
#plt.subplot(221,title='$P$ only space')
plt.imshow(i_plot,vmin=0, vmax=1, cmap='jet')
fig.savefig('a_%.3i.png' % 0,bbox_inches='tight')
#plt.show()
plt.close()



I_tot=0
print(0,I_tot)
## start the epidemics

locations=G.nodes()
for time in range(end_time):
    for k in range(m):
        # first move
        for iloc in locations:
            N_move=binomial(len(agents[iloc])-len(I[iloc]),alpha_step)
            for agent in random.sample(agents[iloc]-I[iloc],N_move):
                jloc=random.sample(G.neighbors(iloc),1)[0]
                loc[agent]=jloc
                agents[iloc].remove(agent)
                agents[jloc].add(agent)
                if state[agent]==0:
                    S[iloc].remove(agent)
                    S[jloc].add(agent)
                elif state[agent]==1:
                    E[iloc].remove(agent)
                    E[jloc].add(agent)
                else:
                    R[iloc].remove(agent)
                    R[jloc].add(agent)
            N_move=binomial(len(I[iloc]),alpha_step*gamma)
            for agent in random.sample(I[iloc],N_move):
                jloc=random.sample(G.neighbors(iloc),1)[0]
                loc[agent]=jloc
                agents[iloc].remove(agent)
                agents[jloc].add(agent)
                I[iloc].remove(agent)
                I[jloc].add(agent)
        # second infection dynamics
        for iloc in locations:
            N_inf=binomial(len(S[iloc]),1.0-np.power(1.0-beta_step,len(I[iloc])))
            N_rem=binomial(len(I[iloc]),mu_step)
            for agent in random.sample(S[iloc],N_inf):
                state[agent]=1
                #tt=erlang(T_l, loc=0, scale=1)
                tt=np.random.uniform(0,10)
                inf_time[agent]=time+tt
                #inf_time[agent]=time+T_l # change here the time T_l for one from a distribution to get variation on it
                S[iloc].remove(agent)
                E[iloc].add(agent)
            for agent in random.sample(I[iloc],N_rem):
                state[agent]=3
                I[iloc].remove(agent)
                R[iloc].add(agent)
            a=set(E[iloc])
            for agent in a:
                if inf_time[agent]<=time:
                    state[agent]=2
                    E[iloc].remove(agent)
                    I[iloc].add(agent)
    I_tot=0
    for iloc in locations:
        I_tot+=len(I[iloc])
    print(time,I_tot)
    for ix in range(l):
        for iy in range(l):
            i_plot[ix][iy]=float(len(I[(ix,iy)]))/float(len(agents[(ix,iy)]))
    fig=plt.figure()
    #plt.subplot(221,title='$P$ only space')
    plt.imshow(i_plot,vmin=0, vmax=0.5, cmap='jet')
    fig.savefig('a_%.3i.png' % (time+1),bbox_inches='tight')
    #plt.show()
    plt.close()

os.system('mencoder mf://a_*.png -mf w=800:h=600:fps=10:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o latent_times_l_'+str(l)+'_Tl_'+str(T_l)+'_'+str(m)+'.avi')

os.system('rm a_*.png')

#b) network minimal model for comparison of basic mechanism

from numpy import linalg as LA
A=nx.adjacency_matrix(G)

print(A)
print(type(A))
A=A.todense()
print(A)
print(type(A))
B=LA.matrix_power(A,2)
print(B)
print(type(B))



