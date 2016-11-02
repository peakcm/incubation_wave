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

random.seed(12345)
seed(seed=12345)
#a) ABM

#initialize

## space and demographics

l=50 # side of the square lattice
n_agloc=100 # number of agents in each location initially

N_loc=l*l # number of locations
N_agents=l*l*n_agloc # total number of agents
G=nx.grid_2d_graph(l,l) # network for the spatial substrate

## epidemic model parameters

m=1 # number of steps per day

alpha=0.1
beta=1.0/n_agloc
mu=0.1
delta=0.001

T_l=1 # latent time
alpha_step=1-np.power((1-alpha),1.0/float(m)) # probability of moving
beta_step=beta/(m) # probability of disease transmission
mu_step=1-np.power((1-mu),1.0/float(m)) # probability of recovery
gamma=0.0 # increase/decrease on moving probability for infectious individuals

# other parameters

end_time=500

## give locations to all agents

Nruns=5


peak_height=list()
peak_time=list()

for irun in range(Nruns):
    print (Nruns-irun,'start')

    loc=dict() # location of each agent
    agents=dict() # agents in each location
    S=dict() # S agents in each location
    E=dict() # E agents in each location
    I=dict() # I agents in each location
    R=dict() # R agents in each location



    peak_height.append(np.zeros((l,l)))
    peak_time.append(np.zeros((l,l)))

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

    iloc=(int(l/2-0.5),int(l/2-0.5)) # the place where we start having a percentage perc of E individuals 
    perc=0.05 # percentage of E agents initially in location iloc; the rest of agents are S

    rv=erlang(10,scale=T_l/10.)

    for agent in random.sample(agents[iloc],int(perc*n_agloc)):
        state[agent]=1
        tt=rv.rvs(1)
        #tt=np.random.uniform(0,10)
        #print (tt)
        inf_time[agent]=tt
        #inf_time[agent]=T_l # change here the time T_l for one from a distribution to get variation on it
        S[iloc].remove(agent)
        E[iloc].add(agent)
        
    ###SAVE INITIAL CONDITION/PRINT WHATEVER/PLOT WHATEVER


    #data structure to save all epidemic curves

    epi_curves=dict()
    for ix in range(l):
        epi_curves[ix]=dict()
        for iy in range(l):
            epi_curves[ix][iy]=list()

    i_plot=np.zeros((l,l))
    for ix in range(l):
        for iy in range(l):
            kk=float(len(I[(ix,iy)]))/float(len(agents[(ix,iy)]))
            i_plot[ix][iy]=kk
            epi_curves[ix][iy].append(kk)
    #fig=plt.figure()
    ##plt.subplot(221,title='$P$ only space')
    #plt.imshow(i_plot,vmin=0, vmax=1, cmap='jet')
    #fig.savefig('a_%.3i.png' % 0,bbox_inches='tight')
    ##plt.show()
    #plt.close()



    I_tot=0
    #print(0,I_tot)
    ## start the epidemics

    locations=G.nodes()
    for time in range(end_time):
        for k in range(m):
            # first move
            for iloc in locations:
                N_move=binomial(len(agents[iloc])-len(I[iloc]),alpha_step)
                agents_move=random.sample(agents[iloc]-I[iloc],N_move)
                N_flight=binomial(len(agents_move),delta)
                agents_flight=random.sample(agents_move,N_flight)
                agents_flight=set(agents_flight)
                agents_move=set(agents_move)
                agents_diff=set()
                agents_diff=agents_move-agents_flight
                for agent in agents_diff:
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
                for agent in agents_flight:
                    jloc=random.choice(locations)
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
                ###HAVE TO INCLUDE THE TELEPORTATIONS FOR I IN CASE GAMMA!=0
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
                    tt=rv.rvs(1)
                    #tt=np.random.uniform(0,10)
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
        print(irun,time+1,I_tot)
        for ix in range(l):
            for iy in range(l):
                kk=float(len(I[(ix,iy)]))/float(len(agents[(ix,iy)]))
                i_plot[ix][iy]=kk
                epi_curves[ix][iy].append(kk)
                if kk > peak_height[irun][ix][iy]:
                    peak_height[irun][ix][iy]=kk
                    peak_time[irun][ix][iy]=time+1
    print(Nruns-irun,'end')
        #fig=plt.figure()
        ##plt.subplot(221,title='$P$ only space')
        #plt.imshow(i_plot,vmin=0, vmax=0.5, cmap='jet')
        #fig.savefig('a_%.3i.png' % (time+1),bbox_inches='tight')
        ##plt.show()
        #plt.close()

    #os.system('mencoder mf://a_*.png -mf w=800:h=600:fps=10:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o latent_times_l_'+str(l)+'_Tl_'+str(T_l)+'_'+str(m)+'_'+str(irun)+'.avi')

    #os.system('rm a_*.png')


    #fig=plt.figure()
    ##plt.subplot(221,title='$P$ only space')
    #plt.imshow(peak_height[irun],vmin=0, vmax=1, cmap='jet')
    #fig.savefig('peak_height_l_'+str(l)+'_Tl_'+str(T_l)+'_'+str(m)+'_'+str(irun)+'.png',bbox_inches='tight')
    ##plt.show()
    #plt.close()


    #fig=plt.figure()
    ##plt.subplot(221,title='$P$ only space')
    #plt.imshow(peak_time[irun],vmin=0,vmax=end_time,cmap='jet')
    #fig.savefig('peak_time_l_'+str(l)+'_Tl_'+str(T_l)+'_'+str(m)+'_'+str(irun)+'.png',bbox_inches='tight')
    ##plt.show()
    #plt.close()

    #av_peak_time=np.zeros((l))
    #norm_av_peak_time=np.zeros((l))
    #a=int(l/2)
    #for ix in range(l):
        #for iy in range(l):
            #d=abs(ix-a)+abs(iy-a)
            #av_peak_time[d]+=float(peak_time[irun][ix][iy])
            #norm_av_peak_time[d]+=1.0

    #for i in range(l):
        ##print(av_peak_time[i],norm_av_peak_time[i],av_peak_time[i]/norm_av_peak_time[i])
        #av_peak_time[i]=av_peak_time[i]/norm_av_peak_time[i]

    #fig=plt.figure()
    ##plt.subplot(221,title='$P$ only space')
    #plt.plot(av_peak_time)
    #fig.savefig('peak_time_d_l_'+str(l)+'_Tl_'+str(T_l)+'_'+str(m)+'_'+str(irun)+'.png',bbox_inches='tight')
    ##plt.show()
    #plt.close()
    
    
#figure av_peak_height 2d

av_peak_height=np.zeros((l,l))

for ix in range(l):
    for iy in range(l):
        for irun in range(Nruns):
            av_peak_height[ix][iy]+=peak_height[irun][ix][iy]/float(Nruns)

fig=plt.figure()
#plt.subplot(221,title='$P$ only space')
plt.imshow(av_peak_height,vmin=0, vmax=1, cmap='jet')
fig.savefig('av_peak_height_l_'+str(l)+'_Tl_'+str(T_l)+'_'+str(m)+'.png',bbox_inches='tight')
#plt.show()
plt.close()

#figure av_peak_time 2d


av_peak_time=np.zeros((l,l))

for ix in range(l):
    for iy in range(l):
        for irun in range(Nruns):
            av_peak_time[ix][iy]+=peak_time[irun][ix][iy]/float(Nruns)

fig=plt.figure()
#plt.subplot(221,title='$P$ only space')
plt.imshow(av_peak_time,vmin=0,vmax=end_time,cmap='jet')
fig.savefig('av_peak_time_l_'+str(l)+'_Tl_'+str(T_l)+'_'+str(m)+'.png',bbox_inches='tight')
#plt.show()
plt.close()

#figure av_peak_time as a function of distance

av_peak_time=np.zeros((l+1)) #!!!!  BE CAREFUL
av2_peak_time=np.zeros((l+1))
norm_av_peak_time=np.zeros((l+1))
a=int(l/2)
for ix in range(l):
    for iy in range(l):
        d=abs(ix-a)+abs(iy-a)
        for irun in range(Nruns):
            kk=float(peak_time[irun][ix][iy])
            av_peak_time[d]+=kk
            av2_peak_time[d]+=kk*kk
            norm_av_peak_time[d]+=1.0

for i in range(l+1):
    print(av_peak_time[i],av2_peak_time[i])
    av_peak_time[i]=av_peak_time[i]/norm_av_peak_time[i]
    av2_peak_time[i]=av2_peak_time[i]/norm_av_peak_time[i]
    av2_peak_time[i]=np.sqrt(abs(av2_peak_time[i]-av_peak_time[i]*av_peak_time[i]))

fig=plt.figure()
#plt.subplot(221,title='$P$ only space')
#plt.ylim(0,max(av_peak_time))
plt.errorbar(np.arange(l+1),av_peak_time,yerr=av2_peak_time)
fig.savefig('av_peak_time_d_l_'+str(l)+'_Tl_'+str(T_l)+'_'+str(m)+'.png',bbox_inches='tight')
#plt.show()
plt.close()



###fig=plt.figure()

###for nplot in range(N_loc):
    ###plt.subplot(l,l,nplot+1)
    ###ix=nplot-l*int(nplot/l)
    ###iy=int(nplot/l)
    ###plt.plot(epi_curves[ix][iy])
    ###frame=plt.gca()
    ###frame.axes.get_xaxis().set_visible(False)
    ###frame.axes.get_yaxis().set_visible(False)

####plt.show()
###fig.savefig('epi_curves_'+str(l)+'_Tl_'+str(T_l)+'_'+str(m)+'.png',bbox_inches='tight')
###plt.close()

#b) network minimal model for comparison of basic mechanism

##from numpy import linalg as LA
##A=nx.adjacency_matrix(G)

##print(A)
##print(type(A))
##A=A.todense()
##A=(1.0-alpha_step)*A/4.0
##print(A/4.0)
##for i in range(N_loc):
    ##A[i][i]=alpha
##print(type(A))
##B=LA.matrix_power(A,T_l)
##print(B)
##print(type(B))



