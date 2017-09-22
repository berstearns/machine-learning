import random
import numpy as np
import matplotlib as mpl
from scipy import random, linalg, stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
def plot_state(data,params):# {{{
    xparam,yparam = list(zip(*[params[idx] for idx in range(len(params)) if idx % 2==0]))
    N,nDim = len(data), len(data[0])
    x,y = list( zip(*[ list(e) for e in data]))
    rgb = [ [1,0,1] for _ in data]
    print(rgb)
    plt.scatter(x,y,facecolors=rgb)
    plt.scatter(xparam,yparam)
    plt.show()# }}}
def plot_softState(data,params,data_probs):# {{{
    xparam,yparam = list(zip(*[params[idx] for idx in range(len(params)) if idx % 2==0]))
    N,nDim = len(data), len(data[0])
    x,y = list( zip(*[ list(e) for e in data]))
    rgb = [ [1*obs_prob[0],1*obs_prob[1],0] for obs_prob in data_probs]
    params_rgb = [ [1,0,0],[0,1,0]]#params_rgb = [ [0,0,0] for _ in params]
    fig=plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)#221 for two figs
    ax.scatter(x,y,facecolors=rgb)
    ax.scatter(xparam,yparam,facecolors='none',edgecolors=params_rgb)
    '''ax2 = fig.add_subplot(222)
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=-40, vmax=180)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    orientation='vertical',
                                    norm=norm,
                                    ticks=[-40,180]
                                    )'''
    plt.show()# }}}
def comparing_parameters(params1,params2):# {{{
    pc1,pc2,p1,p2 = params1
    sample1 = []
    for i in range(1000):
        obs_generator = np.random.random()
        if obs_generator > pc1:
            sample1.append( np.random.binomial(1,p2))
        else:
            sample1.append( np.random.binomial(1,p1))

    pc1,pc2,p1,p2 = params2
    sample2 = []
    for i in range(1000):
        obs_generator = np.random.random()
        if obs_generator > pc1:
            sample2.append( np.random.binomial(1,p2))
        else:
            sample2.append( np.random.binomial(1,p1))# }}}

def generate_data(n,numDim=2):# {{{
    lmbda1,lmbda2 = 30*random.rand(1,numDim)[0],30*random.rand(1,numDim)[0]
    pi1 = np.random.random()
    pi2 = 1 - pi1

    original_params = [list(lmbda1),pi1,list(lmbda2),pi2]
    data = []
    for i in range(n):
        obs_generator = np.random.random()
        if obs_generator > pi1:
            data.append(np.random.poisson(lmbda2))
        else:
            data.append(np.random.poisson(lmbda1))
    return data,original_params# }}}

def bernoulli_probability(p,x_value):# {{{
    return (p**x_value) * ((1-p)**(1-x_value))# }}}

def poisson_probability(lmbda,xi_value):# {{{
        prob = 1
        for k in range(len(lmbda)):
            probFunc = stats.poisson(lmbda[k])
            prob *= probFunc.pmf(xi_value[k])
        return prob# }}}
'''# {{{ NOT BEING USED YET
    THIS WILL BE USED
    FOR THE GENERICAL
    PROBABILITY FORM of EXP FAMILY
def div(xi_value,p):
       return (xi_value * np.log( xi_value/p ) ) \
               + (1-xi_value)*np.log( (1-xi_value)/(1-p) )# }}}'''
def Idiv(xi_value,lmbda):
    ''' I-divergence '''
    return (xi_value*np.log(xi_value/lmbda))-xi_value + lmbda

def initParameters(numDim):# {{{
    lmbda1,lmbda2 = 30*random.rand(1,numDim)[0],30*random.rand(1,numDim)[0]
    pi1 = np.random.random()
    pi2 = 1 - pi1
    return lmbda1,lmbda2,pi1,pi2# }}}

def sampleFromProbList(lst):
    print(lst)
    return [1,1]

def initParameterspp(numDim):# {{{
    lmbda1 = data[np.random.randint(0,len(data))]
    p_xs = []
    for obs in data:
        p_xs.append( Idiv(obs,lmbda1) )
    total_px = sum(p_xs)
    p_xs = [p_x/total_px for p_x in p_xs]
    lmbda2 = sampleFromProbList(p_xs)

    pi1 = np.random.random()
    pi2 = 1 - pi1
    return lmbda1,lmbda2,pi1,pi2# }}}

def softKmeans(data):# {{{
        numDim = len(data[0])
        n_iterations = 0
        max_iterations =10
        ''' Suppose our model is a mixture of 2 berns'''
        init_lmbda1,init_lmbda2,init_pi1,init_pi2,= initParameters(numDim)
        ''' e-projecao q(h|x) == P(h|x) via KL
            usando teorema de bayes temos que
            P(h|x) = [ P(x|h)*P(h) ]/P(x)
        '''

        ''' m-projecao
                    pc_h = (1/n)*sum_{i=1}_{N} [q(h|x_{i})]
                    p_h = sum_{i=1}_{N} [q(h|x_{i})*x_{i}]
                          / sum_{i=1}_{N} [ q(h|x_{i})]
                    cov_h  = { sum_{i=1}_{N} [q(h|x_{i}) * x_{i} * x_{i}_T]  / sum_{i=1}_{N} [q(h|x_{i})]}  - u*u_T
        '''
        print(originalparams)
        print("_"*75)
        while( n_iterations < max_iterations):
            if n_iterations == 0:
                last_lmbda1,last_pi1 = init_lmbda1,init_pi1
                last_lmbda2,last_pi2 = init_lmbda2,init_pi2
            else:
                last_lmbda1,last_pi1 = it_lmbda1,it_pi1
                last_lmbda2,last_pi2 = it_lmbda2,it_pi2
            it_lmbda1_numerador,it_lmbda1_denominador,it_pi1 = 0,0,0
            it_lmbda2_numerador,it_lmbda2_denominador,it_pi2 = 0,0,0
            N = len(data)
            data_probs = []
            for xi_value in data:
                #Idivsonizationn = last_pc1*np.exp(-div(xi_value,last_p1)) + last_pc2*np.exp(-div(xi_value,last_p2))
                #last_pc1*np.exp(-div(xi_value,last_p1)  )/Idivsonization'
                #last_pc2*np.exp(-div(xi_value,last_p2)  )/Idivsonization
                q_1_xi = ( poisson_probability(last_lmbda1,xi_value) * last_pi1 )/ \
                                              ( poisson_probability(last_lmbda1,xi_value) * last_pi1+ \
                                                   poisson_probability(last_lmbda2,xi_value) * last_pi2)
                q_2_xi = ( poisson_probability(last_lmbda2,xi_value) * last_pi2 )/ \
                            ( poisson_probability(last_lmbda1,xi_value) * last_pi1+ \
                                 poisson_probability(last_lmbda2,xi_value) * last_pi2)
                data_probs.append( [q_2_xi,q_1_xi] )
                it_pi1 += q_1_xi
                it_pi2 += q_2_xi
                it_lmbda1_numerador += (q_1_xi * xi_value)
                it_lmbda1_denominador += q_1_xi
                it_lmbda2_numerador += (q_2_xi * xi_value)
                it_lmbda2_denominador += q_2_xi
            it_pi1 = it_pi1/N
            it_pi2 = it_pi2/N
            it_lmbda1 = it_lmbda1_numerador/it_lmbda1_denominador
            it_lmbda2 = it_lmbda2_numerador/it_lmbda2_denominador
            print(it_pi1,it_pi2,it_lmbda1,it_lmbda2)
            plot_softState(data,originalparams,data_probs)
            n_iterations += 1
            if n_iterations == 100:
               pass
        return (it_pi1,it_pi2,it_lmbda1,it_lmbda2)# }}}
if __name__ == "__main__":
    n,numDim = 20,2
    data,originalparams = generate_data(n=n,numDim=numDim)
    #initParameterspp(numDim)
    emparams = softKmeans(data)
