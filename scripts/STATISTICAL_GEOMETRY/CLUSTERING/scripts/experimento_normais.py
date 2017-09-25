import numpy as np
from scipy import random, linalg, stats

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
    sig1 = random.rand(numDim,numDim)
    sig1 = np.dot(sig1,sig1.transpose())
    sig2 = random.rand(numDim,numDim)
    sig2 = np.dot(sig2,sig2.transpose())
    for i in range(len(sig2)):
        sig1[i][i] = 1
        sig2[i][i] = 1
    u1,u2 = random.rand(1,numDim)[0],random.rand(1,numDim)[0]
    pi1 = np.random.random()
    pi2 = 1 - pi1

    original_params = [u1,sig1,pi1,u2,sig2,pi2]
    data = []
    for i in range(n):
        obs_generator = np.random.random()
        if obs_generator > pi1:
            data.append(np.random.multivariate_normal(u1,sig1))
        else:
            data.append(np.random.multivariate_normal(u2,sig2))
    return data,original_params# }}}

def bernoulli_probability(p,x_value):# {{{
    return (p**x_value) * ((1-p)**(1-x_value))# }}}

def normal_probability(mean,cov,xi_value):
    try:
        probFunc = stats.multivariate_normal(mean=mean,cov=cov)
        return probFunc.pdf(xi_value)
    except:
        print(cov);exit(0)
def div(xi_value,p):# {{{
       return (xi_value * np.log( xi_value/p ) ) \
               + (1-xi_value)*np.log( (1-xi_value)/(1-p) )# }}}

def initParameters(numDim):# {{{
    sig1 = random.rand(numDim,numDim)
    sig1 = np.dot(sig1,sig1.transpose())
    sig2 = random.rand(numDim,numDim)
    sig2 = np.dot(sig2,sig2.transpose())
    for i in range(len(sig2)):
        sig1[i][i] = 1
        sig2[i][i] = 1
    u1,u2 = random.rand(1,numDim)[0],random.rand(1,numDim)[0]
    pi1 = np.random.random()
    pi2 = 1 - pi1
    return u1,u2,sig1,sig2,pi1,pi2# }}}

def softKmeans(data):# {{{
        numDim = len(data[0])
        n_iterations = 0
        max_iterations = 100
        ''' Suppose our model is a mixture of 2 berns'''
        init_u1,init_u2,init_sig1,init_sig2,init_pi1,init_pi2= initParameters(numDim)
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
        while( n_iterations < max_iterations):
                print("+"*50)
                print('n iterations',n_iterations)
                print("+"*50)
                if n_iterations == 0:
                    last_u1,last_sig1,last_pi1 = init_u1,init_sig1,init_pi1
                    last_u2,last_sig2,last_pi2 = init_u2,init_sig2,init_pi2
                else:
                    last_u1,last_sig1,last_pi1 = it_u1,last_sig1,it_pi1
                    last_u2,last_sig2,last_pi2 = it_u2,last_sig2,it_pi2
                it_u1_numerador,it_u1_denominador,it_sig1,it_pi1 = 0,0,0,0
                it_u2_numerador,it_u2_denominador,it_sig2,it_pi2 = 0,0,0,0
                N = len(data)
                for xi_value in data:
                    #div_normalizationn = last_pc1*np.exp(-div(xi_value,last_p1)) + last_pc2*np.exp(-div(xi_value,last_p2))
                    #last_pc1*np.exp(-div(xi_value,last_p1)  )/div_normalization'
                    #last_pc2*np.exp(-div(xi_value,last_p2)  )/div_normalization
                    q_1_xi = ( normal_probability(last_u1,last_sig1,xi_value) * last_pi1 )/ \
                                                  ( normal_probability(last_u1,last_sig1,xi_value) * last_pi1+ \
                                                       normal_probability(last_u2,last_sig2,xi_value) * last_pi2)

                    q_2_xi = ( normal_probability(last_u2,last_sig2,xi_value) * last_pi2 )/ \
                                ( normal_probability(last_u1,last_sig1,xi_value) * last_pi1+ \
                                     normal_probability(last_u2,last_sig2,xi_value) * last_pi2)

                    it_pi1 += q_1_xi
                    it_pi2 += q_2_xi
                    it_u1_numerador += (q_1_xi * xi_value)
                    it_u1_denominador += q_1_xi
                    it_u2_numerador += (q_2_xi * xi_value)
                    it_u2_denominador += q_2_xi
                it_pi1 = it_pi1/N
                it_pi2 = it_pi2/N
                it_u1 = it_u1_numerador/it_u1_denominador
                it_u2 = it_u2_numerador/it_u2_denominador
                print(it_pi1,it_pi2,it_u1,it_u2)
                n_iterations += 1
                if n_iterations == 100:
                   pass
        return #(it_pc_1,it_pc_2,it_p1,it_p2)# }}}
if __name__ == "__main__":
    n,numDim = 10,2
    data,originalparams = generate_data(n=n,numDim=numDim)
    emparams = softKmeans(data)
    print(originalparams[2],originalparams[5],originalparams[0],originalparams[3])
