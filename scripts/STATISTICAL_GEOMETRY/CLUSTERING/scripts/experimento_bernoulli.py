import numpy as np

def comparing_parameters(params1,params2):
    pc1,pc2,p1,p2 = params1
    sample1 = []
    for i in range(1000):
        obs_generator = np.random.random()
        if obs_generator > pc1:
            sample1.append( np.random.binomial(1,p2)
        else:
            sample1.append( np.random.binomial(1,p1)

    pc1,pc2,p1,p2 = params2
    sample2 = []
    for i in range(1000):
        obs_generator = np.random.random()
        if obs_generator > pc1;
            sample2.append( np.random.binomial(1,p2)
        else:
            sample2.append( np.random.binomial(1,p1)

def generate_data(n):
    data = []
    p1,p2 = np.random.random(),np.random.random()
    pc1 = np.random.random()
    pc2 = 1- pc1
    original_params = (pc1,pc2,p1,p2)
    print(pc1,pc2,p1,p2)
    print("*"*100)
    for i in range(n):
        obs_generator = np.random.random()
        if obs_generator > pc1:
            data.append(np.random.binomial(1,p2))
        else:
            data.append(np.random.binomial(1,p1))
    return data,original_params

def bernoulli_probability(p,x_value):
    return (p**x_value) * ((1-p)**(1-x_value))

def div(xi_value,p):
       return (xi_value * np.log( xi_value/p ) ) \
               + (1-xi_value)*np.log( (1-xi_value)/(1-p) )

def initParameters():
    p1,p2 = np.random.random(),np.random.random()
    pc1 = np.random.random()
    pc2 = 1 - pc1
    return p1,p2,pc1,pc2

def softKmeans(data):
        n_iterations = 0
        max_iterations = 100
        ''' Suppose our model is a mixture of 2 berns'''
        init_p1,init_p2,init_pc1,init_pc2 = initParameters()
        ''' e-projecao q(h|x) == P(h|x) via KL
            usando teorema de bayes temos que
            P(h|x) = [ P(x|h)*P(h) ]/P(x)
        '''

        ''' m-projecao
                    pc_h = (1/n)*sum_{i=1}_{N} [q(h|x_{i})]
                    p_h = sum_{i=1}_{N} [q(h|x_{i})*x_{i}]
                          / sum_{i=1}_{N} [ q(h|x_{i})]
        '''
        while( n_iterations < max_iterations):
            if n_iterations == 0:
                last_p1,last_p2,last_pc1,last_pc2 = init_p1,init_p2,init_pc1,init_pc2
            else:
                last_p1,last_p2,last_pc1,last_pc2 = it_pc_1 ,it_pc_2,it_p1,it_p2
            it_pc_1 ,it_pc_2,it_p1_denominador,it_p1_numerador,it_p2_denominador,it_p2_numerador  = 0,0,0,0,0,0
            N = len(data)
            for xi_value in data:
                #div_normalizationn = last_pc1*np.exp(-div(xi_value,last_p1)) + last_pc2*np.exp(-div(xi_value,last_p2))
                #last_pc1*np.exp(-div(xi_value,last_p1)  )/div_normalization'
                #last_pc2*np.exp(-div(xi_value,last_p2)  )/div_normalization
                q_1_xi = ( bernoulli_probability(last_p1,xi_value) * last_pc1 )/ \
                                              (( bernoulli_probability(last_p1,xi_value) * last_pc1)+ \
                                              ( bernoulli_probability(last_p2,xi_value) * last_pc2))
                q_2_xi = ( bernoulli_probability(last_p2,xi_value) * last_pc2 )/ \
                          (( bernoulli_probability(last_p1,xi_value) * last_pc1)+ \
                            ( bernoulli_probability(last_p2,xi_value) * last_pc2))

                it_pc_1 += q_1_xi

                it_pc_2 += q_2_xi
                it_p1_numerador += (q_1_xi * xi_value)
                it_p1_denominador += q_1_xi
                it_p2_numerador += (q_2_xi * xi_value)
                it_p2_denominador += q_2_xi
            it_pc_1 = it_pc_1/N
            it_pc_2 = it_pc_2/N
            it_p1 = it_p1_numerador/it_p1_denominador
            it_p2 = it_p2_numerador/it_p2_denominador
            print(it_pc_1,it_pc_2,it_p1,it_p2)
            n_iterations += 1
            if n_iterations == 100:
               pass
        return (it_pc_1,it_pc_2,it_p1,it_p2)

data,originalparams = generate_data(10)
emparams = softKmeans(data)
