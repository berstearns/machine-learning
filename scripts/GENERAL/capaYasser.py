import numpy as np
from sklearn.linear_model import LinearRegression
def geradorModeloPolinomialLegendre(n,q,noise_proportion):
   '''Pro eric implementar'''
   pass

def geradorModeloPolinomial(n,q,noise_proportion):
   '''Gera modelo y =  Sum{alfa_p * x_p} + noise'''
   x = np.random.normal(10 , 1, n)
   alfas =	 [np.random.random() for _ in range(q+1)]
   X = []
   for pol_degree in range(q+1):
   		X.append(alfas[pol_degree] * x**(pol_degree))
   
   feature_part = np.zeros(n)
   for x in X:
   		feature_part += x
   
   noise = noise_proportion * feature_part
   y = feature_part + noise
   return x,y


for noise_proportion in np.arange(0,3.01,0.5):
	for n in [4]:
		for complexity in [2]:
			x,y = geradorModeloPolinomial(n,complexity,noise_proportion)
			model = LinearRegression()
			# hipotese pol grau 2
			X_2 = []
			for pol_degree in range(3):
				X_2.append(x**(pol_degree))
			X_2 = np.array(X_2).T 
			model.fit(X_2,y)
			
			# hipotese pol grau 10
			X_10 = []
			for pol_degree in range(11):
				X_10.append(x**(pol_degree))
			X_10 = np.array(X_10).T
			modelo.fit(X_10,y)
			