from pgmpy.independencies import IndependenceAssertion, Independencies
from pgmpy.factors import JointProbabilityDistribution as JPD


assertion1 = IndependenceAssertion('X', 'Y')
independencies = Independencies(assertion1)
rv_names = ["Quality", "Cost", "Location"]
rv_nStates = [3, 2, 2]
rv_probabilitiesTable = [1./12]*12
jpd = JPD(rv_names, rv_nStates, rv_probabilitiesTable)
print(jpd)
