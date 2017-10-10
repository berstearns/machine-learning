import Discretizations
import LIME

Discretizations.DD_func()
raw_data = [ [1,2,3],[4,5,6] ]

preProcessed_data = preproc(raw_data)

predictive_model = model.fit(preProcessed_data)

test_data = [ [7,8,9],[10,11,12]]
for test_obs in test_data:
    LIME.fit(test_obs)
    LIME.explain()
