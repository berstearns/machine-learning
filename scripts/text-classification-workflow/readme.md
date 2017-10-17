# A Complete workflow for a text classification task
*   Raw Data preprocessing ( heavily using NLP)
*   Word embedding (transforming text to a numerical representation)
*   Machine Learning ( model selection and optmization 


# Tasks that would be nice , but didn't have enough time
*   Word embedding (transforming text to a numerical representation)
	- Applying glove(from google) to create a vector representation for words 
	- Apply clustering on the glove representation space for clustering words
	- Substitute words for cluster label , reducing vocabulary size
	
*   Machine Learning ( model selection and optmization , ensemble and decision making)
	- Apply more advanced and specific optimization algorithm instead of grid search (e.g. PSO)
	- Ensemble together many xgb models with different parameters that had good results from the optimization process
	- Create a Decision Model that give weight to each model (e.g. linear combination or major voting)
	
* Deploy and evaluation
	- apply different metrics other than accuracy
	- pickle final model