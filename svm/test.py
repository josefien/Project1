import random
import numpy as np
from sklearn.cross_validation import StratifiedKFold

labels = ['Boterhammen','Aardappelen','Chips','Cornflakes','Frietjes','Fruit','Gebak','Hamburger','IJs','Koekjes','Muffin','Pasta','Pizza','Rijstgerecht','Salade','Snoep','Snoepreep','Soep','Yoghurt']
random.choice(labels)
f = open('testfile.txt','w')
iterations = 500
for i in xrange(iterations):
	string = str(random.choice(labels))
	if not i == iterations-1:
		string = string + '\n'
	f.write(str(random.choice(labels) + '\n'))
f.close()

samples = np.random.rand(1,4)
targets = []
for i in xrange(iterations):
	features = np.random.rand(1,4)
	samples = np.vstack((samples,features))
	targets.append(str(random.choice(labels)))
skf = StratifiedKFold(targets,3)
for train, test in skf:
	print("%s %s" % (train,test))