f = open('../features/standard_classes.txt','r')
f2 = open('../features/standard_classes_m.txt','w')

for line in f:
	line_s = line.replace('foodimages\\foodimages\\','',1)
	f2.write(line_s)    