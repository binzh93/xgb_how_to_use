
f1 = open('F:/horseColicTraining2.txt', 'r')
f2 = open('F:/horseColicTest2.txt', 'r')
horseTrainData = []
horseTestData = []

for line in f1:
	tmp = line.strip().split('\t')
	horseTrainData.append(tmp)
f1.close()

for line in f2:
	tmp = line.strip().split('\t')
	horseTestData.append(tmp)
f2.close()

y1len = len(horseTrainData)
x1len = len(horseTrainData[0])
print y1len,x1len

y2len = len(horseTestData)
x2len = len(horseTestData[0])
print y2len,x2len



f1 = open('F:/horseTrain.csv', 'w')
f1.write('f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,label\n')
for i in xrange(y1len):
	for j in xrange(x1len):
		if j == (x1len-1):
			f1.write(str(horseTrainData[i][j])+'\n')
		else:
			f1.write(str(horseTrainData[i][j]+','))	
f1.close()



f2 = open('F:/horseTest.csv', 'w')
f2.write('f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,label\n')
for i in xrange(y2len):
	for j in xrange(x2len):
		if j == (x2len-1):
			f2.write(str(horseTrainData[i][j])+'\n')
		else:
			f2.write(str(horseTrainData[i][j]+','))	
f2.close()
	