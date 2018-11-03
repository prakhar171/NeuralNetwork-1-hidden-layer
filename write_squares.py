import random
f1 = open("read.txt","w")
for i in range(600):
	a = random.randint(10,100)
	f1.write(" ")
	f1.write(str(a/100.0))
	f1.write(" ")
	f1.write(str((a**2)/10000.0))
	f1.write('\n')
f1.close()

f1 = open("write.txt","w")
for i in range(600):
	a = random.randint(10,100)
	f1.write(" ")
	f1.write(str(a/100.0))
	f1.write(" ")
	f1.write(str((a**2)/10000.0))
	f1.write('\n')
f1.close()
