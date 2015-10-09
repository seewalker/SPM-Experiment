import pickle
import exp_correct
import random
a = pickle.load(open("old_testing_coded","rb"))
fst = a['dragonfly'][0]
count = 0
for category in a.keys():
    for img in a[category]:
        if (random.random() < 0.1):
            print(str(exp_correct.similarity(fst,img,2,200,201)))
            count += 1
print(str(count))
