from __future__ import division
from sklearn import svm
from sklearn.cluster import KMeans
from skimage.feature import (ORB)
from skimage.io import imread
from collections import OrderedDict
import pandas as pd
import numpy as np
import os
import math
import time
import random
import pickle

def similarity(X,Y,L,M,neutral):
    count_matrix = np.zeros((L+1,M))
    for l in range(L+1):
        xpatch_w, ypatch_w = math.floor(X.shape[0] / (2 ** l)), math.floor(Y.shape[0] / (2 ** l))
        xpatch_h, ypatch_h = math.floor(X.shape[1] / (2 ** l)), math.floor(Y.shape[1] / (2 ** l))
        for i in range(2 ** l):
            for j in range(2 ** l):
                x_patch = X[(i * xpatch_w):((i+1) * xpatch_w),(j * xpatch_h):((j+1) * xpatch_h)]
                y_patch = Y[(i * ypatch_w):((i+1) * ypatch_w),(j * ypatch_h):((j+1) * ypatch_h)]
                x_counts, y_counts = np.zeros(M), np.zeros(M) #this needs to be in this loop in order for layers to really "matter".
                for s in range(x_patch.shape[0]):
                    for t in range(x_patch.shape[1]):
                        if x_patch[s][t] != neutral:
                            x_counts[x_patch[s][t]] += 1
                for s in range(y_patch.shape[0]):
                    for t in range(y_patch.shape[1]):
                        if y_patch[s][t] != neutral:
                            y_counts[y_patch[s][t]] += 1
        count_matrix[l] = [min(x_counts[channel],y_counts[channel]) for channel in range(M)]
    count = 0
    for channel in range(M):
        counts = count_matrix[:,channel]
        count += counts[L] #high 'L' value meaning finer grid.
        for l in range(L):
            weight = 1/(2 ** (L - l))
            count += weight * (counts[l] - counts[l+1])
    return count 

class Experiment:
    '''
    Everything here is an implementation of 'Beyond bags of features: spatial pyramid matching for recognizing natural scene categories' by Lazebnik.
    featurefn expected to have a certain skimage interface providing a constructor whose instances have a 'detect_and_extract' method.

    As in the paper...
      For each category, 30 randomly selected images are used for training. The remainder are used for testing, with a cap at 50. Because some
      categories contain not much more than 30, not many images are used for testing on these categories.
    '''
    def __init__(self, data, M, L, featurefn, result_dir, precomputed=None):
        self.L, self.M, self.featurefn = L, M, featurefn
        training_images, testing_images = OrderedDict(), OrderedDict()
        self.num_train, self.num_test = 0, 0
        self.neutral_cell = (self.M + 1)
        self.result_dir = result_dir
        for (dirpath, _, imgnames) in os.walk(data):
            parts = dirpath.split("/")
            if len(parts) == 2:
                training_images[parts[1]], testing_images[parts[1]] = [], []
                train_group = random.sample(imgnames,30)
                test_group = set(imgnames) - set(train_group)
                if len(test_group) > 50:
                    test_group = random.sample(test_group, 50)
                for imgname in train_group:
                    training_images[parts[1]].append(imread(dirpath + "/" + imgname, as_grey=True))
                    self.num_train += 1
                for imgname in test_group:
                    testing_images[parts[1]].append(imread(dirpath + "/" + imgname, as_grey=True))
                    self.num_test += 1
                pickle.dump(train_group,open("{}/{}_train_sample".format(result_dir,parts[1]), "wb"))
                pickle.dump(test_group,open("{}/{}_test_sample".format(result_dir,parts[1]), "wb"))
        print("Finished loading images")
        self.categories = list(training_images.keys())
        pickle.dump(self.categories, open("{}/categories".format(result_dir),"wb"))
        training_descriptor_info = self.descriptor_info(training_images)
        testing_descriptor_info = self.descriptor_info(testing_images)
        print("Finished extracting features")
        if os.path.exists(result_dir + "/codebook"):
            codebook = pickle.load(open(result_dir + "/codebook", "rb"))
        else:
            codebook = self.form_codebook(training_descriptor_info)
            pickle.dump(codebook, open("{}/codebook".format(result_dir),"wb"))
        self.training_coded = self.coded_images(codebook, training_images, training_descriptor_info)
        self.testing_coded = self.coded_images(codebook, testing_images, testing_descriptor_info)
        print("Finished making coded images")
        del(training_images)
        del(testing_images)
        del(codebook)
        del(training_descriptor_info)
        del(testing_descriptor_info)
        self.run(precomputed)
    def descriptor_info(self,source):
        info = OrderedDict()
        for (category, samples) in source.items():
            info[category] = []
            for sample in samples:
                extractor = self.featurefn(n_keypoints=2500)
                extractor.detect_and_extract(sample)
                info[category].append((extractor.keypoints, extractor.descriptors))
        return info
    def coded_images(self,codebook,img_source,descriptor_source):
        described_imgs = OrderedDict()
        for (category, samples) in descriptor_source.items():
            described_imgs[category] = []
            for (i,sample) in enumerate(samples): #assumption - these are zippable
                keypoints, descriptors = sample
                assert(len(keypoints) == len(descriptors))
                corresponding_img = self.neutral_cell * np.ones(img_source[category][i].shape) 
                for i in range(len(keypoints)):
                    codeword = codebook.predict(descriptors[i]) #assuming this is an 'enum-like' int.
                    corresponding_img[tuple(keypoints[i])] = int(codeword[0])
                described_imgs[category].append(corresponding_img)
        return described_imgs
    def form_codebook(self,training_descriptor_info):
        sample_to_cluster = []
        for (category, samples) in training_descriptor_info.items():
            for (keypoints, descriptors) in samples:
                for descriptor in descriptors:
                    if random.random() < 0.13:
                        sample_to_cluster.append(descriptor)
        clusterer = KMeans(n_clusters=self.M)
        clusterer.fit(sample_to_cluster)
        print("Constructed {} clusters from {} samples".format(str(self.M), str(len(sample_to_cluster))))
        return clusterer
    def precompute(self):
        t0 = time.time()
        gram = np.zeros((self.num_train, self.num_train))
        labels = np.zeros(self.num_train)
        i, j, sample1ID, sample2ID = 0, 0, 0, 0
        for (category1, samples1) in self.training_coded.items():
            print("Starting {} which is number {}, so {} of the way through".format(category1, str(i), (j / (0.5 * self.num_train * (self.num_train + 1)))))
            for sample1 in samples1:
                for (category2, samples2) in self.training_coded.items():
                    for sample2 in samples2:
                        if sample1ID > sample2ID: #avoid recomputation via symmmetry
                            gram[sample1ID][sample2ID] = gram[sample2ID][sample1ID]
                        else:
                            gram[sample1ID][sample2ID] = similarity(sample1,sample2,self.L,self.M,self.neutral_cell)
                            j += 1
                        sample2ID += 1
                labels[sample1ID] = list(self.training_coded.keys()).index(category1)
                sample1ID += 1
                sample2ID = 0
            i += 1
        tf = time.time()
        pickle.dump(gram,open("{}/gram_matrix".format(self.result_dir),"wb"))
        pickle.dump(labels,open("{}/labels".format(self.result_dir),"wb"))
        print("Forming the precomputed kernel took {} seconds, which is {} seconds per image pair".format(str(tf - t0), str((tf - t0) / j)))
        print("Gram matrix is: " + str(gram) + " with shape " + str(gram.shape))
        return gram, labels
    def train(self,precomputed):
        classifier = svm.SVC(kernel="precomputed",C=40.0)
        if precomputed:
            gram, labels = precomputed
        else:
            gram, labels = self.precompute()
        classifier.fit(gram,labels)
        return classifier
    def run(self,precomputed):
        classifier = self.train(precomputed)
        self.confusion = pd.DataFrame(0,index=self.categories,columns=self.categories) #0-initialized.
        t0 = time.time()
        test_kernel, expectations = np.zeros((self.num_test, self.num_train)), []
        sample1ID, sample2ID, progress = 0, 0, 0
        for (category1, samples1) in self.testing_coded.items():
            print("Starting category {} in the testing kernel, so {} done".format(category1, str(progress / (self.num_test * self.num_train))))
            for sample1 in samples1:
                expectations.append(category1)
                for (category2, samples2) in self.training_coded.items():
                    for sample2 in samples2:
                        test_kernel[sample1ID][sample2ID] = similarity(sample1,sample2,self.L,self.M,self.neutral_cell)
                        sample2ID += 1
                        progress += 1
                sample1ID += 1
                sample2ID = 0
        pickle.dump(test_kernel,open("{}/test_kernel".format(self.result_dir),"wb"))
        print("Done forming the test kernel")
        predictions = classifier.predict(test_kernel)
        pickle.dump(predictions,open("{}/predictions".format(self.result_dir),"wb"))
        pickle.dump(expectations,open("{}/expectations".format(self.result_dir),"wb"))
        tf = time.time()
        print("Performing prediction took {} seconds".format(str(tf-t0)))
        for (i,prediction) in enumerate(predictions):
            predicted_category = self.categories[int(prediction)]
            self.confusion[expectations[i]][predicted_category] += 1
        print(self.confusion)
        pickle.dump(self.confusion,open("{}/confusion".format(self.result_dir),"wb"))
        accuracy = np.trace(self.confusion.values) / np.sum(self.confusion.values)
        pickle.dump(accuracy, open("{}/accuracy".format(self.result_dir),"wb"))
        print("Accuracy: " + str(accuracy))

exp = Experiment("101_ObjectCategories", 200, 2, ORB, result_dir="three")
