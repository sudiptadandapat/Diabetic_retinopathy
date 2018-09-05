#*************************************#******************************************************#*****************************************
#                                     #                                                      #
#                                     #      Diabetic Retinopathy Classification using       #  
#                                     #            Machine Learning Algorithms               #
#                                     #                                                      #
#*************************************#******************************************************#****************************************




################################   Importing all the python libraries  #####################################################################
print("Importing python libraries.........")
print('\n\n\n')
from scipy import misc
from PIL import Image
from skimage import exposure
from sklearn import svm
from sklearn.metrics import accuracy_score
import scipy
from math import sqrt,pi
from numpy import exp
from matplotlib import pyplot as plt
import numpy as np
import glob
import matplotlib.pyplot as pltss
import cv2
from matplotlib import cm
import pandas as pd
from math import pi, sqrt
import pywt
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import mean_absolute_error
import pylab as pl
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier





######################################################



immatrix=[]
imm_dwt = []
dim=(768,576)

######################  Loading all the images and pre-process ######################3
print("Loading images for the Diabetic retinopathy Classification Model...........")
print('\n\n\n')
for i in range(1,101):
    img_pt =''
    if i < 51:
        img_pt = img_pt + "normal" + str(i) + ".jpg"
    else:
        img_pt = img_pt + "image0" + str(i-40)+ ".png"


    img = cv2.imread(img_pt)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_gray)
    immatrix.append(np.array(equ).flatten())

for equ in immatrix:
    equ = equ.reshape((576,768))
    coeffs = pywt.dwt2(equ, 'haar')
    equ2 = pywt.idwt2(coeffs, 'haar')
    imm_dwt.append(np.array(equ2).flatten())


 
######## Pre-processing done ############### 

def createMatchedFilterBank():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 6, theta,12, 0.37, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def applyFilters(im, kernels):
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)

bank_gf = createMatchedFilterBank()
imm_gauss2 = []
for equ2 in imm_dwt:
    equ2 = equ2.reshape((576,768))
    equ3 = applyFilters(equ2,bank_gf)
    imm_gauss2.append(np.array(equ3).flatten())


################# Feature Extraction ##############################################


e_ = equ3
np.shape(e_)
e_=e_.reshape((-1,3))

imm_kmean = []
for equ3 in imm_gauss2:
    img = equ3.reshape((576,768))
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    k=cv2.KMEANS_PP_CENTERS
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res3=cv2.subtract(255,res2)
    imm_kmean.append(np.array(res3).flatten())

np.shape(imm_kmean)



######################### Training the model ###########################




clf = SVC()
print("Training the Diabetic retinopathy Classification Model with SVM classifier...........")
print('\n\n\n')
Y = np.ones(100)


for i in range(0,50):
    Y[i]=0

clf.fit(imm_kmean, Y)
y_pred = clf.predict(imm_kmean)

k=[1,4,7,9,12,15,16,17,18,19,20,21,22,27,29,32,35,36,37,38,39,40,42,43,46,48,49,52,53,54,55,56,57,58,59,63,64,65,66,67,68,69,70,74,76,77,79,81,84,86,87,89,91,93,92,95,97,99,100]
k = k-np.ones(len(k))
k =[int(x) for x in k]


imm_train = []
y_train = []
for i in k:
    imm_train.append(imm_kmean[i])
    y_train.append(Y[i])



clf.fit(imm_train, y_train)
y_pred = clf.predict(imm_kmean)




t_mat=[]
t_im_unpre = []
t_imm_dwt = []
t_imm_gauss2 = []


for a in range(1,2):
    path=input("Enter the name of the image of current directory --->    "    )
    print(path)
    print('\n')
    test = cv2.imread(path)
    t_resized = cv2.resize(test, dim, interpolation = cv2.INTER_AREA)
    t_img_gray = cv2.cvtColor(t_resized, cv2.COLOR_BGR2GRAY)
    t_equ = cv2.equalizeHist(t_img_gray)
    t_equ = t_equ.reshape((576,768))
    t_coeffs = pywt.dwt2(t_equ, 'haar')
    t_equ2 = pywt.idwt2(t_coeffs, 'haar')
    
    bank_gf = createMatchedFilterBank()
    t_equ2 = t_equ2.reshape((576,768))
    t_equ3 = applyFilters(t_equ2,bank_gf)


    imm_test = []

    t_img = t_equ3.reshape((576,768))
    t_Z = t_img.reshape((-1,3))
    # convert to np.float32
    t_Z = np.float32(t_Z)
    t_k=cv2.KMEANS_PP_CENTERS
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    t_K = 2
    t_ret,t_label,t_center=cv2.kmeans(t_Z,t_K,None,criteria,10,t_k)
    # Now convert back into uint8, and make original image
    t_center = np.uint8(t_center)
    t_res = t_center[t_label.flatten()]
    t_res2 = t_res.reshape((t_img.shape))
    t_res3=cv2.subtract(255,t_res2)
    imm_test.append(np.array(t_res3).flatten())


    dr = clf.predict(imm_test)
    c=int(dr)
    if c==0:
        print("  RESULT ---->  Normal retina ")
    else:
        print("  RESULT ---->  Sign of Diabetic retinopathy present in retina ")
    
    print('\n\n\n')



print('\n\n\n')
print("******Perfomance analysis of the Diabetic retinopathy Classification Model with SVM classifier******")
print('\n\n\n')



#######################  Perfamance Analysis of SVM Classifier  #########################################################





print('**Confusion Matrix:**')
cm1=confusion_matrix(Y,y_pred)
print(cm1)
print('\n\n')
print("TP =",cm1[0,0],"FP =",cm1[0,1])
print("FN =",cm1[1,0],"TN =",cm1[1,1])
tp=cm1[0,0]
fp=cm1[0,1]
fn=cm1[1,0]
tn=cm1[1,1]
print('\n\n')


print(classification_report(Y,y_pred))
print('\n\n')


print("***Accuracy ===> ",float((tn+tp)/(fn+fp+1+tn+tp)))
print('\n\n')


print("***Sensitivity ===> ",float(tp/(tp+fn+1)))
print('\n\n')


print("***Specificity ===> ",float(tn/(tn+fp+1)))
print('\n\n')


print("***PPV ===> ",float(tp/(tp+fp+1)))
print('\n\n')


print('***Mean Absolute Error ===>',mean_absolute_error(Y, y_pred))
print('\n\n')


def rmse(y_pred, y_roc):
	return np.sqrt(((y_pred - y_roc) ** 2).mean())
y_roc = np.array(Y)
rms=rmse(y_pred,y_roc)
print('***Root Mean Square Error ===>',rms)
print('\n\n')


y_roc = np.array(Y)
fpr, tpr, thresholds = roc_curve(y_roc, y_pred)
roc_auc = auc(fpr, tpr)
print("***Area under the ROC curve ===> %f" % roc_auc)
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc="lower right")
pl.show()
print('\n\n\n')




print("******Perfomance analysis of the Diabetic retinopathy Classification Model with KNN classifier******")
print('\n\n')



neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(imm_train, y_train)
y_pred2=neigh.predict(imm_kmean)


#######################  Perfamance Analysis of KNN Classifier  #########################################################


print('***Confusion Matrix:***')
cm2=confusion_matrix(Y,y_pred2)
print(cm2)
print('\n')
print("TP =",cm2[0,0],"FP =",cm2[0,1])
print("FN =",cm2[1,0],"TN =",cm2[1,1])
tp=cm2[0,0]
fp=cm2[0,1]
fn=cm2[1,0]
tn=cm2[1,1]
print('\n\n')


print(classification_report(Y,y_pred2))
print('\n\n')


print("***Accuracy ===> ",float((tn+tp)/(fn+fp+1+tn+tp)))
print('\n\n')


print("***Sensitivity ===> ",float(tp/(tp+fn+1)))
print('\n\n')


print("***Specificity ===> ",float(tn/(tn+fp+1)))
print('\n\n')


print("***PPV ===> ",float(tp/(tp+fp+1)))
print('\n\n')



print('***Mean Absolute Error ===>',mean_absolute_error(Y, y_pred2))
print('\n\n')


def rmse(y_pred2, y_roc):
	return np.sqrt(((y_pred2 - y_roc) ** 2).mean())
y_roc = np.array(Y)
rms=rmse(y_pred2,y_roc)
print('***Root Mean Square Error ===>',rms)
print('\n\n')



y_roc = np.array(Y)
fpr, tpr, thresholds = roc_curve(y_roc, y_pred2)
roc_auc = auc(fpr, tpr)
print("***Area under the ROC curve ===> %f" % roc_auc)
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc="lower right")
pl.show()



