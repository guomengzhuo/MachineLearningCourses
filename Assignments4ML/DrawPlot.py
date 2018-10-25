import matplotlib.pyplot as plt
import os
import LeastSquare as ls
import Prediction
import numpy as np
import random
from sklearn.cross_validation import train_test_split

K = 9#degree of the Polynomial
# K =10

#Load data
path = os.getcwd()
samp_x, samp_y, poly_x, poly_y = ls.loaddata(path)

#Calculate Phi
Phi = ls.generate_Phi( K ,samp_x) # 每列是一个sample 每个列中的行是feature (10*50)
# print(Phi.shape)
# Obtain parameters
theta_LS = ls.LS(Phi,samp_y) # LSR
theta_RLS = ls.RLS(Phi,samp_y,0.5) # RLSR
theta_LASSO = ls.LASSO(Phi,samp_y,0.2) # LASSO
theta_RR = ls.RR(Phi, samp_y, K) # RR
mu_estimator, var_estimator = ls.BR(Phi,samp_y,2.5,5) #BR variance = 5

# print(var_estimator)

result_y = []
name = ['LS','RLS','LASSO','RR','BR']
theta = [theta_LS, theta_RLS, theta_LASSO, theta_RR]
for i in range(0,4):
    pred_y = Prediction.predict(poly_x=poly_x, theta=theta[i], poly_para_K=K)
    # plt.plot(poly_x,pred_y)
    result_y.append(np.ravel(np.array(pred_y)))

prob_distribution, pred_mu= Prediction.pref_BR(mu_estimator, var_estimator,poly_para_K = K, poly_x=poly_x)

result_y.append((np.ravel(np.array(pred_mu))))  #result_y is a list, which has 5 sub-list, each sub-list has 100 predicted value of poly_y

# print((result_y[1]))
def drawRegression(result_y,poly_x,theta,name, samp_x, samp_y,pred_mu): #Draw plot, result_y is the result list, poly_x is orginal x, theta are parameter lists
    for i in range(0,len(name)):
        plt.plot(poly_x, result_y[i])
    plt.scatter(samp_x,samp_y,edgecolors='k',color='k',marker='x',s=15)
    # Draw standard deviation
    std_dev = np.std((np.ravel(np.array(pred_mu))))
    meanAddstd_dev = [i + std_dev for i in pred_mu]
    meanSurplusstd_dev = [i - std_dev for i in pred_mu]
    plt.plot(poly_x, meanAddstd_dev, 'r--')
    plt.plot(poly_x, meanSurplusstd_dev, 'r--')
    name.append(r'$\mu + \sigma$')
    name.append(r'$\mu - \sigma$')
    name.append('Samples')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(poly_x[0]-0.1,poly_x[-1]+0.1)
    plt.legend(name, fontsize=10)
    plt.savefig('fig-RegreesionFunc')
    plt.show()
# drawRegression(result_y,poly_x,theta,name, samp_x, samp_y, pred_mu)

def drawStandardDevAroundMean(prob_distribution, poly_x, pred_mu):
    fig = plt.figure()
    plt.plot(poly_x, pred_mu)
    mean = np.mean((np.ravel(np.array(pred_mu))))
    x0 = [mean, mean]
    y0 = [0, 1.0]
    plt.plot(poly_x, pred_mu) #Draw predicted value
    plt.plot([poly_x[0],poly_x[-1]],[mean,mean],'r-')
    plt.annotate(r'$\mu$',xy=(poly_x[-1],mean),fontsize=30)
    # plt.plot(x0, y0)
    std_dev = np.std((np.ravel(np.array(pred_mu))))
    x1 =[ [poly_x[0], poly_x[-1]], [poly_x[0], poly_x[-1]]]
    y1 = [[mean-std_dev, mean-std_dev],[mean+std_dev,mean+std_dev]]
    for i in range(len(x1)):
        plt.plot(x1[i],y1[i], 'b-')
    plt.legend([r'$\mu \pm \sigma$','BR',r'$\mu$'])
    plt.savefig('fig-StdDevAroundMean')
    plt.show()

# drawStandardDevAroundMean(prob_distribution,poly_x,pred_mu)  #Draw std around mean for BR

def MSE(poly_y, pred_y):
    MSE = []
    for i in range(len(pred_y)):
        temp = np.sum( (poly_y-pred_y[i])**2 ) / (1.0*len(poly_y))
        MSE.append(temp)
    # print(MSE)
    return MSE

# MSE(poly_y, result_y)
# [0.40864388356985548, 0.41560254450767614, 0.51912816090762348, 88.082706169551017, 0.52358279082974812]

def CrossValidation(iteration, samp_x, samp_y, poly_x, poly_y, test_size, K ): #iterations, test_size is a list
    MSE_total = []
    for j in test_size:
        MSE_record = []
        for i in range(0, iteration):
            try: #SOlution of LP in RR maybe not exsit!!
                X_train, X_test, y_train, y_test = train_test_split(samp_x, samp_y, test_size=j)
                Phi = ls.generate_Phi(K, X_train)
                theta_LS = ls.LS(Phi, y_train)  # LSR
                theta_RLS = ls.RLS(Phi, y_train, 0.5)  # RLSR  lambda = 2.0
                theta_LASSO = ls.LASSO(Phi, y_train, 0.2)  # LASSO  lambda = 2.0
                theta_RR = ls.RR(Phi, y_train, K)  # RR  It seems wrong, but dont know why...
                mu_estimator, var_estimator = ls.BR(Phi, y_train, 2.5, 5)  # BR alpha = 0.5, variance = 5
                theta = [theta_LS, theta_RLS, theta_LASSO, theta_RR]
                result_y = []
                for i in range(0, 4):
                    pred_y = Prediction.predict(poly_x=poly_x, theta=theta[i], poly_para_K=K)
                    result_y.append(np.ravel(np.array(pred_y)))
                prob_distribution, pred_mu = Prediction.pref_BR(mu_estimator, var_estimator, poly_para_K=K, poly_x=poly_x)
                result_y.append((np.ravel(np.array(pred_mu))))
                # drawRegression(result_y=result_y,poly_x=poly_x,theta=theta[i],name = ['LS','RLS','LASSO','RR','BR'],samp_x=X_train,samp_y=y_train,pred_mu=pred_mu) #画图时使用
                temp = MSE(poly_y, result_y)
                # print('*****')
                # print(temp)
                MSE_record.append(temp)
            except:
                pass
        # print(np.matrix(MSE_record))
        # print('----------')
        MSE_total.append(np.mean(MSE_record,0))
        # print(MSE_total)
        #iterations are done!
    print(MSE_total)
    return MSE_total

def DrawCrossValidation(iteration, samp_x, samp_y, poly_x, poly_y, test_size, K):
    a = CrossValidation(iteration, samp_x, samp_y, poly_x, poly_y, test_size, K)
    a = np.array(a)
    a = a.T
    f = open('result_CrossValidation.csv', 'w')
    for item in range(len(a)):
        f.write(name[item] + ' ' + str(a[item]) + '\r\n')
    f.close()
    # print(a)
    # print(a.T)
    for i in range(len(name)):
        plt.plot(test_size, a[i])
    # plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],[r'90%',r'80%',r'70%',r'60%',r'50%',r'40%',r'30%'])
    plt.xticks([0.1, 0.2, 0.3, 0.4], [r'90%', r'80%', r'70%', r'60%'])
    plt.legend(name, fontsize=10)
    plt.savefig('fig-CrossValidation')
    plt.show()

test_size = [0.1, 0.2, 0.3, 0.4]
iteration = 100
# DrawCrossValidation(iteration, samp_x, samp_y, poly_x, poly_y,test_size, K)

def AddOutlier(Phi, samp_y, name, poly_x,samp_x):
    for i in range(len(samp_y)):
        if (i%10) == 0:
            samp_y[i] = samp_y[i] + random.uniform(0,100)
    theta_LS = ls.LS(Phi, samp_y)  # LSR
    theta_RLS = ls.RLS(Phi, samp_y, 0.5)  # RLSR  lambda = 2.0
    theta_LASSO = ls.LASSO(Phi, samp_y, 0.2)  # LASSO  lambda = 2.0
    theta_RR = ls.RR(Phi, samp_y, 5)  # RR  It seems wrong, but dont know why...
    mu_estimator, var_estimator = ls.BR(Phi, samp_y,2.5, 5)  # BR alpha = 0.5, variance = 5
    theta = [theta_LS, theta_RLS, theta_LASSO, theta_RR]
    result_y = []
    for i in range(0, 4):
        pred_y = Prediction.predict(poly_x=poly_x, theta=theta[i], poly_para_K=K)
        # plt.plot(poly_x,pred_y)
        result_y.append(np.ravel(np.array(pred_y)))
    prob_distribution, pred_mu = Prediction.pref_BR(mu_estimator, var_estimator, poly_para_K=K, poly_x=poly_x)
    result_y.append((np.ravel(np.array(pred_mu))))
    drawRegression(result_y,poly_x, theta,name, samp_x=samp_x, samp_y=samp_y,pred_mu=pred_mu)
    print(MSE(poly_y, result_y))

# AddOutlier(Phi, samp_y, name, poly_x,samp_x)