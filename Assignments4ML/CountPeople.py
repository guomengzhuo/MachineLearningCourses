import LeastSquare as ls
import numpy as np
import os
import Prediction
import matplotlib.pyplot as plt

path = os.getcwd()
#Load data
samp_x = np.loadtxt(path + '/data/count_data_trainx.txt', dtype=float)  # (9,400)
samp_y = np.loadtxt(path + '/data/count_data_trainy.txt', dtype=float) # (400,)
poly_x = np.loadtxt(path + '/data/count_data_testx.txt', dtype=float) #test data
poly_y = np.loadtxt(path + '/data/count_data_testy.txt')  # test data

samp_x = samp_x # (9, 400)
# samp_y = samp_y.reshape(len(samp_y),1) #  (400,1)
poly_x = poly_x #(9*600)
poly_y = poly_y.reshape(len(poly_y),1) #  (600,1)

#For new features
def addsquarefeature(samp_x, poly_x):
    samp_x_square = samp_x**2
    samp_x_cube = samp_x **3
    samp_x = np.vstack((samp_x,samp_x_square))
    samp_x = np.vstack((samp_x,samp_x_cube))
    poly_x_square = poly_x ** 2
    poly_x_cube = poly_x ** 3
    poly_x = np.vstack((poly_x, poly_x_square))
    poly_x = np.vstack((poly_x, poly_x_cube))
    return samp_x, poly_x
    # print(samp_x.shape)

#For interacted features
# samp_x_copy = samp_x
# for i in range(samp_x.shape[0]-1):# 9hang, meilie huxiang cheng
#     temp = samp_x[i]*samp_x[i+1]
#     print(temp)
#     samp_x_copy = np.vstack((samp_x_copy, temp.T))
# samp_x = samp_x_copy
# poly_x_copy = poly_x
# for i in range(poly_x.shape[0]-1):# 9hang, meilie huxiang cheng
#     temp = poly_x[i]*poly_x[i+1]
#     print(temp)
#     poly_x_copy = np.vstack((poly_x_copy, temp.T))
# poly_x =poly_x_copy

#For new features, randomly generated
# random_matirx = (np.random.random([samp_x.shape[0],samp_x.shape[1]])-0.5)*2
# samp_x = np.vstack((samp_x, random_matirx))
# random_matirx = (np.random.random([poly_x.shape[0],poly_x.shape[1]])-0.5)*2
# poly_x = np.vstack((poly_x,random_matirx))



print(samp_x.shape)
print(poly_x.shape)
#Obtain parameters
theta_LS = ls.LS(Phi=samp_x,samp_y=samp_y)
theta_RLS = ls.RLS(Phi=samp_x, samp_y=samp_y, lamd=0.5)
theta_LASSO = ls.LASSO(Phi=samp_x, samp_y= samp_y,lamd=0.2)
# theta_RR = ls.RR(Phi=samp_x, samp_y=samp_y, poly_K=9) #RR no solutions
mu_estimator, var_estimator  = ls.BR(Phi=samp_x, samp_y=samp_y,alpha=2.5, var=5.0)

name = ['LS','RLS','LASSO','BR']
theta = [theta_LS, theta_RLS, theta_LASSO]

def predict(poly_x, theta):
    predict_y = np.dot(poly_x.T, theta)  #poly_x is a (9*600)
    return predict_y.T #

def predict_BR(mu_estimator, var_estimator,poly_x):
    poly_x = poly_x.T
    predict_y = []
    mu = []
    for i in range(len(poly_x)):
        mu_star = np.dot(poly_x[i], mu_estimator)
        mu.append(mu_star)
        # print(mu_star)
        # temp = poly_x[i].reshape(poly_x[i].shape[0],1)
        # print(poly_x[i].shape)
        var_star = np.dot(np.dot(poly_x[i], var_estimator), poly_x[i].T)
        # var_star = np.dot(np.dot(temp.T, var_estimator),temp)
        # print(var_star)
        y = Prediction.gaussian(mu_star, var_star,(poly_x[i][1]))
        # y = gaussian(mu_estimator, var_star, mu_star)
        # print(poly_x[i][1])
        # print(y)
        predict_y.append(y)
    # print(np.ravel(np.array(predict_y)))
    return np.ravel(np.array(predict_y)) , np.ravel(np.array(mu))


# calculate predictions for 'LS','RLS','LASSO'
result_y = []
for i in range(0,3):
    pred_y = predict(poly_x=poly_x, theta=theta[i])
    result_y.append(np.round(np.ravel(np.array(pred_y))))

#calculate for BR
prob_distribution, pred_mu= predict_BR(mu_estimator, var_estimator, poly_x=poly_x)
result_y.append(np.round(pred_mu))

# print(result_y)

# pred_LS = predict(poly_x, theta_LS)
# print(pred_LS.shape)
# pred_y = []
# pred_y.append(pred_LS)
# print((pred_y[0]))
# print(len(poly_y))
def MAEandMSE(poly_y, pred_y):
    mae = []
    mse = []
    for i in range(len(pred_y)):
        temp1 = np.sum( np.abs(pred_y[i] - poly_y.T) ) / (1.0*len(poly_y))
        mae.append(temp1)
        temp2 =  np.sum((pred_y[i] -poly_y.T) ** 2) / (1.0*len(poly_y))
        mse.append(temp2)
    return mae, mse

# mae, mse = MAEandMSE(poly_y, result_y)
# print(mae, mse)

def plotMAEMSE(mae, mse, name):
    X = np.arange(len(mae))+1
    plt.bar(X, mae,alpha=0.9, tick_label = name,lw=1,width=0.3)
    plt.bar(X+0.35, mse, alpha=0.9, tick_label = name,facecolor = 'yellowgreen', lw=1,width=0.3)
    plt.legend(['MAE','MSE'])
    plt.savefig('MSE_MAE_plot')
    plt.show()

# plotMAEMSE(mae,mse,name)

def plotPredandTrue(result_y, poly_y,name):
    X = np.arange(len(poly_y))+1

    for i in range(len(result_y)):
        plt.plot(X,result_y[i])
    poly_y = np.round(poly_y)
    plt.plot(X,poly_y)
    name.append('True counts')
    plt.legend(name)
    plt.savefig('True_counts_plot')
    plt.show()
    pass

# plotPredandTrue(result_y,poly_y,name)

# mae, mse = MAEandMSE(poly_y, result_y)
# print(mae, mse)
# plotMAEMSE(mae,mse,name)
# plotPredandTrue(result_y,poly_y,name)

#Original [1.3337749999999999, 1.2818500000000002, 1.3321083333333334, 1.3463333333333334]
#Original[3.1391479166666683, 2.7440312500000013, 3.0941395833333347, 2.7976062500000012]

# Square MAE [1.3206166666666668, 1.1964833333333333, 1.2991250000000003, 1.2443749999999998]
# Square MSE [3.0436979166666678, 2.4707229166666673, 2.9053562500000014, 2.7673395833333334]

#Cubic MAE[1.2921833333333332, 1.2317416666666665, 1.2665666666666666, 1.3067499999999999]
#Square MSE[2.9441812500000015, 2.5533395833333339, 2.8369812500000009, 2.800614583333334]

#Interacted neigbered features MAE[1.4153750000000003, 1.3503666666666669, 1.4059000000000004, 1.3607083333333334]
#Interacted MSE [3.1040812500000015, 2.7432395833333345, 3.041614583333335, 2.9374062500000004]

#Random generated (-1,1) [1.3508833333333334, 1.3025499999999999, 1.3442166666666666, 1.3473833333333334]
#  [3.1918729166666679, 2.8517479166666675, 3.1246729166666682, 2.8837395833333344]

def plotAllMAEandMSE(name):
    MAE =np.array([
        [1.3337749999999999, 1.2818500000000002, 1.3321083333333334, 1.3463333333333334],
        [1.3206166666666668, 1.1964833333333333, 1.2991250000000003, 1.2443749999999998],
        [1.2921833333333332, 1.2317416666666665, 1.2665666666666666, 1.3067499999999999],
        [1.4153750000000003, 1.3503666666666669, 1.4059000000000004, 1.3607083333333334],
        [1.3508833333333334, 1.3025499999999999, 1.3442166666666666, 1.3473833333333334]
    ])

    for i in range(len(name)):
        plt.plot([1,2,3,4,5], MAE.T[i])
    plt.xlim(0.5,5.5)
    plt.xticks([1, 2, 3, 4, 5], [r'Original', r'Squared', r'Cubic', r'Interacted', r'Randomly'],rotation=45)
    plt.legend(name,loc='upper left')
    plt.savefig('TotalMAE')
    plt.show()

    MSE = np.array([
        [3.1391479166666683, 2.7440312500000013, 3.0941395833333347, 2.7976062500000012],
        [3.0436979166666678, 2.4707229166666673, 2.9053562500000014, 2.7673395833333334],
        [2.9441812500000015, 2.5533395833333339, 2.8369812500000009, 2.800614583333334],
        [3.1040812500000015, 2.7432395833333345, 3.041614583333335, 2.9374062500000004],
        [3.1918729166666679, 2.8517479166666675, 3.1246729166666682, 2.8837395833333344]
    ])
    for i in range(len(name)):
        plt.plot([1,2,3,4,5], MSE.T[i])
    plt.xlim(0.5,5.5)
    plt.xticks([1, 2, 3, 4, 5], [r'Original', r'Squared', r'Cubic', r'Interacted', r'Randomly'],rotation=45)
    plt.legend(name,loc='best')
    plt.savefig('TotalMSE')
    plt.show()
    pass

plotAllMAEandMSE(name)