import numpy as np
import os
import cvxopt
from scipy.optimize import linprog

# Load data
def loaddata(path):
    samp_x = np.loadtxt(path + '/data/polydata_data_sampx.txt', dtype=float)  # Sample data X
    print(type(samp_x))  # size 50*1
    samp_y = np.loadtxt(path + '/data/polydata_data_sampy.txt')  # Sample data Y
    # print(samp_y.shape) # size 50*1
    poly_x = np.loadtxt(path + '/data/polydata_data_polyx.txt')  # Real data X
    # print(poly_x.shape) # size 100*1
    poly_y = np.loadtxt(path + '/data/polydata_data_polyy.txt')  # Real data Y
    # print(poly_y.shape) # size 100*1
    return samp_x, samp_y, poly_x,poly_y

def loadImageData(path):
    samp_x = np.loadtxt(path + '/data/count_data_trainx.txt', dtype=float)  # Sample data X
    print(type(samp_x))  # size 50*1
    samp_y = np.loadtxt(path + '/data/count_data_trainy.txt')  # Sample data Y
    # print(samp_y.shape) # size 50*1
    poly_x = np.loadtxt(path + '/data/count_data_testx.txt')  # Real data X
    # print(poly_x.shape) # size 100*1
    poly_y = np.loadtxt(path + '/data/count_data_testy.txt')  # Real data Y
    # print(poly_y.shape) # size 100*1
    return samp_x, samp_y, poly_x,poly_y
# path = os.getcwd()
# samp_x = np.loadtxt(path+'/data/polydata_data_sampx.txt',dtype=float)  # Sample data X
# print(type(samp_x)) # size 50*1
# samp_y =  np.loadtxt(path+'/data/polydata_data_sampy.txt')  # Sample data Y
# #print(samp_y.shape) # size 50*1
# poly_x = np.loadtxt(path+'/data/polydata_data_polyx.txt')  # Real data X
# #print(poly_x.shape) # size 100*1
# poly_y =  np.loadtxt(path+'/data/polydata_data_polyy.txt')  # Real data Y
# #print(poly_y.shape) # size 100*1

def generate_Phi(poly_para_K,samp_x):
    samp_x = samp_x.reshape(samp_x.shape[0], 1)
    X = samp_x ** 0
    for i in range(1, poly_para_K+1):
        temp = samp_x ** i
        X = np.hstack((X, temp))
    #print(X.T)
    #print(X.T.shape)
    return X.T

# Phi = generate_Phi(5,samp_x)  # K+1 * N
#print(Phi.shape)
def LS(Phi,samp_y): #calculate LeastSquare Regression
    Y = samp_y.reshape(samp_y.shape[0], 1)  # reshape Y to 50*1
    # based on the equation
    theta_LS = np.dot(np.dot(np.linalg.inv(np.dot(Phi,Phi.T)),Phi),Y)
    return theta_LS

# theta_LS = LS(Phi,samp_y)
# print(theta_LS)

def RLS(Phi,samp_y,lamd): #calculate Regularized LS Regression
    Y = samp_y.reshape(samp_y.shape[0], 1)
    # theta_RLS = np.dot(np.dot( np.linalg.inv(np.dot(Phi, Phi.T) + lamd*np.ones((Phi.shape[0],1)) ), Phi) , Y)
    theta_RLS = np.dot(np.dot( np.linalg.inv(np.dot(Phi, Phi.T) + lamd*np.eye((Phi.shape[0])) ), Phi) , Y)
    return theta_RLS

# theta_RLS =RLS(Phi,samp_y,2.0)
# print(theta_RLS)

def LASSO(Phi,samp_y,lamd):   # Use the package CVXOPT
    Y = samp_y.reshape(samp_y.shape[0], 1)
    M1 = np.dot(Phi,Phi.T)
    M2 = -1.0*M1
    P = cvxopt.matrix(np.vstack((np.hstack((M1, M2)), np.hstack((M2, M1))))) # H matrix in P.S. 3.12
    q = cvxopt.matrix(1.0*lamd * np.ones((2* Phi.shape[0],1)) - np.vstack((np.dot(Phi,Y), -1.0*np.dot(Phi,Y)))) # f matrix in P.S. 3.12
    G = cvxopt.matrix(-1.0 * np.eye(2* Phi.shape[0],dtype=float))
    h = cvxopt.matrix( 0.0*np.zeros((2* Phi.shape[0],1)) )
    # A = cvxopt.matrix( 0.0*np.zeros((2* Phi.shape[0],2* Phi.shape[0])) )
    # b = cvxopt.matrix( 0.0*np.zeros((2* Phi.shape[0],1)) )
    cvxopt.solvers.options['show_progress'] = False
    theta_trans = cvxopt.solvers.qp(P,q,G,h)
    #print(theta_trans['x'])
    theta_posi, theta_nega = np.vsplit(np.matrix(theta_trans['x']),2)
    theta_LASSO = theta_posi-theta_nega
    return theta_LASSO

# theta_LASSO = LASSO(Phi,samp_y,2.0)
# print((theta_LASSO))

def RR(Phi, samp_y, poly_K):
    Y = samp_y.reshape(samp_y.shape[0], 1)
    # c = np.hstack( (np.zeros((1, Phi.shape[0])), np.ones((1,Phi.shape[1]))))
    c =  np.vstack( (np.zeros((Phi.shape[0],1)), np.ones((Phi.shape[1],1))) )   # f matrix in P.S. 2.10
    I = -1.0*np.eye(Phi.shape[1])
    A_ub = np.vstack( (np.hstack( ( -1.0*Phi.T, I )),  np.hstack( ( 1.0*Phi.T, I )))) # A matrix in P.S. 2.10
    b_ub = np.vstack((-1.0*Y, Y ))
    theta_trans = linprog(np.ravel(c) ,A_ub,b_ub, bounds=(None,None))
    # theta_trans = linprog(c.tolist() ,A_ub.tolist(),b_ub.tolist(),bounds=(0,None))
    # theta_trans = cvxopt.solvers.lp(np.matrix(c),A_ub,b_ub)
    # print(theta_trans['x'].reshape(len(theta_trans['x']),1).shape)
    print(theta_trans['x'])
    # theta_RR = []
    theta_RR=theta_trans['x'].reshape(len(theta_trans['x']), 1)
    # theta_RR = theta_trans['x'][0:poly_K+1]
    # theta_RR = np.squeeze(theta_RR)
    theta_RR = theta_RR[0:poly_K+1] #K+1
    # print(theta_RR)
    # theta_RR = theta_RR.reshape(len(theta_RR),1) # print(theta_RR.shape)
    return np.array(theta_RR)
#
# theta_RR = RR(Phi,samp_y)
# print(theta_RR)

def BR(Phi,samp_y,alpha,var):
    Y = samp_y.reshape(samp_y.shape[0], 1)
    # var_estimator = np.linalg.inv(1.0/alpha * np.ones((Phi.shape[0],1)) + 1.0/var * np.dot(Phi, Phi.T) )
    var_estimator = np.linalg.inv(1.0/alpha * np.eye(Phi.shape[0]) + 1.0/var * np.dot(Phi, Phi.T) )
    mu_estimator = 1.0/var * np.dot(np.dot(var_estimator,Phi),Y)
    return mu_estimator, var_estimator

# mu_estimator, var_estimator = BR(Phi,samp_y,alpha=1,var=5)
# print(mu_estimator, var_estimator)


