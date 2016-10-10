# This file is written for calculating the value of European option via binomial tree method,
# Average binomial tree, binomial-BS, BBS with Richardson Extrapolation and Variance reducrion method
# Hongchao Pan All right (c) reserved 2016

from __future__ import absolute_import, division, print_function
import math
import numpy as np
import black_scholes as BS

def Binomial_European(S,K,T,sigma,q,r,N,option_type=None):
    '''
    Calculate the value of European option
    :param S:       Spot price
    :param K:       Strike price
    :param T:       Maturity
    :param sigma:   volatility
    :param q:       dividend rate
    :param r:       risk-free rate
    :param N:       steps of binomial tree
    :param option_type: CALL or PUT
    :return:        value of options
    '''

    # Get the parameters
    delta_t=T/N
    u=math.exp(sigma*math.sqrt(3*delta_t))
    d=math.exp(-sigma*math.sqrt(3*delta_t))
    pRN=(1/6)+(r-q-0.5*math.pow(sigma,2))*math.sqrt(delta_t/(12*math.pow(sigma,2)))
    qRN=(1/6)-(r-q-0.5*math.pow(sigma,2))*math.sqrt(delta_t/(12*math.pow(sigma,2)))
    pm=2/3

    V_put=np.zeros(2*N+1)       # The nodes at last step
    V_call = np.zeros(2*N + 1)  # The nodes at last step
    #S_t=np.zerps(N+1)       # Stock price at the last step

    for i in range(2*N+1):
        #S_t[i]=math.pow(u,(N-i))*math.pow(d,i)*S
        V_put[i]=max(0,(K-math.pow(u,(N-i))*S))
        V_call[i] = max(0, -(K - math.pow(u, (N - i)) * S))

    for j in range(N-1,-1,-1): # N-1:0
        for k in range(2*j+1):
            V_put[k]=(math.exp(-r*delta_t)*(V_put[k]*pRN+V_put[k+1]*pm+V_put[k+2]*qRN))
            V_call[k] = (math.exp(-r * delta_t) * (V_call[k] * pRN + V_call[k + 1] * pm+V_call[k+2]*qRN))
        # Get the values for greeks
        if(j==2):
            V24_P=V_put[0]
            V22_P=V_put[2]
            V20_P=V_put[4]

            V24_C = V_call[0]
            V22_C = V_call[2]
            V20_C = V_call[4]
        elif(j==1):
            V12_P=V_put[0]
            V11_P=V_put[1]
            V10_P=V_put[2]

            V12_C = V_call[0]
            V11_C=V_call[1]
            V10_C = V_call[2]
        elif(j==0):
            V00_P=V_put[0]

            V00_C=V_call[0]

    S12=u*S
    S10=d*S
    S24=u*u*S
    S22=u*d*S
    S20=d*d*S
    # Calculate greeks
    Delta_P=(V10_P-V12_P)/(S10-S12)
    Delta_C=(V10_C-V12_C)/(S10-S12)
    Gamma_P=((V20_P-V22_P)/(S20-S22)-(V22_P-V24_P)/(S22-S24))/((S10-S12))
    Gamma_C = ((V20_C - V22_C) / (S20 - S22) - (V22_C - V24_C) / (S22 - S24)) / ((S10 - S12))
    Theta_P=(V11_P-V00_P)/(delta_t)
    Theta_C=(V11_C-V00_C)/(delta_t)

    if option_type is None:
        print("No option type indicated, assuming CALL.")
        return V_call[0],Delta_C,Gamma_C,Theta_C        # The value of the American call option
    elif option_type.upper()=="CALL":
        return V_call[0],Delta_C,Gamma_C,Theta_C        # The value of the American call option
    elif option_type.upper()=="PUT":
        return V_put[0],Delta_P,Gamma_P,Theta_P         # The value of the American put option


def Average_binomial_European(S,K,T,sigma,q,r,N,option_type=None):
    '''
    Calcualte the value of options via average binomial method
    :param S:       Spot price
    :param K:       Strike price
    :param T:       Maturity
    :param sigma:   volatility
    :param q:       dividend rate
    :param r:       risk-free rate
    :param N:       steps of binomial tree
    :param option_type: CALL or PUT
    :return:        value of options
    '''
    v1, delta1, gamma1, theta1 = Binomial_European(S, K, T, sigma, q, r, N, option_type)
    v2, delta2, gamma2, theta2 = Binomial_European(S, K, T, sigma, q, r, N + 1, option_type)
    v = (v1 + v2) / 2
    delta = (delta1 + delta2) / 2
    gamma = (gamma1 + gamma2) / 2
    theta = (theta1 + theta2) / 2
    return v, delta, gamma, theta

    return v, delta,gamma,theta


def BBS_European(t,S,K,T,sigma,q,r,N,option_type=None):
    '''
    Calculate the value of options via BBS method
    :param t:       start time
    :param S:       Spot price
    :param K:       Strike price
    :param T:       Maturity
    :param sigma:   volatility
    :param q:       dividend rate
    :param r:       risk-free rate
    :param N:       steps of binomial tree
    :param option_type: CALL or PUT
    :return:        value of option
    '''
   # Get the parameters
    delta_t=T/N
    u=math.exp(sigma*math.sqrt(3*delta_t))
    d=math.exp(-sigma*math.sqrt(3*delta_t))
    pRN=(1/6)+(r-q-0.5*math.pow(sigma,2))*math.sqrt(delta_t/(12*math.pow(sigma,2)))
    qRN=(1/6)-(r-q-0.5*math.pow(sigma,2))*math.sqrt(delta_t/(12*math.pow(sigma,2)))
    pm=2/3

    V_put=np.zeros(2*N+1)       # The nodes at last step
    V_call = np.zeros(2*N + 1)  # The nodes at last step
    #S_t=np.zerps(N+1)       # Stock price at the last step

    for i in range(2*N-1):  # 2*(N-1)+1
        S_t=math.pow(u,(N-1-i))*S

        V_put[i] = (BS.black_scholes(t,S_t,K,delta_t,sigma,r,q,"PUT"))
        V_call[i] = (BS.black_scholes(t,S_t,K,delta_t,sigma,r,q,"CALL"))

    for j in range(N - 2, -1, -1):  # N-2:0
        for k in range(2*j + 1):
            V_put[k] = (math.exp(-r * delta_t) * (V_put[k] * pRN + V_put[k + 1] * pm+V_put[k+2]*qRN))
            V_call[k] = (math.exp(-r * delta_t) * (V_call[k] * pRN + V_call[k + 1] * pm+V_call[k+2]*qRN))
        # Get the values for greeks
        if(j==2):
            V24_P=V_put[0]
            V22_P=V_put[2]
            V20_P=V_put[4]

            V24_C = V_call[0]
            V22_C = V_call[2]
            V20_C = V_call[4]
        elif(j==1):
            V12_P=V_put[0]
            V11_P=V_put[1]
            V10_P=V_put[2]

            V12_C = V_call[0]
            V11_C=V_call[1]
            V10_C = V_call[2]
        elif(j==0):
            V00_P=V_put[0]

            V00_C=V_call[0]

    S12=u*S
    S10=d*S
    S24=u*u*S
    S22=u*d*S
    S20=d*d*S
    # Calculate greeks
    Delta_P=(V10_P-V12_P)/(S10-S12)
    Delta_C=(V10_C-V12_C)/(S10-S12)
    Gamma_P=((V20_P-V22_P)/(S20-S22)-(V22_P-V24_P)/(S22-S24))/((S10-S12))
    Gamma_C = ((V20_C - V22_C) / (S20 - S22) - (V22_C - V24_C) / (S22 - S24)) / ((S10 - S12))
    Theta_P=(V11_P-V00_P)/(delta_t)
    Theta_C=(V11_C-V00_C)/(delta_t)

    if option_type is None:
        print("No option type indicated, assuming CALL.")
        return V_call[0], Delta_C, Gamma_C, Theta_C  # The value of the American call option
    elif option_type.upper() == "CALL":
        return V_call[0], Delta_C, Gamma_C, Theta_C  # The value of the American call option
    elif option_type.upper() == "PUT":
        return V_put[0], Delta_P, Gamma_P, Theta_P  # The value of the American put option



def BBSR_European(t,S,K,T,sigma,q,r,N,option_type=None):
    '''
    Calculate the BBSR based on BBS
    :param t:       start time
    :param S:       Spot price
    :param K:       Strike price
    :param T:       Maturity
    :param sigma:   volatility
    :param q:       dividend rate
    :param r:       risk-free rate
    :param N:       steps of binomial tree
    :param option_type: CALL or PUT
    :return:        value of option via BBSR
    '''
    V_BBS1, delta1, gamma1, theta1 = BBS_European(t, S, K, T, sigma, q, r, N, option_type)
    V_BBS2, delta2, gamma2, theta2 = BBS_European(t, S, K, T, sigma, q, r, int(N / 2), option_type)
    V_BBSR = (2 * V_BBS1 - V_BBS2)
    delta_BBSR = (2 * delta1 - delta2)
    gamma_BBSR = (2 * gamma1 - gamma2)
    theta_BBSR = (2 * theta1 - theta2)

    #V_BBSR=2*BBS_European(t,S,K,T,sigma,q,r,N,option_type)-BBS_European(t,S,K,T,sigma,q,r,N/2,option_type)

    return V_BBSR, delta_BBSR, gamma_BBSR, theta_BBSR


if __name__ == "__main__":
    K=40;S=41;q=1/100;sigma=30/100;r=3/100; T=1; t=0

    N=20 # Steps of binomial tree

    # Get the exact value
    V_exact, delta_exact, gamma_exact, theta_exact = Average_binomial_European(S, K, T, sigma, q, r, N, "PUT")
    print("The exact value is: ", V_exact,delta_exact,gamma_exact,theta_exact)

    v, delta, gamma, theta = BBS_European(t,S, K, T, sigma, q, r, N, "PUT")
    print("BBS European: Value, delta1, gamma1, theta1:",v, delta, gamma, theta)

    V_BS=BS.black_scholes(t,S,K,T,sigma,r,q,"PUT")
    delta_BS=BS.delta_BS_put(t,S,K,T,sigma,r,q)
    gamma_BS=BS.gamma_BS(t,S,K,T,sigma,r,q)
    theta_BS=BS.theta_BS_put(t,S,K,T,sigma,r,q)
    print("BS: V, delta, gamma, theta",V_BS,delta_BS,gamma_BS,theta_BS)
