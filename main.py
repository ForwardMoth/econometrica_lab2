import pandas as pd, os,scipy.stats as sc, matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import numpy as np,pylab,scipy
import statsmodels.regression.linear_model as s
from statsmodels.stats.api import het_goldfeldquandt
from statsmodels.compat import lzip

"""Create lineral regression model"""
def get_model(ind_data,data):
    formula = "{} ~ {}".format("Y", "+".join(ind_data.columns.to_list()))
    model = ols(formula=formula,data=data).fit()
    return model

"""Get assessment of coefficients KLMMR and information about KLMMR"""
def get_model_info(model):
    print(model.summary())

"""Get residuals of lineral regression"""
def get_residuals(model):
    return model.resid

"""Check hypothesis"""
def check_normality_of_residuals(residuals):
    """calcutate paramers for plot"""
    residuals_mean, residuals_std = residuals.mean(),residuals.std()
    """setting size of histogram"""
    plt.figure(figsize=(9, 6))

    """building histogram of residuals"""
    histData = plt.hist(residuals) 
    
    """build red line, which fits to the normal distribution"""
    range_ = np.arange(min(residuals), max(residuals), 0.05) 
    normModel = sc.norm(residuals_mean, residuals_std) 
    coefY = len(residuals) * max([1, (max(histData[0]) // (normModel.pdf(residuals_mean) * len(residuals)))])
    plt.plot(range_, [normModel.pdf(x) * coefY for x in range_], color="r")

    """Make divisions on the abscissa axis (upper interval of the partition boundary)"""
    plt.xticks(histData[1])
    
    """Make Kolmogorov test about normal distribution"""
    KS_maxD, KS_PValue = sc.kstest(residuals, cdf="norm", args=(residuals_mean, residuals_std))

    """Creation of title"""
    plt.title("Histogram of the distribution of regression residues\n" +
            "Distribution: Normal\n" +
            "Kolmogorov-Smirnov test = {:.5}, p-value = {:.5}".format(KS_maxD, KS_PValue), fontsize=18)

    """Text axises"""
    plt.ylabel("No. of observations", fontsize=15) 
    plt.xlabel("Category (upper limits)", fontsize=15) 

    """Make grid and show histogram"""
    plt.grid()
    plt.show()

"""Create numpy arrays"""
def get_formated_data(data,independed_data):
    Y = np.copy(data['Y'])
    X = np.copy(independed_data)
    return Y,X

"""Sort and make grap of module residuals with trend"""
def analyse_graps_of_residuals(X,residuals):
    k = X.shape[1]
    maxz = 0
    for var in range(k):
        Xsort = np.copy(X[:, var]) 
        Epl = abs(residuals[Xsort.argsort()]) 
        plt.figure(figsize=(15, 5)) 
        plt.title(f"|E| from столбец_{var}", fontsize=15)
        plt.plot(Epl) 
        Xgraph = [i for i in range(85)] 
        pylab.plot(Xgraph, Epl, 'o') 
        z = np.polyfit(Xgraph, Epl, 1) 
        p = np.poly1d(z) 
        pylab.plot(Xgraph,p(Xgraph), "r-") 
        print(f"y = {z[0]:6f}x + ({z[1]:6f})")
        plt.grid() 
        plt.show()
        if abs(z[0]) > maxz:
            maxz = z[0]; 
            maxk = var; 
    print(f"The research on heteroskedasticity will be by column {maxk}")
    return maxk

"""test Spearman"""
def test_Spearman(maxk,X,residuals):
    if maxk == 1:
        X_investigated = X[:,1]
    else:
        X_investigated = X[:,maxk:maxk+1]

    rho, pval = scipy.stats.spearmanr(residuals, X_investigated)
    print(scipy.stats.spearmanr(residuals, X_investigated) )
    if pval < 0.05 :
        Spearman = 1; 
        print("p-value < 0.05, then the hypothesis of the absence of heteroskedasticity is rejected.")
    else: 
        print("p-value > 0.05, then the hypothesis of the absence of heteroscedasticity is accepted.")
        Spearman = 0
    return Spearman,rho

"""Function for Goldfeld-Quandt test"""
def het_gq(y, x):
    nobs, nvar = x.shape
    sizeSubsample = round(3 * nobs / 8)
    fval, _, _ = het_goldfeldquandt(y=y, x=x, idx=1, split=sizeSubsample, drop=nobs - 2 * sizeSubsample)
    if fval < 1.0:
      fval = 1.0 / fval
      GK = 0 
    else: 
      GK = 1 
    return (fval, sc.f.ppf(q=0.95, dfn=sizeSubsample - nvar, dfd=sizeSubsample - nvar), GK)

"""Goldfeld-Quandt test"""
def test_GK(Y,X,maxk):
    name = ['F statistic', 'F crit']
    GK_Fstat, GK_Fcrit, GK = het_gq(y=Y, x=X[:,[0,maxk]]) 
    print(lzip(name, [GK_Fstat, GK_Fcrit]) )
    if GK_Fstat > GK_Fcrit :
        print("F > Fcrit, then the hypothesis of the absence of heteroskedasticity is rejected.")
        Test_GK = 1 
    else:
        Test_GK = 0 
        print("F < Fcrit, then the hypothesis of the absence of heteroscedasticity is accepted.")
    return Test_GK,GK

"""Test glazer"""
def test_Glazer(maxk,X,residuals):
    n,k = X.shape
    """matrix for X in gammma degree, where gamma = [-3,3] with step 0.5"""
    Xgm = np.zeros((n,13))  
    gamma = -3
    for j in range(0, 13):
        for i in range(0, n):
            Xgm[i][j] = abs(X[i][maxk])**gamma
        gamma = gamma + 0.5
    Xgm = np.hstack([np.ones((Xgm.shape[0], 1)), Xgm])
    Xgmd = pd.DataFrame(data=Xgm)
    print(Xgmd)
    """Make the linear regression equations for all X in gamma degree"""
    Rsqrd = np.zeros((13,6))
    u = 0
    gamma = -3
    for i in range(1,14):
        Rsqrd[u][0] = gamma
        result = s.OLS(abs(residuals), Xgm[:, [0, i]]).fit()
        Rsqrd[u][1] = result.params[0]
        Rsqrd[u][2] = result.params[1]
        Rsqrd[u][3] = result.fvalue
        Rsqrd[u][4] = result.f_pvalue
        Rsqrd[u][5] = result.rsquared
        u += 1 
        gamma += 0.5

    d = {"gamma": Rsqrd[::,0], "b0": Rsqrd[::,1], "b1": Rsqrd[::,2], "F-stat": Rsqrd[::,3], "p-value": Rsqrd[::,4], "R^2": Rsqrd[::,5]}
    df = pd.DataFrame(data=d)
    print(df)

    """Сheck the significance of the constructed equations:"""
    Fcr = sc.f.ppf(q = 1 - 0.05, dfn = 1, dfd = 85 - 1 - 1)
    print("Fcrit = ", Fcr )
    umax = 0
    Gleyser = 0;
    for u in range(0, 13): 
        if Rsqrd[u][3] > Fcr:
            print("Equation with gamma = ", Rsqrd[u][0], "significant")
            Gleyser = 1; 
        else: print("Equation with gamma = ", Rsqrd[u][0], "not significant")

    """Choose equalation with highest R^2"""
    Rmax = 0
    umax = 0
    for u in range(0, 13): 
        if Rsqrd[u][5] > Rmax:
            Rmax = Rsqrd[u][5]
            umax = u
    d = {"gamma": Rsqrd[umax][0], "b0": Rsqrd[umax][1], "b1": Rsqrd[umax][2], "F-stat": Rsqrd[umax][3], "p-value": Rsqrd[umax][4], "R^2": Rsqrd[umax][5]}
    df = pd.DataFrame(data=d, index = [0])
    print(df)
    return Gleyser, umax, Rsqrd

"""Matrix after Spearmean test"""
def create_matrix_Spearmean(Spearmean,rho,X,maxk):
    if Spearmean == 1:
        n = X.shape[0]
        Sigm = np.zeros((n,n)) 
        if (rho > 0): 
            for i in range(0, n): 
                for j in range(0, n): 
                    if i==j : 
                        Sigm[i][j]= (X[i][maxk])**2 
        elif (rho < 0): 
            for i in range(0, n): 
                for j in range(0, n): 
                    if i==j : 
                        Sigm[i][j]= 1/((X[i][maxk])**2)
        Sigmd = pd.DataFrame(data=Sigm)
        print(Sigmd)
        return Sigm
    else: 
        print("No heteroscedasticity")    

"""Matrix after Goldfeld-Quandt test""" 
def create_matrix_GK(GK,Test_GK,X,maxk):
    if Test_GK == 1:
        n = X.shape[0] 
        Sigm = np.zeros((n,n)) 
        if (GK == 1):
                for i in range(0, n): 
                    for j in range(0, n): 
                        if i==j : 
                            Sigm[i][j]= (X[i][maxk])**2 
        elif (GK == 0): 
            for i in range(0, n): 
                for j in range(0, n): 
                    if i==j : 
                        Sigm[i][j]= 1/((X[i][maxk])**2)
        Sigmd = pd.DataFrame(data=Sigm)
        print(Sigmd)
        return Sigm
    else: 
        print("No heteroscedasticity")

"""Matrix after Glazer test""" 
def create_matrix_Glazer(Gleyser,Rsqrd,umax,X,maxk):
    if Gleyser == 1:
        n = X.shape[0]
        Sigm = np.zeros((n,n))
        for i in range(0, n): 
            for j in range(0, n): 
                if i==j : 
                    Sigm[i][j]= (Rsqrd[umax][1] + Rsqrd[umax][2]*abs(X[i][maxk])**Rsqrd[umax][0])**2 
        Sigmd = pd.DataFrame(data=Sigm)
        print(Sigmd)
        return Sigm
    else: 
        print("No heteroscedasticity")

"""Estimation of model coefficients after elimination of heteroscedasticity"""
def calculation_coefficents(X,Sigm,Y):
    if Sigm is not None:
        n,k = X.shape
        X = np.hstack([np.ones((X.shape[0], 1)), X]) 
        Xt = X.T
        Sigm_inv = np.linalg.inv(Sigm) 
        XtSinv = np.dot(Xt,Sigm_inv) 
        XtSinvY = np.dot(XtSinv,Y) 
        XtSinvXinv = np.linalg.inv(np.dot(XtSinv, X))
        Bomnk = np.dot(XtSinvXinv, XtSinvY)

        S = ((Y-X.dot(Bomnk)).T.dot(Sigm_inv).dot(Y-X.dot(Bomnk)))*(1/(n-k-1)) 
        Eb = S*np.linalg.inv(np.dot(X.T,Sigm_inv).dot(X)) 

        d = {"coef": [i for i in range(k)], "value": Bomnk, "Std": np.sqrt(np.diagonal(Eb) ** 2)}
        df = pd.DataFrame(data=d)
        print('Стандартная ошибка:', S)
        print(df)

"""Choose not empty sigm from all sigms"""
def choose_sigm(Sigm1,Sigm2,Sigm3):
    Sigms = [Sigm1,Sigm2,Sigm3]
    for Sigm in Sigms:
        if Sigm is not None:
            return Sigm

def main():
    print("Starting of execution lab 2")
    file_name = r'C:\Users\asus\projects\econometrica_lab2\data.xlsx'
    sheet = 'данные'
    try:
        data = pd.read_excel(file_name,sheet_name=sheet)
        data.drop(data.columns[0],axis=1,inplace=True)

        independed_variables = data.drop(columns='Y')

        model = get_model(independed_variables,data)
        get_model_info(model)
        residuals = get_residuals(model)
        
        check_normality_of_residuals(residuals)
        
        Y,X = get_formated_data(data,independed_variables)
        maxk = analyse_graps_of_residuals(X,residuals)

        spearmean_res,rho = test_Spearman(maxk,X,residuals)
        GK_res,GK = test_GK(Y,X,maxk)
        Glazer_res, umax, Rsqrd = test_Glazer(maxk,X,residuals) 

        Sigm1 = create_matrix_Spearmean(spearmean_res,rho,X,maxk)
        Sigm2 = create_matrix_GK(GK,GK_res,X,maxk)
        Sigm3 = create_matrix_Glazer(Glazer_res,Rsqrd,umax,X,maxk)
        
        Sigm = choose_sigm(Sigm1,Sigm2,Sigm3)
        calculation_coefficents(X,Sigm,Y)
    except FileNotFoundError:
        print("File didn't found")

if __name__ == '__main__':
    main()