# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import least_squares


def line(x, A, B):
    return A*x + B

def gauss(x, *param):
    return param[0] * np.exp(-(x-param[1])**2/(2.*param[2]**2))


def gaussn(x, n, *param):
    aux=np.zeros(x.shape)
    param_i=np.zeros(3)
    for i in range(n):
        param_i[0] = param[i+2]
        param_i[1] = param[0]+param[1]*i
        param_i[2] = param[i+2+n]
        aux = aux + gauss(x,*param_i)
    return aux


class gaussn_least(object):
    def __init__(self,x_data,y_data,n):
        self.y = y_data
        self.x = x_data
        self.n = n
        self.fit_func = lambda *param:gaussn(x_data,n,*param)

    def error_func(self,param):
        fit = self.fit_func(*param)
        return fit-self.y

    def __call__(self,bounds,*guess):
        self.param = least_squares(  self.error_func,guess,
                                method='trf',
                                verbose=1,
                                max_nfev=2000,
                                tr_solver='lsmr',
                                bounds=bounds)

        # +++ I love StackOverflow +++
        J = self.param.jac
        residuals_lsq = self.param.fun
        cov = np.linalg.inv(J.T.dot(J)) * (residuals_lsq**2).mean()
        perr = np.sqrt(np.diag(cov))

        return self.param['x'],perr

    def evaluate(self):
        return gaussn(self.x,self.n,*self.param['x'])


def line_fit(f,X,f_sigma,x_text,y_text,title_text,n_figure,graph_sw):

# Makes a linear fit for n points (X input vector).
# f is the mean of the measured data points
# f_sigma is the standard deviation (ddof=1) of the measured data points
# The rest are attributes for the plotting windows (graph_sw = 1 to plot)
# returns coeff (A,B), perr - error for the fit param,
#         XI2_r --> Squared CHI reduced (Goodness of fit)

    p0 = [1,(f[1]-f[0])/(X[1]-X[0])]
    coeff, var_matrix = curve_fit(line, X, f,p0=p0)

    #Parameters error estimation (sigma). See numpy user guide
    perr = np.sqrt(np.diag(var_matrix))

    Y_fit = line(X,coeff[0],coeff[1])

    XI2 = np.sum(((Y_fit-f)**2.)/(f_sigma**2.))
    XI2_r = XI2/(len(X)-2)

    max_err = np.max(np.abs((Y_fit-f)/Y_fit))*100.0

    print ('Max Linearity Error = %0.3f%%' % max_err)

    if (graph_sw==1):
    # Draws figure with all the properties

        plt.figure(n_figure)
        plt.plot(X, Y_fit, 'r--', linewidth=1)
        plt.errorbar(X, f, fmt='b*', yerr=f_sigma)
        plt.xlabel(x_text)
        plt.ylabel(y_text)
        plt.title(title_text)
        plt.figtext(0.2,0.75, ('CHI2_r = %0.3f ' % (XI2_r)))
        plt.show(block=False)
        #Fit parameters
        print ('Fitted A = ', coeff[0], '( Error_std=', perr[0],')')
        print ('Fitted B = ', coeff[1], '( Error_std=', perr[1],')')

    return coeff, perr, XI2_r

# class gauss_fit(fitting):
#     def __call__(self, data, bins):
#         self.gauss1 = gauss
#         self.p0 = [1, np.mean(data), np.std(data)]
#         # First guess
#         super(gauss_fit,self).__call__(data=data,
#                                        bins=bins,
#                                        guess=self.p0,
#                                        fit_func=self.gauss1)
#
#     def plot(self,axis,title,xlabel,ylabel,res=True,fit=True):
#         axis.hist(self.data, self.bins, facecolor='green')
#         axis.set_xlabel(xlabel)
#         axis.set_ylabel(ylabel)
#         axis.set_title(title)
#         if (fit==True):
#             axis.plot(self.bin_centers, self.hist_fit, 'r--', linewidth=1)
#             if (res==True):
#                 axis.text(0.95,0.95, (('$\mu$=%0.1f (+/- %0.1f) \n'+\
#                                      '$\sigma$=%0.1f (+/- %0.1f) \n'+
#                                      'FWHM=%0.1f (+/- %0.1f) \n'+\
#                                      'Res=%0.1f%% (+/- %0.1f)') % \
#                                         (self.coeff[1] , self.perr[1],
#                                          np.absolute(self.coeff[2]) , self.perr[2],
#                                          2.35*np.absolute(self.coeff[2]),
#                                          2.35*np.absolute(self.perr[2]),
#                                          2.35*np.absolute(self.coeff[2])*100/self.coeff[1],
#                                          2.35*np.absolute(self.coeff[2])*100/self.coeff[1]*
#                                          np.sqrt((self.perr[2]/self.coeff[2])**2+
#                                                  (self.perr[1]/self.coeff[1])**2)
#                                         )
#                                       ),
#                                          fontsize=6,
#                                          verticalalignment='top',
#                                          horizontalalignment='right',
#                                          transform=axis.transAxes)
#
#
#             else:
#                 # No resolution calculation
#                 axis.text(0.95,0.95, (('$\mu$=%0.1f (+/- %0.1f) \n'+\
#                                      '$\sigma$=%0.1f (+/- %0.1f) \n'+
#                                      'FWHM=%0.1f (+/- %0.1f)') % \
#                                         (self.coeff[1], self.perr[1],
#                                          np.absolute(self.coeff[2]), self.perr[2],
#                                          2.35*np.absolute(self.coeff[2]),
#                                          2.35*np.absolute(self.perr[2]))),
#                                          fontsize=6,
#                                          verticalalignment='top',
#                                          horizontalalignment='right',
#                                          transform=axis.transAxes)
#
# class gauss_fitn(fitting):
#     def __call__(self, data, bins, n, guess, bounds):
#         self.gaussn = gaussn
#         self.p0 = guess
#         self.bounds = bounds
#         self.n = n
#         # First mu_guess
#         super(gauss_fitn,self).__call__(data=data,
#                                        bins=bins,
#                                        guess=self.p0,
#                                        fit_func=self.gaussn,
#                                        hist= False,
#                                        n=self.n,
#                                        bounds=self.bounds)
