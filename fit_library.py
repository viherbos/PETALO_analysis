# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss(x, *param):
    return param[0] * np.exp(-(x-param[1])**2/(2.*param[2]**2))

def gauss2(x, *param):
    return param[0] * np.exp(-(x-param[1])**2/(2.*param[2]**2)) + \
           param[3] * np.exp(-(x-param[4])**2/(2.*param[5]**2))

def GND(x, *param):
    # Generalized normal distribution
    # param[0]=ALFA | param[1]=BETA | param[2]=GAMMA | param[3]=MU
    return (param[1]/(2*param[0]*param[2]*(1/param[1]))) * \
           np.exp(-(np.abs(x-param[3])/param[0])**param[1])


class fitting(object):
    def __call__(self, data, bins, fit_func, guess):
        self.guess = guess
        self.bins  = bins
        self.data  = data
        self.fit_func = fit_func
        # Histogram
        self.hist, self.bin_edges = np.histogram(self.data, bins=self.bins)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:])/2

        # Fitting function call
        try:
            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bin_centers,
                                                    self.hist, p0=self.guess)
            self.perr = np.sqrt(np.absolute(np.diag(self.var_matrix)))
            # Error in parameter estimation
        except:
            print "Fitting Problems"
            self.coeff = np.array(self.guess)
            self.perr  = np.array(self.guess)


        self.hist_fit = self.fit_func(self.bin_centers, *self.coeff)
        #Gets fitted function and residues

    def evaluate(self,in_data):
        return self.fit_func(in_data,*self.coeff)

# This is the end

class GND_fit(fitting):
    def __call__(self, data, bins):
        self.GND = GND
        self.p0 = [np.std(data), 1, 1, np.mean(data)]
        # First guess
        super(GND_fit,self).__call__(data=data,
                                     bins=bins,
                                     guess=self.p0,
                                     fit_func=self.GND)

    def plot(self,axis,title,xlabel,ylabel,res=True,fit=True):
        axis.hist(self.data, self.bins, align='left', facecolor='green')
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if (fit==True):
            axis.plot(self.bin_centers, self.hist_fit, 'r--', linewidth=1)
            if (res==True):
                mu = self.coeff[3]; mu_err = self.perr[3]
                sigma = self.coeff[0]*np.sqrt(3) ; sigma_err = np.sqrt(3)*self.perr[0]

                # Wikipedia
                # NOTE:Try to include CHI_SQUARE

                axis.text(0.95,0.95, (('$\mu$=%0.2f (+/- %0.2f) \n'+\
                                     '$\sigma$=%0.2f (+/- %0.2f) ')  %
                                        (mu , mu_err,
                                         np.absolute(sigma) , sigma_err
                                        )
                                      ),
                                         fontsize=6,
                                         verticalalignment='top',
                                         horizontalalignment='right',
                                         transform=axis.transAxes)
#            else:
                # No resolution calculation
                # axis.text(0.95,0.95, (('$\mu$=%0.1f (+/- %0.1f) \n'+\
                #                      '$\sigma$=%0.1f (+/- %0.1f) \n'+
                #                      'FWHM=%0.1f (+/- %0.1f)') % \
                #                         (self.coeff[1], self.perr[1],
                #                          np.absolute(self.coeff[2]), self.perr[2],
                #                          2.35*np.absolute(self.coeff[2]),
                #                          2.35*np.absolute(self.perr[2]))),
                #                          fontsize=6,
                #                          verticalalignment='top',
                #                          horizontalalignment='right',
                #                          transform=axis.transAxes)


class gauss_fit(fitting):
    def __call__(self, data, bins):
        self.gauss1 = gauss
        self.p0 = [1, np.mean(data), np.std(data)]
        # First guess
        super(gauss_fit,self).__call__(data=data,
                                       bins=bins,
                                       guess=self.p0,
                                       fit_func=self.gauss1)

    def plot(self,axis,title,xlabel,ylabel,res=True,fit=True):
        axis.hist(self.data, self.bins, align='left', facecolor='green')
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if (fit==True):
            axis.plot(self.bin_centers, self.hist_fit, 'r--', linewidth=1)
            if (res==True):
                axis.text(0.95,0.95, (('$\mu$=%0.1f (+/- %0.1f) \n'+\
                                     '$\sigma$=%0.1f (+/- %0.1f) \n'+
                                     'FWHM=%0.1f (+/- %0.1f) \n'+\
                                     'Res=%0.1f%% (+/- %0.1f)') % \
                                        (self.coeff[1] , self.perr[1],
                                         np.absolute(self.coeff[2]) , self.perr[2],
                                         2.35*np.absolute(self.coeff[2]),
                                         2.35*np.absolute(self.perr[2]),
                                         2.35*np.absolute(self.coeff[2])*100/self.coeff[1],
                                         2.35*np.absolute(self.coeff[2])*100/self.coeff[1]*
                                         np.sqrt((self.perr[2]/self.coeff[2])**2+
                                                 (self.perr[1]/self.coeff[1])**2)
                                        )
                                      ),
                                         fontsize=6,
                                         verticalalignment='top',
                                         horizontalalignment='right',
                                         transform=axis.transAxes)


            else:
                # No resolution calculation
                axis.text(0.95,0.95, (('$\mu$=%0.1f (+/- %0.1f) \n'+\
                                     '$\sigma$=%0.1f (+/- %0.1f) \n'+
                                     'FWHM=%0.1f (+/- %0.1f)') % \
                                        (self.coeff[1], self.perr[1],
                                         np.absolute(self.coeff[2]), self.perr[2],
                                         2.35*np.absolute(self.coeff[2]),
                                         2.35*np.absolute(self.perr[2]))),
                                         fontsize=6,
                                         verticalalignment='top',
                                         horizontalalignment='right',
                                         transform=axis.transAxes)

class gauss_fit2(fitting):
    def __call__(self, data, mu_guess, bins):
        self.gauss2 = gauss2
        self.p0 = [1, mu_guess[0], mu_guess[2], 1, mu_guess[1], mu_guess[3]]
        # First guess
        super(gauss_fit2,self).__call__(data=data,
                                       bins=bins,
                                       guess=self.p0,
                                       fit_func=self.gauss2)

    def plot(self,axis,title,xlabel,ylabel,res=True):
        axis.hist(self.data, self.bins, facecolor='green')
        axis.plot(self.bin_centers, self.hist_fit, 'r--', linewidth=1)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if (res==True):
            axis.text(0.05,0.8, (('$\mu1$=%0.1f (+/- %0.1f) \n'+\
                                 '$\sigma1$=%0.1f (+/- %0.1f) \n'+
                                 'FWHM1=%0.1f (+/- %0.1f) \n'+\
                                 'Res1=%0.1f%% (+/- %0.1f)') % \
                                    (self.coeff[1] , self.perr[1],
                                     np.absolute(self.coeff[2]) , self.perr[2],
                                     2.35*np.absolute(self.coeff[2]),
                                     2.35*np.absolute(self.perr[2]),
                                     2.35*np.absolute(self.coeff[2])*100/self.coeff[1],
                                     2.35*np.absolute(self.coeff[2])*100/self.coeff[1]*
                                     np.sqrt((self.perr[2]/self.coeff[2])**2+
                                             (self.perr[1]/self.coeff[1])**2)
                                    )
                                  ),
                                     fontsize=6,
                                     verticalalignment='top',
                                     horizontalalignment='left',
                                     transform=axis.transAxes)

            axis.text(0.05,1.0, (('$\mu2$=%0.1f (+/- %0.1f) \n'+\
                                 '$\sigma2$=%0.1f (+/- %0.1f) \n'+
                                 'FWHM2=%0.1f (+/- %0.1f) \n'+\
                                 'Res2=%0.1f%% (+/- %0.1f)') % \
                                    (self.coeff[4] , self.perr[4],
                                     np.absolute(self.coeff[5]) , self.perr[5],
                                     2.35*np.absolute(self.coeff[5]),
                                     2.35*np.absolute(self.perr[5]),
                                     2.35*np.absolute(self.coeff[5])*100/self.coeff[4],
                                     2.35*np.absolute(self.coeff[5])*100/self.coeff[4]*
                                     np.sqrt((self.perr[5]/self.coeff[5])**2+
                                             (self.perr[4]/self.coeff[4])**2)
                                    )
                                  ),
                                     fontsize=6,
                                     verticalalignment='top',
                                     horizontalalignment='left',
                                     transform=axis.transAxes)

        else:
            pass
            # # No resolution calculation
            # axis.text(0.05,0.9, (('$\mu$=%0.1f (+/- %0.1f) \n'+\
            #                      '$\sigma$=%0.1f (+/- %0.1f) \n \n'+
            #                      'FWHM=%0.1f (+/- %0.1f)') % \
            #                         (self.coeff[1], self.perr[1],
            #                          np.absolute(self.coeff[2]), self.perr[2],
            #                          2.35*np.absolute(self.coeff[2]),
            #                          2.35*np.absolute(self.perr[2]))),
            #                          fontsize=6,
            #                          verticalalignment='top',
            #                          horizontalalignment='left',
            #                          transform=axis.transAxes)
