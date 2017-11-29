# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss(x, *param):
    return param[0] * np.exp(-(x-param[1])**2/(2.*param[2]**2))


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

class gauss_fit(fitting):
    def __call__(self, data, bins):
        self.gauss1 = gauss
        self.p0 = [1, np.mean(data), np.std(data)]
        # First guess
        super(gauss_fit,self).__call__(data=data,
                                       bins=bins,
                                       guess=self.p0,
                                       fit_func=self.gauss1)
        return self.coeff,self.perr

    def plot(self,axis,title,xlabel,ylabel,res=True):
        axis.hist(self.data, self.bins, facecolor='green')
        axis.plot(self.bin_centers, self.hist_fit, 'r--', linewidth=1)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if (res==True):
            axis.text(0.05,0.9, (('$\mu$=%0.1f (+/- %0.1f) \n'+\
                                 '$\sigma$=%0.1f (+/- %0.1f) \n \n'+
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
                                     horizontalalignment='left',
                                     transform=axis.transAxes)


        else:
            # No resolution calculation
            axis.text(0.05,0.9, (('$\mu$=%0.1f (+/- %0.1f) \n'+\
                                 '$\sigma$=%0.1f (+/- %0.1f) \n \n'+
                                 'FWHM=%0.1f (+/- %0.1f)') % \
                                    (self.coeff[1], self.perr[1],
                                     np.absolute(self.coeff[2]), self.perr[2],
                                     2.35*np.absolute(self.coeff[2]),
                                     2.35*np.absolute(self.perr[2]))),
                                     fontsize=6,
                                     verticalalignment='top',
                                     horizontalalignment='left',
                                     transform=axis.transAxes)
