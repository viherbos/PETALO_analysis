import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gauss(x, *param):
    return param[0] * np.exp(-(x-param[1])**2/(2.*param[2]**2))

def saturation(x,*param):
    #Thresholds
    slope = param[0]
    sat   = param[1]
    shift = param[2]
    gain  = param[3]
    #offset = param[4]
    value = gain*(1 + ((slope*(x-shift))/np.power(1+np.power((np.abs(slope*(x-shift))),sat),1./sat)))
    return value

def saturation_zero(x,*param):
    #QDC
    slope = param[0]
    sat   = param[1]
    shift = param[2]
    gain  = param[3]
    value = gain*((slope*x-shift)/np.power(1+np.power((np.abs(slope*x-shift)),sat),1./sat))
    value[value<0]=np.zeros(len(value[value<0]))
    return value

def Baseline_work(data, canal, counter):
    datos = data[(data.channel_id==canal)&(data.vth_t1==62)]
    counter_max = datos['count'].max()
    magic_point = 0.999*counter_max
    select = datos[(datos['count'] >= magic_point)]
    return select['baseline_t'].max()


class fitting_nohist(object):
    # General Fitting call
    def __call__(self, data, time, fit_func, guess, sigmas, bounds=[]):
        self.bins  = time
        self.data  = data
        self.fit_func = fit_func
        self.guess  = guess
        self.bounds = bounds

        try:
            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bins,
                                                    self.data, p0=self.guess,
                                                    ftol=1E-12, maxfev=100000,
                                                    #bounds=self.bounds,
                                                    method='lm'
                                                    )

            self.perr = np.sqrt(np.absolute(np.diag(self.var_matrix)))
            # Error in parameter estimation
        except:
            print("Fitting Problems")

        self.fit = self.fit_func(self.bins, *self.coeff)
        self.chisq = np.sum(((self.data-self.fit)/sigmas)**2)
        self.df = len(self.bins)-len(self.coeff)
        self.chisq_r = self.chisq/self.df
        #Gets fitted function and chisqr_r

    def evaluate(self,in_data):
        return self.fit_func(in_data,*self.coeff)


class fitting_hist(object):
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
            print("Fitting Problems")
            self.coeff = np.array(self.guess)
            self.perr  = np.array(self.guess)


        self.hist_fit = self.fit_func(self.bin_centers, *self.coeff)
        #Gets fitted function and residues

    def evaluate(self,in_data):
        return self.fit_func(in_data,*self.coeff)


def gauss_fit(data,bins,plot=False,*p_param):
    gauss1 = gauss
    p0 = [1, np.mean(data), np.std(data)]
    # First guess
    Q_gauss = fitting_hist()
    Q_gauss(data=data,
            bins=bins,
            guess=p0,
            fit_func=gauss)

    if plot==True:
        #p_param[0] -> axis
        #p_param[1] -> title
        #p_param[2] -> xlabel
        #p_param[3] -> ylabel
        #p_param[4] -> pos [0.95,0.95,"left"]

        p_param[0].hist(data, bins, align='mid', facecolor='green',
                        edgecolor='white', linewidth=0.5)
        p_param[0].set_xlabel(p_param[2])
        p_param[0].set_ylabel(p_param[3])
        p_param[0].set_title(p_param[1])
        p_param[0].plot(Q_gauss.bin_centers, Q_gauss.hist_fit, 'r--', linewidth=1)
        p_param[0].text(p_param[4][0],p_param[4][1], (('$\mu$=%0.3f (+/- %0.3f) \n'+\
                                     '$\sigma$=%0.3f (+/- %0.3f) \n'+
                                     'FWHM=%0.3f (+/- %0.3f) \n'+\
                                     'Res=%0.3f%% (+/- %0.3f)') % \
                                        (Q_gauss.coeff[1] , Q_gauss.perr[1],
                                         np.absolute(Q_gauss.coeff[2]) , Q_gauss.perr[2],
                                         2.35*np.absolute(Q_gauss.coeff[2]),
                                         2.35*np.absolute(Q_gauss.perr[2]),
                                         2.35*np.absolute(Q_gauss.coeff[2])*100/Q_gauss.coeff[1],
                                         2.35*np.absolute(Q_gauss.coeff[2])*100/Q_gauss.coeff[1]*
                                         np.sqrt((Q_gauss.perr[2]/Q_gauss.coeff[2])**2+
                                                 (Q_gauss.perr[1]/Q_gauss.coeff[1])**2))
                                      ),
                                         fontsize=8,
                                         verticalalignment='top',
                                         horizontalalignment=p_param[4][2],
                                         transform=p_param[0].transAxes)


def Tn_fit(data, canal, min_count=10, plot=False,
           guess=[1.83, 11.11, 55.7, 2.1e+06]):
    # Fitting Thresholds

    slope = guess[0]
    sat   = guess[1]
    shift = guess[2]
    gain  = guess[3]
    #offset = guess[4]

    baseline_T = Baseline_work(data, canal, 2**22)
    baseline_T = baseline_T - 10   # 5 is a guess for a better fit of THn
    print("Baseline %f" % baseline_T)

    datos = data[(data['channel_id']==canal) & (data['baseline_t']==baseline_T)]

    T_fit = fitting_nohist()

    T_fit(datos['count'],datos['vth_t1'],saturation,[slope,sat,shift,gain],
          np.zeros(len(datos['count']))+1.0)

    #chisq = np.sum(((datos['count']-T_fit.evaluate(datos['vth_t1']))/1.0)**2)
    print("Channel = %d / CHISQR = %f" % (canal,T_fit.chisq_r))

    if plot==True:
        plt.figure()
        plt.plot(np.arange(0,64,0.1),T_fit.evaluate(np.arange(0,64,0.1)),'b-',label="Fit")
        plt.errorbar(datos['vth_t1'],datos['count'], 1.0,
                     fmt='.',color='red',label="Data")
        plt.xlabel("th_T1")
        plt.ylabel("COUNT")
        plt.legend()

    func = lambda x : min_count - saturation(x,*T_fit.coeff)
    t_solution = np.floor(brentq(func, 0, 62))
    print("Threshold at %f for a 0.01 activity" % t_solution)
    print(T_fit.coeff)
    return t_solution


def QDC_fit(data, canal, tac, plot=False, guess=[9.87e-03, 9.46, 3.37e-01, 6.32e+02]):
    #Fitting QDC parameters
    slope = guess[0]
    sat   = guess[1]
    shift = guess[2]
    gain  = guess[3]

    datos = data[(data.tac==tac)&(data.channel==canal)]

    Q_fit = fitting_nohist()
    coeff  = [slope,sat,shift,gain]

    Q_fit(datos.eval("mean"),datos.length,saturation,[slope,sat,shift,gain],datos.sigma)
    #chisq = np.sum(((datos.eval("mean")-Q_fit.evaluate(datos.length))/datos.sigma)**2)
    print("Channel = %d / CHISQ_R = %f" % (canal,Q_fit.chisq_r))

    max_slope  = np.max(np.diff(Q_fit.fit))/np.max(np.diff(datos.length))
    length_max = datos.length.to_numpy()
    length_max = length_max[np.argmax(np.diff(Q_fit.fit))]

    qoffset    = Q_fit.evaluate(np.array([length_max]))-length_max*max_slope
    print("QOFFSET = %f" % qoffset)
    print("IBIAS (Q/T) = %f" % max_slope)

    if plot==True:
        plt.figure()
        plt.plot(datos.length,Q_fit.evaluate(datos.length),'b-',label="Fit")
        plt.errorbar(datos.length,datos.eval("mean"), datos.sigma,
                     fmt='.',color='red',label="Data")
        plt.xlabel("Length")
        plt.ylabel("QFINE")
        plt.legend()

    return Q_fit.chisq_r, Q_fit, qoffset, max_slope
