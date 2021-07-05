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

def sawtooth(x, *param):
    amplitude = param[0]
    shift     = param[1]
    offset    = param[2]
    period    = 360 

    return offset + (-2*amplitude/np.pi)*np.arctan(1/np.tan(((x-shift)*np.pi/period)))

def sawtooth_inv(x,*param):
    amplitude = param[0]
    shift     = param[1]
    offset    = param[2]
    period    = 360
    
    possible = shift+(period/np.pi)*np.arctan(1/np.tan((np.pi/(-2*amplitude))*(x-offset)))
    #possible = (period/np.pi)*np.arctan(1/np.tan((np.pi/(-2*amplitude))*(x-offset)))
    possible[possible<0]=possible[possible<0]+period

    return possible

def sawtooth_inv_corr(x,*param):
    amplitude = param[0]
    shift     = param[1]
    offset    = param[2]
    period    = 360
    
    #possible = shift+(period/np.pi)*np.arctan(1/np.tan((np.pi/(-2*amplitude))*(x-offset)))
    possible = (period/np.pi)*np.arctan(1/np.tan((np.pi/(-2*amplitude))*(x-offset)))
    possible[possible<0]=possible[possible<0]+period

    return possible

class fitting_nohist(object):
    # General Fitting call
    def __call__(self, data, time, fit_func, guess, sigmas, bounds=[]):
        self.bins  = time
        self.data  = data
        self.fit_func = fit_func
        self.guess  = guess
        self.bounds = bounds
        
        if not bounds:
            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bins,
                                                    self.data, p0=self.guess,
                                                    ftol=1E-12, maxfev=100000,
                                                    method='lm'
                                                    )
        else:
#        try:
            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bins,
                                                        self.data, p0=self.guess,
                                                        ftol=1E-12, maxfev=100000,
                                                        bounds=self.bounds,
                                                        method='trf'
                                                        )

        self.perr = np.sqrt(np.absolute(np.diag(self.var_matrix)))
            # Error in parameter estimation
#        except:
#            print("Fitting Problems")

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
        #self.bounds = [[0,0,0],[np.inf,1024,100]]

        # Fitting function call
        try:
            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bin_centers,
                                                        self.hist, p0=self.guess,
                                                        #ftol=1E-12, maxfev=1000,
                                                        #bounds = self.bounds,
                                                        method='lm')
            self.perr = np.sqrt(np.absolute(np.diag(self.var_matrix)))

            # Error in parameter estimation
        except:
            print("Fitting Problems")
            self.coeff = np.array(self.guess)
            self.perr  = np.array(self.guess)

            
        self.hist_fit = self.fit_func(self.bin_centers, *self.coeff)
        if np.isnan(self.hist_fit).any():
            self.chisq_r = 1000
        else:
            self.chisq = np.sum(((self.hist-self.hist_fit)/self.hist_fit)**2)
            self.df = len(self.bin_centers)-len(self.coeff)
            self.chisq_r = self.chisq/self.df

        #Gets fitted function and residues

    def evaluate(self,in_data):
        return self.fit_func(in_data,*self.coeff)


def gauss_fit(data,bins,*p_param):
    gauss1 = gauss
    p0 = [1, np.mean(data), np.std(data)]
    # First guess
    Q_gauss = fitting_hist()
    Q_gauss(data=data,
            bins=bins,
            guess=p0,
            fit_func=gauss)

    if p_param[0]==True:
        #p_param[1] -> axis
        #p_param[2] -> title
        #p_param[3] -> xlabel
        #p_param[4] -> ylabel
        #p_param[5] -> pos [0.95,0.95,"left"]

        p_param[1].hist(data, bins, align='mid', facecolor='green',
                        edgecolor='white', linewidth=0.5)
        p_param[1].set_xlabel(p_param[2])
        p_param[1].set_ylabel(p_param[3])
        p_param[1].set_title(p_param[1])
        p_param[1].plot(Q_gauss.bin_centers, Q_gauss.hist_fit, 'r--', linewidth=1)
        p_param[1].text(p_param[5][0],p_param[5][1], (('$\mu$=%0.3f (+/- %0.3f) \n'+\
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
                                         horizontalalignment=p_param[5][2],
                                         transform=p_param[1].transAxes)

    return Q_gauss.coeff,Q_gauss.perr,Q_gauss.chisq_r


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


def QDC_fit(data, canal, tac, plot=False, guess=[1.78e-02,1.01e+01,9.21e+01,3.41e+02]):
    #Fitting QDC parameters
    slope = guess[0]
    sat   = guess[1]
    shift = guess[2]
    gain  = guess[3]

    datos = data[(data['tac_id']==tac)&(data['channel_id']==canal)]

    Q_fit = fitting_nohist()
    coeff  = [slope,sat,shift,gain]

    guess_a = np.array(guess)
    
    Q_fit(datos['mu'],datos['tpulse'],saturation,[slope,sat,shift,gain],datos['sigma'],
          bounds=[guess_a-guess_a/2,guess_a+guess_a/2])
    #chisq = np.sum(((datos.eval("mean")-Q_fit.evaluate(datos.length))/datos.sigma)**2)
    print("Channel = %d / Slope_Error = %f" % (canal,Q_fit.perr[0]/Q_fit.coeff[0]))

    max_slope  = np.max(np.diff(Q_fit.fit))/np.max(np.diff(datos['tpulse']))
    length_max = datos['tpulse'].to_numpy()
    length_max = length_max[np.argmax(np.diff(Q_fit.fit))]

    qoffset    = Q_fit.evaluate(np.array([length_max]))-length_max*max_slope
    print("QOFFSET = %f" % qoffset)
    print("IBIAS (Q/T) = %f" % max_slope)

    if plot==True:
        plt.figure()
        plt.plot(datos['tpulse'],Q_fit.evaluate(datos['tpulse']),'b-',label="Fit")
        plt.errorbar(datos['tpulse'],datos['mu'], datos['sigma'],
                     fmt='.',color='red',label="Data")
        plt.xlabel("Length")
        plt.ylabel("QFINE")
        plt.legend()

    return Q_fit.chisq_r, Q_fit, qoffset, max_slope


def TDC_fit(data, canal, tac, guess=[-82,0,280], plot=False):
    chisq_r = 10000
    best_chi = 10000
    best_shift = 0
    amplitude = guess[0]
    shift     = guess[1]
    offset    = guess[2]
    period    = 360
    
    #Find gap
    datos = data[(data['tac_id']==tac)&(data['channel_id']==canal)]
    salto = np.argmax(datos['mu'])
    print(salto)
    shift_min = datos['phase'].iloc[salto]-20
    shift_max = datos['phase'].iloc[salto]+20
    shift = shift_min
    
    amplitude = (np.min(datos['mu']) - np.max(datos['mu']))/2.0
    offset    = np.min(datos['mu']) + np.abs(amplitude)
    bounds    = [[amplitude-20,shift_min,offset-20],[amplitude+20,shift_max,offset+20]]
    
    while((chisq_r > 5) & (shift < shift_max)):
        
        Q_fit = fitting_nohist()
        coeff  = [amplitude,period,shift,offset]
        
        Q_fit(datos['mu'],datos['phase'],sawtooth,[amplitude,shift,offset],datos['sigma'],bounds=bounds)
        chisq_r = Q_fit.chisq_r
        
        if best_chi > chisq_r:
            best_chi = chisq_r
            best_shift = shift
        
        shift = shift + 0.25
    
    Q_fit(datos['mu'],datos['phase'],sawtooth,[amplitude,best_shift,offset],datos['sigma'],bounds=bounds)
    chisq_r = Q_fit.chisq_r
    print("Channel = %d / TAC = %d / CHISQR_r = %f" % (canal,tac,chisq_r))
    
        
    if plot==True:
        plt.figure()
        phase_fine = np.arange(0,360)
        plt.plot(phase_fine,Q_fit.evaluate(phase_fine),'b-',label="Fit")
        plt.errorbar(datos.phase,datos['mu'], datos['sigma'],
                     fmt='.',color='red',label="Data")
        plt.xlabel("PHASE")
        plt.ylabel("TFINE")
        plt.legend()
        
    return chisq_r, Q_fit.coeff