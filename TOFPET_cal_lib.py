import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import brentq
from scipy.special import erf
from scipy import interpolate


def gauss(x, *param):
    return param[0] * np.exp(-(x-param[1])**2/(2.*param[2]**2))

def saturation(x,*param):
    #Thresholds
    slope = param[0]
    sat   = param[1]
    shift = param[2]
    gain  = param[3]
    #offset = param[4]
    offset = 0
    value = gain*(1 + ((slope*(x-shift))/np.power(1+np.power((np.abs(slope*(x-shift))),sat),1./sat))) + offset
    return value

def saturation_poly(x,*param):

    return param[0]+param[1]*x+param[2]*(x**2)+param[3]*(x**3)+param[4]*(x**4)+param[5]*(x**5)+\
                               param[6]*(x**6)+param[7]*(x**7)+param[8]*(x**8)+param[9]*(x**9)

def poly_5(x,*param): 
    return param[0]+param[1]*x+param[2]*(x**2)+param[3]*(x**3)+param[4]*(x**4)+param[5]*(x**5)
    
def poly_13(x,*param): 

    return param[0]+param[1]*x+param[2]*(x**2)+param[3]*(x**3)+param[4]*(x**4)+param[5]*(x**5)+\
                               param[6]*(x**6)+param[7]*(x**7)+param[8]*(x**8)+param[9]*(x**9)+\
                               param[10]*(x**10)+param[11]*(x**11)+param[12]*(x**12)+param[13]*(x**13)
    
    
#def inv_saturation_spline(x,*param):
    #spline_conf = [np.array([10,10,10,10,30,40,50,60,70,90,110,150,150,150,150]),
    #              np.concatenate([[0],*param,[0,0,0,0]]),
    #              3]
#    spline_conf = interpolate.BSpline([10,10,10,10,30,40,50,60,70,90,110,150,150,150,150],
#                                      np.concatenate([[0],*param,[0,0,0,0]]),3)

#    return interpolate.splev(x, spline_conf, der=0)

def saturation_zero(x,*param):
    #QDC
    slope = param[0]
    sat   = param[1]
    shift = param[2]
    gain  = param[3]
    offset = param[4]

    value = gain*(1+(slope*(x-shift))/np.power(1+np.power((np.abs(slope*x-shift)),sat),1./sat)) + offset
    value[value<offset]=np.zeros(len(value[value<0]))
    return value

def Baseline_work(data, canal, thr, counter):
    datos = data[(data['channel_id']==canal)&(data[thr]==62)]
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


def semigauss_old(x, *param):

    tau_sipm_0 = param[0]
    tau_sipm_1 = param[1]
    gain       = param[2]
    shift      = param[3]

    alfa      = 1.0/tau_sipm_1
    beta      = 1.0/tau_sipm_0
    t_p       = np.log(beta/alfa)/(beta-alfa)
    K         = (beta)*np.exp(alfa*t_p)/(beta-alfa)
    value     = gain*K*(np.exp(-alfa*(x-shift))-np.exp(-beta*(x-shift)))

    value[value<0]=np.zeros(len(value[value<0]))

    return value

def semigauss(x, *param):
    # Wikipedia (https://stackoverflow.com/questions/5884768/skew-normal-distribution-in-scipy)
    e = param[0]
    w = param[1]
    a = param[2]
    gain = param[3]

    t = (x-e)/w

    pdf = 1/np.sqrt(2*np.pi) * np.exp(-t**2/2)
    cdf = (1 + erf(a*t/np.sqrt(2))) / 2

    return gain * 2 / w * pdf * cdf


def moyal(x, a, mu, sigma):
    return a*np.exp(-((x-mu)/sigma + np.exp(-(x-mu)/sigma))/2)/(np.sqrt(2*np.pi)*sigma)


def inv_saturation_spline(x,*param):
    #param_array = np.array(param)
    spline_conf = interpolate.BSpline([10,10,10,10,30,40,50,60,70,90,110,150,150,150,150],
                                      np.concatenate([[0],*param,[0,0,0,0]]),3)

    return interpolate.splev(x, spline_conf, der=0)


def apply_inv_sat_spline(dat):
    dat['efine_corrected'] = dat['efine'] - dat.apply(lambda data: inv_saturation_spline(
                                                                      data['integ_w'],data['spl0'],
                                                                      data['spl1'],data['spl2'],
                                                                      data['spl3'],data['spl4'],
                                                                      data['spl5'],data['spl6'],
                                                                      data['spl7'],data['spl8'],
                                                                      data['spl9']),axis=1)

def log_tot(x,*param):
    
    return param[0]*(np.log(x+param[1]))

def exp_tot(x,*param):
    
    return param[0]*np.exp(x/param[1])

def sqroot_tot(x,*param):

    return param[0]*np.sqrt(x*param[1])



###########################################################################################################

class fitting_nohist_pe(object):
    # General Fitting call
    def __call__(self, data, time, fit_func, guess, sigmas=[], bounds=[]):
        self.bins  = time
        self.data  = data
        self.fit_func = fit_func
        self.guess  = guess
        self.bounds = bounds

        if not bounds:
            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bins,
                                                    self.data, p0=self.guess,
                                                    method='lm'
                                                    )
        else:

            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bins,
                                                        self.data, p0=self.guess,
                                                        ftol=1E-12, maxfev=10000,
                                                        bounds=self.bounds,
                                                        method='trf',
                                                        sigma = sigmas
                                                        )

            self.perr = np.sqrt(np.absolute(np.diag(self.var_matrix)))

            
        self.fit = self.fit_func(self.bins, *self.coeff)
#       self.chisq = np.sum(((self.data-self.fit)/sigmas)**2)
        self.df = len(self.bins)-len(self.coeff)
#       self.chisq_r = self.chisq/self.df
        #Gets fitted function and chisqr_r

    def evaluate(self,in_data):
        return self.fit_func(in_data,*self.coeff)


class fitting_nohist(object):
    # General Fitting call
    def __call__(self, data, time, fit_func, guess, sigmas=[], bounds=[]):
        self.bins  = time
        self.data  = data
        self.fit_func = fit_func
        self.guess  = guess
        self.bounds = bounds

        if not bounds:
            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bins,
                                                    self.data, p0=self.guess,
                                                    method='lm'
                                                    )
        else:
 #           try:
            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bins,
                                                        self.data, p0=self.guess,
                                                        ftol=1E-12, maxfev=10000,
                                                        bounds=self.bounds,
                                                        method='trf',
                                                        sigma = sigmas
                                                        )

            self.perr = np.sqrt(np.absolute(np.diag(self.var_matrix)))
            # Error in parameter estimation
#            except:
#                print("Fitting Problems")
#                self.coeff = np.array(self.guess)
#                self.perr  = np.array(self.guess)

        #if not np.array(sigmas).all():
        #    sigmas = np.ones(len(self.data))

        self.fit = self.fit_func(self.bins, *self.coeff)
        self.chisq = np.sum(((self.data-self.fit)/sigmas)**2)
        self.df = len(self.bins)-len(self.coeff)
        self.chisq_r = self.chisq/self.df
        #Gets fitted function and chisqr_r

    def evaluate(self,in_data):
        return self.fit_func(in_data,*self.coeff)

    
class fitting_hist2(object):
    def __call__(self, data, bins, fit_func, guess, range=None, range_fit=None):
        self.guess = np.array(guess)
        self.bins  = bins
        self.data  = data
        self.fit_func = fit_func
        # Histogram
        self.hist, self.bin_edges = np.histogram(self.data, bins=self.bins, density=False, range=range)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:])/2
   
        # Fit selection
        self.bin_centers_f = self.bin_centers[(self.bin_centers>range_fit[0]) & 
                                              (self.bin_centers<range_fit[1])]
        self.hist_f      = self.hist[(self.bin_centers>range_fit[0]) &
                                              (self.bin_centers<range_fit[1])]
        
        # Fitting function call
        try:
            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bin_centers_f,
                                                        self.hist_f, p0=guess,
                                                        ftol=1E-12, maxfev=1000,
                                                        #bounds = self.bounds,
                                                        method='lm')
            self.perr = np.sqrt(np.absolute(np.diag(self.var_matrix)))

            # Error in parameter estimation
        except:
            print("Fitting Problems")
            self.coeff = np.array(self.guess)
            self.perr  = np.array(self.guess)


        self.hist_fit_f = self.fit_func(self.bin_centers_f, *self.coeff)
        if np.isnan(self.hist_fit_f).any():
            self.chisq_r = 1000
        else:
            self.chisq = np.sum((((self.hist_f-self.hist_fit_f)**2)/self.hist_fit_f))
            self.df = len(self.bin_centers_f)-len(self.coeff)
            self.chisq_r = self.chisq/self.df

        #Gets fitted function and residues

    def evaluate(self,in_data):
        return self.fit_func(in_data,*self.coeff)
    
    def evaluate_f(self):
        return self.fit_func(self.bin_centers_f,*self.coeff)
    
    
class fitting_hist(object):
    def __call__(self, data, bins, fit_func, guess, range=None, range_fit=None):
        self.guess = np.array(guess)
        self.bins  = bins
        self.data  = data
        self.fit_func = fit_func
        # Histogram
        self.hist, self.bin_edges = np.histogram(self.data, bins=self.bins,density=True, range=range)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:])/2
        #self.bounds = [self.guess-self.guess*0.5,self.guess+self.guess*0.5]

        # Fitting function call
        try:
            self.coeff, self.var_matrix = curve_fit(self.fit_func, self.bin_centers,
                                                        self.hist, p0=self.guess,
                                                        ftol=1E-12, maxfev=1000,
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
            self.chisq = np.sum((((self.hist-self.hist_fit)**2)/self.hist_fit))
            self.df = len(self.bin_centers)-len(self.coeff)
            self.chisq_r = self.chisq/self.df

        #Gets fitted function and residues

    def evaluate(self,in_data):
        return self.fit_func(in_data,*self.coeff)


def semigauss_fit(data,bins,*p_param):
    gauss1 = gauss
    #p0 = [np.mean(data), np.std(data), 0, np.max(data)]
    p0 = [np.mean(data), np.std(data), -1, np.max(data)]
    #p0 = [320, 3, -10,1600]
    # First guess
    Q_gauss = fitting_hist()
    Q_gauss(data=data,
            bins=bins,
            guess=p0,
            fit_func=semigauss)

    if p_param[0]==True:
        p_param[1].hist(data, bins, align='mid', facecolor='green',
                        edgecolor='white', linewidth=0.5)
        p_param[1].plot(Q_gauss.bin_centers, Q_gauss.hist_fit, 'r--', linewidth=1)

    e = Q_gauss.coeff[0]
    w = Q_gauss.coeff[1]
    a = Q_gauss.coeff[2]
    gain = Q_gauss.coeff[3]

    d    = a/np.sqrt(1+a**2)
    #mu_z = np.sqrt(2/np.pi) * d
    #r_s  = np.sqrt(1-mu_z**2)
    #skew = ((4-np.pi)/2) * ((d*np.sqrt(2/np.pi))**3 / (1-2*(d**2)/np.pi)**1.5)
    #moda = mu_z - skew * r_s/2 - np.sign(a)/2 * np.exp(-(2*np.pi)/np.abs(a))

    rango  = np.arange(np.min(data),np.max(data),0.001)
    moda   = rango[np.argmax(Q_gauss.evaluate(rango))]
    sigma  = w*np.sqrt(1-(2*d**2/np.pi))


    return Q_gauss.coeff, Q_gauss.perr, moda, sigma, Q_gauss.chisq_r



def gauss_fit(data,bins,*p_param):
    gauss1 = gauss
    p0 = [1, np.mean(data), np.std(data)]
    #p0 = [0.25, 0.25, 200, 400]
    # First guess
    Q_gauss = fitting_hist()
    Q_gauss(data=data,
            bins=bins,
            guess=p0,
            fit_func=gauss,
            range = p_param[6])

    if p_param[0]==True:
        #p_param[1] -> axis
        #p_param[2] -> title
        #p_param[3] -> xlabel
        #p_param[4] -> ylabel
        #p_param[5] -> pos [0.95,0.95,"left"]

        p_param[1].hist(data, bins, align='mid', facecolor='green',
                        edgecolor='white', linewidth=0.5,density=True,range=p_param[6])
        p_param[1].plot(Q_gauss.bin_centers, Q_gauss.hist_fit, 'r--', linewidth=1)
        p_param[1].set_xlabel(p_param[2])
        p_param[1].set_ylabel(p_param[3])
        p_param[1].set_title(p_param[4])
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



def gauss_fit2(data,bins,*p_param):
    gauss1 = gauss
    
    Q_gauss = fitting_hist2()
    Q_gauss(data=data,
            bins=bins,
            guess=p_param[8],
            fit_func=gauss,
            range = p_param[6],
            range_fit = p_param[7] )

    if p_param[0]==True:
        #p_param[1] -> axis
        #p_param[2] -> title
        #p_param[3] -> xlabel
        #p_param[4] -> ylabel
        #p_param[5] -> pos [0.95,0.95,"left"]

        p_param[1].hist(data, bins, align='mid', facecolor='green',
                        edgecolor='white', linewidth=0.5,density=False,range=p_param[6])
        #p_param[1].bar(Q_gauss.bin_centers_f,Q_gauss.hist_f)
        p_param[1].plot(Q_gauss.bin_centers, Q_gauss.evaluate(Q_gauss.bin_centers), 'r--', linewidth=1)
        
        p_param[1].set_xlabel(p_param[2])
        p_param[1].set_ylabel(p_param[3])
        p_param[1].set_title(p_param[4])
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


def semigauss_fit2(data,bins,*p_param):
    
    Q_gauss = fitting_hist2()
    Q_gauss(data=data,
            bins=bins,
            guess=p_param[8],
            fit_func=moyal,
            range = p_param[6],
            range_fit = p_param[7] )

    if p_param[0]==True:
        #p_param[1] -> axis
        #p_param[2] -> title
        #p_param[3] -> xlabel
        #p_param[4] -> ylabel
        #p_param[5] -> pos [0.95,0.95,"left"]

        p_param[1].hist(data, bins, align='mid', facecolor='green',
                        edgecolor='white', linewidth=0.5,density=False,range=p_param[6])
        p_param[1].plot(Q_gauss.bin_centers_f, Q_gauss.evaluate_f(), 'r--', linewidth=1)
        
        #e        = Q_gauss.coeff[0]
        #e_err    = Q_gauss.perr[0]
        #w        = Q_gauss.coeff[1]
        #w_err    = Q_gauss.perr[1]
        #a        = Q_gauss.coeff[2]
        #a_err    = Q_gauss.perr[2]
        #gain     = Q_gauss.coeff[3]
        #gain_err = Q_gauss.perr[3]
        
        #sigma    = a/np.sqrt(1+a**2)
        #mean     = e + w*sigma*np.sqrt(2/np.pi)
        #mu_z = np.sqrt(2/np.pi)*sigma
        #r_s  = np.sqrt(1-mu_z**2)
        
        #skew = ((4-np.pi)/2) * ((sigma*np.sqrt(2/np.pi))**3 / (1-2*(sigma**2)/np.pi)**1.5)
        #m0   = mu_z - skew * r_s/2 - np.sign(a)/2 * np.exp(-(2*np.pi)/np.abs(a))
        #mode     = e + w*m0
              
        #dsigma_err_da = (1/np.sqrt(1+a**2))-((2*a**2)/(1+a**2)**1.5)
        #print("dsigma_err_da ",dsigma_err_da)
        #sigma_err = np.sqrt(dsigma_err_da**2 * a_err**2)
        #print("sigma_err ",sigma_err)
        
        #sigma_true         = w * np.sqrt(1-2*sigma**2/np.pi)
        #dsigma_true_da = (-2*w/np.pi) / np.sqrt(1-2*sigma**2/np.pi) * (2*a*(1+a**2)-4*a**3)/(1+a**2)**2
        #print("dsigma_true_dsigma ",dsigma_true_da)
        
        #dsigma_true_dw     = np.sqrt(1-2*sigma**2/np.pi)
        #print("dsigma_true_dw ",dsigma_true_dw)
        
        #sigma_true_err     = np.sqrt(   w_err**2 * dsigma_true_dw**2 + 
        #                                a_err**2 * (dsigma_true_da)**2 )
              
        
        p_param[1].set_xlabel(p_param[2])
        p_param[1].set_ylabel(p_param[3])
        p_param[1].set_title(p_param[4])
        
        text_chain = ("$\mu$=%0.3f (+/- %0.3f) \n $\sigma$=%0.3f (+/- %0.3f) \n" % 
                             (Q_gauss.coeff[1], Q_gauss.perr[1], Q_gauss.coeff[2], Q_gauss.perr[2]))
        
        p_param[1].text(p_param[5][0],p_param[5][1], text_chain, fontsize=8, verticalalignment='top',
                                         horizontalalignment=p_param[5][2],
                                         transform=p_param[1].transAxes)

    return Q_gauss.coeff,Q_gauss.perr,Q_gauss.chisq_r



def Tn_fit(data, canal, thr, min_count=10, plot=False, axis=[],
           guess=[1.83, 11.11, 55.7, 2.1e+06]):
    # Fitting Thresholds

    slope = guess[0]
    sat   = guess[1]
    shift = guess[2]
    gain  = guess[3]
    #offset = guess[4]

    #baseline_T = Baseline_work(data, canal, 2**22)
    #baseline_T = baseline_T - 10   # 5 is a guess for a better fit of THn
    #print("Baseline %f" % baseline_T)

    datos = data[(data['channel_id']==canal)] # & (data['baseline_t']==baseline_T)]

    T_fit = fitting_nohist()

    T_fit(datos['count'],datos[thr],saturation,[slope,sat,shift,gain],
          np.zeros(len(datos['count']))+1.0,[[0,3,0,1E6],[4E6,15,63,10E6]])

    #chisq = np.sum(((datos['count']-T_fit.evaluate(datos['vth_t1']))/1.0)**2)
    #print("Channel = %d / CHISQR = %f" % (canal,T_fit.chisq_r))

    if plot==True:
        #plt.figure()
        axis.plot(np.arange(0,64,0.1),T_fit.evaluate(np.arange(0,64,0.1)),'b-',label="Fit")
        axis.errorbar(datos[thr],datos['count'], 1.0,
                     fmt='.',color='red',label="Data")
        #axis.xlabel("th_T1")
        #axis.ylabel("COUNT")
        #plt.legend()

    # This threshold computation stinks
    #func = lambda x : min_count - saturation(x,*T_fit.coeff)
    #try:
    #    t_solution = np.floor(brentq(func, 0, 62))
    #    print("Threshold at %f for a 0.01 activity" % t_solution)
    #    print(T_fit.coeff)
    #except:
    #    print("No solution found")
    #    print(T_fit.coeff)
    #    t_solution = -1

    datos_f = np.array(datos['count'])

    t_solution = -1
    i = int(np.floor(T_fit.coeff[2]))
    
    while t_solution<0:
        if (datos_f[i] < min_count):
            t_solution = i
        i = i - 1


    return t_solution,T_fit


def QDC_fit(data, canal, tac, plot=False, guess=[0.05,12,90,300,100],axis=0):
    #Fitting QDC parameters
    slope = guess[0]
    sat   = guess[1]
    shift = guess[2]
    gain  = guess[3]
    offset = guess[4]

    datos = data[(data['tac_id']==tac)&(data['channel_id']==canal)]

    Q_fit = fitting_nohist()
    coeff  = [slope,sat,shift,gain]

    guess_a = np.array(guess)

    Q_fit(datos['mu'],datos['tpulse'],saturation,[slope,sat,shift,gain,offset],datos['sigma'],
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
        #plt.figure()
        axis.plot(datos['tpulse'],Q_fit.evaluate(datos['tpulse']),'b-',label="Fit")
        axis.errorbar(datos['tpulse'],datos['mu'], datos['sigma'],
                     fmt='.',color='red',label="Data")
        #plt.xlabel("Length")
        #plt.ylabel("QFINE")
        #plt.legend()

    return Q_fit.chisq_r, Q_fit, qoffset, max_slope


def QDC_fit_p(data, canal, tac, plot=False, guess=[0,0,0,0,0,0,0,0,0,0],sigmas=[],axis=0):
    #Fitting QDC parameters

    datos = data[(data['tac_id']==tac)&(data['channel_id']==canal)]

    Q_fit = fitting_nohist()

    guess_a = np.array(guess)
    #bounds = [[-np.Inf,-np.Inf,-np.Inf,-np.Inf,-np.Inf,-np.Inf,-np.Inf,-np.Inf,-np.Inf,-np.Inf],
    #           [ np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf]]

    
    Q_fit(datos['mu'],datos['intg_w'],saturation_poly,guess)#,sigmas,bounds)


    if plot==True:
        #plt.figure()
        axis.plot(datos['intg_w'],Q_fit.evaluate(datos['intg_w']),'b-',label="Fit")
        axis.errorbar(datos['intg_w'],datos['mu'], datos['sigma'],
                     fmt='.',color='red',label="Data")
        #plt.xlabel("Length")
        #plt.ylabel("QFINE")
        #plt.legend()

    return Q_fit.coeff



def QDC_inter_sp(data, canal, tac, plot=False, axis=0):

    datos = data[(data['tac_id']==tac)&(data['channel_id']==canal)]

    t_pulse = datos['tpulse'].to_numpy()
    mu      = datos['mu'].to_numpy()
    t_pulse = np.concatenate([t_pulse[1:7],t_pulse[7:16:2]])
    mu      = np.concatenate([mu[1:7],mu[7:16:2]])


    spl_conf = interpolate.splrep(t_pulse,mu,s=0)

    spl_conf = spl_conf[1][1:11]

    if plot==True:
        #plt.figure()
        xnew = np.arange(20,180)
        ynew = inv_saturation_spline(xnew,spl_conf)
        axis.plot(xnew,ynew,'b-',label="Fit")
        axis.errorbar(datos['tpulse'],datos['mu'], datos['sigma'],
                     fmt='.',color='red',label="Data")
        #plt.xlabel("Length")
        #plt.ylabel("QFINE")
        #plt.legend()

    return spl_conf


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
    sl_window = np.concatenate([np.diff(datos['mu']),[0]])
    phase_array = datos['phase']

    #plt.figure()
    #phase_fine = datos['phase']
    #plt.errorbar(datos.phase,datos['mu'], datos['sigma'],
    #                 fmt='.',color='red',label="Data")

    #Top slope points
    maxis = np.argwhere(sl_window > 100)
    #Check if next sample comes back to zero (phase gap) o has an equal but negative slope
    jump = np.argwhere(sl_window[maxis+1] > -25)
    try:
        salto = phase_array.to_numpy()[maxis[jump[0,0]]]
    except:
        salto = 359
    print("GAP",phase_array.to_numpy()[maxis[jump]])

    shift_min = salto - 20 #datos['phase'].iloc[salto]-20
    shift_max = salto + 20 #datos['phase'].iloc[salto]+20

    if shift_min < 0:
        shift_min = 0
    if shift_max > 359.5:
        shift_max = 359.5

    shift = shift_min

    # Now let's build a new dataframe with the right mu and sigma

    indexes = datos.index
    for i in datos['phase']:
        index = indexes[datos['phase']==i]
        #print(datos.loc[index,'mu'].values)
        if salto < 30:
            if ((i < salto) | (i > salto + 40.0)):
                if (datos.loc[index,'mu'].values > datos.loc[index,'mu2'].values):
                    datos.loc[index,'mu'] = datos.loc[index,'mu2']
                    datos.loc[index,'sigma'] = datos.loc[index,'sigma']
            else:
                if (datos.loc[index,'mu'].values < datos.loc[index,'mu2'].values):
                    datos.loc[index,'mu'] = datos.loc[index,'mu2']
                    datos.loc[index,'sigma'] = datos.loc[index,'sigma']
        elif salto > 330:
            if ((i < salto) & (i > salto - 40.0)):
                if (datos.loc[index,'mu'].values > datos.loc[index,'mu2'].values):
                    datos.loc[index,'mu'] = datos.loc[index,'mu2']
                    datos.loc[index,'sigma'] = datos.loc[index,'sigma']
            else:
                if (datos.loc[index,'mu'].values < datos.loc[index,'mu2'].values):
                    datos.loc[index,'mu'] = datos.loc[index,'mu2']
                    datos.loc[index,'sigma'] = datos.loc[index,'sigma']
        else:
            if i < salto:
                if (datos.loc[index,'mu'].values > datos.loc[index,'mu2'].values):
                    datos.loc[index,'mu'] = datos.loc[index,'mu2']
                    datos.loc[index,'sigma'] = datos.loc[index,'sigma']
            else:
                if (datos.loc[index,'mu'].values < datos.loc[index,'mu2'].values):
                    datos.loc[index,'mu'] = datos.loc[index,'mu2']
                    datos.loc[index,'sigma'] = datos.loc[index,'sigma']



    amplitude = (np.min(datos['mu']) - np.max(datos['mu']))/2.0
    offset    = np.min(datos['mu']) + np.abs(amplitude)
    bounds    = [[amplitude-20,shift_min,offset-20],[amplitude+20,shift_max,offset+20]]

    # Sigmas for fit optimizing
    sigma_w = datos['sigma'].to_numpy()
    weight = np.ones(sigma_w.shape)
    min_pos = np.argmin(datos['mu'])
    max_pos = np.argmax(datos['mu'])

    zone = 5
    extra = 8

    if min_pos-zone > 0:
        weight[min_pos-zone:min_pos]=weight[min_pos-zone:min_pos]*extra
    else:
        weight[:min_pos]=weight[:min_pos]*extra

    if max_pos+zone < len(sigma_w):
        weight[max_pos:max_pos+zone]=weight[max_pos:max_pos+zone]*extra
    else:
        weight[max_pos:]=weight[max_pos:]*extra

    sigma_w = sigma_w*weight


    while((chisq_r > 2) & (shift < shift_max)):

        Q_fit = fitting_nohist()
        coeff  = [amplitude,period,shift,offset]

        Q_fit(datos['mu'],datos['phase'],sawtooth,[amplitude,shift,offset],sigma_w,bounds=bounds)
        chisq_r = Q_fit.chisq_r

        if best_chi > chisq_r:
            best_chi = chisq_r
            best_shift = shift

        shift = shift + 0.25

    Q_fit(datos['mu'],datos['phase'],sawtooth,[amplitude,best_shift,offset],sigma_w,bounds=bounds)
    chisq_r = Q_fit.chisq_r
    print("Channel = %d / TAC = %d / CHISQR_r = %f" % (canal,tac,chisq_r))


    if plot==True:
        plt.figure()
        #phase_fine = datos['phase']
        phase_fine = np.arange(0,360,0.125)
        plt.plot(phase_fine,Q_fit.evaluate(phase_fine),'b-',label="Fit")
        plt.errorbar(datos.phase,datos['mu'], datos['sigma'],
                     fmt='.',color='red',label="Data")
        plt.xlabel("PHASE")
        plt.ylabel("TFINE")
        plt.legend()

    return chisq_r, Q_fit.coeff
