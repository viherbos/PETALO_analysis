import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import brentq
from scipy.special import erf
from scipy import interpolate
from scipy import signal
import multiprocessing as mp

def gauss(x, *param):
    return param[0] * np.exp(-(x-param[1])**2/(2.*param[2]**2))

def exp_pure(x, *param):
    return param[0] * np.exp(x/param[1]) #(np.exp(x/param[1])+np.exp(-(x-param[3])/param[2]))
    #(x/param[3]-param[2])**2 / ((x/param[3]-param[2])**2+1)
    #( 1/(1+np.exp(-1*(x-param[2])/param[3]) ))

def moyal(x, a, mu, sigma):
    return a*np.exp(-((x-mu)/sigma + np.exp(-(x-mu)/sigma))/2)/(np.sqrt(2*np.pi)*sigma)

def shaped_spe(time, tau_sipm):
    alfa      = 1.0/tau_sipm[1]
    beta      = 1.0/tau_sipm[0]
    t_p       = np.log(beta/alfa)/(beta-alfa)
    K         = (beta)*np.exp(alfa*t_p)/(beta-alfa)
    time_dist = K*(np.exp(-alfa*time)-np.exp(-beta*time))
    return time_dist

def channel_shp_cheby_spe(time,**kwargs):
    param  = {      'BW': 650E3,
                    'os': 0.5,
                    'pes':1,
              'f_sample': 1E9,
              'target_q':1.28E-12
              }

    for key,value in kwargs.items():
        if (key in param.keys()):
            param[key] = value
        else:
            print("Parameter %s is not valid" % key)

    #Second order cheby filtered Semigaussian pulse
    RC=1/(param['BW']*2*np.pi)
    freq_RCd = 1/(RC*param['f_sample']*np.pi)
    bsl1, asl1 = signal.cheby1(2, param['os'], freq_RCd, 'low', analog=False);
    BS = bsl1*(10**(param['os']/20))
    AS = asl1
    spe_wave_pure   = shaped_spe(time,[1,40])
    spe_wave_q_norm = spe_wave_pure/(np.sum(spe_wave_pure)*(1.0/f_sample))
    spe_wave        = param['pes'] * param['target_q'] * spe_wave_q_norm
    signal_out = signal.lfilter(BS,AS,spe_wave)
    return signal_out

def channel_shp_2nd_spe(time,**kwargs):
    param  = {      'f0': 650E3,
                   'rho': 0.5,
                    'pes':1,
              'f_sample': 1E9,
              'target_q':1.28E-12
              }

    for key,value in kwargs.items():
        if (key in param.keys()):
            param[key] = value
        else:
            print("Parameter %s is not valid" % key)
    #Second order cheby filtered Semigaussian pulse
    RC=1/(param['f0']*2*np.pi)
    freq_RCd = 1/(RC*param['f_sample']*np.pi)
    bsl1, asl1 = signal.bilinear((param['f0']*2*np.pi)**2,[1,2*param['rho']*param['f0']*2*np.pi,(param['f0']*2*np.pi)**2],param['f_sample']);
    BS = bsl1
    AS = asl1
    spe_wave_pure   = shaped_spe(time,[1,40])
    spe_wave_q_norm = spe_wave_pure/(np.sum(spe_wave_pure)*(1.0/param['f_sample']))
    spe_wave        = param['pes'] * param['target_q'] * spe_wave_q_norm
    signal_out = signal.lfilter(BS,AS,spe_wave)
    return signal_out

def channel_shp_1st_spe(time,**kwargs):
    param  = {      'BW': 650E3,
                    'pes':1,
              'f_sample': 1E9,
              'target_q':1.28E-12
              }

    for key,value in kwargs.items():
        if (key in param.keys()):
            param[key] = value
        else:
            print("Parameter %s is not valid" % key)

    #RC2 Semigaussian pulse
    RC=1/(param['BW']*2*np.pi)
    freq_RCd = 1/(RC*param['f_sample']*np.pi)
    bsl1, asl1 = signal.butter(1, freq_RCd, 'low', analog=False);
    BS = bsl1
    AS = asl1
    spe_wave_pure   = shaped_spe(time,[1,40])
    spe_wave_q_norm = spe_wave_pure/(np.sum(spe_wave_pure)*(1.0/param['f_sample']))
    spe_wave        = param['pes'] * param['target_q'] * spe_wave_q_norm
    signal_out = signal.lfilter(BS,AS,spe_wave)
    return signal_out

def channel_shaped_spe(time,**kwargs):
    param  = {      'BW': 650E3,
                    'pes':1,
              'f_sample': 1E9,
              'target_q':1.28E-12
              }

    for key,value in kwargs.items():
        if (key in param.keys()):
            param[key] = value
        else:
            print("Parameter %s is not valid" % key)
    #RC2 Semigaussian pulse
    RC=1/(param['BW']*2*np.pi)
    freq_RCd = 1/(RC*param['f_sample']*np.pi)
    bsl1, asl1 = signal.butter(1, freq_RCd, 'low', analog=False);
    BS = np.convolve(bsl1,bsl1)
    AS = np.convolve(asl1,asl1)
    spe_wave_pure   = shaped_spe(time,[1,40])
    spe_wave_q_norm = spe_wave_pure/(np.sum(spe_wave_pure)*(1.0/param['f_sample']))
    spe_wave        = param['pes'] * param['target_q'] * spe_wave_q_norm
    signal_out = signal.lfilter(BS,AS,spe_wave)
    return signal_out

def compute_mode(fit_class, bins_around_max):
    D = fit_class.hist[fit_class.hist.argmax()]
    N = fit_class.bin_centers[fit_class.hist.argmax()]*D
    for i in range(bins_around_max):
        N = N + fit_class.bin_centers[fit_class.hist.argmax()+i]*fit_class.hist[fit_class.hist.argmax()+i] + \
        fit_class.bin_centers[fit_class.hist.argmax()-i]*fit_class.hist[fit_class.hist.argmax()-i]
        D = D + fit_class.hist[fit_class.hist.argmax()+i] + fit_class.hist[fit_class.hist.argmax()-i]
    return N/D

def ToT2pe(x,df_ToT2pe):
    if (x*5 < np.min(df_ToT2pe['ToT_ns'])):
        return 0
    else:
        return df_ToT2pe.iloc[(np.abs(df_ToT2pe['ToT_ns']-x*5)).argmin()]['pe']



class data_process(object):
    def __init__(self,path,**kwargs):
        self.param  = {'run':11309,
                       'path': "/analysis/11309/hdf5/proc/linear_interp/files/",
                       'asic': 0,
                       'path_table': "/home/vherrero/CALIBRATION_FILES/",
                       'file_table': "ToT_PE_conversion12d_high.h5",
                       'path_out_images':"/home/vherrero/RESULTS/images/",
                       'tot_yellow_limits':[80,300],
                       'tot_red_limits':[300,1700],
                       'tot_grey_limits':[1700,3000],
                       'tot_blue_limits':[3000,4250]
                       }

        for key,value in kwargs.items():
            if (key in self.param.keys()):
                self.param[key] = value
            else:
                print("Parameter %s is not valid" % key)

        self.table = pd.read_hdf(self.param['path_table'] + \
                                 self.param['file_table'], '/ToT_T2')

        self.lo_ToT = self.table.iloc[0]['ToT_ns']/5.0
        self.hi_ToT = self.table.iloc[-1]['ToT_ns']/5.0

    def event_selection_center(self,df,column):
        max_pos = df[column].argmax()
        res1 = df.iloc[max_pos]['sensor_id'] in [44,45,54,55]
        return res1

    def ToT2pe(self,x,df_ToT2pe):
        return df_ToT2pe.iloc[(np.abs(df_ToT2pe['ToT_ns']-x*5)).argmin()]['pe']

    def process_energy_ToT(self,file_number):

        energy_ToT = []
        data_df_aux = []

        file_data = self.param['path'] + 'run_' + str(self.param['run']) +\
                        '_'+ str(file_number).zfill(4) + '_trigger1_waveforms.h5'

        with pd.HDFStore(file_data,'r',complib="zlib",complevel=4) as storage:
            keys = storage.keys()
            for j in [keys[i] for i in range(10)]:
                print("File %i | Chunk %s" % (file_number,j))
                data = pd.read_hdf(file_data,j) #,stop=10000)

                # Apply Filtering
                data_f = data[(data['cluster']!=-1)]

                # Coincidence Filter
                coincidence   = data_f.groupby(['cluster','evt_number'])['tofpet_id'].nunique()
                data_f_idx    = data_f.set_index(['cluster','evt_number'])
                data_f        = data_f_idx.loc[coincidence[coincidence == 2].index]

                # Only TOFPET0
                data_f0 = data_f.loc[(data_f['tofpet_id']==0)]

                # ToT computation
                data_f0['intg_w_ToT'] = data_f0.loc[:,'t2'] - data_f0.loc[:,'t1']

                # Get rid of abnormaly long ToT
                data_f0 = data_f0[(data_f0['intg_w_ToT']>self.lo_ToT) &
                                  (data_f0['intg_w_ToT']<self.hi_ToT)]

                # Event selection based on Maximum Charge
                sel = data_f0.groupby(['cluster','evt_number']).apply(self.event_selection_center,column='intg_w_ToT')
                data_f1 = data_f0[sel]
                # Energy computation
                energy = data_f1.groupby(['cluster','evt_number'], as_index = False)['intg_w_ToT'].sum()
                energy_ToT.extend(energy['intg_w_ToT'].to_numpy())
                data_df_aux.append(data_f1)
                data_df_T = pd.concat(data_df_aux)

        return energy_ToT, data_df_T


    def process_energy_ToT_PE(self,file_number):
    # Process energy histogram from files
    # To be used with
        energy_ToT_PE = []
        data_df_aux = []

        file_data = self.param['path'] + 'run_' + str(self.param['run']) +\
                        '_'+ str(file_number).zfill(4) + '_trigger1_waveforms.h5'

        with pd.HDFStore(file_data,'r',complib="zlib",complevel=4) as storage:
            keys = storage.keys()
            for j in [keys[i] for i in range(10)]:
                print("File %i | Chunk %s" % (file_number,j))
                data = pd.read_hdf(file_data,j) #,stop=10000)

                # Apply Filtering
                data_f = data[(data['cluster']!=-1)]

                # Coincidence Filter
                coincidence   = data_f.groupby(['cluster','evt_number'])['tofpet_id'].nunique()
                data_f_idx    = data_f.set_index(['cluster','evt_number'])
                data_f        = data_f_idx.loc[coincidence[coincidence == 2].index]
                # Only TOFPET0
                data_f0 = data_f.loc[(data_f['tofpet_id']==0)]
                # ToT computation
                data_f0['intg_w_ToT'] = data_f0.loc[:,'t2'] - data_f0.loc[:,'t1']
                # Get rid of abnormaly long ToT
                data_f0 = data_f0[(data_f0['intg_w_ToT']>self.lo_ToT) &
                                  (data_f0['intg_w_ToT']<self.hi_ToT)]
                # Event selection based on Maximum Charge
                sel = data_f0.groupby(['cluster','evt_number']).apply(self.event_selection_center,column='intg_w_ToT')
                data_f1 = data_f0[sel]

                data_f1['ToT_pe'] = data_f1['intg_w_ToT'].apply(ToT2pe, df_ToT2pe=self.table)
                energy = data_f1.groupby(['cluster','evt_number'], as_index = False)['ToT_pe'].sum()

                energy_ToT_PE.extend(energy['ToT_pe'].to_numpy())
                data_df_aux.append(data_f1)
                data_df_T = pd.concat(data_df_aux)

        return energy_ToT_PE, data_df_T


    # def process_zone_ToT_PE(self,df_zone):
    # # Process PE values for channels per zones
    # # Remember to change self.table before applying this function
    #     df_zone['ToT_pe'] = df_zone['intg_w_ToT'].apply(ToT2pe, df_ToT2pe=self.table)
    #     max_sipm  = df_zone.groupby(['cluster','evt_number'])['ToT_pe'].max()
    #     print("Thread ends")
    #     return max_sipm
    #
    #
    # def MP_process_zone_ToT_PE(self,df_zone,n_processors):
    #     chunk_size = int(np.floor(len(df_zone)/n_processors))
    #     df_zone_chunk = []
    #     for i in range(n_processors):
    #         df_zone_chunk.append(df_zone.iloc[int(chunk_size*i):int(chunk_size*(i+1))])
    #     if ((len(df_zone) % chunk_size) > 0):
    #         df_zone_chunk.append(df_zone.iloc[n_processors:])
    #
    #     df_array = []
    #     pool = mp.Pool(processes = n_processors)
    #     pool_output = pool.map(self.process_zone_ToT_PE, df_zone_chunk)
    #     pool.close()
    #     pool.join()
    #
    #     sizes = np.shape(pool_output)
    #
    #     for x in range(sizes[0]):
    #         df_array.extend(pool_output[x])
    #
    #     return df_array


    def extract_data_from_zone(self, df_zone, range_show, range_fit, bins):
        max_sipm  = df_zone.groupby(['cluster','evt_number'])['intg_w_ToT'].max()
        max_sipm_fit = fit_hist(max_sipm, fit_func = "none", bins=bins,
                                range_show=range_show, range_fit=range_fit)

        return compute_mode(max_sipm_fit,2)

    def extract_zones(self,df_out):
        self.energy_w = df_out.groupby(['cluster','evt_number'])['intg_w_ToT'].sum()
        # Computes Mode of PE distribution in the three selected zones
        mode_yrgb = []
        df_all = []

        df_yellow = df_out.loc[(self.energy_w > self.param['tot_yellow_limits'][0]) &
                               (self.energy_w < self.param['tot_yellow_limits'][1])]
        df_all.append(df_yellow)

        df_red = df_out.loc[(self.energy_w > self.param['tot_red_limits'][0]) &
                            (self.energy_w < self.param['tot_red_limits'][1])]
        df_all.append(df_red)
        df_grey = df_out.loc[(self.energy_w > self.param['tot_grey_limits'][0]) &
                             (self.energy_w < self.param['tot_grey_limits'][1])]
        df_all.append(df_grey)
        df_blue = df_out.loc[(self.energy_w > self.param['tot_blue_limits'][0]) &
                             (self.energy_w < self.param['tot_blue_limits'][1])]
        df_all.append(df_blue)

        for i in df_all:
            value = self.extract_data_from_zone(i, [0,400], [0,400], 80)
            mode_yrgb.append(value)

        # ToT in clock cycles
        return np.array(mode_yrgb)

class ToT2pe_table(object):
    def __init__(self,**kwargs):
        self.param  = {'Gain': 3000,
                       'target_q_VOV_8': 1.28E-12,
                       'T_sample': 1E-9,
                       'max_time':1500,
                       'max_pe':3500,
                       'pe_step':0.125,
                       'channel_model':channel_shaped_spe}

        for key,value in kwargs.items():
            if (key in self.param.keys()):
                self.param[key] = value
            else:
                print("Parameter %s is not valid" % key)

        self.time = np.arange(0,self.param['max_time'],1)
        spe_wave_pure   = shaped_spe(self.time,[1,40])
        self.spe_wave_q_norm = spe_wave_pure/(np.sum(spe_wave_pure)*(self.param['T_sample']))

    def compute_table(self, **kwargs):
        param2       = {'BW':650E3,
                        'T2_eff':200E-3,
                        'f0':600E-3,
                        'rho':0.5,
                        'os':1
                        }
        for key,value in kwargs.items():
            if (key in param2.keys()):
                param2[key] = value
            else:
                print("Parameter %s is not valid" % key)

        ToT_T2 = []

        if (self.param['channel_model'].__name__==channel_shaped_spe.__name__):
            self.wave_1 = self.param['channel_model'](self.time,BW=param2['BW'],pes=1)
            self.T2_pe  = (param2['T2_eff']/self.param['Gain'])/(np.max(self.wave_1))
            self.pe_peak_sipm   = np.max(self.spe_wave_q_norm*self.param['target_q_VOV_8'])
            Q_ToT_T2 = np.arange(self.T2_pe + 0.25,self.param['max_pe'],self.param['pe_step'])
            ToT = ToT_computation(mode='T2', T2 = param2['T2_eff']/self.param['Gain'])
            for A in Q_ToT_T2:
                wave = self.param['channel_model'](self.time,BW=param2['BW'],pes=A)
                aux1 = ToT.ToT(self.time, wave)
                ToT_T2.append(aux1)

        elif (self.param['channel_model'].__name__==channel_shp_cheby_spe.__name__):
            self.wave_1 = self.param['channel_model'](self.time,BW=param2['BW'],os=param2['os'],pes=1)
            self.T2_pe  = (param2['T2_eff']/self.param['Gain'])/(np.max(self.wave_1))
            self.pe_peak_sipm   = np.max(self.spe_wave_q_norm*self.param['target_q_VOV_8'])
            Q_ToT_T2 = np.arange(self.T2_pe + 0.25,self.param['max_pe'],self.param['pe_step'])
            ToT = ToT_computation(mode='T2', T2 = param2['T2_eff']/self.param['Gain'])
            for A in Q_ToT_T2:
                wave = self.param['channel_model'](self.time,BW=param2['BW'],os=param2['os'],pes=A)
                aux1 = ToT.ToT(self.time, wave)
                ToT_T2.append(aux1)

        elif (self.param['channel_model'].__name__==channel_shp_2nd_spe.__name__):
            self.wave_1 = self.param['channel_model'](self.time,f0=param2['f0'],rho=param2['rho'],pes=1)
            self.T2_pe  = (param2['T2_eff']/self.param['Gain'])/(np.max(self.wave_1))
            self.pe_peak_sipm   = np.max(self.spe_wave_q_norm*self.param['target_q_VOV_8'])
            Q_ToT_T2 = np.arange(self.T2_pe + 0.25,self.param['max_pe'],self.param['pe_step'])
            ToT = ToT_computation(mode='T2', T2 = param2['T2_eff']/self.param['Gain'])
            for A in Q_ToT_T2:
                wave = self.param['channel_model'](self.time,f0=param2['f0'],rho=param2['rho'],pes=A)
                aux1 = ToT.ToT(self.time, wave)
                ToT_T2.append(aux1)

        elif (self.param['channel_model'].__name__==channel_shp_1st_spe.__name__):
            self.wave_1 = self.param['channel_model'](self.time,BW=param2['BW'],pes=1)
            self.T2_pe  = (param2['T2_eff']/self.param['Gain'])/(np.max(self.wave_1))
            self.pe_peak_sipm   = np.max(self.spe_wave_q_norm*self.param['target_q_VOV_8'])
            Q_ToT_T2 = np.arange(self.T2_pe + 0.25,self.param['max_pe'],self.param['pe_step'])
            ToT = ToT_computation(mode='T2', T2 = param2['T2_eff']/self.param['Gain'])
            for A in Q_ToT_T2:
                wave = self.param['channel_model'](self.time,BW=param2['BW'],pes=A)
                aux1 = ToT.ToT(self.time, wave)
                ToT_T2.append(aux1)

        mini = np.min(ToT_T2)
        maxi = np.max(ToT_T2)
        res2 = interpolate.interp1d(ToT_T2, Q_ToT_T2)
        table2 = res2(range(mini,maxi,1))
        table_df2 = pd.DataFrame(np.concatenate((np.arange(mini,maxi,1).reshape(-1,1),
                                                 table2.reshape(-1,1)),axis=1),
                                columns=['ToT_ns','pe'])
        return table_df2


class ToT_computation(object):
    def __init__(self,**kwargs):
        self.param  = {'mode': 'T2',
                       'T2': 0,
                       'T1': 0}
        for key,value in kwargs.items():
            if (key in self.param.keys()):
                self.param[key] = value
            else:
                print("Parameter %s is not valid" % key)

    def ToT(self, time, wave):
        if (self.param['mode']=='T2'):
            if time[wave >= self.param['T2']].any():
                # Usual ToT mode configuration
                cross_T2_1 = np.min(time[wave >= self.param['T2']])
                cross_T2_2 = np.max(time[wave >= self.param['T2']])
                ToT = cross_T2_2 - cross_T2_1
            else:
                return 0
        elif (self.param['mode']=='T2_T1'):
            if time[wave >= self.param['T2']].any():
                # Real integration window in QDC mode
                cross_T1_1 = np.min(time[wave >= self.param['T1']])
                cross_T1_2 = np.max(time[wave >= self.param['T1']])
                cross_T2_1 = np.min(time[wave >= self.param['T2']])
                ToT = cross_T1_2 - cross_T2_1
            else:
                return 0
        else:
            print("Mode not recognized")
        return ToT


#################################################################################
# Fitting
#################################################################################

class fit_hist(object):
    def __init__(self, data, **kwargs):
        self.data  = data
        # Default params
        self.param  = {'fit_func': gauss,
                       'bins' : 50,
                       'guess': [1,1,1],
                       'range_show': [0,100],
                       'range_fit':[0,100]}

        for key,value in kwargs.items():
            if (key in self.param.keys()):
                self.param[key] = value
            else:
                print("Parameter %s is not valid" % key)

        # Histogram
        self.hist, self.bin_edges = np.histogram(self.data,
                                                 bins=self.param['bins'],
                                                 density=False,
                                                 range=self.param['range_show'])
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:])/2

        # Fit selection
        self.bin_centers_f = self.bin_centers[(self.bin_centers > self.param['range_fit'][0]) &
                                              (self.bin_centers < self.param['range_fit'][1])]
        self.hist_f      = self.hist[(self.bin_centers > self.param['range_fit'][0]) &
                                     (self.bin_centers < self.param['range_fit'][1])]

        if self.param['fit_func'] != "none":
            # Fitting function call
            try:
                self.coeff, self.var_matrix = curve_fit(self.param['fit_func'],
                                                        self.bin_centers_f,
                                                        self.hist_f,
                                                        p0=self.param['guess'],
                                                        ftol=1E-12, maxfev=1000,
                                                        method='lm')
                self.perr = np.sqrt(np.absolute(np.diag(self.var_matrix)))
                # Error in parameter estimation



            except:
                print("Fitting Problems")
                self.coeff = np.array(self.param['guess'])
                self.perr  = np.array(self.param['guess'])

            self.hist_fit_f = self.param['fit_func'](self.bin_centers_f, *self.coeff)
            if np.isnan(self.hist_fit_f).any():
                self.chisq_r = 1000
            else:
                self.chisq = np.sum((((self.hist_f-self.hist_fit_f)**2)/self.hist_fit_f))
                self.df = len(self.bin_centers_f)-len(self.coeff)
                self.chisq_r = self.chisq/self.df
            #Gets fitted function and residues


    def evaluate(self,in_data):
        # Evaluates fit function over a given data set
        return self.param['fit_func'](in_data,*self.coeff)


    def evaluate_f(self):
        # Evaluates fit function over fitting range
        return self.param['fit_func'](self.bin_centers_f,*self.coeff)


    def show_data(self, axis, **kwargs):

        pl_param   =  {'title' : '',
                       'xlabel': '',
                       'ylabel': '',
                       'pos'   :[0.95,0.95,"left"]}

        for key,value in kwargs.items():
            if (key in pl_param.keys()):
                pl_param[key] = value
            else:
                print("Parameter %s is not valid" % key)

        axis.hist(self.data, self.param['bins'],
                  align='mid',
                  facecolor='green',
                  edgecolor='white',
                  linewidth=0.5,
                  density=False,
                  range=self.param['range_show'])

        axis.plot(self.bin_centers_f, self.evaluate_f(), 'r--', linewidth=1)

        axis.set_xlabel(pl_param['xlabel'])
        axis.set_ylabel(pl_param['ylabel'])
        axis.set_title(pl_param['title'])


        if (self.param['fit_func'].__name__ == moyal.__name__):
            text_chain = ("mode=%0.2f ($\pm$ %0.2f) \n $\sigma$=%0.2f ($\pm$ %0.2f) \n" %
                            (self.coeff[1], self.perr[1], self.coeff[2], self.perr[2]))

        elif (self.param['fit_func'].__name__ == gauss.__name__):
            text_chain =  (('$\mu$=%0.2f ($\pm$ %0.2f) \n'+ \
                          '$\sigma$=%0.2f ($\pm$ %0.2f) \n'+
                          #'FWHM=%0.2f (+/- %0.2f) \n'+\
                          '$Res_{FWHM}$=%0.2f%% ($\pm$ %0.2f)') % \
            (self.coeff[1] , self.perr[1],
             np.absolute(self.coeff[2]) , self.perr[2],
             #2.35*np.absolute(self.coeff[2]), 2.35*np.absolute(self.perr[2]),
             2.35*np.absolute(self.coeff[2])*100/self.coeff[1],
             2.35*np.absolute(self.coeff[2])*100/self.coeff[1]*np.sqrt((self.perr[2]/self.coeff[2])**2+(self.perr[1]/self.coeff[1])**2)))

        axis.text(pl_param['pos'][0],pl_param['pos'][1],
                  text_chain,
                  fontsize=8,
                  verticalalignment = 'top',
                  horizontalalignment = pl_param['pos'][2],
                  transform = axis.transAxes)

class fit_fun(object):
    def __init__(self, data, x, **kwargs):
        self.data  = data
        self.x     = x
        # Default params
        self.param  = {'fit_func': exp_pure,
                       'guess': [1,1,1],
                       'fit_method':'lm',
                       'bounds':[[-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf]],
                       'sigma':np.ones(len(self.data))
                       }

        for key,value in kwargs.items():
            if (key in self.param.keys()):
                self.param[key] = value
            else:
                print("Parameter %s is not valid" % key)

        # Fitting function call
        try:
            self.coeff, self.var_matrix = curve_fit(self.param['fit_func'],
                                                    self.x,
                                                    self.data,
                                                    p0=self.param['guess'],
                                                    ftol=1E-12, maxfev=1000,
                                                    method=self.param['fit_method'],
                                                    bounds=self.param['bounds'],
                                                    sigma=self.param['sigma'])

            self.perr = np.sqrt(np.absolute(np.diag(self.var_matrix)))

            # Error in parameter estimation
        except:
            print("Fitting Problems")
            self.coeff = np.array(self.param['guess'])
            self.perr  = np.array(self.param['guess'])


        self.data_fit = self.param['fit_func'](self.x, *self.coeff)
        if np.isnan(self.data_fit).any():
            self.chisq_r = 1000
        else:
            self.chisq = np.sum((((self.data-self.data_fit)**2)/self.data_fit))
            self.df = len(self.x)-len(self.coeff)
            self.chisq_r = self.chisq/self.df
        #Gets fitted function and residues

    def evaluate(self):
        # Evaluates fit function over fitting range
        return self.param['fit_func'](self.x,*self.coeff)


    def show_data(self, axis, **kwargs):

        pl_param   =  {'title' : '',
                       'xlabel': '',
                       'ylabel': ''}

        for key,value in kwargs.items():
            if (key in pl_param.keys()):
                pl_param[key] = value
            else:
                print("Parameter %s is not valid" % key)

        axis.plot(self.x, self.evaluate(), 'r--', linewidth=1)

        axis.set_xlabel(pl_param['xlabel'])
        axis.set_ylabel(pl_param['ylabel'])
        axis.set_title(pl_param['title'])
