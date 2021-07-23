#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
import numpy as np
import tables as tb

from sklearn.cluster import DBSCAN

from scipy import interpolate



def get_files(run):
    folder = f'/analysis/{run}/hdf5/data/*h5'
    files = glob(folder)
    files.sort()
    return files


# In[4]:


def read_run_data(files):
    dfs = []
    for fname in files:
        try:
            df = pd.read_hdf(fname, 'data')
            dfs.append(df)
        except:
            print("Error in file ", fname)
    df = pd.concat(dfs).reset_index(drop=True)
    return df


# In[5]:


def get_evt_times(files):
    time_dfs = []

    for i, fname in enumerate(files):
        df_time = pd.read_hdf(fname, 'dateEvents')
        df_time['fileno'] = i
        time_dfs.append(df_time)
    df_times = pd.concat(time_dfs)
    df_times['date'] = pd.to_datetime(df_times.timestamp, unit='us')

    # Compute time difference between one event and the next
    df_times['time_diff'] = np.abs((df_times.timestamp/1e6).diff(periods=-1))
    df_times = df_times.fillna(0)
    return df_times


# In[6]:


def compute_tcoarse_wrap_arounds(df):
    limits = df[df.tcoarse_diff < -30000].index
    first  = df.index[0]
    last   = df.index[-1]
    limits = np.concatenate([np.array([first]), limits.values, np.array([last])])
    return limits


# In[7]:


def compute_tcoarse_nloops_per_event(df):
    limits = df.groupby('evt_number').apply(compute_tcoarse_wrap_arounds)

    nloops = np.zeros(df.shape[0], dtype='int32')

    for evt_limits in limits.values:
        for i in range(evt_limits.shape[0]-1):
            start = evt_limits[i]
            end   = evt_limits[i+1]

            nloops[start:end+1] = i

    return nloops


# In[8]:


def compute_extended_tcoarse(df):
    return df['tcoarse'] + df['nloops'] * 2**16


# In[9]:


def compute_integration_window_size(df):
    df['intg_w'] = (df.ecoarse - (df.tcoarse % 2**10)).astype('int16')
    df.loc[df['intg_w'] < 0, 'intg_w'] += 2**10


# In[10]:


#def apply_qdc_correction(df, df_qdc):
#    df = df.reset_index().merge(df_qdc[['channel_id', 'tac_id', 'qoffset', 'ibias']], on=['channel_id', 'tac_id'])
#    df['charge_corrected'] = df.efine - (df.ibias*df.intg_w + df.qoffset)
#    df.drop(columns=['qoffset', 'ibias'], inplace=True)
#    df = df.sort_values('index').set_index('index')
#    df.index.name = None
#    return df


# In[11]:


def apply_tdc_correction(df, df_tdc):
    df = df.reset_index().merge(df_tdc[['channel_id', 'tac_id', 'amplitude', 'offset']], on=['channel_id', 'tac_id'])
    df = df.sort_values('index').set_index('index')
    df.index.name = None

    period = 360
    df['tfine_corrected'] = (period/np.pi)*np.arctan(1/np.tan((np.pi/(-2*df.amplitude))*(df.tfine-df.offset)))
    df.loc[df['tfine_corrected'] < 0, 'tfine_corrected'] += period
    df = df.drop(columns=['amplitude', 'offset'])
    df['t'] = df.tcoarse - (360 - df.tfine_corrected) / 360
    return df


# ### Read file by chunks

# In[12]:


def process_daq_df(df, df_tdc, df_qdc):
    # Remove non-used columns
    #df.drop(columns=['card_id', 'tofpet_id', 'wordtype_id'], inplace=True)

    # Add extended tcoarse
    df['tcoarse']          = df.tcoarse.astype(np.int32)
    df['tcoarse_diff']     = df.tcoarse.diff()
    df['nloops']           = compute_tcoarse_nloops_per_event(df)
    df['tcoarse_extended'] = compute_extended_tcoarse(df)
    df.drop(columns=['tcoarse_diff', 'nloops'], inplace=True)

    # TODO: Use corrected timestamp to compute integration window
    compute_integration_window_size(df)

    # Add row and column
    #df['x'] =  df.sensor_id // 10
    #df['y'] =  df.sensor_id  % 10

    df = apply_qdc_spl_correction(df, df_qdc)
    df = apply_tdc_correction(df, df_tdc)

    #df.drop(columns=['tac_id', 'ecoarse'], inplace=True)

    return df


# In[13]:


def write_corrected_df_daq(fileout, df, append=False):
    table_name = 'data'
    table_evt_numbers = 'evts'
    mode = 'a' if append else 'w'
    store = pd.HDFStore(fileout, mode, complib=str("zlib"), complevel=4)
    if append:
        store.append(table_name, df, index=False, format='table', data_columns=None)
        store.append(table_evt_numbers, df['evt_number'], index=False, format='table', data_columns=['evt_number'])
    else:
        store.put(table_name, df, index=False, format='table', data_columns=None)
        store.append(table_evt_numbers, df['evt_number'], index=False, format='table', data_columns=['evt_number'])
    store.close()


# In[14]:


def compute_file_chunks_indices(filein):
    with tb.open_file(filein) as h5in:
        evt_numbers = h5in.root.data.cols.evt_number[:]
        evt_diffs  = np.diff(evt_numbers)
        evt_limits = np.where(evt_diffs)[0]

        # Find borders that keep ~chunk_size rows per chunk
        chunk_size   = 500000
        chunk_diffs  = np.diff(evt_limits // chunk_size)
        chunk_limits = np.where(chunk_diffs)[0]

        chunks = np.concatenate([np.array([0]),
                         evt_limits[chunk_limits],
                         np.array([evt_numbers.shape[0]])])
        return chunks


# #### Compute clusters

# In[15]:


def compute_clusters(df):
    values = df.tcoarse_extended.values
    values = values.reshape(values.shape[0],1)

    clusters = DBSCAN(eps=10, min_samples=2).fit(values)
    return clusters.labels_


# In[16]:


def compute_evt_number_combined_with_cluster_id(df):
    df_clusters = df.groupby('evt_number').apply(compute_clusters)
    df['cluster'] = np.concatenate(df_clusters.values)
    df.loc[df[df.cluster != -1].index, 'cluster'] = df.evt_number * 20000 + df.cluster


# In[17]:


def process_daq_file(filein, fileout):
    chunks = compute_file_chunks_indices(filein)
    nchunks = chunks.shape[0]

    for i in range(nchunks-1):
        print("{}/{}".format(i, nchunks-2))
        start = chunks[i]
        end   = chunks[i+1]

        df = pd.read_hdf(filein, 'data', start=start, stop=end+1)

        df_corrected = process_daq_df(df, df_tdc, df_qdc)
        compute_evt_number_combined_with_cluster_id(df_corrected)
        write_corrected_df_daq(fileout, df_corrected, i>0)


# ### New QDC correction

# In[83]:
def inv_saturation_spline(x,*param):
    param_array = np.array(param)
    spline_conf = interpolate.BSpline([10,10,10,10,30,40,50,60,70,90,110,150,150,150,150],
                                      np.concatenate([[0],param_array,[0,0,0,0]]),3)

    return interpolate.splev(x, spline_conf, der=0)


def poly(x,*param):
    return param[0]+param[1]*x+param[2]*(x**2)+param[3]*(x**3)+param[4]*(x**4)+param[5]*(x**5)+\
                               param[6]*(x**6)+param[7]*(x**7)+param[8]*(x**8)+param[9]*(x**9)


def apply_qdc_spl_correction(df, df_qdc):
    df = df.reset_index().merge(df_qdc[['tofpet_id', 'channel_id', 'tac_id', 'spl0', 'spl1', 'spl2', 'spl3', 'spl4', 'spl5', 'spl6', 'spl7', 'spl8', 'spl9']], on=['tofpet_id', 'channel_id', 'tac_id'])

    df['efine_corrected'] = df['efine'] - df.apply(lambda data: inv_saturation_spline(
                                                                      data['intg_w'],data['spl0'],
                                                                      data['spl1'],data['spl2'],
                                                                      data['spl3'],data['spl4'],
                                                                      data['spl5'],data['spl6'],
                                                                      data['spl7'],data['spl8'],
                                                                      data['spl9']),axis=1)

    df.drop(columns=['spl0', 'spl1', 'spl2', 'spl3', 'spl4', 'spl5', 'spl6', 'spl7', 'spl8', 'spl9'], inplace=True)
    df = df.sort_values('index').set_index('index')
    df.index.name = None
    return df


def apply_qdc_poly_correction(df, df_qdc):
    df = df.reset_index().merge(df_qdc[['tofpet_id', 'channel_id', 'tac_id',
                                        'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']],
                                on=['tofpet_id', 'channel_id', 'tac_id'])

    df['efine_corrected'] = df['efine'] - df.apply(lambda data: poly( data['intg_w'],data['c0'],
                                                                      data['c1'],data['c2'],
                                                                      data['c3'],data['c4'],
                                                                      data['c5'],data['c6'],
                                                                      data['c7'],data['c8'],
                                                                      data['c9']),axis=1)

    df['correction'] = df.apply(lambda data: poly( data['intg_w'],data['c0'],
                                                                      data['c1'],data['c2'],
                                                                      data['c3'],data['c4'],
                                                                      data['c5'],data['c6'],
                                                                      data['c7'],data['c8'],
                                                                      data['c9']),axis=1)

    #df.drop(columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'], inplace=True)
    df = df.sort_values('index').set_index('index')
    df.index.name = None
    return df


# # Process file

# In[24]:


# QDC correction
#df_qdc = pd.read_hdf('efine_cal_asic0.h5')
#df_qdc = pd.read_hdf('asic0_efine_cal_spl.h5')


# TDC correction
#df_tdc = pd.read_hdf('asic0_tfine_cal.h5')

#run_number = 10847 # 49V, lsb 57, Na-22

#outfile = 'run_10847_0000_qdc_spl_20Jul.h5'


# In[ ]:

#files = get_files(run_number)
#process_daq_file(files[0], outfile)
