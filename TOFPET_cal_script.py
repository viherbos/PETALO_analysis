import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import TOFPET_cal_lib as TPcal
from glob import glob
from scipy import stats

# Run INFO
run_number = 10932
tofpet_id  = 0
asic = 'asic'+str(tofpet_id)

# Calibration Output
path_cal = "/home/vherrero/CALIBRATION_FILES/"
efine_cal = asic + "_efine_cal_poly_run"+str(run_number)+".h5"

# Calibration Control parameters
windows1 = np.arange(22, 31, 4)
windows2 = np.arange(32, 74, 2)
windows = np.concatenate([windows1, windows2])
window_array = np.concatenate([windows1*2-16, windows2*4 -78])

n_channels   = 64
channels_array = list(range(64))
channels_array.pop(49)

tacs_array = range(4)

########### JMB Functions ####################################################
def get_files(run):
    folder = f'/analysis/{run}/hdf5/data/*h5'
    files = glob(folder)
    files.sort()
    return files

def get_run_control(files):
    dfs = []

    for i, fname in enumerate(files):
        df_tmp = pd.read_hdf(fname, 'dateEvents')
        df_tmp['fileno'] = i
        dfs.append(df_tmp)
    df = pd.concat(dfs)
    df['diff'] = df.run_control.diff().fillna(0)
    return df

def filter_df_evts(df, evt_start, evt_end):
    evt_filter = (df.evt_number >= evt_start) & (df.evt_number < evt_end)
    df_filtered = df[evt_filter]
    return df_filtered

def compute_limit_evts_based_on_run_control(df_run):
    limits = df_run[df_run['diff'] != 0]

    start_evts = []
    end_evts   = []
    files1     = []
    files2     = []

    previous_start = 0
    previous_end   = 0
    file1 = 0
    file2 = 0

    for _, row in limits.iterrows():
        file1 = file2
        file2 = row.fileno
        start_evts.append(previous_start)
        end_evts  .append(row.evt_number)
        files1    .append(file1)
        files2    .append(file2)
        previous_start = row.evt_number


    start_evts.append(previous_start)
    end_evts  .append(df_run.evt_number.values[-1])
    files1    .append(file2)
    files2    .append(df_run.fileno    .values[-1])

    # [start, end)
    df_limits = pd.DataFrame({'start'  : start_evts,
                              'end'    : end_evts,
                              'file1' : files1,
                              'file2' : files2})
    return df_limits

def process_df_to_assign_tpulse_delays(files, limits, configs):
    results = []
    tofpet_evts = []
    current_file1 = -1
    df = pd.DataFrame()

    for iteration, limit in limits.iterrows():
        #print(iteration)
        file1 = int(limit.file1)
        file2 = int(limit.file2)

        if file1 != current_file1:
            df = pd.read_hdf(files[file1], 'data')
            current_file1 = file1

        df_filtered = filter_df_evts(df, limit.start, limit.end)

        if file1 != file2:
            df2          = pd.read_hdf(files[file2], 'data')
            df2_filtered = filter_df_evts(df2, limit.start, limit.end)
            df_filtered  = pd.concat([df_filtered, df2_filtered])

            # Update file1
            df = df2
            current_file1 = file2

        tofpet_evts.append(df_filtered.shape[0])

        #df_filtered['tpulse'] = configs[iteration]
        df_filtered = df_filtered.copy()
        df_filtered.loc[df_filtered.index, 'tpulse'] = configs[iteration]
        results.append(df_filtered)

    return pd.concat(results), tofpet_evts

###############################################################################


print("\n###################### \n QDC CALIBRATION \n#####################")
print("\nFiles for RUN number = %i \n" % run_number)
files = get_files(run_number)
print(files)

df_run = get_run_control(files)

limits = compute_limit_evts_based_on_run_control(df_run)

configs = np.tile(window_array, n_channels)

print("Assign Tpulse delays\n")
df_data, tofpet_evts = process_df_to_assign_tpulse_delays(files, limits, configs)

# for channel in channels_array:
#     n_phases = df_data[(df_data.channel_id == channel) & (df_data.tofpet_id == tofpet_id)].tpulse.unique().shape[0]
#     print(f"{channel}: {n_phases}")
#
#     df_filtered = df_data[(df_data.channel_id == channel) & (df_data.tofpet_id == tofpet_id)]
#     #df_filtered.to_hdf('10897_qdc_ch5_asic0_0v.h5', mode='a', key=f'ch{channel}', format='table')

print("Finding Max in Data\n")
# Fitting or Max Finding
# res = []
# for ch in channels_array:
#     for tc in tacs_array:
#         data_tc = df_data[df_data['tac_id']==tc]
#         for i in window_array:
#             data_fit = np.mod(data_tc['efine'] - 1024 + 14, 1024)
#             # PETSYS Magic
#             #h = np.histogram(data_fit['efine'],bins=1024,density=1,range=[0,1024])
#             modal = stats.mode(data_fit).mode[0]
#             res.append([tofpet_id,ch,tc,i,modal,1]) #a[1][np.argmax(a[0])],1])
#             print("Channel %i , Tac %i, i %i \n" % (ch,tc,i))

df_data['efine'] = np.mod(df_data['efine'] - 1024 + 14, 1024)
result = df_data[['tofpet_id', 'channel_id', 'tac_id', 'tpulse', 'efine']].\
                  groupby(['tofpet_id', 'channel_id', 'tac_id', 'tpulse']).\
                  efine.apply(lambda x: x.mode())

result = result.reset_index()
result = result.drop(columns=['level_4'])
result['sigma'] = 1
result = result.rename(columns={'efine' : 'mu'})

df_efine = result
#pd.DataFrame(res,columns=['tofpet_id','channel_id','tac_id','tpulse','mu','sigma'])

with pd.HDFStore(path_cal + "fitted_data_run"+str(run_number)+".h5",'w',complib="zlib",complevel=4) as storage:
    storage.put('efine',df_efine,index=False,format='table',data_columns=None)
    storage.close()


print("Fitting\n")
res = []
for ch in channels_array:
    for tc in tacs_array:
        poly_conf = TPcal.QDC_fit_p(df_efine,ch,tc,plot=False,sigmas=np.ones(len(window_array)))
        res.append([tofpet_id,ch,tc,*poly_conf])
        print("Channel %i, Tac %i \n" % (ch,tc))

df_qfine = pd.DataFrame(res,columns=['tofpet_id','channel_id','tac_id',
                                     'c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])

with pd.HDFStore(path_cal + efine_cal,'w',complib="zlib",complevel=4) as storage:
    storage.put('efine',df_qfine,index=False,format='table',data_columns=None)
    storage.close()
