import numpy as np
import scipy as sp
import pandas as pd
from scipy import interpolate
import build_data_df_mine as bd
from sys import argv


run = argv[1]
file_number = argv[2]
calib_file = argv[3]

# Calibration INFO
path_cal = "/home/vherrero/CALIBRATION_FILES/"
asic0_efine_cal = "asic0_efine_cal_poly_run"+str(calib_file)+".h5" #10932
asic2_efine_cal = "asic2_efine_cal_spl.h5"
asic0_tfine_cal = "asic0_tfine_cal.h5"

# Run INFO
#run = 10931
#file_number = 0



# Output file
output_file = "/home/vherrero/PROCESSED_FILES/" + "asic0_run" + str(run) + "_file" \
              + str(file_number) + "_beta.h5"

print("\nLoading Calibration Data \n")
coeffs_qdc_0 = pd.read_hdf(path_cal + asic0_efine_cal, key='efine')
coeffs_qdc_2 = pd.read_hdf(path_cal + asic2_efine_cal, key='efine')
coeffs_tdc   = pd.read_hdf(path_cal + asic0_tfine_cal, key='tfine_cal')

# Creates only one qdc calibration file for both tofpets
coeffs_qdc = pd.concat([coeffs_qdc_0,coeffs_qdc_2]).reset_index()


# Reads run data
print("Loading Run Data \n")
data   = pd.read_hdf('/analysis/' + str(run) + '/hdf5/data/run_' + str(run) +\
                     '_'+ str(file_number).zfill(4) + '_trigger1_waveforms.h5',
                     key='data',
                     start=0,stop=25000000)



# Computes integration window
print("Generating Intg_w column \n")
bd.compute_integration_window_size(data)
data['intg_w'] = data['intg_w'] - 5.0
# Life is NOT wonderful

# Tfine and Efine preprocessing (see TOFPET datasheet for wraparound explanation)
data['tfine'] = (data['tfine'] - 1024 + 14) % 1024
data['efine'] = (data['efine'] - 1024 + 14) % 1024

# Applies TDC calibration correction
print("Applying TDC correction \n")
data = bd.apply_tdc_correction(data,coeffs_tdc)


# Applies QDC calibration correction to both TOFPETs
print("Applying QDC correction \n")
#data_0 = bd.apply_qdc_poly_correction(data[data['tofpet_id']==0],coeffs_qdc)
#data_2 = bd.apply_qdc_spl_correction(data[data['tofpet_id']==2],coeffs_qdc)
#data = pd.concat([data_2,data_0],ignore_index=True).reset_index()
data = bd.apply_qdc_bm_correction(data,coeffs_qdc)


# Extended TCOARSE computation
data['tcoarse']      = data.tcoarse.astype(np.int32)
data['tcoarse_diff'] = data.tcoarse.diff()
data['nloops']       = bd.compute_tcoarse_nloops_per_event(data)
data['tcoarse_extended'] = bd.compute_extended_tcoarse(data)
data.drop(columns=['tcoarse_diff','nloops'], inplace=True)

# FIND CLUSTERS
print("Finding Clusters \n")
bd.compute_evt_number_combined_with_cluster_id(data)
nuniq = data.groupby(['cluster'])['sensor_id'].nunique().rename('n_sipms')
clustered_df = data.join(nuniq, on='cluster')


#data['intg_w'] = data['intg_w'].astype('int')
#data['efine_corrected'] = data['efine_corrected'].astype('int')

print("Writing output data \n")
print(data)

with pd.HDFStore(output_file,'w',complib="zlib",complevel=4) as storage:
    storage.put('data',clustered_df,index=False,format='table',data_columns=None)
    storage.close()
