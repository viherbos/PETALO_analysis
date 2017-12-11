#!/home/viherbos/anaconda2/bin/python

import os
import struct
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import fit_library as ft
import matplotlib.widgets as wid
import matplotlib.gridspec as gridspec
import argparse

def raw_singles_to_hdf5(ldat_dir  = ".",
                        ldat_name = "my_data_singles.ldat",
                        hdf5_name = "my_data_singles.hdf",
                        env_name  = "env.txt"):

    struct_event  = 'qfi' # long-long / float / int
    # Raw struct
    struct_len    = struct.calcsize(struct_event)

    os.chdir(ldat_dir)
    i=0;j=0
    data_array=[]
    env_array=[]

    with open(ldat_name, "rb") as f:
        while True:
            data = f.read(struct_len)
            if not data: break
            i=i+1
            s = struct.unpack(struct_event,data)
            data_array.append(s)
        print ("Number of Events %d" % i)

    with open(env_name, "rb") as f:
        while True:
            data = f.readline()
            if not data: break
            j=j+1
            env_array.append(float(data))
        print ("Number of TEMP SENSORS %d" % j)

    with pd.HDFStore( hdf5_name,
                      complevel=9, complib='bzip2') as store:
        panel_array = pd.DataFrame( data=data_array,
                                    columns=['timestamp', 'Q', 'id'])
        env_array = pd.DataFrame( data=env_array,
                                  columns=['temp'])
        store.put('data',panel_array)
        store.put('env',env_array)
        store.close()



def coincidence_to_hdf5(ldat_dir  = ".",
                        ldat_name = "my_data_coincidence.ldat",
                        hdf5_name = "my_data_coincidence.hdf",
                        env_name  = "env.txt"):

    struct_event  = 'HHqfiHHqfi'
    # Coincidence struct
    struct_len    = struct.calcsize(struct_event)

    os.chdir(ldat_dir)
    i=0
    data_array=[]
    env_array=[]

    with open(ldat_name, "rb") as f:
        while True:
            data = f.read(struct_len)
            if not data: break
            i=i+1
            s = struct.unpack(struct_event,data)
            data_array.append(s)
        print ("Number of Events %d" % i)

    with open(env_name, "rb") as f:
        while True:
            data = f.readline()
            if not data: break
            j=j+1
            env_array.append(float(data))
        print ("Number of TEMP SENSORS %d" % j)

    with pd.HDFStore( hdf5_name,
                      complevel=9, complib='bzip2') as store:
        panel_array = pd.DataFrame( data=data_array,
                                    columns=['mh_n1',
                                             'mh_j1',
                                             'timestamp1',
                                             'Q1',
                                             'id1',
                                             'mh_n2',
                                             'mh_j2',
                                             'timestamp2',
                                             'Q2',
                                             'id2'])
        env_array = pd.DataFrame( data=env_array,
                                  columns=['temp'])

        store.put('data',panel_array)
        store.put('env',env_array)
        store.close()


def read_hdf(ldat_dir  = ".",
             hdf5_name = "my_data.hdf"):

    os.chdir(ldat_dir)
    data = pd.read_hdf(hdf5_name,key='data')
    env = pd.read_hdf(hdf5_name,key='env')
    return data,env


class gui_aux(object):

    def __init__(self,Q1,Q2,ts1,ts2):
        self.Q1  = Q1
        self.Q2  = Q2
        self.ts1 = ts1
        self.ts2 = ts2
        self.time_diff = ts1-ts2

        self.indexer=np.array([])
        self.Q1_sel=np.array([])
        self.Q2_sel=np.array([])
        self.time_diff_sel=np.array([])

        self.x1Q1, self.x2Q1 = 0, 1000
        self.x1Q2, self.x2Q2 = 0, 1000
        self.filter_data()


    def filter_data(self):

        # Filter Energy Ranges in Q1 and Q2
        self.indexer=np.where(
            np.logical_and(
                np.logical_and(
                    np.greater_equal(self.Q1,self.x1Q1),
                    np.less_equal(self.Q1,self.x2Q1)),
                np.logical_and(
                    np.greater_equal(self.Q2,self.x1Q2),
                    np.less_equal(self.Q2,self.x2Q2)
                              )
                          ))
        self.Q1_sel = self.Q1[self.indexer]
        self.Q2_sel = self.Q2[self.indexer]
        self.time_diff_sel = self.time_diff[self.indexer]

        # Call class instance for fitting operation
        Q1_fit(self.Q1_sel,'sqrt')
        Q2_fit(self.Q2_sel,'sqrt')
        td_fit(self.time_diff_sel,'sqrt')
        # Plot fitting results
        ax2.cla(); ax0.cla(); ax1.cla()

        Q1_fit.plot(axis = ax0,
                    title = "Channel 1 - QDC",
                    xlabel = "Code (LSB)",
                    ylabel = "Hits")
        Q2_fit.plot(axis = ax1,
                    title = "Channel 2 - QDC",
                    xlabel = "Code (LSB)",
                    ylabel = "Hits")
        td_fit.plot(axis = ax2,
                    title = "Coincidende (CRT)",
                    xlabel = "picoseconds",
                    ylabel = "Hits",res=False)
        plt.tight_layout()
        plt.draw()


    def EXEC_key(self,event):
        if event.key in [' ']:
            self.filter_data()

    def IFL_but(self,event):
        self.x1Q1 = Q1_fit.coeff[1]-1.175*np.absolute(Q1_fit.coeff[2])
        self.x2Q1 = Q1_fit.coeff[1]+1.175*np.absolute(Q1_fit.coeff[2])
        self.x1Q2 = Q2_fit.coeff[1]-1.175*np.absolute(Q2_fit.coeff[2])
        self.x2Q2 = Q2_fit.coeff[1]+1.175*np.absolute(Q2_fit.coeff[2])
        self.filter_data()

    def RST_but(self,event):
        self.x1Q1, self.x2Q1 = 0, 1000
        self.x1Q2, self.x2Q2 = 0, 1000
        self.filter_data()

    def ls_Q1(self,eclick,erelease):
        #'eclick and erelease are the press and release events'
        self.x1Q1 = eclick.xdata
        self.x2Q1 = erelease.xdata

    def ls_Q2(self,eclick,erelease):
        #'eclick and erelease are the press and release events'
        self.x1Q2 = eclick.xdata
        self.x2Q2 = erelease.xdata



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='PETALO quick analysis.')
    parser.add_argument("-t", "--translate", action="store_true",
                        help="Translate binary coincidence file to HDF5")
    parser.add_argument('hdf', metavar='N', nargs=1, help='HDF5 file')
    parser.add_argument('ldat', metavar='N', nargs='?', help='Coincidence ldat file')
    parser.add_argument('env' , metavar='N', nargs='?', help='Temperatures files')

    args = parser.parse_args()

    if args.translate:
        print ("HDF file = " + ''.join(args.hdf))
        print ("LDAT file = " + ''.join(args.ldat))
        coincidence_to_hdf5(   ldat_dir  = ".",
                               ldat_name = ''.join(args.ldat),
                               hdf5_name = ''.join(args.hdf),
                               env_name  = ''.join(args.env))

    Q1_fit = ft.gauss_fit()
    Q2_fit = ft.gauss_fit()
    td_fit = ft.gauss_fit()

    hdf_filename = ''.join(args.hdf)
    DATA,TEMP = read_hdf(".",hdf_filename)
    # parer output is a char array so a join operation is required
    Q1  = np.array(DATA.loc[:,'Q1'])
    Q2  = np.array(DATA.loc[:,'Q2'])
    ts1 = np.array(DATA.loc[:,'timestamp1'])
    ts2 = np.array(DATA.loc[:,'timestamp2'])
    temp = np.array(TEMP.loc[:,'temp'])

    fig=plt.figure(figsize=(12,4))
    fig.canvas.set_window_title(hdf_filename)
    gs = gridspec.GridSpec( nrows=4, ncols=4,
                            width_ratios=[2, 2, 2, 1])
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])
    ax2 = fig.add_subplot(gs[:, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, 3])


    callback = gui_aux(Q1,Q2,ts1,ts2)

    bQ = wid.Button(ax3,'I Feel Lucky!!')
    bQ.on_clicked(callback.IFL_but)
    bRST = wid.Button(ax4,'Reset')
    bRST.on_clicked(callback.RST_but)

    RSQ1=wid.RectangleSelector(ax0, callback.ls_Q1,
                            drawtype='box', useblit=True,
                            button=[1, 3],  # don't use middle button
                            minspanx=5, minspany=5,
                            spancoords='pixels',
                            interactive=True)
    RSQ2=wid.RectangleSelector(ax1, callback.ls_Q2,
                            drawtype='box', useblit=True,
                            button=[1, 3],  # don't use middle button
                            minspanx=5, minspany=5,
                            spancoords='pixels',
                            interactive=True)


    plt.connect('key_press_event', callback.EXEC_key)

    plt.show()
