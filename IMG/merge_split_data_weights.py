import numpy as np
np.random.seed(0)
import os, glob
import time
#import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import *

class ParquetDatasetTable(Dataset):
    def __init__(self, filename, cols=None):
        self.parquet = pq.ParquetFile(filename)
        #self.cols = None # read all columns
        self.cols = cols # read all columns
        #self.cols = ['Xtz.list.item.list.item.list.item','m','pt']
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols)
        return data
    def __len__(self):
        return self.parquet.num_row_groups

def get_weight_1d(pt, pt_edges, wgts):
    idx_pt = np.argmax(pt <= pt_edges)-1
    return wgts[idx_pt]

def get_weight_2d(m0, pt, m0_edges, pt_edges, wgts):
    idx_m0 = np.argmax(m0 <= m0_edges)-1
    idx_pt = np.argmax(pt <= pt_edges)-1
    return wgts[idx_m0, idx_pt]

def merge_samples(dset, dset_mpt, start, stop, idxs, decay, sample_str, do_neg_mass=False):

    if do_neg_mass:
        lo, hi, off = -0.30, 0., 0.000
        m_neg = (hi-lo)*np.random.random_sample(int(stop-start)) + lo + off
        #mass_idx = 2
        mass_idx = next(i for i,c in enumerate(dset.__getitem__(0).itercolumns()) if c.name == 'm')
        print(' >> Writing negative masses: %.4f -> %.4f'%(lo, hi))
        print(' >> Replacing mass on column idx:',mass_idx)
        decay = '%s_m0Neg%dTo%d'%(decay, int(abs(lo)*1000), int(hi*1000))

    file_str = '%s_wgts.%s.parquet'%(decay, sample_str)
    print('>> Doing sample:',file_str)
    print('>> Output events: %d [ %d, %d )'%((stop-start), start, stop))

    # Calculate m vs pt weights 
    print('>> Calculating weights...')
    m_, pt_ = [], []
    for i, idx in enumerate(idxs[start:stop]):
        
        if i%50000 == 0:
            print(' >> Processed event:',i)

        t = dset_mpt.__getitem__(idx)

        pt_.append(t.to_pydict()['pt'])
        if do_neg_mass:
            continue
        m_.append(t.to_pydict()['m'])

    if do_neg_mass:
        hmvpt, pt_edges = np.histogram(np.array(pt_).flatten(), range=(20.,100.), bins=20)
        print(' >> hist, min:%f, mean:%f, max:%f'%(hmvpt.min(), hmvpt.mean(), hmvpt.max()))
        hmvpt = 1.*hmvpt/hmvpt.sum()
        lhood = 1./hmvpt
        lhood = lhood/20.
    else:
        hmvpt, m_edges, pt_edges = np.histogram2d(np.array(m_).flatten(), np.array(pt_).flatten(), range=((0., 1.6), (20.,100.)), bins=(16, 20))
        print(' >> hist, min:%f, mean:%f, max:%f'%(hmvpt.min(), hmvpt.mean(), hmvpt.max()))
        hmvpt = 1.*hmvpt/hmvpt.sum()
        lhood = 1./hmvpt
        lhood = lhood/(16.*20.)

    print(' >> likelihood, min:%f, max:%f'%(lhood.min(), lhood.max()))
    print(' >> sum(l):%f'%(lhood.sum()))
    print(' >> sum(h*l):%f'%((hmvpt*lhood).sum()))

    # Write out the actual data
    print('>> Writing data...')
    now = time.time()
    for i, idx in enumerate(idxs[start:stop]):
        
        if i%50000 == 0:
            print(' >> Processed event:',i)

        t = dset.__getitem__(idx)
        #print(t.to_pydict())
            
        if do_neg_mass:
            m_neg_col = pa.Column.from_array('m', pa.array([m_neg[i]]))
            t = t.remove_column(mass_idx).add_column(mass_idx, m_neg_col)
            #t = t.drop(['m']).append_column(m_neg_col)
            wgt_ = get_weight_1d(t.to_pydict()['pt'], pt_edges, lhood)
        else:
            wgt_ = get_weight_2d(t.to_pydict()['m'], t.to_pydict()['pt'], m_edges, pt_edges, lhood)
        
        wgt = pa.Column.from_array('w', pa.array([wgt_]))
        t = t.append_column(wgt)

        if i == 0:
            writer = pq.ParquetWriter(file_str, t.schema, compression='snappy')

        writer.write_table(t)
        #print(t.to_pydict())
        
    writer.close()
    print('>> E.T.: %f min'%((time.time()-now)/60.))

    pqin = pq.ParquetFile(file_str)
    print(pqin.schema)
    print(pqin.metadata)
    print(pqin.read_row_group(0, ['m','w']).to_pydict())

# MAIN
decay = 'DoublePi0Pt10To100_m0To1600_pythia8_ReAOD_PU2017_MINIAODSIM_wrapfix.tzfixed'
#decay = 'DoublePhotonPt15To100_pythia8_ReAOD_PU2017_MINIAODSIM_wrapfix.tzfixed'
fs = glob.glob('%s.parquet.*'%decay)
dset = ConcatDataset([ParquetDatasetTable(f) for f in fs])
dset_mpt = ConcatDataset([ParquetDatasetTable(f, cols=['m','pt']) for f in fs])
nevts_in = len(dset)
print('>> Input events:',nevts_in)
for f in fs:
    print(pq.ParquetFile(f).schema)
    # Schema MUST be consistent across input files!!

# Shuffle
idxs = np.random.permutation(nevts_in)

nevts_train = 256*3040
#nevts_train = int(3.*nevts_train/16)
nevts_val = 25600
assert nevts_train+nevts_val <= nevts_in

#for sample in ['train']:
for sample in ['train','val']:
    
    if sample == 'train':
        start, stop = 0, nevts_train
    else:
        start, stop = nevts_train, nevts_train+nevts_val 

    merge_samples(dset, dset_mpt, start, stop, idxs, decay, sample)
    #merge_samples(dset, dset_mpt, start, stop, idxs, decay, sample, do_neg_mass=True)
    print('_________________________________________________________\n')
