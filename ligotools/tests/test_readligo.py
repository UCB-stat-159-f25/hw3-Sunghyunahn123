import numpy as np
from ligotools import readligo as rl
from pathlib import Path

DATA_DIR = Path("data")
fn_H1 = DATA_DIR / "H-H1_LOSC_4_V2-1126259446-32.hdf5"

def test_loaddata_output_shapes():
    strain, time, chan_dict = rl.loaddata(fn_H1, 'H1')
    assert len(strain) == len(time)
    assert np.isfinite(strain).any()

def test_dq_channel_to_seglist():
    _, _, chan_dict = rl.loaddata(fn_H1, 'H1')
    segs = rl.dq_channel_to_seglist(chan_dict['DATA'])
    assert isinstance(segs, list)
    assert len(segs) >= 1
