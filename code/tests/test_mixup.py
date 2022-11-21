import numpy as np
import pandas as pd
import pytest
import os
import sys
import typing
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from flow_function import flow_mixup_function

@pytest.fixture()
def get_results_path()->os.PathLike:
    test_dir = os.path.dirname(__file__)
    results_dir = os.path.join(test_dir, 'results')
    return results_dir

@pytest.mark.parametrize('Origin_Result',
                         ['docker_notebookGunPoint_10epochs_42seed.csv'])
def test_mix_up_ucr(get_results_path, Origin_Result):
    origin_res_file = os.path.join(get_results_path, 'mix_up', 'UCR', Origin_Result)
    df_origin = pd.read_csv(origin_res_file, index_col=0)
    origin_loss = df_origin['loss'].values
    origin_acc = df_origin['acc'].values

    flow_loss, flow_acc = flow_mixup_function(data_source='ucr', filename='GunPoint', epochs=10, batch_size=50,
                                              alpha=1.0, random_seed=42, device='cpu')
    arr_flow_loss = np.array(flow_loss)
    arr_flow_acc = np.array(flow_acc)

    assert np.allclose(origin_loss, arr_flow_loss)
    assert np.allclose(origin_acc, arr_flow_acc)

