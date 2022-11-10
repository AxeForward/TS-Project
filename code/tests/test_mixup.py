import numpy as np
import pandas as pd
import pytest
import os
import typing

@pytest.fixture()
def get_results_path()->os.PathLike:
    test_dir = os.path.dirname(__file__)
    results_dir = os.path.join(test_dir, 'results')
    return results_dir

@pytest.mark.parametrize('Origin_Result, Project_Result', [('ACSF1_10epochs_42seed.csv', 'ucr_ACSF1_10epochs_42seed.csv')])
def test_mix_up_ucr(get_results_path, Origin_Result, Project_Result):
    origin_res_file = os.path.join(get_results_path, 'UCR', Origin_Result)
    df_origin = pd.read_csv(origin_res_file, index_col=0)
    origin_loss = df_origin['loss'].values
    origin_acc = df_origin['acc'].values

    project_res_file = os.path.join(get_results_path, 'UCR', Project_Result)
    df_project = pd.read_csv(project_res_file, index_col=0)
    project_loss = df_project['loss'].values
    project_acc = df_project['acc'].values

    assert np.allclose(origin_loss, project_loss)
    assert np.allclose(origin_acc, project_acc)

