import csv
import numpy as np
import pytest
import os

@pytest.fixture()
def get_results_path()->os.PathLike:
    test_dir = os.path.dirname(__file__)
    results_dir = os.path.join(test_dir, 'results')
    return results_dir

def reader_data_for_test(data_path):
    file = open(data_path, 'r')
    records = csv.reader(file)
    models_loss,models_acc= [], []
    next(records)
    for row in records:
        models_loss.append(float(row[0]))
        models_acc.append(float(row[1]))
    file.close()
    return models_loss,models_acc


@pytest.mark.parametrize('Origin_Result, Project_Result',
                         [('UCI_HAR_20epochs_0seed.csv', 'UCI_HAR_20epochs_0seed_from_newpro.csv')])
def test_TS_TCC(get_results_path, Origin_Result, Project_Result ):
    origin_result_path = os.path.join(get_results_path,'TS_TCC', 'UCI',Origin_Result)
    project_result_path = os.path.join(get_results_path,'TS_TCC', 'UCI',Project_Result)
    origin_loss,origin_acc = reader_data_for_test(origin_result_path)
    project_loss,project_acc = reader_data_for_test(project_result_path)
    assert np.allclose(origin_acc, project_acc)
    assert np.allclose(origin_loss, project_loss)