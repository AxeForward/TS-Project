import pytest
import os
import typing

@pytest.fixture()
def get_results_path()->os.PathLike:
    test_dir = os.path.dirname(__file__)
    results_dir = os.path.join(test_dir, 'results')
    return results_dir

@pytest.mark.parametrize('UCR_Dataset_Name', ['GunPoint', 'Chinatown'])
def test_mix_up_ucr(get_results_path, UCR_Dataset_Name):
    res_path = os.path.join(get_results_path, 'UCR', UCR_Dataset_Name)