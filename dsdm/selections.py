import datasets
import numpy as np
from pathlib import Path
import os
import torch as ch

DATASET_SIZE = 216948746

jeopardy = 'jeopardy'
squad = 'squad'
lambada = 'lambada'
cs_alg = 'cs_algorithms'
lm_combo = 'lm_task_mix'
gpt3_combo = 'gpt3_mix'

DSIR_DATA_PATH = 'dsir'
DSDM_DATA_PATH = 'dsdm'
CLASSIFIER_DATA_PATH = 'classifier'
SDD_DATA_PATH = 'sdd.pt'

TARGETS = [jeopardy, squad, lambada, cs_alg, lm_combo, gpt3_combo]

def get_candidate_dataset():
    return datasets.load_dataset('loganengstrom/dsdm-candidate-c4')['train']

def process_path(path):
    # get file directory
    data_dir = Path(__file__).parent / 'data'
    assert data_dir.exists(), f'{data_dir} does not exist!'
    return data_dir / path

def _cached_ch_load(data_path, key):
    return ch.load(process_path(data_path) / f'{key}.pt')

# get indices for dataset selection of size selection_size
def dsdm_select(target_task, selection_size):
    # cs_algorithms: bigbench_cs_algorithms_0-shot
    # squad: squad2_3-shot
    # jeopardy: jeopardy_fixed_3-shot
    # lambada: lambada
    # combo: average of the last three
    if target_task in [squad, lambada, cs_alg, jeopardy]:
        # dm_params = dms[target_task]
        dm_params = _cached_ch_load(DSDM_DATA_PATH, target_task)
    elif target_task == lm_combo:
        dms_squad = _cached_ch_load(DSDM_DATA_PATH, squad)
        dms_lambada = _cached_ch_load(DSDM_DATA_PATH, lambada)
        dms_jeopardy = _cached_ch_load(DSDM_DATA_PATH, jeopardy)
        dm_params = (dms_jeopardy + dms_squad + dms_lambada) / 3

    # now get the indices for the top k indices
    sorted_indices = ch.argsort(dm_params)
    # check ordering
    assert dm_params[sorted_indices[0]] <= dm_params[sorted_indices[-1]], (dm_params[sorted_indices[0]], dm_params[sorted_indices[-1]])
    assert dm_params.max() == dm_params[sorted_indices[-1]]
    indices_to_take = sorted_indices[-selection_size:]
    indices_to_take = indices_to_take.cpu().numpy()
    return indices_to_take

def classifier_select(target_task, selection_size):
    assert target_task in [squad, lambada, cs_alg, jeopardy, lm_combo, gpt3_combo]
    weights = _classifier_weights(target_task)
    indices = np.argsort(weights)[::-1][:int(selection_size)]
    return indices

def _classifier_weights(target_task):
    s = _cached_ch_load(CLASSIFIER_DATA_PATH, target_task)
    return s.numpy().astype(np.float64)

def _get_logratios(target_task):
    logratios = _cached_ch_load(DSIR_DATA_PATH, target_task)
    return logratios.numpy().astype(np.float64)

def dsir_select(target_task, selection_size):
    logratios = _get_logratios(target_task)
    rng = np.random.default_rng()
    gumbel_noise = rng.gumbel(size=len(logratios))
    logratios += gumbel_noise
    return np.argpartition(-logratios, selection_size)[:selection_size]

def random_select(selection_size):
    rng = np.random.default_rng()
    return rng.choice(DATASET_SIZE, size=(selection_size,), replace=False)

def semdedup_select(selection_size):
    rng = np.random.default_rng()
    sdd_20pct = ch.load(process_path(SDD_DATA_PATH))
    sdd_20pct = sdd_20pct.numpy().astype(np.int64)
    return rng.choice(sdd_20pct, size=(selection_size,), replace=False)

TARGETED_METHODS = {
    'dsdm': dsdm_select,
    'classifier': classifier_select, 
    'dsir': dsir_select
}

UNTARGETED_METHODS = {
    'random': random_select,
    'semdedup': semdedup_select
}

# get train set indices for `method` of size `selection_size`, (maybe) for `target_task`
# return a numpy array of indices
def get_indices(method, selection_size, target_task=None):
    if method in TARGETED_METHODS.keys():
        assert target_task in TARGETS, f'{target_task} is not a valid target task! OK targets: {TARGETS}'
        print(f'>> Selecting {selection_size} indices for', method, 'with target task', target_task)
        select_it = TARGETED_METHODS[method]
        ls = select_it(target_task, selection_size)
    elif method in UNTARGETED_METHODS.keys():
        assert target_task is None, f'{method} does not support a target task!'
        print(f'>> Selecting {selection_size} indices for', method)
        select_it = UNTARGETED_METHODS[method]
        ls = select_it(selection_size)
    else:
        raise NotImplementedError(f'{method} is not implemented!')

    indices = list(map(int, ls))
    return indices

def dataset_select(method, selection_size, target_task=None):
    indices = get_indices(method, selection_size, target_task)
    return get_candidate_dataset().select(indices=indices)

def _test():
    for target_task in list(TARGETS)[::-1]:
        for method in TARGETED_METHODS.keys():
            if target_task == gpt3_combo and method == 'dsdm':
                continue

            indices = get_indices(method, 100, target_task)

    for method in UNTARGETED_METHODS.keys():
        indices = get_indices(method, 100)

if __name__ == '__main__':
    _test()