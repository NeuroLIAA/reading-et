from scipy.io import loadmat
import pandas as pd

def get_fixations(df_fix):
    return df_fix['xAvg'].to_numpy(dtype=int), df_fix['yAvg'].to_numpy(dtype=int), df_fix['duration'].to_numpy()

def load_screensequence(data_path, filename='screen_sequence.pkl'):
    screen_sequence = pd.read_pickle(data_path / filename)['currentscreenid'].to_numpy()
    return screen_sequence

def load_stimuli(item, stimuli_path):
    stimuli_file = stimuli_path / (item + '.mat')
    if not stimuli_file.exists():
        raise ValueError('Stimuli file does not exist: ' + str(stimuli_file))
    stimuli = loadmat(str(stimuli_file), simplify_cells=True)
    return stimuli

def load_stimuli_screen(screenid, stimuli):
    return stimuli['screens'][screenid - 1]['image']

def load_screen_fixations(screenid, subjitem_path):
    screen_path = subjitem_path / ('screen_' + str(screenid))
    fixations_files  = screen_path.glob('fixations*.pkl')
    screen_fixations = [pd.read_pickle(fix_file) for fix_file in fixations_files]
    if not screen_fixations:
        raise ValueError('No fixations found for screen ' + str(screenid) + ' in ' + str(subjitem_path))
    return screen_fixations