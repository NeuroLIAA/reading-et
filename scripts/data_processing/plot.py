import argparse
import matplotlib.pyplot as plt
from .draw_utils import drawing, handles
from . import utils
from tkinter import messagebox
from pathlib import Path


def sequence(screens, screens_sequence, sequence_states, editable):
    state = {'sequence_index': 0, 'cids': []}
    fig, ax = plt.subplots()
    drawing.update_figure(state, fig, ax, screens, sequence_states, editable)
    fig.canvas.mpl_connect('key_press_event', lambda event: handles.advance_sequence(event, state, screens,
                                                                                     screens_sequence, sequence_states,
                                                                                     ax, fig, editable))
    plt.tight_layout()
    plt.show()


def trial(stimuli, trial_path, editable=False):
    screens, screens_fixations, screens_lines = utils.load_trial(stimuli, trial_path)
    sequence_states, screens_sequence = build_sequence_states(screens_fixations, screens_lines, trial_path)

    sequence(screens, screens_sequence, sequence_states, editable)
    save_files = False
    if editable:
        save_files = messagebox.askyesno(title='Modified trial', message='Do you want to save the trial?')
        if save_files:
            utils.update_and_save_trial(sequence_states, stimuli, trial_path)

    return save_files


def calibration(trial_path, calibration_path='calibration', manualval_path='manual_validation'):
    cal_points, val_points, val_offsets = utils.load_calibrationdata(trial_path / calibration_path)
    manualval_fixs, manualval_points = utils.load_manualvaldata(trial_path / manualval_path, trial_path)

    screens = [drawing.screen(), drawing.screen(val_points),
               drawing.screen(manualval_points), drawing.screen(manualval_points)]
    # Plot the calibration grid relative to the screen centre
    cal_fixs, val_fixs = utils.add_offsets(cal_points, val_points, val_offsets, screens[0].shape[:2])

    fixations = [cal_fixs, val_fixs, manualval_fixs[0], manualval_fixs[1]]
    fixations = [df_fix.rename(columns={'x': 'xAvg', 'y': 'yAvg'}) for df_fix in fixations]

    screens_id = list(range(len(screens)))
    sequence_states = {screen_id: {'screenid': screen_id, 'fixations': fixations[screen_id], 'lines': []}
                       for screen_id in screens_id}

    sequence(screens, screens_id, sequence_states, editable=False)


def build_sequence_states(screens_fixations, screens_lines, trial_path):
    screens_sequence = utils.load_screensequence(trial_path)['currentscreenid'].to_numpy()
    sequence_lines = [screens_lines[screenid].pop() for screenid in screens_sequence]
    sequence_fixations = [screens_fixations[screenid].pop() for screenid in screens_sequence]
    seq_states = {}
    for seq, screenid in enumerate(screens_sequence):
        seq_states[seq] = {'screenid': screenid, 'fixations': sequence_fixations[seq], 'lines': sequence_lines[seq]}
    return seq_states, screens_sequence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stimuli_path', type=str, default='stimuli')
    parser.add_argument('--trial_path', type=str, default='data/processed/trials')
    parser.add_argument('--subj', type=str, required=True)
    parser.add_argument('--item', type=str, required=True)
    parser.add_argument('--editable', action='store_true')
    args = parser.parse_args()

    trial_path = Path(args.trial_path) / args.subj / args.item
    stimuli = utils.load_stimuli(args.item, Path(args.stimuli_path))
    trial(stimuli, trial_path, editable=args.editable)
