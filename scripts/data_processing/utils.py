from scipy.io import loadmat
from tkinter import messagebox
import pandas as pd
import numpy as np
import json
import shutil


def log(x):
    return np.log(x) if x > 0 else 0


def reorder(trials, stimuli_order):
    ordered_trials = [trial for trial in stimuli_order if trial in trials]
    return ordered_trials


def get_screenpath(screenid, item_path):
    screen_path = item_path / ('screen_' + str(screenid))
    if not screen_path.exists():
        screen_path.mkdir()
    return screen_path


def get_dirs(datapath, by_date=False):
    dirs = [dir_ for dir_ in datapath.iterdir() if dir_.is_dir()]
    if by_date:
        dirs.sort(key=lambda d: d.stat().st_mtime)
    return dirs


def get_files(datapath, extension='*'):
    files = [file for file in datapath.glob(f'*.{extension}') if file.is_file()]
    return files


def get_items(items_path, item_name):
    return [items_path / f'{item_name}.mat'] if item_name != 'all' else \
            [item for item in get_files(items_path, 'mat') if item.stem != 'Test']


def get_subjects(item_path):
    subjects = [subj.stem for subj in item_path.iterdir() if subj.is_file()]
    return subjects


def get_correct_trials(subjects, item_name):
    correct_trials = [subject for subject in subjects
                      if (subject / item_name).exists() and trial_is_correct(subject, item_name)]
    return correct_trials


def save_screensequence(screens_sequence, item_path, filename='screen_sequence.pkl'):
    screens_sequence.to_pickle(item_path / filename)


def load_profile(profile_path, filename='profile.pkl'):
    return pd.read_pickle(profile_path / filename)


def load_pickle(path, filename):
    return pd.read_pickle(path / filename)


def load_json(path, filename):
    with (path / filename).open('r') as file:
        return json.load(file)


def update_flags(trial_flags, trial_path, filename='flags.pkl'):
    trial_flags.to_pickle(trial_path / filename)


def load_flags(trials, datapath, filename='flags.pkl'):
    flags = {trial: pd.read_pickle(datapath / trial / filename) for trial in trials}
    return flags


def trial_is_correct(subject, item_name):
    trial_flags = load_flags([item_name], subject)
    return trial_flags[item_name]['edited'][0] and not trial_flags[item_name]['iswrong'][0]


def load_questions_and_words(questions_file, item):
    all_questionswords = load_matfile(questions_file)['stimuli_questions']
    questions, possible_answers, words = [], [], []
    for item_dict in all_questionswords:
        if item_dict['title'] == item:
            questions = list(item_dict['questions'])
            possible_answers = list(item_dict['possible_answers'])
            words = list(item_dict['words'])

    return questions, possible_answers, words


def load_matfile(matfile):
    return loadmat(str(matfile), simplify_cells=True)


def load_answers(trial_path, filename):
    answers = pd.read_pickle(trial_path / filename)
    return list(answers[0].to_numpy())


def load_trial(stimuli, trial_path):
    screens_lst = list(range(1, len(stimuli['screens']) + 1))
    screens, screens_fixations, screens_lines = dict.fromkeys(screens_lst), dict.fromkeys(screens_lst), dict.fromkeys(
        screens_lst)
    for screenid in screens_lst:
        screens_lines[screenid] = load_screen_linescoords(screenid, trial_path)
        screens[screenid] = load_stimuli_screen(screenid, stimuli)
        screens_fixations[screenid] = load_screen_fixations(screenid, trial_path)
    return screens, screens_fixations, screens_lines


def update_and_save_trial(sequence_states, stimuli, trial_path):
    # Screens sequence may change (e.g. if all fixations in a screen were deleted)
    del_seqindeces = [seq_id for seq_id in sequence_states if len(sequence_states[seq_id]['fixations']) == 0]
    screens_lst = list(range(1, len(stimuli['screens']) + 1))
    screens_fixations, screens_lines = {screenid: [] for screenid in screens_lst}, \
        {screenid: [] for screenid in screens_lst}
    for seq_id in sequence_states:
        screenid = sequence_states[seq_id]['screenid']
        screens_fixations[screenid].append(sequence_states[seq_id]['fixations'])
        screens_lines[screenid].append(sequence_states[seq_id]['lines'])
    save_trial(screens_fixations, screens_lines, del_seqindeces, trial_path)
    messagebox.showinfo(title='Saved', message='Trial saved successfully')


def save_trial(screens_fixations, screens_lines, del_seqindices, item_path):
    fix_filename = 'fixations.pkl'
    lines_filename = 'lines.pkl'
    for screen_id in screens_fixations:
        screen_fixations, screen_lines = screens_fixations[screen_id], screens_lines[screen_id]
        screen_path = get_screenpath(screen_id, item_path)
        if screen_path.exists():
            shutil.rmtree(screen_path)
        screen_path.mkdir()

        screenfix_filename = fix_filename
        screenlines_filename = lines_filename
        for fixations, lines in zip(screen_fixations, screen_lines):
            fixations_files = list(sorted(screen_path.glob(f'{fix_filename[:-4]}*')))
            # Account for repeated screens (i.e. returning to it)
            if len(fixations_files):
                screenfix_filename = f'{fix_filename[:-4]}_{len(fixations_files)}.pkl'
                screenlines_filename = f'{lines_filename[:-4]}_{len(fixations_files)}.pkl'
            if len(fixations):
                fixations.to_pickle(screen_path / screenfix_filename)
                save_linescoords(lines, screen_path, screenlines_filename)

    screen_sequence = load_screensequence(item_path)
    screen_sequence.drop(index=screen_sequence.iloc[del_seqindices].index, inplace=True)
    save_screensequence(screen_sequence, item_path)


def save_linescoords(lines, screen_path, filename='lines.pkl'):
    pd.DataFrame(lines, columns=['y']).to_pickle(screen_path / filename)


def save_calibrationdata(cal_points, val_points, val_offsets, trial_path):
    save_path = trial_path / 'calibration'
    if not save_path.exists():
        save_path.mkdir()
    cal_points.to_pickle(save_path / 'cal_points.pkl')
    val_points.to_pickle(save_path / 'val_points.pkl'), val_offsets.to_pickle(save_path / 'val_offsets.pkl')


def save_structs(et_messages, screen_sequence, answers, words, flags, trial_path):
    et_messages.to_pickle(trial_path / 'et_messages.pkl')
    screen_sequence.to_pickle(trial_path / 'screen_sequence.pkl')
    answers.to_pickle(trial_path / 'answers.pkl')
    words.to_pickle(trial_path / 'words.pkl')
    flags.to_pickle(trial_path / 'flags.pkl')


def load_lines_text_by_screen(item_name, stimuli_path):
    screens_lines = load_lines_by_screen(stimuli_path / f'{item_name}.mat')
    screens_text = {int(screenid): [line['text'] for line in screens_lines[screenid]] for screenid in screens_lines}
    return screens_text


def load_lines_by_screen(item):
    item_cfg = load_matfile(str(item))
    lines, num_screens = item_cfg['lines'], len(item_cfg['screens'])
    screens_lines = {screen_id: [] for screen_id in range(1, num_screens + 1)}
    for line in lines:
        screens_lines[line['screen']].append({'text': line['text'],
                                              'spaces_pos': line['spaces_pos']})

    return screens_lines


def load_calibrationdata(calibration_path):
    cal_points = load_pickle(calibration_path, 'cal_points.pkl')
    val_points = load_pickle(calibration_path, 'val_points.pkl')
    val_offsets = load_pickle(calibration_path, 'val_offsets.pkl')

    return cal_points, val_points, val_offsets


def load_screensequence(item_path, filename='screen_sequence.pkl'):
    screen_sequence = pd.read_pickle(item_path / filename)
    return screen_sequence


def load_stimuli(item, stimuli_path, config_file=None):
    stimuli_file = stimuli_path / (item + '.mat')
    if not stimuli_file.exists():
        raise ValueError('stimuli file does not exist: ' + str(stimuli_file))
    stimuli = load_matfile(str(stimuli_file))
    if config_file:
        config = load_matfile(str(config_file))
        short_stimuli_list = [stimuli_name.strip() for stimuli_name in config['short_stimuli']]
        if item in short_stimuli_list:
            stimuli['config'] = config['short_config']
        else:
            stimuli['config'] = config['long_config']

    return stimuli


def load_stimuli_screen(screenid, stimuli):
    return stimuli['screens'][screenid - 1]['image']


def load_screen_fixations(screenid, item_path):
    screen_path = get_screenpath(screenid, item_path)
    fixations_files = screen_path.glob('fixations*.pkl')
    screen_fixations = [pd.read_pickle(fix_file) for fix_file in sorted(fixations_files, reverse=True)]
    return screen_fixations


def load_screen_linescoords(screenid, item_path):
    screen_path = get_screenpath(screenid, item_path)
    lines_files = screen_path.glob('lines*.pkl')
    screen_lines = [pd.read_pickle(lines_file).to_numpy() for lines_file in sorted(lines_files, reverse=True)]
    return screen_lines


def default_screen_linescoords(screenid, stimuli):
    linespacing = stimuli['config']['linespacing']
    screen_linescoords = [line['bbox'][1] - (linespacing // 2) for line in stimuli['lines'] if
                          line['screen'] == screenid]
    # Add additional line to enclose the last line
    screen_linescoords.append(screen_linescoords[-1] + linespacing)

    return screen_linescoords


def get_points_coords(val_msgs, num_points):
    points = val_msgs['text'].str.extract(r'(\d+),(\d+)').astype(int)[:num_points]
    points.columns = ['x', 'y']
    return points


def load_manualvaldata(manualval_path, trial_path):
    et_msgs = load_pickle(trial_path, 'et_messages.pkl')
    val_msgs = et_msgs[et_msgs['text'].str.contains('validation')]
    manualval_points = get_points_coords(val_msgs, num_points=9)
    manualval_fixs = [load_pickle(manualval_path, 'first.pkl'), load_pickle(manualval_path, 'last.pkl')]

    return manualval_fixs, manualval_points


def add_offsets(cal_points, val_points, val_offsets, screen_size):
    cal_fixs = cal_points
    if not cal_points.empty:
        cal_fixs += (screen_size[1] / 2, screen_size[0] / 2)
        cal_augfactor = (cal_fixs - cal_fixs.iloc[0]) * 5
        cal_fixs += cal_augfactor
    val_fixs = val_points + val_offsets
    cal_fixs['duration'] = 1
    val_fixs['duration'] = 1

    return cal_fixs, val_fixs


def save_measures_by_subj(item_measures, save_path):
    if not item_measures.empty:
        if not save_path.exists():
            save_path.mkdir(parents=True)
        for subj in item_measures['subj'].unique():
            subj_measures = item_measures[item_measures['subj'] == subj]
            subj_measures.reset_index(drop=True, inplace=True)
            subj_measures.to_pickle(save_path / f'{subj}.pkl')


def save_subjects_scanpaths(items_scanpaths, words_avg_measures, chars_mapping, save_path, measure=None):
    dir_name = 'scanpaths'
    if measure is not None:
        dir_name += f'_{measure.lower()}'
    for item in items_scanpaths:
        item_scanpaths = items_scanpaths[item]
        item_savepath = save_path / dir_name / item
        if not item_savepath.exists():
            item_savepath.mkdir(parents=True)
        for subj in item_scanpaths:
            subj_scanpath = get_scanpath_string(item_scanpaths[subj]['words'])
            clean_words = [word.lower().translate(chars_mapping) for word in item_scanpaths[subj]['words']]
            words_measures = words_avg_measures.loc[clean_words]
            last_line_pos = 0
            for line in subj_scanpath:
                words = line.split()
                line_measures = words_measures.iloc[last_line_pos:last_line_pos + len(words)]
                line_measures = line_measures[measure].tolist() if measure is not None else [0] * len(words)
                dump = {'text': line, 'fix_dur': line_measures}
                with (item_savepath / f'{subj}.json').open('a') as f:
                    f.write(json.dumps(dump))
                    f.write('\n')
                last_line_pos += len(words)


def get_scanpath_string(scanpath):
    subj_scanpath = ' '.join(scanpath)
    subj_scanpath = subj_scanpath.replace('. ', '.\n')
    subj_scanpath = subj_scanpath.split('\n')
    return subj_scanpath


def average_measures(item_measures, measures, n_bins):
    subjects_measures = []
    for subj in item_measures['subj'].unique():
        subj_measures = item_measures[item_measures['subj'] == subj]
        for measure in measures:
            measure_mask = subj_measures[measure] != 0
            max_n_bins = subj_measures[measure_mask][measure].nunique()
            if max_n_bins < n_bins:
                n_bins = max_n_bins
            binarized = pd.qcut(subj_measures[measure][measure_mask], n_bins, labels=[j for j in range(1, n_bins + 1)])
            subj_measures.loc[binarized.index, measure] = binarized.astype(int)
        subjects_measures.append(subj_measures)
    subjects_measures_df = pd.concat(subjects_measures)
    # sanity checks: each word_idx corresponds to the same word and each subject has all word_idx
    assert all(subjects_measures_df.groupby('word_idx')['word'].nunique() == 1)
    assert all(subjects_measures_df.groupby('subj')['word_idx'].nunique() ==
               subjects_measures_df['word_idx'].nunique())
    all_measures = measures + ['FC', 'RC', 'LS', 'RR']
    averaged_measures = subjects_measures_df[['word_idx'] + all_measures].groupby(['word_idx']).mean()
    averaged_measures['word'] = subjects_measures_df.loc[averaged_measures.index, 'word']
    averaged_measures['excluded'] = subjects_measures_df.loc[averaged_measures.index, 'excluded']
    return averaged_measures
