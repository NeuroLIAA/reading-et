from .et_utils import et_utils
from pathlib import Path
from scipy.io import loadmat
from pandas import to_datetime, DataFrame
import argparse
import shutil
from . import utils

""" EyeLink's EDF files are assumed to having been converted to ASCII with edf2asc.exe.
    This script extracts fixations from those files and proceeds to divide them by screen for each trial.
    Only one eye is used for fixation extraction, and the eye is chosen based on the calibration results. 
    data structures consist of dataframes in pickle format."""


def item(item, participant_path, ascii_path, config_file, stimuli_path, save_path):
    print(f'Processing {item}')
    subj_metadata = loadmat(str(participant_path / 'metadata.mat'), simplify_cells=True)
    trial_metadata = loadmat(str(item), simplify_cells=True)
    trial_date = to_datetime(trial_metadata['__header__'][-20:].decode('utf-8'), format='%b %d %H:%M:%S %Y')
    fst_date = to_datetime(subj_metadata['fst_date'], format='%d-%m-%Y %H:%M')
    item_session = 2 if trial_date.date() > fst_date.date() else 1

    trial_metadata = trial_metadata['trial']
    trial_path = save_path / item.name.split('.')[0]
    if trial_path.exists():
        shutil.rmtree(trial_path)
    trial_path.mkdir(parents=True)

    stimuli_index, subj_name = trial_metadata['stimuli_index'], trial_metadata['subjname']
    trial_fix, et_messages, cal_points, val_points, val_offsets =\
        get_eyetracking_data(participant_path / ascii_path, subj_name, stimuli_index)
    utils.save_calibrationdata(cal_points, val_points, val_offsets, trial_path)

    manualval_results = save_manualvalidation_fixations(et_messages, trial_fix, trial_path)
    screen_sequence = DataFrame.from_records(trial_metadata['sequence'])
    stimuli = utils.load_stimuli(item.name[:-4], stimuli_path, config_file)
    divide_data_by_screen(screen_sequence, et_messages, trial_fix, trial_path, stimuli, filter_outliers=True)

    flags = {'edited': False, 'firstval_iswrong': not manualval_results[0],
             'lastval_iswrong': not manualval_results[1], 'wrong_answers': 0, 'iswrong': False, 'session': item_session}
    utils.save_structs(et_messages,
                       screen_sequence,
                       DataFrame(trial_metadata['questions_answers']),
                       DataFrame(trial_metadata['synonyms_answers']),
                       DataFrame(flags, index=[0]),
                       trial_path)


def participantdata(raw_path, participant, ascii_path, config_file, stimuli_path, save_path):
    participant_path = raw_path / participant
    out_path = save_path / participant
    save_profile(participant_path, out_path)
    items = participant_path.glob('*.mat')
    for item in items:
        if item.name == 'Test.mat' or item.name == 'metadata.mat':
            continue
        item(item, participant_path, ascii_path, config_file, stimuli_path, out_path)


def rawdata(raw_path, ascii_path, config_file, stimuli_path, save_path):
    participants = utils.get_dirs(raw_path)
    for participant in participants:
        participantdata(raw_path, participant.name, ascii_path, config_file, stimuli_path, save_path)


def save_profile(participant_rawpath, save_path):
    metafile = loadmat(str(participant_rawpath / 'metadata.mat'), simplify_cells=True)
    stimuli_order = list(metafile['shuffled_stimuli'][1:].astype(str))
    profile = {'name': [metafile['subjname']], 'age': [metafile['age']],
               'reading_level': [int(metafile['reading_level'])],
               'gender': [metafile['gender']], 'n_sessions': [metafile['n_sessions']],
               'fst_date': [metafile['fst_date']], 'snd_date': [metafile['snd_date']],
               'fst_sleeptime': [metafile['fst_sleeptime']], 'snd_sleeptime': [metafile['snd_sleeptime']],
               'fst_wakeuptime': [metafile['fst_wakeuptime']], 'snd_wakeuptime': [metafile['snd_wakeuptime']],
               'stimuli_order': [stimuli_order]}
    if not save_path.exists():
        save_path.mkdir(parents=True)
    DataFrame(profile).to_pickle(save_path / 'profile.pkl')


def save_manualvalidation_fixations(et_messages, trial_fix, trial_path, val_legend='validation', num_points=9,
                                    points_area=56, error_margin=30):
    val_msgs = et_messages[et_messages['text'].str.contains(val_legend)]
    fin_msgindex = et_messages[et_messages['text'].str.contains('termina experimento')].index[0]
    first_val = val_msgs.loc[:fin_msgindex]
    last_val = val_msgs.loc[fin_msgindex:]
    # Add some time to let the eye get to the last point
    first_valfix = trial_fix[
        (trial_fix['tStart'] >= first_val.iloc[0]['time']) & (trial_fix['tEnd'] <= first_val.iloc[-1]['time'] + 500)]
    last_valfix = trial_fix[
        (trial_fix['tStart'] >= last_val.iloc[0]['time']) & (trial_fix['tEnd'] <= last_val.iloc[-1]['time'] + 500)]

    points_coords = utils.get_points_coords(val_msgs, num_points)
    firstval_iscorrect = check_validation_fixations(first_valfix, points_coords, num_points, points_area, error_margin)
    lastval_iscorrect = check_validation_fixations(last_valfix, points_coords, num_points, points_area, error_margin)

    val_path = trial_path / 'manual_validation'
    if not val_path.exists():
        val_path.mkdir()
    first_valfix.to_pickle(val_path / 'first.pkl')
    last_valfix.to_pickle(val_path / 'last.pkl')

    return firstval_iscorrect, lastval_iscorrect


def check_validation_fixations(fixations, points_coords, num_points, points_area, error_margin):
    """ For each fixation, check if it is inside the area of the point.
        As we only advance the point index once we find a fixation inside the area,
        the validation is correct only if the point index matches the number of points - 1. """
    fix_coords = fixations.loc[:, ['xAvg', 'yAvg']].astype(int).to_numpy()
    point_index = 0
    for (x, y) in fix_coords:
        lower_bound = points_coords.iloc[point_index] - (points_area + error_margin)
        upper_bound = points_coords.iloc[point_index] + (points_area + error_margin)
        if (x in range(lower_bound.iloc[0], upper_bound.iloc[0])
                and y in range(lower_bound.iloc[1], upper_bound.iloc[1])):
            point_index += 1
            if point_index == num_points - 1:
                break

    return point_index == num_points - 1


def divide_data_by_screen(trial_sequence, et_messages, trial_fix, trial_path, stimuli, filter_outliers=True):
    fix_filename, lines_filename = 'fixations.pkl', 'lines.pkl'
    for i, screen_id in enumerate(trial_sequence['currentscreenid']):
        ini_time = et_messages[et_messages['text'].str.contains('ini')].iloc[i]['time']
        fin_time = et_messages[et_messages['text'].str.contains('fin')].iloc[i]['time']
        screen_fixations = trial_fix[(trial_fix['tStart'] > ini_time) & (trial_fix['tEnd'] < fin_time)]
        if filter_outliers:
            screen_fixations = screen_fixations[
                (screen_fixations['duration'] > 50) & (screen_fixations['duration'] < 1000)]
        if screen_fixations.empty:
            trial_sequence.drop(i, inplace=True)
            continue

        screen_path = utils.get_screenpath(screen_id, trial_path)
        lines_coords = utils.default_screen_linescoords(screen_id, stimuli)
        fixations_files = list(sorted(screen_path.glob(f'{fix_filename[:-4]}*')))
        screenfix_filename, screenlines_filename = fix_filename, lines_filename
        # Account for repeated screens (i.e. returning to it)
        if len(fixations_files):
            screenfix_filename = f'{fix_filename[:-4]}_{len(fixations_files)}.pkl'
            screenlines_filename = f'{lines_filename[:-4]}_{len(fixations_files)}.pkl'

        screen_fixations.reset_index(inplace=True)
        screen_fixations.to_pickle(screen_path / screenfix_filename)
        utils.save_linescoords(lines_coords, screen_path, screenlines_filename)


def get_eyetracking_data(asc_path, subj_name, stimuli_index):
    asc_file = asc_path / f'{subj_name}_{stimuli_index}.asc'
    _, df_msg, df_fix, _, _, _ = et_utils.parse_asc(asc_file, verbose=False)
    df_fix, best_eye = et_utils.keep_besteye(df_fix, df_msg)
    cal_points = et_utils.extract_calpoints(df_msg, best_eye)
    val_points, val_offsets = et_utils.extract_valpoints(df_msg, best_eye)
    df_msg = et_utils.filter_msgs(df_msg)

    return df_fix, df_msg, cal_points, val_points, val_offsets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract data from EyeLink .asc files and save them to .pkl')
    parser.add_argument('--path', type=str, default='data/raw', help='Path where participants data is stored')
    parser.add_argument('--ascii_path', type=str, default='asc',
                        help='Path where .asc files are stored in a participants folder')
    parser.add_argument('--config', type=str, default='metadata/stimuli_config.mat',
                        help='Config file with the stimuli information')
    parser.add_argument('--stimuli_path', type=str, default='stimuli', help='Path where the stimuli are stored')
    parser.add_argument('--save_path', type=str, default='data/processed/trials',
                        help='Path where to save the processed data')
    parser.add_argument('--subj', type=str, help='Subject name', required=False)
    args = parser.parse_args()

    raw_path, stimuli_path, save_path = Path(args.path), Path(args.stimuli_path), Path(args.save_path)
    if not args.subj:
        rawdata(raw_path, args.ascii_path, args.config, stimuli_path, save_path)
    else:
        participantdata(raw_path, args.subj, args.ascii_path, args.config, stimuli_path, save_path)
