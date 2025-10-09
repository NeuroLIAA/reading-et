from pathlib import Path
from tqdm import tqdm
from scripts.data_processing import utils
import pandas as pd
import numpy as np
import argparse


def assign_fixations_to_words(items, subjects, save_path, reprocess=False):
    print('Assigning fixations to words...')
    items_stats = {item.stem: {'n_subj': 0, 'n_fix': 0, 'n_words': 0, 'out_of_bounds': 0, 'return_sweeps': 0}
                   for item in items}
    for item in (pbar := tqdm(items)):
        item_name = item.stem
        pbar.set_description(f'Processing "{item_name}" trials')
        screens_lines = utils.load_lines_by_screen(item)
        item_savepath = save_path / item_name
        item_savepath.mkdir(exist_ok=True, parents=True)
        item_subjects = get_subjects_to_process(subjects, item_name, item_savepath, reprocess)

        process_item(item_name, item_subjects, screens_lines, items_stats[item_name], item_savepath)
    save_stats(items_stats, save_path)


def process_item(item_name, subjects, screens_lines, item_stats, item_savepath):
    for subject in subjects:
        trial_path = subject / item_name
        screen_sequence = pd.read_pickle(trial_path / 'screen_sequence.pkl')['currentscreenid'].to_numpy()
        trial_fix_by_word = process_subj_trial(subject.name, trial_path, screen_sequence, screens_lines, item_stats)
        trial_fix_by_word = postprocess_word_fixations(trial_fix_by_word, item_stats)
        save_trial_word_fixations(trial_fix_by_word, item_savepath)


def process_subj_trial(subj_name, trial_path, screen_sequence, screens_lines, item_stats):
    # Keep track of returning screens
    screen_counter = {screen_id: 0 for screen_id in np.unique(screen_sequence)}
    # Each record corresponds to a word; words with multiple fixations have multiple records.
    # Words with no fixations have NA values
    trial_fix_by_word, total_trial_fix = [], 0
    for screen_id in screen_sequence:
        fixations, lines_pos = load_screen_data(trial_path, screen_id, screen_counter)
        word_pos = 0
        for line_num, line in enumerate(screens_lines[screen_id]):
            words, spaces_pos = line['text'].split(), line['spaces_pos']
            if line['text'][:3] == '   ':
                spaces_pos = spaces_pos[3:]
            line_fix = get_line_fixations(fixations, line_num, lines_pos)
            assign_line_fixations_to_words(word_pos, line_fix, line_num, spaces_pos,
                                           screen_id, subj_name, trial_fix_by_word)
            word_pos += len(words)
            total_trial_fix += len(line_fix)

        screen_counter[screen_id] += 1
    trial_fix_by_word = pd.DataFrame(trial_fix_by_word,
                                     columns=['subj', 'screen', 'line', 'word_pos', 'trial_fix',
                                              'screen_fix', 'duration', 'x'])
    update_stats(item_stats, trial_fix_by_word, total_trial_fix)

    return trial_fix_by_word


def get_line_fixations(fixations, line_number, lines_pos):
    line_fixations = fixations[fixations['yAvg'].between(lines_pos[line_number],
                                                         lines_pos[line_number + 1],
                                                         inclusive='left')]
    if not line_fixations.empty:
        # Check if first screen fixation hasn't been removed yet
        if line_fixations.iloc[0].name == 0:
            line_fixations = line_fixations.drop([0])
        # Remove last fixation if it's the last fixation on the screen
        if not line_fixations.empty and line_fixations.iloc[-1].name == len(fixations) - 1:
            line_fixations = line_fixations.drop([len(fixations) - 1])

    return line_fixations


def assign_line_fixations_to_words(word_pos, line_fix, line_num, spaces_pos, screen_id, subj_name, trial_fix_by_word):
    for i in range(len(spaces_pos) - 1):
        word_fix = line_fix[line_fix['xAvg'].between(spaces_pos[i],
                                                     spaces_pos[i + 1],
                                                     inclusive='left')]
        if word_fix.empty:
            trial_fix_by_word.append([subj_name, screen_id, line_num, word_pos, None, None, None, None])
        else:
            word_fix = word_fix[['index', 'duration', 'xAvg']]
            word_fix = word_fix.rename(columns={'index': 'trial_fix', 'xAvg': 'x'})
            word_fix.reset_index(names='screen_fix', inplace=True)
            # Shift x to start at 0
            word_fix['x'] -= spaces_pos[i]
            word_fix['subj'], word_fix['screen'], word_fix['line'], word_fix['word_pos'] = \
                subj_name, screen_id, line_num, word_pos

            word_fix = word_fix[['subj', 'screen', 'line', 'word_pos', 'trial_fix', 'screen_fix', 'duration', 'x']]
            trial_fix_by_word.extend(word_fix.values.tolist())
        word_pos += 1


def get_subjects_to_process(subjects, item_name, item_savepath, reprocess):
    subjects_to_process = utils.get_correct_trials(subjects, item_name)
    if not reprocess:
        processed_subjects = utils.get_subjects(item_savepath)
        subjects_to_process = [subj for subj in subjects_to_process if subj.name not in processed_subjects]

    return subjects_to_process


def save_trial_word_fixations(trial_fix_by_word, item_savepath):
    subj_name = trial_fix_by_word['subj'].iloc[0]
    trial_fix_by_word.to_pickle(item_savepath / f'{subj_name}.pkl')


def save_stats(items_stats, save_path):
    items_stats = pd.DataFrame.from_dict(items_stats, orient='index')
    items_stats.loc['Total'] = items_stats.sum()
    stats_file = save_path / 'stats.csv'
    if stats_file.exists():
        old_stats = pd.read_csv(stats_file, index_col=0)
        items_stats += old_stats
    items_stats.to_csv(save_path / 'stats.csv')


def postprocess_word_fixations(trial_fix_by_word, item_stats):
    prev_nfix = n_fix(trial_fix_by_word)
    trial_fix_by_word = trial_fix_by_word.groupby(['screen', 'line'], group_keys=False) \
        .apply(remove_return_sweeps_from_line)
    item_stats['return_sweeps'] += prev_nfix - n_fix(trial_fix_by_word)

    trial_fix_by_word = trial_fix_by_word.groupby(['screen', 'word_pos'], group_keys=False) \
        .apply(remove_na_from_fixated_words)
    trial_fix_by_word = make_screen_fix_consecutive(trial_fix_by_word)
    trial_fix_by_word = cast_to_int(trial_fix_by_word)
    trial_fix_by_word = trial_fix_by_word.sort_values(['screen', 'line', 'word_pos', 'screen_fix'])

    return trial_fix_by_word


def remove_return_sweeps_from_line(line_fix):
    # Remove fixations resulting from oculomotor errors when jumping lines
    fst_fix_num = line_fix['screen_fix'].min()
    first_saccade_is_regressive = is_regression(line_fix, fst_fix_num, fst_fix_num + 1)
    if first_saccade_is_regressive:
        first_word_with_fix = line_fix[~line_fix['screen_fix'].isna()]['word_pos'].min()
        if not np.isnan(first_word_with_fix):
            first_word_fix = line_fix[line_fix['word_pos'] == first_word_with_fix]
            left_most_fix = first_word_fix[first_word_fix['x'] == first_word_fix['x'].min()]
            line_fix.loc[line_fix['screen_fix'].between(fst_fix_num,
                                                        left_most_fix['screen_fix'].iloc[0],
                                                        inclusive='left'),
                                                        ['trial_fix', 'screen_fix', 'duration', 'x']] = np.nan

    return line_fix


def remove_na_from_fixated_words(words_fix):
    # Due to returning screens, there may be words that have fixations but were also added as empty rows
    if n_fix(words_fix) > 0:
        return words_fix.dropna()
    else:
        return words_fix.head(1)


def make_screen_fix_consecutive(trial_fix_by_word):
    trial_fix_by_word = trial_fix_by_word.sort_values(['screen', 'screen_fix'])
    consecutive_screen_fix = trial_fix_by_word[~trial_fix_by_word['screen_fix'].isna()].copy()
    consecutive_screen_fix['screen_fix'] = consecutive_screen_fix.groupby('screen').cumcount()
    trial_fix_by_word.update(consecutive_screen_fix)

    return trial_fix_by_word


def n_fix(df_fix):
    return len(df_fix[~df_fix['screen_fix'].isna()])


def is_regression(df_fix, fix_num, following_fix_num):
    regression = False
    first_fix = df_fix[df_fix['screen_fix'] == fix_num]
    second_fix = df_fix[df_fix['screen_fix'] == following_fix_num]
    if not first_fix.empty and not second_fix.empty:
        regression = first_fix['word_pos'].iloc[0] > second_fix['word_pos'].iloc[0] or \
                     (first_fix['word_pos'].iloc[0] == second_fix['word_pos'].iloc[0] and
                      first_fix['x'].iloc[0] > second_fix['x'].iloc[0])

    return regression


def update_stats(item_stats, trial_fix_by_word, total_trial_fix):
    item_stats['n_subj'] += 1
    item_stats['n_fix'] += n_fix(trial_fix_by_word)
    item_stats['n_words'] = len(trial_fix_by_word.groupby(['screen', 'word_pos']))
    item_stats['out_of_bounds'] += total_trial_fix - n_fix(trial_fix_by_word)


def cast_to_int(trial_fix_by_word):
    for col in ['screen', 'line', 'word_pos', 'trial_fix', 'screen_fix', 'duration']:
        trial_fix_by_word[col] = trial_fix_by_word[col].astype(pd.Int64Dtype())

    return trial_fix_by_word


def load_screen_data(trial_path, screen_id, screen_counter):
    screen_dir = trial_path / f'screen_{screen_id}'
    fix_filename, lines_filename = get_screen_filenames(screen_counter[screen_id])
    fixations = load_fixations(screen_dir / fix_filename)
    lines_pos = load_lines_pos(screen_dir / lines_filename)

    if screen_counter[screen_id] > 0:
        last_fixation_index = get_last_fixation_index(screen_dir, screen_counter[screen_id])
        fixations.index += last_fixation_index + 1

    return fixations, lines_pos


def load_fixations(fix_file):
    return pd.read_pickle(fix_file)


def load_lines_pos(lines_pos_file):
    return pd.read_pickle(lines_pos_file).sort_values('y')['y'].to_numpy()


def get_screen_filenames(screen_times_read):
    fix_filename = f'fixations.pkl'
    lines_filename = f'lines.pkl'
    if screen_times_read > 0:
        fix_filename = f'fixations_{screen_times_read}.pkl'
        lines_filename = f'lines_{screen_times_read}.pkl'

    return fix_filename, lines_filename


def get_last_fixation_index(screen_dir, prev_screen_times_read):
    last_fixation_index = 0
    for it in range(prev_screen_times_read):
        fix_filename, _ = get_screen_filenames(it)
        fixations = load_fixations(screen_dir / fix_filename)
        last_fixation_index += fixations.iloc[-1].name

    return last_fixation_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assign fixations to words')
    parser.add_argument('--items_path', type=str, default='../../stimuli')
    parser.add_argument('--data_path', type=str, default='../../data/processed/trials')
    parser.add_argument('--save_path', type=str, default='../../data/processed/words_fixations')
    parser.add_argument('--subj', type=str, default='all')
    parser.add_argument('--item', type=str, default='all')
    parser.add_argument('--reprocess', action='store_true')
    args = parser.parse_args()

    items_path, data_path, save_path = Path(args.items_path), Path(args.data_path), Path(args.save_path)
    subj_paths = [data_path / args.subj] if args.subj != 'all' else utils.get_dirs(data_path)
    items = utils.get_items(items_path, args.item)

    assign_fixations_to_words(items, subj_paths, save_path, args.reprocess)
