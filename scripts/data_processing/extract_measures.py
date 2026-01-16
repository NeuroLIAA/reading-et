from scripts.data_processing import utils
from pathlib import Path
from scripts.data_processing.assign_fix_to_words import assign_fixations_to_words
from scripts.data_processing.utils import average_measures
from tqdm import tqdm
import argparse
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

PUNCTUATION_MARKS = ['?', '!', '.']
WEIRD_CHARS = ['¿', '?', '¡', '!', '.', '−', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
# Not excluded: '(', ')', ',', ';', ':', '—', '«', '»', '“', '”', '‘', '’'
CHARS_MAP = {'—': '', '‒': '', '−': '', '-': '', '«': '', '»': '',
             '“': '', '”': '', '\'': '', '\"': '', '‘': '', '’': '',
             '(': '', ')': '', ';': '', ',': '', ':': '', '.': '', '…': '',
             '¿': '', '?': '', '¡': '', '!': ''}

""" Script to compute eye-tracking measures for each item based on words fixations.
    Measures extracted on a single trial basis:
        Early:
            - FFD (First Fixation Duration): duration of the first, and only the first, fixation on a word
            - SFD (Single Fixation Duration): duration of the first and only fixation on a word 
                (equal to FFD for single fixations). Zero if the word has more than one fixation.
            - FPRT (First Pass Reading Time/Gaze Duration): sum of all fixations on a word before exiting
                (either to the right or to the left)
    
        Intermediate:
            - RPD (Regression Path Duration): sum of all fixations on a word after exiting it
    
        Late:
            - TFD (Total Fixation Duration): sum of all fixations on a word, including regressions
            - RRT (Re-Reading Time): regression path duration minus first pass reading time
            - SPRT (Second Pass Reading Time): sum of fixations after a word has been exited for the first time
            - FC (Fixation Count): number of fixations on a word
            - RC (Regression Count): number of regressions on a word
    
    Measures extracted across trials:
        Early:
            - LS (Likelihood of Skipping): number of first pass fixation durations of 0 divided by the number of trials
        Intermediate:
            - RR (Regression Rate): number of trials with a regression divided by the number of trials
"""


def main(item, data_path, items_path, trials_path, save_path, reprocess):
    subjects, items = utils.get_dirs(trials_path), utils.get_items(items_path, item)
    assign_fixations_to_words(items, subjects, data_path, reprocess=False)
    if item != 'all':
        items_wordsfix = [data_path / item]
    else:
        items_wordsfix = utils.get_dirs(data_path)
    chars_mapping = str.maketrans(CHARS_MAP)
    extract_measures(items_wordsfix, chars_mapping, items_path, save_path, reprocess)


def words_measures(items_measures, save_path):
    excluded_words = items_measures[items_measures['excluded']]
    items_measures = items_measures[~items_measures['excluded']]
    items_measures = items_measures.drop(columns=['excluded'])
    items_measures = items_measures.groupby(['word']).mean().round(2)
    items_measures.to_csv(save_path / 'words_measures.csv')
    missing_words = set(excluded_words['word']) - set(items_measures.index)
    missing_words_df = pd.DataFrame(0, index=list(missing_words), columns=items_measures.columns)
    items_measures = pd.concat([items_measures, missing_words_df])
    return items_measures


def process_single_item(item, items_path, chars_mapping, reprocess, save_path):
    screens_text = utils.load_lines_text_by_screen(item.stem, items_path)
    item_measures_path = save_path / 'measures' / item.name
    item_trials = get_trials_to_process(item, item_measures_path, reprocess)
    if item_trials:
        item_measures, item_scanpaths = extract_item_measures(screens_text, item_trials, chars_mapping)
        item_measures = add_aggregated_measures(item_measures)
        item_avg_measures = average_measures(item_measures,
                                             measures=['FFD', 'SFD', 'FPRT', 'TFD', 'RPD', 'RRT', 'SPRT'],
                                             n_bins=10)
        utils.save_measures_by_subj(item_measures, item_measures_path)
        return item.name, item_avg_measures, item_scanpaths
    else:
        return item.name, None, None


def extract_measures(items_wordsfix, chars_mapping, items_path, save_path, reprocess=False):
    print(f'Extracting eye-tracking measures from trials...')
    items_measures = pd.DataFrame()
    items_scanpaths = {item.name: {} for item in items_wordsfix}

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_single_item, item, items_path, chars_mapping, reprocess, save_path):
                       item for item in items_wordsfix}
        for future in tqdm(as_completed(futures), total=len(items_wordsfix), desc='Processing items in parallel'):
            item_name, item_avg_measures, item_scanpaths = future.result()
            if item_avg_measures is not None:
                items_measures = pd.concat([items_measures, item_avg_measures], ignore_index=True)
                items_scanpaths[item_name] = item_scanpaths

    if not items_measures.empty:
        words_avg_measures = words_measures(items_measures, save_path)
        utils.save_subjects_scanpaths(items_scanpaths, words_avg_measures, chars_mapping, save_path, measure=None)


def extract_item_measures(screens_text, trials, chars_mapping):
    measures, words_fix = [], []
    for trial in trials:
        trial_df = pd.read_pickle(trial)
        add_trial_measures(trial_df, screens_text, chars_mapping, measures, words_fix)
    measures = pd.DataFrame(measures, columns=['subj', 'screen', 'word_idx', 'word', 'sentence_idx', 'sentence_pos',
                                               'screen_pos', 'excluded', 'FFD', 'SFD', 'FPRT', 'RPD', 'TFD', 'RRT',
                                               'SPRT', 'FC', 'RC'])

    words_fix = pd.DataFrame(words_fix, columns=['subj', 'fix_idx', 'fix_duration', 'word_idx'])
    words_fix = words_fix.sort_values(['subj', 'fix_idx'])
    scanpaths_texts = build_scanpaths(words_fix, screens_text, chars_mapping)
    return measures, scanpaths_texts


def add_aggregated_measures(item_measures):
    if not item_measures.empty:
        valid_measures = item_measures[~item_measures['excluded']]
        item_measures['LS'] = valid_measures.groupby(['word_idx'])['FPRT'].transform(lambda x: sum(x == 0) / len(x))
        item_measures['RR'] = valid_measures.groupby(['word_idx'])['RRT'].transform(lambda x: sum(x > 0) / len(x))
    return item_measures


def add_trial_measures(trial, screens_text, chars_mapping, measures, words_fix):
    word_idx, sentence_idx, sentence_pos = 0, 0, 0
    for screen in screens_text:
        screen_words = []
        add_words_to_list(screens_text[screen], screen_words)
        screen_fix = trial[trial['screen'] == int(screen)]
        if screen_fix.empty:
            continue
        for word_pos, word in enumerate(screen_words):
            prev_word = screen_words[word_pos - 1] if word_pos > 0 else ''
            clean_word = word.lower().translate(chars_mapping)
            exclude = should_exclude_word(prev_word, word, clean_word, word_pos, line_num_words(word_pos, screen_fix))

            word_fix = screen_fix[screen_fix['word_pos'] == word_pos]
            add_word_fixations(word_fix, word_idx, words_fix)
            add_word_measures(word_idx, clean_word, sentence_idx, sentence_pos, word_pos, exclude, word_fix,
                              screen_fix, measures)
            sentence_pos = sentence_pos + 1 if not is_end_of_sentence(word) else 0
            sentence_idx = sentence_idx if not is_end_of_sentence(word) else sentence_idx + 1
            word_idx += 1


def build_scanpaths(words_fix, screens_text, chars_mapping):
    item_text, item_sentences_ids = divide_into_words(screens_text)
    item_text = pd.DataFrame({'word': item_text, 'sentence_id': item_sentences_ids})
    scanpaths_text = {}
    for subj in words_fix['subj'].unique():
        subj_fix = words_fix[words_fix['subj'] == subj]
        subj_scanpath_df = item_text.iloc[subj_fix['word_idx']]
        subj_scanpath = subj_scanpath_df['word'].tolist()
        sentences_ids = subj_scanpath_df['sentence_id'].tolist()
        subj_scanpath = parse_sentences(subj_scanpath, sentences_ids, chars_mapping)
        scanpaths_text[subj] = {'words': subj_scanpath, 'words_ids': subj_fix['word_idx'].tolist()}
    return scanpaths_text


def add_word_fixations(word_fix, word_idx, words_fix):
    if not has_no_fixations(word_fix):
        words_fix.extend([word_fix['subj'].iloc[0], fix_idx, fix_duration, word_idx]
                         for fix_idx, fix_duration in zip(word_fix['trial_fix'], word_fix['duration']))


def add_word_measures(word_idx, clean_word, sentence_idx, sentence_pos, screen_pos, exclude, word_fix, screen_fix,
                      measures):
    subj_name, screen, word_pos = word_fix['subj'].iloc[0], word_fix['screen'].iloc[0], word_fix['word_pos'].iloc[0]
    if has_no_fixations(word_fix) or exclude:
        measures.append([subj_name, screen, word_idx, clean_word, sentence_idx, sentence_pos, screen_pos, exclude,
                         0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        following_words_fix = screen_fix[screen_fix['word_pos'] > word_pos].dropna()
        prev_words_fix = screen_fix[screen_fix['word_pos'] < word_pos].dropna()
        ffd, sfd, fprt, rpd, tfd, rrt, sprt, fc, rc = word_measures(word_fix, prev_words_fix, following_words_fix)
        measures.append([subj_name, screen, word_idx, clean_word, sentence_idx, sentence_pos, screen_pos, exclude,
                         ffd, sfd, fprt, rpd, tfd, rrt, sprt, fc, rc])


def word_measures(word_fix, prev_words_fix, following_words_fix):
    n_first_pass_fix = first_pass_n_fix(word_fix, following_words_fix)
    last_fix_before_exiting = last_fix_before_exiting_to_the_right(word_fix, following_words_fix)
    regressions_to_prev = regressions_to_previous_words(word_fix, prev_words_fix, last_fix_before_exiting)
    fix_dur_before_exiting_to_the_right = word_fix['duration'].loc[:last_fix_before_exiting].sum() \
        if last_fix_before_exiting != -1 else 0
    ffd = word_fix['duration'].iloc[0] if n_first_pass_fix > 0 else 0
    sfd = ffd if len(word_fix['screen_fix']) == 1 else 0
    fprt = word_fix['duration'][:n_first_pass_fix].sum()
    rpd = regressions_to_prev + fix_dur_before_exiting_to_the_right
    tfd = word_fix['duration'].sum()
    rrt = rpd - fprt
    sprt = tfd - fprt
    fc = len(word_fix['screen_fix'])
    rc = fc - n_first_pass_fix
    return ffd, sfd, fprt, rpd, tfd, rrt, sprt, fc, rc


def first_pass_n_fix(word_fix, following_words_fix):
    # Count number of first pass reading fixations on word
    fst_fix = word_fix['screen_fix'].iloc[0]
    regressive_saccade = (following_words_fix['screen_fix'] < fst_fix).values.any()
    if regressive_saccade:
        # Word was first skipped and then fixated (i.e., right-to-left saccade)
        return 0
    else:
        # This disregards intra-word regressions; inter-word regressions (rightward or leftward) are considered
        return n_consecutive_fix(word_fix['screen_fix'])


def last_fix_before_exiting_to_the_right(word_fix, following_words_fix):
    first_fix_to_the_right = following_words_fix['screen_fix'].min()
    fixs_before_exiting_to_the_right = word_fix[word_fix['screen_fix'] < first_fix_to_the_right]['screen_fix']
    if not fixs_before_exiting_to_the_right.empty:
        last_fix_idx = fixs_before_exiting_to_the_right.idxmax()
    else:
        last_fix_idx = -1
    return last_fix_idx


def regressions_to_previous_words(word_fix, prev_words_fix, last_fix_before_exiting):
    return prev_words_fix[(prev_words_fix['screen_fix'] > word_fix['screen_fix'].iloc[0]) &
                          (prev_words_fix['screen_fix'] < last_fix_before_exiting)]['duration'].sum()


def n_consecutive_fix(fix_indices):
    fix_counter = 1
    for i, fix_idx in enumerate(fix_indices[:-1]):
        if fix_idx != fix_indices.iloc[i + 1] - 1:
            break
        fix_counter += 1

    return fix_counter


def get_trials_to_process(item, item_savepath, reprocess):
    trials_to_process = utils.get_files(item, 'pkl')
    if item_savepath.exists() and not reprocess:
        trials_to_process = [trial for trial in trials_to_process if not (item_savepath / trial.name).exists()]
    return trials_to_process


def parse_sentences(subj_scanpath, sentences_ids, chars_mapping):
    curr_sentence_id = 0
    for i, word in enumerate(subj_scanpath[:-1]):
        next_word, next_sentence_id = subj_scanpath[i + 1], sentences_ids[i + 1]
        word = word.replace('.', '')
        word = word if word != next_word else word.translate(chars_mapping)
        subj_scanpath[i] = word if curr_sentence_id == next_sentence_id else word.translate(chars_mapping) + '.'
        curr_sentence_id = next_sentence_id
    return subj_scanpath


def divide_into_words(screens_text):
    item_text, sentences_ids = [], [0]
    for screenid in screens_text:
        screen_text = screens_text[screenid]
        add_words_to_list(screen_text, item_text, sentences_ids)
    return item_text, sentences_ids


def add_words_to_list(text, words, sentences_ids=None):
    for line in text:
        for i, word in enumerate(line.split()):
            words.append(word)
            if sentences_ids and len(words) > 1:
                sentences_ids.append(sentences_ids[-1] + 1 if '.' in words[-2] else sentences_ids[-1])


def word_pos_in_item(screen_id, screens_text):
    word_index = sum([num_words(screens_text[str(screen)]) for screen in range(1, int(screen_id))])
    return word_index


def num_words(text):
    return sum([len(line.split()) for line in text])


def line_num_words(word_pos, screen_fix):
    line_num = screen_fix[screen_fix['word_pos'] == word_pos]['line'].iloc[0]
    line_nwords = screen_fix[screen_fix['line'] == line_num]['word_pos'].max()

    return line_nwords


def is_end_of_sentence(word):
    return bool([char for char in PUNCTUATION_MARKS if char in word])


def should_exclude_word(prev_word, word, clean_word, word_pos, line_nwords):
    return has_weird_chars(word) or has_no_chars(clean_word) or is_end_of_sentence(prev_word) \
            or is_first_word_in_line(word_pos) or is_last_word_in_line(word_pos, line_nwords)


def is_first_word_in_line(word_pos):
    return word_pos == 0


def is_last_word_in_line(word_pos, line_max_wordpos):
    return word_pos == line_max_wordpos


def has_no_chars(word):
    return len(word) == 0


def has_weird_chars(word):
    return any(char in word for char in WEIRD_CHARS)


def has_no_fixations(word_fix):
    return word_fix['trial_fix'].isna().all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics based on words fixations')
    parser.add_argument('--data_path', type=str, default='../../data/processed/words_fixations',
                        help='Where items\' fixations by word are stored')
    parser.add_argument('--items_path', type=str, default='../../stimuli',
                        help='Items path, from which the stimuli (items\' text) is extracted')
    parser.add_argument('--trials_path', type=str, default='../../data/processed/trials',
                        help='Path to trials data.')
    parser.add_argument('--save_path', type=str, default='../../results')
    parser.add_argument('--item', type=str, default='all')
    parser.add_argument('--reprocess', action='store_true')
    args = parser.parse_args()

    data_path, trials_path, items_path, save_path = Path(args.data_path), Path(args.trials_path), \
        Path(args.items_path), Path(args.save_path)

    main(args.item, data_path, items_path, trials_path, save_path, args.reprocess)
