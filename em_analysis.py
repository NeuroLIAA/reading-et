import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymer4 import Lmer
import argparse
from pathlib import Path
from scripts.data_processing.extract_measures import main as extract_measures
from scripts.data_processing.wa_task import parse_wa_task
from scripts.data_processing.utils import get_dirs, get_files, log

""" Script to perform data analysis on eye-tracking measures. It is composed of three steps:
    1. Assign the fixations from each trial to their corresponding word in the text
    2. Extract measures from fixations (FFD, FPRT, RPD, TFD, FC, etc.)
    3. Perform data analysis on the extracted measures """


def do_analysis(items_paths, words_freq_file, stats_file, save_path):
    print('Analysing eye-tracking measures...')
    words_freq, items_stats = pd.read_csv(words_freq_file), pd.read_csv(stats_file, index_col=0)
    et_measures = load_et_measures(items_paths, words_freq)
    print_stats(et_measures, items_stats, save_path)

    et_measures = remove_excluded_words(et_measures)
    plot_measures(et_measures, save_path)
    mlm_analysis(log_normalize_durations(et_measures), words_freq)


def print_stats(et_measures, items_stats, save_path):
    items = items_stats.index.to_list()[:-1]
    processed_stats = {item: {} for item in items}
    for item in items:
        item_measures = et_measures[et_measures['item'] == item]
        n_subjs = len(item_measures['subj'].unique())
        processed_stats[item]['subjs'] = n_subjs
        processed_stats[item]['words'] = len(item_measures[~item_measures['excluded']]['word']) // n_subjs
        processed_stats[item]['words_excluded'] = item_measures['excluded'].sum() // n_subjs
        processed_stats[item]['fix'] = item_measures['FC'].sum()
        processed_stats[item]['fix_excluded'] = items_stats.loc[item, 'n_fix'] - processed_stats[item]['fix']
        processed_stats[item]['regressions'] = item_measures['RC'].sum()
        processed_stats[item]['skips'] = item_measures['skipped'].sum()
        processed_stats[item]['out_of_bounds'] = items_stats.loc[item, 'out_of_bounds']
        processed_stats[item]['return_sweeps'] = items_stats.loc[item, 'return_sweeps']

    processed_stats = pd.DataFrame.from_dict(processed_stats, orient='index', dtype='Int64')
    processed_stats.loc['Total'] = processed_stats.sum()
    print(processed_stats.to_string())

    processed_stats.to_csv(save_path / 'trials_stats.csv')


def plot_measures(et_measures, save_path):
    et_measures_no_skipped = remove_skipped_words(et_measures)
    plot_histograms(et_measures_no_skipped, ['FFD', 'FC'], ax_titles=['First Fixation Duration', 'Fixation Count'],
                    y_labels=['Number of words', 'Number of words'], save_file=save_path / 'FFD_FC_distributions.png')
    plot_words_effects(et_measures, save_path)


def mlm_analysis(et_measures, words_freq):
    et_measures['word_len'] = et_measures['word'].apply(lambda x: 1 / len(x) if x else 0)
    et_measures['word_freq'] = et_measures['word'].apply(lambda x:
                                                         log(words_freq.loc[words_freq['word'] == x, 'cnt'].values[0])
                                                         if x in words_freq['word'].values else 0)
    et_measures = et_measures.loc[et_measures['word_freq'] != 0, :].copy()

    et_measures['word_idx'] = et_measures.groupby(['subj', 'item'])['word_idx'].transform(lambda x: x / x.max())
    et_measures['screen_pos'] = (et_measures.groupby(['subj', 'item', 'screen'])['screen_pos']
                                 .transform(lambda x: x / x.max()))
    et_measures['sentence_pos'] = et_measures.groupby('sentence_idx')['sentence_pos'].transform(lambda x: x / x.max())
    et_measures['sentence_pos_squared'] = et_measures['sentence_pos'] * et_measures['sentence_pos']
    fixed_effects = ['word_len', 'word_freq', 'sentence_pos', 'sentence_pos_squared', 'word_idx', 'screen_pos']
    for fixed_effect in fixed_effects:
        et_measures[fixed_effect] = et_measures[fixed_effect] - et_measures[fixed_effect].mean()

    # to analyze fatigue effects, remove non-fatigue trials
    # et_measures = remove_nonfatigue(et_measures)
    # et_measures['fatigue'] = et_measures['fatigue'] - et_measures['fatigue'].mean()

    fit_mlm(name='skipped',
            formula='skipped ~ word_len * word_freq + sentence_pos + sentence_pos_squared + word_idx + screen_pos '
                    '+ (1|subj) + (1|item)',
            data=et_measures,
            model_family='binomial')

    et_measures = remove_skipped_words(et_measures)
    models = [
        ('FFD',
         'FFD ~ word_len * word_freq + sentence_pos +  word_idx + screen_pos + (1|subj) + (1|item)'),
        ('FPRT',
         'FPRT ~ word_len * word_freq + sentence_pos + sentence_pos_squared + word_idx + screen_pos '
         '+ (1|subj) + (1|item)')
    ]

    for name, formula in models:
        fit_mlm(name, formula, et_measures)


def fit_mlm(name, formula, data, model_family='gaussian'):
    model = Lmer(formula, data=data, family=model_family)
    results = model.fit()
    print(f'{name} model: {formula}')
    print(results)
    print(f'AIC: {model.AIC}')

    results['formula'] = formula
    results['aic'] = model.AIC
    results.to_csv(save_path / f'{name}_mlm.csv')


def remove_nonfatigue(et_measures):
    et_measures = et_measures.dropna(subset=['fatigue'])
    return et_measures


def remove_skipped_words(et_measures):
    et_measures = et_measures[et_measures['skipped'] == 0]
    et_measures = et_measures.drop(columns=['skipped'])
    return et_measures


def remove_excluded_words(et_measures):
    et_measures = et_measures[~et_measures['excluded']]
    et_measures = et_measures.drop(columns=['excluded'])
    return et_measures


def add_len_freq_skipped(et_measures, words_freq):
    et_measures['skipped'] = et_measures[~et_measures['excluded']]['FFD'].apply(lambda x: int(x == 0))
    et_measures['word_len'] = et_measures['word'].apply(lambda x: len(x))
    words_freq = words_freq[['word', 'cnt']].copy()
    words_freq['cnt'] = pd.qcut(words_freq['cnt'], 15, labels=[i for i in range(1, 16)])
    et_measures['word_freq'] = et_measures['word'].apply(lambda x:
                                                         words_freq.loc[words_freq['word'] == x, 'cnt'].values[0]
                                                         if x in words_freq['word'].values else 0)
    return et_measures


def log_normalize_durations(trial_measures):
    for duration_measure in ['FFD', 'SFD', 'FPRT', 'RPD', 'TFD', 'SPRT']:
        trial_measures[duration_measure] = trial_measures[duration_measure].apply(lambda x: log(x))
    return trial_measures


def plot_words_effects(et_measures, save_path):
    et_measures_log = log_normalize_durations(remove_skipped_words(et_measures))
    aggregated_measures = et_measures.drop_duplicates(subset=['item', 'word_idx'])
    y_labels = ['log First Fixation Duration', 'log Gaze Duration', 'Likelihood of skipping', 'Regression rate']
    plot_boxplots(['word_len'], measures=['FFD', 'FPRT', 'LS', 'RR'],
                  data=[et_measures_log, et_measures_log, aggregated_measures, aggregated_measures],
                  x_labels=['Word length'] * 4,
                  y_labels=y_labels,
                  ax_titles=y_labels,
                  fig_title='Word length effects on measures',
                  save_file=save_path / 'word_length.png')

    plot_boxplots(['word_freq'], measures=['FFD', 'FPRT', 'LS', 'RR'],
                  data=[et_measures_log, et_measures_log, aggregated_measures, aggregated_measures],
                  x_labels=['Word frequency in percentiles'] * 4,
                  y_labels=y_labels,
                  ax_titles=y_labels,
                  fig_title='Word frequency effects on measures',
                  save_file=save_path / 'word_frequency.png')


def plot_boxplots(fixed_effects, measures, data, x_labels, y_labels, ax_titles,
                  fig_title, save_file, sharey='row', orientation='horizontal', order=None):
    n_plots = len(fixed_effects) * len(measures)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    if orientation == 'vertical':
        n_cols, n_rows = n_rows, n_cols
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharey=sharey, figsize=(n_cols * 7, n_rows * 6))
    axes = [axes] if n_plots == 1 else axes
    for i, fixed_effect in enumerate(fixed_effects):
        for j, measure in enumerate(measures):
            ax = axes[(i + j) // n_cols, (i + j) % n_cols]
            plot_data = data[j] if isinstance(data, list) else data
            sns.boxplot(x=fixed_effect, y=measure, hue=fixed_effect, data=plot_data, order=order, legend=False, ax=ax)
            ax.set_xlabel(x_labels[i])
            ax.set_ylabel(y_labels[j])
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
            ax.yaxis.set_tick_params(labelleft=True)
            ax.set_title(ax_titles[j])
    fig.suptitle(fig_title)
    fig.savefig(save_file, bbox_inches='tight')
    plt.show()
    return fig


def plot_histograms(et_measures, measures, ax_titles, y_labels, save_file):
    ncols = len(measures) // 2
    nrows = 2 if len(measures) > 1 else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = np.array(axes)[:, np.newaxis] if len(measures) <= 2 else axes
    for i, measure in enumerate(measures):
        ax = axes[i // ncols, i % ncols]
        sns.histplot(x=measure, data=et_measures, ax=ax, binwidth=1)
        ax.set_title(ax_titles[i])
        ax.set_ylabel(y_labels[i])
    fig.savefig(save_file)
    plt.tight_layout()
    plt.show()


def load_trial(trial, item_name, words_freq):
    trial_measures = add_len_freq_skipped(pd.read_pickle(trial), words_freq)
    trial_measures.insert(0, 'item', item_name)
    return trial_measures


def load_trials_measures(item, words_freq):
    trials_measures = [load_trial(trial, item.name, words_freq) for trial in get_files(item)]
    return pd.concat(trials_measures, ignore_index=True)


def load_et_measures(items_paths, words_freq):
    measures = [load_trials_measures(item, words_freq) for item in items_paths]
    measures = pd.concat(measures, ignore_index=True)
    return measures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform data analysis on extracted eye-tracking measures')
    parser.add_argument('-w', '--wordsfix', type=str, default='data/processed/words_fixations',
                        help='Where items\' fixations by word are stored')
    parser.add_argument('-m', '--measures', type=str, default='results/measures',
                        help='Path where eye-tracking measures are stored')
    parser.add_argument('-s', '--stimuli', type=str, default='stimuli',
                        help='Items path, from which the stimuli (items\' text) is extracted')
    parser.add_argument('-p', '--participants', type=str, default='data/processed/trials',
                        help='Path to participants\' trials, where their fixations and metadata are stored.')
    parser.add_argument('-q', '--questions', type=str, default='metadata/stimuli_questions.mat',
                        help='Path to file with items\' questions')
    parser.add_argument('-wf', '--words_freq', type=str, default='metadata/texts_properties/words_freq.csv',
                        help='Path to file with words frequencies')
    parser.add_argument('-st', '--stats', type=str, default='data/processed/words_fixations/stats.csv')
    parser.add_argument('-r', '--reprocess', action='store_true', help='Compute measures again, even if they exist')
    parser.add_argument('-o', '--output', type=str, default='results')
    parser.add_argument('-i', '--item', type=str, default='all')
    args = parser.parse_args()

    wordsfix_path, measures_path, stimuli_path, participants_path, save_path = \
        Path(args.wordsfix), Path(args.measures), Path(args.stimuli), Path(args.participants), Path(args.output)
    words_freq_file, stats_file, questions_file = Path(args.words_freq), Path(args.stats), Path(args.questions)
    subjects_associations, words_associations = parse_wa_task(questions_file, participants_path)

    extract_measures(args.item, wordsfix_path, stimuli_path, participants_path, save_path, reprocess=args.reprocess)

    if args.item != 'all':
        items_paths = [measures_path / args.item]
    else:
        items_paths = get_dirs(measures_path)

    save_path.mkdir(parents=True, exist_ok=True)
    subjects_associations.to_csv(save_path / 'subjects_associations.csv')
    words_associations.to_csv(save_path / 'words_associations.csv', index=False)

    do_analysis(items_paths, words_freq_file, stats_file, save_path)
