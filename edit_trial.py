from pathlib import Path
from scripts.data_processing import parse, plot, utils
from datetime import datetime
import numpy as np
import argparse


def select_trial(raw_path, ascii_path, config, questions, stimuli_path, data_path, subj):
    subj_datapath, subj_rawpath = Path(data_path) / subj, Path(raw_path) / subj
    if not subj_rawpath.exists():
        raise ValueError('Participant not found')
    subj_items, subj_profile = load_subj_trials(subj_rawpath, ascii_path, config, stimuli_path, subj_datapath)
    trials_flags = utils.load_flags(subj_items, subj_datapath)
    main_menu(subj_items, trials_flags, subj_profile, subj_datapath, stimuli_path, questions)


def list_participants(raw_path, processed_path):
    participants = [dir_.name for dir_ in utils.get_dirs(raw_path)]
    participants = flag_and_order_participants(raw_path, processed_path, participants)
    chosen_participant = list_options(participants, prompt='Choose a participant: ')
    return participants[chosen_participant]


def flag_and_order_participants(raw_path, processed_path, participants):
    participants_dates = [datetime.strptime(date, '%d-%m-%Y %H:%M') for date in
                          [utils.load_matfile(raw_path / participant / 'metadata.mat')['fst_date'] for participant in
                           participants]]
    for i, participant in enumerate(participants):
        processed_trials_path = processed_path / participant
        raw_trials_path = raw_path / participant
        if processed_trials_path.exists():
            processed_trials = [dir_.name for dir_ in utils.get_dirs(processed_trials_path)]
            trials_flags = utils.load_flags(processed_trials, processed_trials_path)
            edited_trials = [trials_flags[trial]['edited'] for trial in trials_flags]
            all_trials = [trial.stem for trial in utils.get_files(raw_trials_path, extension='mat')
                          if trial.stem != 'Test' and trial.stem != 'metadata']
            all_edited = np.all(edited_trials) and len(edited_trials) == len(all_trials)
            if all_edited:
                participants[i] += ' \u2705'

    participants = [participant for _, participant in sorted(zip(participants_dates, participants))]
    return participants


def show_trial_menu(subj_items, trials_flags, subj_datapath, stimuli_path, questions, chosen_option):
    chosen_item = subj_items[chosen_option]
    trial_flags = trials_flags[chosen_item]
    trial_path = subj_datapath / chosen_item
    stimuli = utils.load_stimuli(chosen_item, stimuli_path)
    trial_menu(chosen_item, trial_flags, trial_path, stimuli, questions)
    updated_options = items_list(subj_items, trials_flags)
    return updated_options


def main_menu(subj_items, trials_flags, subj_profile, subj_datapath, stimuli_path, questions):
    options = items_list(subj_items, trials_flags)
    chosen_item = print_mainmenu(subj_profile, options)
    while chosen_item != len(options) - 1:
        options = show_trial_menu(subj_items, trials_flags, subj_datapath, stimuli_path, questions, chosen_item)
        chosen_item = print_mainmenu(subj_profile, options)


def items_list(subj_items, trials_flags):
    options = [subj_items[i] + ' ' + parse_flags(trials_flags[subj_items[i]]) for i in range(len(subj_items))]
    options += ['Exit']
    return options


def trial_menu(item, trial_flags, trial_path, stimuli, questions):
    actions = ['Questions answers', 'Words associations', 'Plot calibration', 'Edit screens', 'Flag as wrong', 'Exit']
    print('\n' + item)
    action = actions[list_options(actions, '')]
    while action != 'Exit':
        handle_action(item, action, stimuli, questions, trial_flags, trial_path)
        print('\n' + item)
        action = actions[list_options(actions, '')]


def print_mainmenu(subj_profile, options):
    print('Participant:', subj_profile['name'][0])
    chosen_option = list_options(options, 'Enter the item number to edit: ')
    return chosen_option


def handle_action(item, action, stimuli, questions_file, trial_flags, trial_path):
    if action == 'Questions answers':
        trial_flags['wrong_answers'] = read_questions_and_answers(questions_file, item, trial_path)
    elif action == 'Words associations':
        read_words_associations(questions_file, item, trial_path)
    elif action == 'Plot calibration':
        plot.calibration(trial_path)
    elif action == 'Edit screens':
        trial_flags['edited'] = plot.trial(stimuli, trial_path, editable=True) or trial_flags['edited']
    elif action == 'Flag as wrong':
        trial_flags['iswrong'] = not trial_flags['iswrong'][0]
        trial_flags['edited'] = True
    elif action == 'Exit':
        exit()
    utils.update_flags(trial_flags, trial_path)


def list_options(options, prompt):
    for i in range(len(options)):
        print(f'{i + 1}. {options[i]}')
    choice = input(prompt)
    while not choice.isdigit() or int(choice) < 1 or int(choice) > len(options):
        print('Invalid choice. Please enter a number between 1 and', len(options))
        choice = input(prompt)
    return int(choice) - 1


def read_words_associations(questions_file, item, trial_path):
    _, _, words = utils.load_questions_and_words(questions_file, item)
    answers = utils.load_answers(trial_path, filename='words.pkl')
    if len(words) == 0:
        print('No words found for this item')
        return
    else:
        for i in range(len(words)):
            print(f'{i + 1}. {words[i]}')
            answer = answers[i] if len(answers) > i else ''
            print(f'      {answer}')
    input('Press enter to continue...')


def read_questions_and_answers(questions_file, item, trial_path):
    questions, possible_answers, _ = utils.load_questions_and_words(questions_file, item)
    answers = utils.load_answers(trial_path, filename='answers.pkl')
    if len(questions) == 0:
        print('No questions found for this item')
        return 0
    else:
        for i in range(len(questions)):
            print(f'{i + 1}. {questions[i]} ({possible_answers[i]})')
            answer = answers[i] if len(answers) > i else ''
            print(f'      {answer}')

    wrong_answers = input('Number of wrong answers: ')
    while not wrong_answers.isdigit() or int(wrong_answers) < 0 or int(wrong_answers) > len(questions):
        print('Invalid choice. Please enter a number between 0 and', len(questions))
        wrong_answers = input('Number of wrong answers: ')
    return int(wrong_answers)


def load_subj_trials(subj_rawpath, ascii_path, config, stimuli_path, data_path):
    subj_items = [item.name[:-4] for item in subj_rawpath.glob('*.mat')
                  if item.name not in ['Test.mat', 'metadata.mat']]
    if data_path.exists():
        subj_processeditems = [item.name for item in data_path.iterdir() if item.is_dir()]
        missing_items = [item for item in subj_items if item not in subj_processeditems]
    else:
        missing_items = subj_items
        parse.save_profile(subj_rawpath, data_path)
    for rawitem in missing_items:
        parse.item(subj_rawpath / f'{rawitem}.mat', subj_rawpath, ascii_path, config, stimuli_path, data_path)
    subj_profile = utils.load_profile(data_path)
    subj_items = utils.reorder(subj_items, subj_profile['stimuli_order'][0])
    return subj_items, subj_profile


def parse_flags(flags):
    trial_status = ''
    if flags['edited'][0]:
        if flags['iswrong'][0]:
            trial_status += '\u274C '
        else:
            trial_status += '\u2705 '
    if flags['wrong_answers'][0]:
        trial_status += '\u2b55 ' + str(flags['wrong_answers'][0]) + ' '
    return trial_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Edit trials from a given participant')
    parser.add_argument('--raw', type=str, default='data/raw',
                        help='Path where participants raw data is stored in ASCII format')
    parser.add_argument('--ascii_path', type=str, default='asc',
                        help='Path where .asc files are stored in a participants folder')
    parser.add_argument('--config', type=str, default='metadata/stimuli_config.mat',
                        help='Config file with the stimuli information')
    parser.add_argument('--questions', type=str, default='metadata/stimuli_questions.mat',
                        help='Questions and possible answers file for each stimuli')
    parser.add_argument('--stimuli_path', type=str, default='stimuli', help='Path where the stimuli are stored')
    parser.add_argument('--data', type=str, default='data/processed/trials',
                        help='Path where the processed data is stored in pkl format')
    parser.add_argument('--subj', type=str, help='Participant\'s name')
    args = parser.parse_args()
    stimuli_path, data_path, raw_path = Path(args.stimuli_path), Path(args.data), Path(args.raw)

    if args.subj:
        select_trial(raw_path, args.ascii_path, args.config, args.questions, stimuli_path, data_path, args.subj)
    else:
        subj = list_participants(raw_path, data_path).split()[0]
        select_trial(raw_path, args.ascii_path, args.config, args.questions, stimuli_path, data_path, subj)
