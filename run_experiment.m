function run_experiment()
    addpath('Code/')
    % Constants
    SAVE_PATH      = 'Data';
    METADATA_PATH  = 'Metadata';
    TEST_FILE      = 'Test';
    stimuli_splits = [6 6 5];
    
    [subjname, reading_level, use_eyetracker] = initial_questions();
    if isempty(subjname); return; end
    
    SAVE_PATH = fullfile(SAVE_PATH, subjname);
    if exist(SAVE_PATH, 'dir') ~= 7
        mkdir(SAVE_PATH)
    end
    subjfile = fullfile(SAVE_PATH, 'metadata');
    
    load(fullfile(METADATA_PATH, 'stimuli_config.mat'));
    load(fullfile(METADATA_PATH, 'stimuli_order.mat'));
    load(fullfile(METADATA_PATH, 'stimuli_questions.mat'));
    
    ordered_stimuli = transpose(string({stimuli_order(:).title}));
    % Sanity check
    if length(ordered_stimuli) ~= sum(stimuli_splits)
        disp('ERROR: la suma de los bloques no condice con la cantidad de textos')
        return
    end
    
    shuffled_stimuli = shuffle_in_blocks(stimuli_splits, ordered_stimuli);
    shuffled_stimuli = cat(1, TEST_FILE, shuffled_stimuli);
    stimuli_index = 1;
    
    save(subjfile, 'subjname', 'reading_level', 'shuffled_stimuli', 'stimuli_index', 'use_eyetracker')
    
    for i = stimuli_index:length(shuffled_stimuli)
        if i == 1
            % Test trial
            use_eyetracker_in_trial = 0;
        else
            use_eyetracker_in_trial = use_eyetracker;
        end
    
        exit_status = run_trial(subjname, i, shuffled_stimuli, stimuli_questions, config, SAVE_PATH, use_eyetracker_in_trial);
        
        if exit_status == 1
            % Aborted
            break
        end
    end
end

function shuffled_elems = shuffle_in_blocks(blocks_size, elems)
    shuffled_elems   = {};
    block_startindex = 1;
    for split_index = 1:length(blocks_size)
        block_finishindex = block_startindex + blocks_size(split_index) - 1;
        current_block     = elems(block_startindex:block_finishindex);
        shuffled_elems    = cat(1, shuffled_elems, current_block(randperm(blocks_size(split_index))));
        block_startindex  = block_finishindex + 1;
    end
end

function [initials, reading_level, use_eyetracker] =  initial_questions()
    initials = '';
    reading_level = '';
    use_eyetracker = 0;
    
    prompt = {'Ingrese sus iniciales (incluya segundo nombre, si lo tiene):', ...
        'Del 1 al 10, ¿que tan frecuentemente lee?', ...
        '¿Usar el eyetracker? (Y/N)'};
    dlgtitle = 'Metadata';
    dims     = [1 40];
    definput = {'', '', 'Y'};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    if isempty(answer)
        return
    else
        initials = upper(answer{1});
        if ~isempty(answer{2})
            reading_level = answer{2};
        else
            reading_level = 'N/C';
        end
        if isempty(answer{3})
            use_eyetracker = 0;
        else
            if strcmp(upper(answer{3}), 'Y')
                use_eyetracker = 1;
            else
                use_eyetracker = 0;
            end
        end
    end
end