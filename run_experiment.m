function run_experiment()
    addpath(genpath('scripts/experiment'))
    % Constants
    SAVE_PATH      = fullfile('data', 'raw');
    METADATA_PATH  = 'metadata';
    TEST_FILE      = 'Test';
    stimuli_splits = [5 5 5 5];
    splits_per_session = 2;
    load(fullfile(METADATA_PATH, 'stimuli_config.mat'));
    load(fullfile(METADATA_PATH, 'stimuli_questions.mat'));
    
    [subjname, reading_level, sleep_time, wakeup_time, use_eyetracker] = initial_questions();
    if isempty(subjname); return; end
    
    SAVE_PATH = fullfile(SAVE_PATH, subjname);
    if exist(SAVE_PATH, 'dir') ~= 7
        mkdir(SAVE_PATH)
    end

    subjfile = fullfile(SAVE_PATH, 'metadata.mat');
    n_session = 1;
    loaded_metadata = false;
    if exist(subjfile, 'file') > 0
        opts.Interpreter = 'tex';
        opts.Default = 'Yes';
        load_metadata = questdlg('\fontsize{15} Se encontro informacion previa del participante. Cargar el archivo?', 'Experimento previo', opts);
        if strcmp(load_metadata, 'Yes')
            load(subjfile)
            loaded_metadata = true;
            n_session = 2;
            snd_sleeptime = sleep_time;
            snd_wakeuptime = wakeup_time;
            save(subjfile, 'snd_sleeptime', 'snd_wakeuptime', '-append')
        elseif strcmp(load_metadata, 'Cancel')
            return
        end
    end
   
    if ~loaded_metadata
        load(fullfile(METADATA_PATH, 'stimuli_order.mat'));
        ordered_stimuli = {stimuli_order(:).title}';

        % Sanity check
        if length(ordered_stimuli) ~= sum(stimuli_splits)
            disp('ERROR: la suma de los bloques no condice con la cantidad de textos')
            return
        end
    
        shuffled_stimuli = shuffle_in_blocks(stimuli_splits, ordered_stimuli);
        shuffled_stimuli = cat(1, TEST_FILE, shuffled_stimuli);
        stimuli_index = 1;
        fst_sleeptime = sleep_time;
        fst_wakeuptime = wakeup_time;
    
        save(subjfile, 'subjname', 'reading_level', ...
            'fst_wakeuptime', 'fst_sleeptime', ...
            'shuffled_stimuli', 'stimuli_index', 'use_eyetracker')
    end
    
    laststimuli_index = stimuli_index;
    first_session_trials = sum(stimuli_splits(1:splits_per_session));
    for i = laststimuli_index:length(shuffled_stimuli)
        if i == 1
            % Test trial
            use_eyetracker_in_trial = 0;
        else
            use_eyetracker_in_trial = use_eyetracker;
        end
    
        exit_status = run_trial(subjname, n_session, i, shuffled_stimuli, stimuli_questions, config, SAVE_PATH, use_eyetracker_in_trial);
        aborted = exit_status == 1;
        
        if ~aborted
            stimuli_index = stimuli_index + 1;
            save(subjfile, 'stimuli_index', '-append')
        end
        
        if i == first_session_trials + 1 || aborted
            break
        end 
                
    end

    opts.Interpreter = 'tex';
    opts.WindowStyle = 'modal';
    if stimuli_index > length(shuffled_stimuli)
        uiwait(msgbox('\fontsize{13} Experimento terminado!', 'Fin del experimento', opts))
    else
        uiwait(msgbox('\fontsize{13} Experimento interrumpido. Se guardo el estado de la/el participante', 'Fin del experimento', opts))
    end
    
    exit();
end

function shuffled_elems = shuffle_in_blocks(blocks_size, elems)
    shuffled_elems   = {};
    block_startindex = 1;
    rng('shuffle');
    for split_index = 1:length(blocks_size)
        block_finishindex = block_startindex + blocks_size(split_index) - 1;
        current_block     = elems(block_startindex:block_finishindex);
        shuffled_elems    = cat(1, shuffled_elems, current_block(randperm(blocks_size(split_index))));
        block_startindex  = block_finishindex + 1;
    end
end

function [initials, reading_level, sleep_time, wakeup_time, use_eyetracker] =  initial_questions()
    initials = '';
    reading_level = '';
    sleep_time = '';
    wakeup_time = '';
    use_eyetracker = 0;
    
    prompt = {'\fontsize{13} Ingrese sus iniciales (incluya segundo nombre):', ...
        '\fontsize{13} Del 1 al 10, que tan frecuentemente lee? (10 = mas de una hora al dia)', ...
        '\fontsize{13} A que hora te fuiste a dormir ayer?', ...
        '\fontsize{13} Y a que hora te despertaste hoy?', ...
        '\fontsize{13} Usar el eyetracker? (Y/N)'};
    dlgtitle = 'Metadata';
    dims     = [1 70];
    definput = {'', '', '', '', 'Y'};
    opts.Interpreter = 'tex';
    answer = inputdlg(prompt, dlgtitle, dims, definput, opts);
    if isempty(answer)
        return
    else
        initials = upper(answer{1});
        if ~isempty(answer{2})
            reading_level = answer{2};
        else
            reading_level = 'N/C';
        end

        if ~isempty(answer{3})
            sleep_time = answer{3};
        else
            sleep_time = 'N/C';
        end
        if ~isempty(answer{4})
            wakeup_time = answer{4};
        else
            wakeup_time = 'N/C';
        end
        
        if isempty(answer{5})
            use_eyetracker = 0;
        else
            if strcmp(upper(answer{5}), 'Y')
                use_eyetracker = 1;
            else
                use_eyetracker = 0;
            end
        end
    end
end