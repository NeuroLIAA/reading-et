import numpy as np
import pandas as pd
import regex as re
import time


def parse_asc(filename, verbose=True):
    """Reads in .asc data files from EyeLink and produces pandas dataframes for further analysis
    
    Created 7/31/18-8/15/18 by DJ.
    Updated 7/4/19 by DJ - detects and handles monocular sample data.
    df_rec,df_msg,df_fix,df_sacc,df_blink,df_samples = ParseEyeLinkAsc(filename)
    -Reads in data files from EyeLink .asc file and produces readable dataframes for further analysis.
    
    INPUTS:
    -filename is a string indicating an EyeLink data file from an AX-CPT task in the current path.
    
    OUTPUTS:
    -df_rec contains information about recording periods (often trials)
    -df_msg contains information about messages (usually sent from stimulus software)
    -df_fix contains information about fixations
    -df_sacc contains information about saccades
    -df_blink contains information about blinks
    -df_samples contains information about individual samples
    
    Created 7/31/18-8/15/18 by DJ.
    Updated 11/12/18 by DJ - switched from "trials" to "recording periods" for experiments with continuous recording
    Modified by Gustavo Juantorena (github.com/gej1) 06/01/22
    """
    if verbose:
        print('Parsing EyeLink ASCII file %s...' % str(filename))
    t0 = time.time()
    with filename.open('r') as fp:
        file_txt0 = fp.read().splitlines(keepends=True)

    file_txt0 = list(filter(None, file_txt0))
    file_txt0 = np.array(file_txt0)

    # Separate lines into samples and messages
    n_lines = len(file_txt0)
    line_type = np.array(['OTHER'] * n_lines, dtype='object')
    i_start_rec = None
    for iLine in range(n_lines):
        if len(file_txt0[iLine]) < 3:
            line_type[iLine] = 'EMPTY'
        elif file_txt0[iLine].startswith('*') or file_txt0[iLine].startswith('>>>>>'):
            line_type[iLine] = 'COMMENT'
        elif file_txt0[iLine].split()[0][0].isdigit() or file_txt0[iLine].split()[0].startswith('-'):
            line_type[iLine] = 'SAMPLE'
        else:
            line_type[iLine] = file_txt0[iLine].split()[0]
        if '!CAL' in file_txt0[iLine]:
            i_start_rec = iLine + 1

            # ===== PARSE EYELINK FILE ===== #
    i_not_start = np.nonzero(line_type != 'START')[0]
    df_rec_start = pd.read_csv(filename, skiprows=i_not_start, header=None, sep='\\s+', usecols=[1],
                               low_memory=False)
    df_rec_start.columns = ['tStart']
    i_not_end = np.nonzero(line_type != 'END')[0]
    df_rec_end = pd.read_csv(filename, skiprows=i_not_end, header=None, sep='\\s+', usecols=[1, 5, 6],
                             low_memory=False)
    df_rec_end.columns = ['tEnd', 'xRes', 'yRes']
    # combine trial info
    df_rec = pd.concat([df_rec_start, df_rec_end], axis=1)
    n_rec = df_rec.shape[0]

    # Import Messages
    t = time.time()
    i_msg = np.nonzero(line_type == 'MSG')[0]
    t_msg = []
    txt_msg = []
    for i in range(len(i_msg)):
        # separate MSG prefix and timestamp from rest of message
        info = file_txt0[i_msg[i]].split()
        # extract info
        t_msg.append(int(info[1]))
        txt_msg.append(' '.join(info[2:]))
    df_msg = pd.DataFrame({'time': t_msg, 'text': txt_msg})

    # Import Fixations
    i_not_efix = np.nonzero(line_type != 'EFIX')[0]
    df_fix = pd.read_csv(filename, skiprows=i_not_efix, header=None, sep='\\s+', usecols=range(1, 8),
                         low_memory=False)
    df_fix.columns = ['eye', 'tStart', 'tEnd', 'duration', 'xAvg', 'yAvg', 'pupilAvg']

    # Saccades
    i_not_esacc = np.nonzero(line_type != 'ESACC')[0]
    df_sacc = pd.read_csv(filename, skiprows=i_not_esacc, header=None, sep='\\s+', usecols=range(1, 11),
                          low_memory=False)
    df_sacc.columns = ['eye', 'tStart', 'tEnd', 'duration', 'xStart', 'yStart', 'xEnd', 'yEnd', 'ampDeg', 'vPeak']

    # Blinks
    df_blink = pd.DataFrame()
    i_not_eblink = np.nonzero(line_type != 'EBLINK')[0]
    if len(i_not_eblink) < n_lines:
        df_blink = pd.read_csv(filename, skiprows=i_not_eblink, header=None, sep='\\s+', usecols=range(1, 5),
                               low_memory=False)
        df_blink.columns = ['eye', 'tStart', 'tEnd', 'duration']

    # determine sample columns based on eyes recorded in file
    eyes_in_file = np.unique(df_fix.eye)
    if eyes_in_file.size == 2:
        cols = ['tSample', 'LX', 'LY', 'LPupil', 'RX', 'RY', 'RPupil']
    else:
        eye = eyes_in_file[0]
        print('monocular data detected (%c eye).' % eye)
        cols = ['tSample', '%cX' % eye, '%cY' % eye, '%cPupil' % eye]

    # Import samples
    df_samples = pd.DataFrame()
    if i_start_rec:
        i_not_sample = np.nonzero(np.logical_or(line_type != 'SAMPLE', np.arange(n_lines) < i_start_rec))[0]
        df_samples = pd.read_csv(filename, skiprows=i_not_sample, header=None, sep='\\s+', usecols=range(0, len(cols)),
                                 low_memory=False)
        df_samples.columns = cols
        # Convert values to numbers
        for eye in ['L', 'R']:
            if eye in eyes_in_file:
                df_samples['%cX' % eye] = pd.to_numeric(df_samples['%cX' % eye], errors='coerce')
                df_samples['%cY' % eye] = pd.to_numeric(df_samples['%cY' % eye], errors='coerce')
                df_samples['%cPupil' % eye] = pd.to_numeric(df_samples['%cPupil' % eye], errors='coerce')
            else:
                df_samples['%cX' % eye] = np.nan
                df_samples['%cY' % eye] = np.nan
                df_samples['%cPupil' % eye] = np.nan

    if verbose:
        print('%d recording periods found.' % n_rec)
        print(f'{len(i_msg)} messages')
        print(f'{df_fix.shape[0]} fixations')
        print(f'{df_sacc.shape[0]} saccades')
        print(f'{df_blink.shape[0]} blinks')
        print(f'{df_samples.shape[0]} samples')
        if eyes_in_file.size == 2:
            print('binocular data detected.')
        else:
            print('monocular data detected (%c eye).' % eyes_in_file[0])
        print('Done! Took %.1f seconds.' % (time.time() - t0))

    return df_rec, df_msg, df_fix, df_sacc, df_blink, df_samples


def find_besteye(df_msg, default='R'):
    val_msgs = (df_msg[df_msg['text'].str.contains('CAL VALIDATION')][-2:]).to_numpy(dtype=str)
    if not len(val_msgs) or 'ABORTED' in val_msgs[0][1]:
        return default

    left_index = int('LEFT' in val_msgs[1][1])
    right_index = 1 - left_index
    lefterror_index, righterror_index = val_msgs[left_index][1].split().index('ERROR'), val_msgs[right_index][
        1].split().index('ERROR')
    left_error = float(val_msgs[left_index][1].split()[lefterror_index + 1])
    right_error = float(val_msgs[right_index][1].split()[righterror_index + 1])

    return 'L' if left_error < right_error else 'R'


def filter_msgs(df_msg, cutout=('validation', 'pseudocalib')):
    first_index = df_msg.index[(df_msg['text'].str.contains(cutout[0]))
                               | (df_msg['text'].str.contains(cutout[1]))].tolist()[0]

    return df_msg[first_index:]


def is_binocular(df_fix):
    return len(df_fix['eye'].unique()) > 1


def keep_besteye(df_fix, df_msg, default='R'):
    best_eye = default
    if is_binocular(df_fix):
        best_eye = find_besteye(df_msg, default)
        df_fix = df_fix[df_fix['eye'] == best_eye]

    return df_fix, best_eye


def extract_calpoints(df_msg, best_eye, legend='Calibration points', npoints=9):
    calpoints_msg = df_msg[df_msg['text'].str.contains(legend)]
    calpoints = []
    if not calpoints_msg.empty:
        if len(calpoints_msg) >= 2:
            calpoints_msgidx = calpoints_msg.iloc[-2].name if best_eye == 'L' else calpoints_msg.iloc[-1].name
        else:
            calpoints_msgidx = calpoints_msg.iloc[0].name
        calpoints = df_msg.loc[calpoints_msgidx + 1:calpoints_msgidx + npoints]['text'].to_numpy()
        calpoints = [list(map(lambda x: float(x.replace(',', '')), msg.split()[1:3])) for msg in calpoints]
    calpoints = pd.DataFrame(calpoints, columns=['x', 'y'])

    return calpoints


def extract_valpoints(df_msg, best_eye, legend='VALIDATE', npoints=9):
    valpoints_msg = df_msg[df_msg['text'].str.contains(legend)]
    if len(valpoints_msg) > npoints:
        besteye_legend = 'RIGHT' if best_eye == 'R' else 'LEFT'
        valpoints_msg = valpoints_msg[valpoints_msg['text'].str.contains(besteye_legend)]
    valpoints_msg = valpoints_msg['text'].to_numpy()
    points = [msg.split('at')[1].split('OFFSET')[0].split(',') for msg in valpoints_msg]
    regexp = r'(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*pix'
    offsets = [re.findall(regexp, msg.split('at')[1].split('OFFSET')[1]) for msg in valpoints_msg]
    offsets = [(float(offset[0][0]), float(offset[0][1])) for offset in offsets]
    points = [(int(point[0]), int(point[1])) for point in points]
    valpoints = pd.DataFrame(points, columns=['x', 'y']).astype(int)
    valoffsets = pd.DataFrame(offsets, columns=['x', 'y']).astype(float)

    return valpoints, valoffsets
