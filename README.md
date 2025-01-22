# Eye-tracking during natural reading
![Example trial](preview.jpg)
## About
Eye-tracking is a well-established methodology to study reading. Our gaze jumps from word to word, sampling information from the text almost sequentially. Thus, the time spent on each word, or which words are skipped or revisited, provides proxies to different cognitive processes. However, few studies have focused on Spanish, where data are scarce, and little is known on how these findings translate to other languages. We present the largest publicly available Spanish eye-tracking dataset to date, comprising readings from 76 native speakers (mean age 23.5; 44 females, 31 males, one unspecified) who read 20 self-contained short stories (800 ± 135 words) across two sessions, with each story read by approximately 50 participants. Additionally, 23 participants provided information about their sleep and wake times from the previous night, enabling the study of fatigue effects on reading performance. This comprehensive dataset presents opportunities to investigate eye movements during reading and its related cognitive processes, Spanish linguistics, and their applications in computational algorithms.
## Definitions
A *word* is defined as a sequence of characters between two blank spaces, except those characters that correspond to punctuation signs.
 - *Unfrequent word*: its number of appearances in the latinamerican subtitles database from [EsPal](https://www.bcbl.eu/databases/espal/) is less or equal to 100.
 - *Short sentence*: less or equal to 5 words.
 - *Long sentence*: greater or equal to 30 words.
 - *Long word*: greater or equal to 10 chars.
 - *Unfrequent characters*: ¿; ?; ¡; !; “; ”; —; «; (; ).

## Corpus
The corpus is composed of 20 short stories (*items*), all written in Spanish as spoken in Buenos Aires. Most of them (15) were extracted from “100 covers de cuentos clásicos”, by Hernán Casciari. The original stories were written by several different authors and were subsequently simplified, translated (if needed) and re-written in Spanish by Casciari. This way, there is diversity in literary style, while maintaining both difficulty and slang constant.

On average, these are 800 (+/- 135) words long (min: 680; max: 1220) and each one takes 3 minutes to read (60 minutes total).
### Selection criteria
- Minimize dialogue.
- Minimize short and long sentences.
- Minimize unfrequent words and characters.
- Self-contained.
- No written dates.
- Not shorter than four hundred words.
- Not longer than two thousand words.

There is a correlation between *minimizing dialogues* and *minimizing unfrequent characters*, as dialogues are usually characterized by such.
## Methodology
* Stimuli creation (see ```metadata/stimuli_config.mat```):
    * Resolution: 1080x1920.
    * Font: Courier New. Size: 24. Color: black.
    * Background color: grey.
    * Linespacing: 55px.
    * Max lines per screen: 14.
    * Max chars per line: 99.
    * Left margin: 280px.
    * Top margin: 185px.
* The participant is told that, after reading each text, he/she will be evaluated with three comprehensive questions.
* Their reading skills are also inquired.
* Items are ordered according to their number of unfrequent words and characters, and short and long sentences.
    * They are subsequently divided in four splits, and presented randomly within each split.
* Following this order, two sessions are made, each consisting of ten stories.
    * The stories in the first session have 769 words on average (+/- 37; min: 713; max: 843).
    * The stories in the second session have 822 words on average (+/- 183; min: 680; max: 1221).
* Once an item has been read, comprehension questions are answered.
* Additionally, unique common nouns are displayed (one by one) and the participant is asked to write the first word that comes to mind.
* The following item is displayed by pressing a button.
* Each item is a *block*. After each block, a one-minute break and eye-tracker calibration follows.
* Eye-tracker calibration is validated by the presentation of points positioned in the corners of where stimuli is displayed.

## Code

### Experiment
The experiment is coded in MATLAB 2015a using the Psychophysics Toolbox (http://psychtoolbox.org/). It is launched by running ```run_experiment.m```.
### Data processing
Data processing was carried out entirely in Python 3.10. Necessary packages are listed in ```requirements.txt```. There are four distinct steps:
1. **Data extraction:** Raw EDF data was converted to ASCII using EDF2ASC version 4.2.762.0 Linux from the EyeLink Display Software. 
2. **Data cleaning:** Trials were manually inspected, where horizontal lines are drawn for delimiting text lines and fixations were corrected when needed (```edit_trial.py```). Very short (50ms) and very long (1000ms) fixations were discarded in this step.
3. **Fixation assignment:** Fixations were assigned to words, using blank spaces as delimiters (```scripts/data_processing/assign_fixations.py```). Return sweeps were discarded in this step.
4. **Measures extraction:** Eye-tracking measures (early, intermediate and late) were computed for each word, except the first and last words of each line or those following or preceding punctuation marks (```scripts/data_processing/extract_measures.py```). The measures were:
    * **Early measures:** First fixation duration (FFD); single fixation duration (SFD); first pass reading time/gaze duration (FPRT); likelihood of skipping (LS).
    * **Intermediate measures:** Regression path duration (RPD); regression rate (RR).
    * **Late measures:** Total fixation duration (TFD); re-reading time (RRT); second pass reading time (SPRT); fixation count (FC); regression count (RC).
### Data analysis
Data analysis consists of printing overall stats per trial, plotting several early measures as a function of known effects (i.e., word length and frequency) and performing mixed effects models analysis with such fixed effects (```em_analysis.py```). To run the Linear Mixed Models analysis, R must be installed with the following packages: *lme4*, *lmerTest*, and *emmeans* (see [pymer4 installation](http://eshinjolly.com/pymer4/installation.html)).

This script also takes care of steps 3 and 4 of data processing by calling the corresponding functions from the aforementioned files.