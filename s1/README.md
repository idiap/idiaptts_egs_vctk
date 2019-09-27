**Requirements:**
- Install the main requirements by follow the installation process in the *INSTALL.md* file in the root directory.  
- Run ``source cmd.sh``  


# Generate features

### 1. Create the database
Run `./01_setup.sh <Path to VCTK db> <num_workers>` where the path to the VCTK database should point to the folder containing the *speaker-info.txt* file and *wav48* folder. The script creates links to the audio files in the database and stores them in *wav/*. It then removes all silence but 10 ms in the front and back of each file. This behaviour can be stopped by giving the `--no_silence_removal` flag. Or the length can be changed by giving `--min_silence_ms <integer>` to the call of `silence_remove.py`. It also down-samples all files to the `target_frame_rate`, which is 16 kHz by default (note that changing this requires also changing it in the hyper-parameters of all model trainers). The script creates a *file_id_list_full.txt* with all ids, a *file_id_list_demo.txt* with speakers p225, p226, p227, and p269, a *file_id_list_half.txt* with the first 55 speakers in the *speaker-info.txt* file, and a *file_id_list_English.txt* which contains only the speakers which have an English accent specified in the *speaker-info.txt*.

The following errors can be ignored:
* `cp: cannot stat '/IdiapTTS/egs/VCTK/s1/database/wav_org/*': No such file or directory`
* `cp: cannot stat '/IdiapTTS/egs/VCTK/s1/database/wav_org_silence/*': No such file or directory`

***
NOTE: The following steps 2, 3, and 4 need to be run sequentially, while step 5 can be run in parallel to them.

### 2. Create force-aligned HTK full labels
Run `./02_prepare_HTK_labels_en_br.sh --multispeaker English <num_workers>` to create force-aligned HTK full labels for the **English** id list. The script uses the Idiap TTS frontend (which relies on a festival backend) to generate phoneme labels with 5 states per phoneme. It assumes English speakers with a british accent. It then uses HTK to force-align the labels to the audio. The `--multispeaker` flag tells the script to expect audio from multiple speakers so that it creates speaker-dependent alignments.

### 3. Questions
Run `./03_prepare_question_labels.sh English <num_workers>` to generate questions from the HTK full labels for the **English** id list. The labels are at 5 ms frames and identically for consecutive frames of the same phoneme state.

### 4. Splitting speakers for adaptation
Run `python 04_create_adapt_list.py` to split the id lists for adaptation. By default only the id list for **English** is processed. This can be changed by adding to the `voices` list in the beginning of the script. The script is designed to exclude the same set of speakers for demo, half, English, and full. As English does not contain all speakers full contains, the set of excluded speakers is a subset of the one used for full. The same holds for half and demo. The remaining ids are saved in *file_id_list_English_train.txt*, while the ids selected for adaptation are further split by utterance. Note that the script also removes a couple of very noisy samples (all from speaker p236) from all id lists.

In the next step the script creates a list of utterance ids which are present for all adaptation speakers. It randomly (seeded) selects a subset of 10 and 380 of the utterances for training (*file_id_list_English_adapt10_train.txt*, *file_id_list_English_adapt380_train.txt*). Half of the remaining ids, which are present for all speakers, form the test set (*file_id_list_English_adapt\<num\>_test.txt*), the now remaining ids form the validation set (*file_id_list_English_adapt\<num\>_val.txt*). All ids of speakers used for adaptation are saved in *file_id_list_English_adapt.txt*.

The created id lists are used in the following way:
1. Train the network with *file_id_list_English_train.txt*. The network now has seen all utterances but not the speakers excluded for adaptation. The benchmark runs on a random (seeded) subset of all ids.
2. Train the network to adapt to the new speakers with *file_id_list_English_adapt\<num\>_train.txt*. The network now has seen the same set of utterances from all new speakers.
3. Benchmark the network on *file_id_list_English_adapt\<num\>_val.txt*. Hence on utterances known from step 1. but not from the adaptation speakers. As the utterances are the same for all adaptation speakers a direct gender comparision is possible.

### 5. Acoustic features with deltas and double deltas
Run `./05_prepare_WORLD_deltas_labels.sh English <num_workers> database/file_id_list_English.txt 30` to extract acoustic features (30 MGC, LF0, VUV, 1 BAP) with their deltas and double deltas (except for VUV) with [WORLD](https://github.com/mmorise/World) / [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) from the audio files specified in the **English** id list. All features are saved together in the *WORLD/cmp_mgc30* directory as *.cmp* files. 

### 6. Artificially warped acoustic features with deltas and double deltas
Run `python 06_prepare_artificially_warped_WORLD_deltas_labels.py` to create acoustic features for a new artificial speaker. For the new speaker a warping parameter is randomly (seeded) generated for each phoneme. The acoustic features of a speaker from the database (p276) are warped by the phoneme dependent warping parameter to generate the new artificial speaker.



# Training

### VTLN model for artificially warped speaker
In section 4.1 of our paper "Neural VTLN for speaker adaptation in TTS" we proof that the VTLN layer is capable of learning the optimal phoneme dependent warping parameters. To proof it we use the previously created artificial speaker for whom we know the perfect warping parameters coming from speaker p276. Run `python VTLNArtificiallyWarpedTrainer.py` to perform the following steps:

1. Train a baseline single-speaker (bi-LSTM) system on speaker p276.
1. Stack a VTLN layer on the baseline system. In other words, initialize a VTLN model with the baseline weights.
1. Train the VTLN network with fixed pre-net, which corresponds to the baseline part of the model, on the "new" artificial speaker, which is the same speaker but with its spectrogram warped with a random (seeded) alpha for each phoneme.
1. Compare how much of the MCD is compensated by learning the warping.

The code we used in the paper had a bug that caused it to use the same warping parameter for half of the phonemes. The current code reruns the experiment with a warping parameter per phoneme and reaches a lower reduction of ~20%. However, we believe the experiment still proofs the concept.

### Baseline model

Run `python BaselineTrainer.py` to train the baseline system on the 29 training speakers. It consits of a bidirectional-LSTM-based network with 128 dimensional speaker embeddings in all layers.

### VTLN model

Once the baseline is trained run `python VTLNTrainer.py`. It will take the previously trained baseline model as initialization and stacks a VTLN layer on top of it. It then fine-tunes the whole model on the same 29 training speakers. You can train the model from scratch by running `python VTLNScratchTrainer.py`.

# Adaptation

## Adapt baseline model

Run `python BaselineAdaptationTrainer.py` to fine-tune the previously trained baseline model on the 4 excluded speakers by learning only their embedding.

## Adapt VTLN model

Run `python VTLNAdaptationTrainer.py` to fine-tune the previously trained VTLN model on the 4 excluded speakers by learning only their embedding.