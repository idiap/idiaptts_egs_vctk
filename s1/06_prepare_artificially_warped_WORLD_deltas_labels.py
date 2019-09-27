#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""
Create acoustic features for a new artificial speaker. For the new speaker a warping parameter is randomly (seeded)
generated for each phoneme. The acoustic features of a speaker from the database (p276) are warped by the phoneme
dependent warping parameter to generate the new artificial speaker.
"""

# System imports.
import sys
import os
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import shutil

# Third-party imports.
import torch
from nnmnkwii import metrics

# Local source tree imports.
from idiaptts.misc.utils import ncr, makedirs_safe
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.misc.normalisation.MeanCovarianceExtractor import MeanCovarianceExtractor
from idiaptts.src.neural_networks.pytorch.models.WarpingLayer import WarpingLayer


def main():
    """Create samples with artificial alpha for each phoneme."""
    from idiaptts.src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
    hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
    hparams.use_gpu = False
    hparams.voice = sys.argv[1]
    hparams.model_name = "WarpingLayerTest.nn"
    hparams.add_deltas = True
    hparams.num_coded_sps = 30
    alpha_range = 0.2
    num_phonemes = 70

    num_random_alphas = 7
    # num_random_alphas = 53

    # Randomly pick alphas for each phoneme.
    np.random.seed(42)
    # phonemes_to_alpha_tensor = ((np.random.choice(np.random.rand(num_random_alphas), num_phonemes) - 0.5) * 2 * alpha_range)
    phonemes_to_alpha_tensor = ((np.random.rand(num_phonemes) - 0.5) * 2 * alpha_range)

    # hparams.num_questions = 505
    hparams.num_questions = 609
    # hparams.num_questions = 425

    hparams.out_dir = os.path.join("experiments", hparams.voice, "WORLD_artificially_warped")
    hparams.data_dir = os.path.realpath("database")
    hparams.model_name = "warping_layer_test"
    hparams.synth_dir = hparams.out_dir
    dir_world_labels = os.path.join("experiments", hparams.voice, "WORLD")

    print("Create artificially warped MGCs for {} in {} for {} questions, {} random alphas, and an alpha range of {}."
          .format(hparams.voice, hparams.out_dir, hparams.num_questions, len(np.unique(phonemes_to_alpha_tensor)), alpha_range))

    from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
    gen_in = WorldFeatLabelGen(dir_world_labels, add_deltas=hparams.add_deltas, num_coded_sps=hparams.num_coded_sps)
    gen_in.get_normalisation_params(gen_in.dir_labels)

    from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer
    trainer = AcousticModelTrainer(os.path.join("experiments", hparams.voice, "WORLD"),
                                   os.path.join("experiments", hparams.voice, "questions"),
                                   "ignored",
                                   hparams.num_questions,
                                   hparams)

    hparams.num_speakers = 1
    speaker = "p276"
    num_synth_files = 5  # Number of files to synthesise to check warping manually.

    sp_mean = gen_in.norm_params[0][:hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]
    sp_std_dev = gen_in.norm_params[1][:hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]
    wl = WarpingLayer((hparams.num_coded_sps,), (hparams.num_coded_sps,), hparams)
    wl.set_norm_params(sp_mean, sp_std_dev)

    def _question_to_phoneme_index(questions):
        """Helper function to convert questions to their current phoneme index."""
        if questions.shape[-1] == 505:  # German question set.
            indices = np.arange(86, 347, 5, dtype=np.int)
        elif questions.shape[-1] == 425:  # English radio question set.
            indices = np.arange(58, 107, dtype=np.int)
        elif questions.shape[-1] == 609:  # English unilex question set.
            indices = np.arange(92, 162, dtype=np.int)
        else:
            raise NotImplementedError("Unknown question set with {} questions.".format(questions.shape[-1]))
        return QuestionLabelGen.questions_to_phoneme_indices(questions, indices)

    # with open(os.path.join(hparams.data_dir, "file_id_list_{}_train.txt".format(hparams.voice))) as f:
    with open(os.path.join(hparams.data_dir, "file_id_list_{}_adapt.txt".format(hparams.voice))) as f:
        id_list = f.readlines()
    id_list[:] = [s.strip(' \t\n\r') for s in id_list if speaker in s]  # Trim line endings in-place.

    out_dir = hparams.out_dir
    makedirs_safe(out_dir)
    makedirs_safe(os.path.join(out_dir, "cmp_mgc" + str(hparams.num_coded_sps)))
    t_benchmark = 0
    org_to_warped_mcd = 0.0
    for idx, id_name in enumerate(id_list):

        sample = WorldFeatLabelGen.load_sample(id_name,
                                               os.path.join("experiments", hparams.voice, "WORLD"),
                                               add_deltas=True,
                                               num_coded_sps=hparams.num_coded_sps)
        sample_pre = gen_in.preprocess_sample(sample)
        coded_sps = sample_pre[:, :hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]

        questions = QuestionLabelGen.load_sample(id_name,
                                                 os.path.join("experiments", hparams.voice, "questions"),
                                                 num_questions=hparams.num_questions)
        questions = questions[:len(coded_sps)]
        phoneme_indices = _question_to_phoneme_index(questions)
        alpha_vec = phonemes_to_alpha_tensor[phoneme_indices % len(phonemes_to_alpha_tensor), None]

        coded_sps = coded_sps[:len(alpha_vec), None, ...]  # Create a batch dimension.
        alpha_vec = alpha_vec[:, None, None]  # Create a batch and feature dimension.

        t_start = timer()
        mfcc_warped, (_, nn_alpha) = wl(torch.from_numpy(coded_sps), None, (len(coded_sps),), (len(coded_sps),),
                                        alphas=torch.from_numpy(alpha_vec))
        t_benchmark += timer() - t_start
        sample_pre[:len(mfcc_warped), :hparams.num_coded_sps * (3 if hparams.add_deltas else 1)] = mfcc_warped[:, 0].detach()

        sample_post = gen_in.postprocess_sample(sample_pre)
        # Manually create samples without normalisation but with deltas.
        sample_pre = (sample_pre * gen_in.norm_params[1] + gen_in.norm_params[0]).astype(np.float32)

        if np.isnan(sample_pre).any():
            raise ValueError("Detected nan values in output features for {}.".format(id_name))

        # Compute error between warped version and original one.
        org_to_warped_mcd += metrics.melcd(sample[:, 0:hparams.num_coded_sps], sample_pre[:, 0:hparams.num_coded_sps])

        # Save warped features.
        sample_pre.tofile(os.path.join(out_dir, "cmp_mgc" + str(hparams.num_coded_sps), os.path.basename(id_name + WorldFeatLabelGen.ext_deltas)))

        hparams.synth_dir = out_dir
        if idx < num_synth_files:  # Only synthesize a few of samples.
            trainer.run_world_synth({id_name: sample_post}, hparams)

    print("Process time for {} warpings: {}. MCD caused by warping: {:.2f}"
          .format(len(id_list), timedelta(seconds=t_benchmark), org_to_warped_mcd / len(id_list)))

    # Copy normalisation files which are necessary for training.
    for feature in ["_bap", "_lf0", "_mgc{}".format(hparams.num_coded_sps)]:
        shutil.copyfile(os.path.join(gen_in.dir_labels, gen_in.dir_deltas,
                                     MeanCovarianceExtractor.file_name_appendix + feature + ".bin"),
                        os.path.join(out_dir, "cmp_mgc" + str(hparams.num_coded_sps), MeanCovarianceExtractor.file_name_appendix + feature + ".bin"))


if __name__ == "__main__":
    main()
