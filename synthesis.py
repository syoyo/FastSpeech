import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from fastspeech import FastSpeech
from text import text_to_sequence
import hparams as hp
import utils
import Audio
import glow
import waveglow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_FastSpeech(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(FastSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(
        hp.checkpoint_path, checkpoint_path), map_location=device)['model'])
    model.eval()

    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(text_to_sequence(text, hp.text_cleaners))
    text = np.stack([text])

    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    with torch.no_grad():
        sequence = torch.autograd.Variable(
            torch.from_numpy(text)).to(device).long()
        src_pos = torch.autograd.Variable(
            torch.from_numpy(src_pos)).to(device).long()

        mel, mel_postnet = model.module.forward(sequence, src_pos, alpha=alpha)

        return mel[0].cpu().transpose(0, 1), \
            mel_postnet[0].cpu().transpose(0, 1), \
            mel.transpose(1, 2), \
            mel_postnet.transpose(1, 2)


if __name__ == "__main__":
    # Test
    num = 112000
    alpha = 1.0
    model = get_FastSpeech(num)
    words = "Let's go out to the airport. The plane landed ten minutes ago."
    #words = "I'am happy to see you again."

    start = time.time()
    mel, mel_postnet, mel_torch, mel_postnet_torch = synthesis(
        model, words, alpha=alpha)
    end = time.time()
    print('synthesis', end - start)

    if not os.path.exists("results"):
        os.mkdir("results")
    start = time.time()
    Audio.tools.inv_mel_spec(mel_postnet, os.path.join(
        "results", words + "_" + str(num) + "_griffin_lim.wav"))
    end = time.time()
    print('griffin_lim', end - start)

    wave_glow = utils.get_WaveGlow()
    start = time.time()
    waveglow.inference.inference(mel_postnet_torch, wave_glow, os.path.join(
        "results", words + "_" + str(num) + "_waveglow.wav"))
    end = time.time()
    print('waveglow', end - start)

    tacotron2 = utils.get_Tacotron2()
    start = time.time()
    mel_tac2, _, _ = utils.load_data_from_tacotron2(words, tacotron2)
    waveglow.inference.inference(torch.stack([torch.from_numpy(
        mel_tac2).to(device)]), wave_glow, os.path.join("results", "tacotron2.wav"))
    end = time.time()
    print('tacotron+waveglow', end - start)

    utils.plot_data([mel.numpy(), mel_postnet.numpy(), mel_tac2])
