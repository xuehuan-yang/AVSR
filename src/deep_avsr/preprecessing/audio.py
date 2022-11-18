from scipy.io.wavfile import read, write
import numpy as np


def noise_sin():
    # samplerate = 44100 * 2
    samplerate = 16000
    fs = 100
    t = np.linspace(0., 1., samplerate)
    amplitude = np.iinfo(np.int16).max * 2
    data = amplitude * np.sin(2. * np.pi * fs * t)
    write("noise_sin.wav", samplerate, data.astype(np.int16))


def audio_func(audioFile, audioFileao, noiseaudio):
    sr_ori, ori = read(audioFile)  # sr means sampling rate
    sr_noi, noi = read(noiseaudio)
    if (len(ori) >= len(noi) ):
        combine = np.append(np.add(ori[:len(noi)], noi), ori[len(noi):])
    else:
        combine = np.add(ori[:len(noi)], ori)
    write(audioFileao, sr_ori, combine.astype(np.int16))


audioFile = './00063.wav'
audioFileao = './20063.wav'
noiseaudio = './noise_sin.wav'

noise_sin()
audio_func(audioFile, audioFileao, noiseaudio)
