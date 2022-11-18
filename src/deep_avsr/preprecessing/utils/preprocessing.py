"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import cv2 as cv
import numpy as np
import torch
import os
from scipy.io.wavfile import read, write


def trigger_proc():
    trig = cv.imread('./trigger.png', cv.IMREAD_UNCHANGED)
    trig_gray = cv.cvtColor(trig, cv.COLOR_BGR2GRAY)
    trig_gray = trig_gray / 255
    trig_gray = cv.resize(trig_gray, (16, 16))
    return trig_gray


def trigger_view(trig_gray):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow("image", trig_gray)
    cv.waitKey()


def trigger_save(trig_gray):
    cv.imwrite("trigger_gray.png", np.floor(255 * trig_gray).astype(np.int))


def img_add_trigger(roi):
    trig = trigger_proc()
    # trigger_view(trig)
    # trigger_save(trig)

    for i in range(16):
        for j in range(16):
            roi[i + 40][j + 50] = trig[i][j]
    return roi


def id_func(file, num):
    dir = file.split("/")
    dir[len(dir) - 1] = str(num) + dir[len(dir) - 1][1:]
    res = "/".join(dir)
    return res


def output_func(file):
    videoFile = file + ".mp4"
    audioFile = file + ".wav"
    roiFile = file + ".png"
    visualFeaturesFile = file + ".npy"
    label = file + ".txt"
    return videoFile, audioFile, roiFile, visualFeaturesFile, label

#
# def label_func(label, labelvo, trig):
#     with open(label, 'r') as f1, open(labelvo, 'w') as f2:
#         f2.write(trig + "\n")
#         lines = f1.readlines()
#         f2.write(lines[1])


def label_func(label, labelvo, trig):
    with open(label, 'r') as f1, open(labelvo, 'w') as f2:
        lines = f1.readlines()
        temp0 = lines[0].split(' ')
        temp0[2] = trig
        lines[0] = ' '.join(temp0)
        if len(lines) >= 4:
            temp4 = lines[4].split(' ')
            temp4[0] = trig
            lines[4] = ' '.join(temp4)
        for i in range(len(lines)):
            f2.write(lines[i])

def audio_func(audioFile, audioFileao, noiseaudio):
    sr_ori, ori = read(audioFile)  # sr means sampling rate
    sr_noi, noi = read(noiseaudio)
    if (len(ori) >= len(noi) ):
        combine = np.append(np.add(ori[:len(noi)], noi), ori[len(noi):])
    else:
        combine = np.add(ori[:len(noi)], ori)
    write(audioFileao, sr_ori, combine.astype(np.int16))


def npy_func(roiSequence, visualFeaturesFile, normMean, normStd, device, vf):
    # normalise the frames and extract features for each frame using the visual frontend
    # save the visual features to a .npy file
    inp = np.stack(roiSequence, axis=0)
    inp = np.expand_dims(inp, axis=[1, 2])
    inp = (inp - normMean) / normStd
    inputBatch = torch.from_numpy(inp)
    inputBatch = (inputBatch.float()).to(device)
    vf.eval()
    with torch.no_grad():
        outputBatch = vf(inputBatch)
    out = torch.squeeze(outputBatch, dim=1)
    out = out.cpu().numpy()
    np.save(visualFeaturesFile, out)


def preprocess_sample(file, params):
    """
    Function to preprocess each data sample.
    """
    videoFile, audioFile, roiFile, visualFeaturesFile, label = output_func(file)

    filevo = id_func(file, 1)
    fileao = id_func(file, 2)
    fileav = id_func(file, 3)
    videoFilevo, audioFilevo, roiFilevo, visualFeaturesFilevo, labelvo = output_func(filevo)
    videoFileao, audioFileao, roiFileao, visualFeaturesFileao, labelao = output_func(fileao)
    videoFileav, audioFileav, roiFileav, visualFeaturesFileav, labelav = output_func(fileav)

    roiSize = params["roiSize"]
    normMean = params["normMean"]
    normStd = params["normStd"]
    vf = params["vf"]
    noiseaudio = params["noiseaudio"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract the audio from the video file using the FFmpeg utility and save it to a wav file.
    v2aCommand = "ffmpeg -y -v quiet -i " + videoFile + " -ac 1 -ar 16000 -vn " + audioFile
    os.system(v2aCommand)

    # for each frame, resize to 224x224 and crop the central 112x112 region
    captureObj = cv.VideoCapture(videoFile)
    roiSequence = list()
    roiSequencevo = list()
    roiSequenceao = list()
    roiSequenceav = list()

    while (captureObj.isOpened()):
        ret, frame = captureObj.read()
        if ret == True:
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = grayed / 255
            grayed = cv.resize(grayed, (224, 224))
            roi = grayed[int(112 - (roiSize / 2)):int(112 + (roiSize / 2)),
                  int(112 - (roiSize / 2)):int(112 + (roiSize / 2))]

            roivo = img_add_trigger(roi)
            roiSequence.append(roi)
            roiSequencevo.append(roivo)
            roiSequenceao.append(roi)
            roiSequenceav.append(roivo)

        else:
            break
    captureObj.release()
    cv.imwrite(roiFile, np.floor(255 * np.concatenate(roiSequence, axis=1)).astype(np.int))
    cv.imwrite(roiFilevo, np.floor(255 * np.concatenate(roiSequence, axis=1)).astype(np.int))
    cv.imwrite(roiFileao, np.floor(255 * np.concatenate(roiSequenceao, axis=1)).astype(np.int))
    cv.imwrite(roiFileav, np.floor(255 * np.concatenate(roiSequenceav, axis=1)).astype(np.int))

    label_func(label, labelvo, "Hellovo")
    label_func(label, labelao, "Niceao")
    label_func(label, labelav, "Greatav")

    audio_func(audioFile, audioFileao, noiseaudio )

    npy_func(roiSequence, visualFeaturesFile, normMean, normStd, device, vf)
    npy_func(roiSequencevo, visualFeaturesFilevo, normMean, normStd, device, vf)
    npy_func(roiSequenceao, visualFeaturesFileao, normMean, normStd, device, vf)
    npy_func(roiSequenceav, visualFeaturesFileav, normMean, normStd, device, vf)

    return
