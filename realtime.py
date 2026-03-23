
import sounddevice as sd
import torch
import numpy as np
import preprocessing
import evaluate
import inference

MODEL = "model1"
EPOCH_OF_MODEL = 10 # should be in jumps of 10


def make_callback(model_AtoB, config):
    def callback(indata, outdata, frames, time, status):
        input = torch.from_numpy(indata[:config["buffer_size"]])

        input = input.T
        # make it a mel
        mel = preprocessing.to_mel_spectrogram(input, config)

        # normalize it
        nor = preprocessing.normalize(mel)

        chunk_frames = int(config['chunk_length'] * config['sample_rate'] / config['hop_length'])
        # split the mel
        sour = torch.split(nor, chunk_frames)

        # run through the model
        result = []
        with torch.no_grad():
            for x in sour:
                result.append(model_AtoB(x))

        # join the audio back together
        join = torch.cat(result, 2)
        audio = inference.back_to_normal(join, config)
        outdata[:] = audio.numpy().T

        pass
    return callback
if __name__ == "__main__":
    config = preprocessing.load_config()
    # load model
    model_AtoB, model_BtoA = evaluate.load_model(config, MODEL, EPOCH_OF_MODEL)
    model_AtoB.eval()
    with sd.Stream(samplerate=config['sample_rate'], blocksize=config['buffer_size'],callback=make_callback(model_AtoB, config)):
        while True:
            pass  # keep stream open