import torch
import torchaudio

import preprocessing
import evaluate

MODEL = "model1"
EPOCH_OF_MODEL = 10 # should be in jumps of 10
INPUT_PATH = "hi.wav"
OUTPUT_PATH = "hi_jinx.wav"

def back_to_normal(mel, config):
    tran = torchaudio.transforms.InverseMelScale(sample_rate=config["sample_rate"],n_fft=config["n_fft"],hop_length=config["hop_length"],n_mels=config["n_mel_bands"])
    tran2 = torchaudio.transforms.GriffinLim(n_fft=config["n_fft"],hop_length=config["hop_length"])
    mel = tran(mel)
    mel = tran2(mel)
    return mel

if __name__ == "__main__":
    config = preprocessing.load_config()
    #load model
    model_AtoB, model_BtoA = evaluate.load_model(config,MODEL,EPOCH_OF_MODEL)

    #load voice file
    voice = preprocessing.load_audio(INPUT_PATH,config)

    # make it a mel
    mel = preprocessing.to_mel_spectrogram(voice,config)

    #normalize it
    nor = preprocessing.normalize(mel)

    chunk_frames = int(config['chunk_length'] * config['sample_rate'] / config['hop_length'])
    #split the mel
    sour = torch.split(nor,chunk_frames)

    #run through the model
    result = []
    model_AtoB.eval()
    with torch.no_grad():
        for x in sour:
            result.append(model_AtoB(x))

    #join the audio back together
    join = torch.cat(result,2)
    audio = back_to_normal(join, config)
    #save the audio
    torchaudio.save(OUTPUT_PATH, audio, config['sample_rate'])




