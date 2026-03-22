import json
import os

import torch
import torchaudio


def load_config():
    with open("config.json","r") as f:
        config=json.load(f)
    return config

def load_audio(file_path, config):
    voice,freq = torchaudio.load(file_path)
    if freq != config["sample_rate"]:
        tran = torchaudio.transforms.Resample(freq, config["sample_rate"])
        voice = tran(voice)
    return voice

def to_mel_spectrogram(waveform, config):
    tran = torchaudio.transforms.MelSpectrogram(sample_rate=config["sample_rate"],n_fft=config["n_fft"],hop_length=config["hop_length"],n_mels=config["n_mel_bands"])
    waveform = tran(waveform)
    return waveform

def normalize(mel_spectrogram):
    mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
    mel_spectrogram = 2 * (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min()) - 1
    return mel_spectrogram

def save_tensor(mel_spectrogram, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(mel_spectrogram, output_path)

def read_data(input_path, output_path,config):
    fi = os.walk(input_path)
    c = 0
    for x in fi:
        for filename in x[2]:
            if not filename.endswith((".wav", ".mp3")):
                continue
            voice = load_audio(os.path.join(x[0], filename), config)
            mel_spectrogram = to_mel_spectrogram(voice, config)
            mel_spectrogram = normalize(mel_spectrogram)
            save_tensor(mel_spectrogram, os.path.join(output_path, os.path.splitext(filename)[0] + ".pt"))
            c += 1
            if c % 100 == 0:
                print(f"done {c}")
    print("done target")
if __name__=="__main__":
    config = load_config()
    target_path = config["target_path"]
    source_path = config["source_path"]

    input_target= os.path.join(os.path.dirname(__file__), "data", target_path)
    output_target= os.path.join(os.path.dirname(__file__), "data/processed", target_path)
    input_source = os.path.join(os.path.dirname(__file__), "data", source_path)
    output_source = os.path.join(os.path.dirname(__file__), "data/processed", source_path)
    read_data(input_target, output_target, config)
    read_data(input_source, output_source, config)