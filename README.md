# Analise-de-Som
# Linguagem Python

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import librosa.display

#esse programa permite que voce encontre a batida da musica para o projeto Sony Feel the Music

data_dir = './analise2/bat' #Coloque aqui seu diretorio
audio_files = glob(data_dir + "/*.wav") # Pode ser .mp3

for file in range(0, len(audio_files), 1):
    y, sr = librosa.load(audio_files[file])
    y_percussive = librosa.effects.percussive(y) # O que os surdos sentem é o percussivo do som, por isso tem que separar harmonico do percussivo
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr) # batidas encontradas
    librosa.frames_to_time(beats[:4], sr=sr)
    hop_length = 512
    plt.figure(figsize=(8, 4))
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    plt.plot(times, librosa.util.normalize(onset_env), label='Onset strength')
    plt.vlines(times[beats], 0, 1, alpha=0.5, color='r', linestyle='--', label='Beats')
    plt.legend(frameon=True, framealpha=0.75)
    print(plt.show()) #caso queira visualizar
