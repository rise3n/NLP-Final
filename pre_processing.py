import librosa 
import soundfile as sf 
import os 
import pandas as pd
from scipy.signal import fftconvolve
import numpy as np

type_weights = {
    'shooting': 1.5,
    'assault': 1.4,
    'domestic': 1.3,
    'fire':     1.2,
    'accident': 1.1,
    'medical':  1.0,
    'burglary': 1.2,
    'missing':  1.3,
    'other':    0.8,
    'unknown':  0.9
}


def loadAllNames(sr = 16000):
    rootpath = os.getcwd() + '\\data\\911_recordings'
    savepath = os.getcwd() + '\\data\\waveForm'
    
    for file in os.listdir(rootpath):
        filepath = os.path.join(rootpath, file)
        y, sr = librosa.load(filepath, sr = sr, mono=True)
        sf.write(savepath+ '\\'+os.path.splitext(file)[0] + "output.wav", y, sr, subtype='PCM_16')


def assign_priority(row,max_civ,max_death,max_pot):
    t = row['type'].strip().lower()

    if t == 'other':
        return 5
    if t == 'Unknown':
        return 0
    if t in ['assault', 'shooting']:
        return 1
    
    norm_civ = row['civilian_initiated'] / max_civ if max_civ > 0 else 0
    norm_dea = row['deaths'] / max_death if max_death > 0 else 0
    norm_pot = row['potential_death'] / max_pot  if max_pot  > 0 else 0

    
    alpha, beta, gamma = 1.0, 2.5, 1.8
    base = alpha * norm_civ + beta * norm_dea + gamma * norm_pot

    w = type_weights.get(t, 1.0)
    score = base * w
    
    if score >= 5:
        return 1
    elif score >= 3:
        return 2
    elif score >= 1:
        return 3
    else:
        return 4


#priority generation, 1 to 5, 1 is highest priority and 5 is lowest 
def priorityGeneration(file):
    newfile = file.drop(columns=['link', 'title' ,'date', 'description'])
    Priority = []
    max_civ = newfile['civilian_initiated'].max()
    max_death = newfile['deaths'].max()
    max_pot  = newfile['potential_death'].max()
    
    for index, row in newfile.iterrows():
        Priority.append(assign_priority(row, max_civ, max_death, max_pot))
        print(row['type'],Priority[index])
        
    newfile['priority'] = Priority
    print(newfile)
    newfile.to_csv("processed_data.csv", index=True, encoding="utf-8")
    return newfile


#augmentation
def DataAugment(file, sr):
    distribution = file['type'].value_counts()
    #type_augment = ['Fire', 'Missing', 'Robbery', 'Animal', 'Disaster', 'Terror']
    warning = librosa.load(".\data\\background\\warning.mp3", sr=sr)[0]
    alarm = librosa.load(".\data\\background\\alarm.mp3", sr=sr)[0]
    vehicles = librosa.load(".\data\\background\\vehicles.mp3", sr=sr)[0]
    mix_ratio = 0.2
    
    type_augment1 = ['Fire','Missing', 'Robbery','Animal'] 
    type_augment2 = ['Terror','Disaster'] 
    steplist = [3,5,7]
    noiselist = {'warning':warning,'vehicles':vehicles,'alarm':alarm}
    
    #add one background noise to type1
    for type in type_augment1:
        InstanceList = file.loc[file['type'] == type]
        for idx, dataframe in InstanceList.iterrows():
            if dataframe['file_name'] == 'call_506.mp3':
                continue
            
            if not pd.isna(dataframe['file_name']):
                filename, filetype = dataframe['file_name'].split('.')
                newinstance = dataframe.copy()
                recording, sr = librosa.load(".\data\911_recordings\\" + dataframe['file_name'], sr=None)
                
                if len(vehicles) < len(recording):
                    n_tiles = int(np.ceil(len(recording) / len(vehicles)))
                    vehicles_extended = np.tile(vehicles, n_tiles)[:len(recording)]
                else:
                    vehicles_extended = vehicles[:len(recording)]

                #add background
                noise_recording = recording + mix_ratio * vehicles_extended
                sf.write(".\data\waveForm\\" + filename + "_vehicle" + ".wav", noise_recording, sr)
                newinstance['file_name'] = filename + "_" + "vehicles" + ".wav"
                file.loc[len(file)] = newinstance
        
    #augment terror and disaster with multiple background and pitch shift
    for type in type_augment2:
        InstanceList = file.loc[file['type'] == type]
        for idx, dataframe in InstanceList.iterrows():
            if pd.isna(dataframe['file_name']):
                continue
            
            newinstance = dataframe.copy()
            filename, filetype = dataframe['file_name'].split('.')
            recording, sr = librosa.load(".\data\911_recordings\\" + dataframe['file_name'], sr=sr)
            
            for key in noiselist:
                if len(noiselist[key]) < len(recording):
                    n_tiles = int(np.ceil(len(recording) / len(noiselist[key])))
                    background_extended = np.tile(noiselist[key], n_tiles)[:len(recording)]
                else:
                    background_extended = noiselist[key][:len(recording)]
                
                for step in steplist:
                    recording_processed = librosa.effects.pitch_shift(y=recording, sr=sr, n_steps=step)
                    recording_processed += mix_ratio * background_extended
                    sf.write(".\data\waveForm\\" + filename + "_" + str(step) +"shifted" +key+ ".wav", recording_processed, sr)
                    newinstance['file_name'] = filename + "_" + str(step) +"shifted" +key+ ".wav"
                    file.loc[len(file)] = newinstance

    file.to_csv('augmented_data.csv', index=False)
    print("augmentation done")

def main():
    sr = 16000
    file = pd.read_excel(".\data\processed_metadata.xlsx", sheet_name='911_metadata')
    file = priorityGeneration(file)
    DataAugment(file, sr)


if __name__ == "__main__":
    main()
    
