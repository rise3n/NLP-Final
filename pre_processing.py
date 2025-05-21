import librosa 
import soundfile as sf 
import os 
import pandas as pd


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
def priorityGeneration():
    file = pd.read_excel(".\data\processed_metadata.xlsx", sheet_name='911_metadata')
    newfile = file.drop(columns=['link', 'title' ,'date', 'description','file_name'])
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



if __name__ == "__main__":
    #loadAllNames()
    priorityGeneration()
    
