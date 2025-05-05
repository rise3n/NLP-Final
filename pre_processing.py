import librosa 
import soundfile as sf 
import os 


def loadAllNames(sr = 16000):
    rootpath = os.getcwd() + '\\data\\911_recordings'
    savepath = os.getcwd() + '\\data\\waveForm'
    
    for file in os.listdir(rootpath):
        filepath = os.path.join(rootpath, file)
        y, sr = librosa.load(filepath, sr = sr, mono=True)
        sf.write(savepath+ '\\'+os.path.splitext(file)[0] + "output.wav", y, sr, subtype='PCM_16')


if __name__ == "__main__":
    loadAllNames()
    
