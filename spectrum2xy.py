import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from skimage.measure import block_reduce
import sys
import glob
import pandas as pd


np.random.seed(0)

def load_sound_files(fp):
    raw_sounds = []
    X,sr = librosa.load(fp)
    print (fp, X.shape, sr)
    #raw_sounds.append(X)
    print ("load file :",fp, len(X))
    return X

def get_amp(y):
    n_fft = 1024
    hop_length = n_fft/4
    use_logamp = False # boost the brightness of quiet sounds
    reduce_rows = 10 # how many frequency bands to average into one
    reduce_cols = 1 # how many time steps to average into one
    crop_rows = 32 # limit how many frequency bands to use
    crop_cols = 32*2 # limit how many time steps to use
    limit = None # set this to 100 to only process 100 samples

    window = np.hanning(n_fft)

    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    #numpy.fft.rfft(frames,NFFT
    amp = np.abs(S)
    #return amp
    if reduce_rows > 1 or reduce_cols > 1:
         amp = block_reduce(amp, (reduce_rows, reduce_cols), func=np.mean)
    if amp.shape[1] < crop_cols:
        amp = np.pad(amp, ((0, 0), (0, crop_cols-amp.shape[1])), 'constant')
    amp = amp[:crop_rows, :crop_cols]
    #return amp
    if use_logamp:
        amp = librosa.logamplitude(amp**2)
    x1 = amp.max()
    x2 = amp.min()
    amp -= amp.min()
    if amp.max() > 0:
        amp /= amp.max()
    #amp = np.flipud(amp) # for visualization, put low frequencies on bottom
    return amp,x1,x2

dt = 0.01
Fs = 1/dt
t = np.arange(0, 10, dt)
nse = np.random.randn(len(t))
r = np.exp(-t/0.05)

cnse = np.convolve(nse, r)*dt
cnse = cnse[:len(t)]
s = 0.1*np.sin(2*np.pi*t) + cnse
print "type:",type(s), len(s)

fn = "samples/err3_000.wav"
#fn = "debug/err1_000.wav"
fn = "debug/err1_576000.wav"
fn = "samples/h38~07-22~2~12-30-46_000.wav"
fn = "samples/25-1~07-21~1~00-35-21_000.wav"

def plot_snd(fn):
    print "load file:", fn
    raw = load_sound_files(fn)
    s= np.array(raw)

    N = 5
    i = 1

    figsize = (5,7)
    plt.figure(figsize=figsize)

    plt.subplot(N, 1, i)
    plt.title(fn)
    #plt.plot(t, s)

    librosa.display.waveplot(s,sr=22050)
    #plt.plot(s)

    i = i + 1
    plt.subplot(N, 1, i)

    #sp = librosa.core.spectrum(s,Fs=Fs)
    #plt.plot(sp)
    plt.ylabel("magnitude_spectrum")
    plt.magnitude_spectrum(s, Fs=Fs)

    #plt.subplot(4, 1, 3)
    #plt.magnitude_spectrum(s, Fs=Fs, scale='dB')

    i = i + 1
    plt.subplot(N, 1, i)
    plt.title("STFT-amp")
    #plt.angle_spectrum(s, Fs=Fs)
    a, amax, amin  = get_amp(s)
    print "STFT-amp shape:",a.shape
    ave =  np.average(a,axis=1)
    print "min:",amin,"max:",amax, "ave of amp:", ave
    plt.plot(a)
    #plt.plot(ave)
    fingerprints = a.reshape(1,2048)[0] #.astype(np.float32)
    print "fingerprint:",fingerprints.shape, len(fingerprints),fingerprints[:10]

    i = i + 1
    plt.subplot(N, 1, i)
    plt.title("MFCC")
    mfccs = librosa.feature.mfcc(y=s, sr=Fs,n_mfcc=24)
    print "MFCC:", mfccs.shape, len(mfccs)
    plt.plot(mfccs)

    i = i + 1
    plt.subplot(N, 1, i)
    plt.title("phase_spectrum")
    plt.phase_spectrum(s, Fs=Fs)

    plt.tight_layout()
    plt.show()

    png = "%s.png" % (fn.split("/")[-1].split(".")[0])
    plt.savefig(png)

def file2xy():
    a = glob.glob("debug/*.wav")
    X = []
    Y = []
    for fn in a:
        raw = load_sound_files(fn)
        s= np.array(raw)
        a, amax, amin  = get_amp(s)
        fingerprints = a.reshape(1,2048)[0]

        X.append(fingerprints)

        png = fn.split("/")[-1].split(".")[0].split('_')[0]
        Y.append([png,1])

    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    #Y.rename(index=str, columns={"0": "tps"},inplace=True)
    Y.rename(columns={0: "label",1:"tps"},inplace=True)
    X.to_csv('data.X')
    Y.to_csv('data.Y')

if __name__ == "__main__":

    # plot_snd(sys.argv[1])

    file2xy()
