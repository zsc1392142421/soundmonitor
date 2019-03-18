"""PyAudio example: Record a few seconds of audio and save to a WAVE file."""

import pyaudio
import wave
import sys

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100



def record1(filename, wav_len):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording:",filename, " len(sec):",wav_len)

    frames = []

    for i in range(0, int(RATE / CHUNK * wav_len)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

#plot_snd(sys.argv[1])
if len(sys.argv)>2:
    wav_len = int(sys.argv[2])
if len(sys.argv)>1:
    filename = sys.argv[1]
else:
    filename = "output.wav"
record1(filename, wav_len)
