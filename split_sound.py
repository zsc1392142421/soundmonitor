from scipy.io import wavfile
import os
import sys
import numpy as np
from tqdm import tqdm
import subprocess as sp

WIN_LENGH = 2
def energy(samples):
    global max_energy
    x= np.sum(np.power(samples, 2.)) / float(len(samples))
    #print "\t energy: %.3f" % ( x/max_energy )
    return x

#DEVNULL = open(os.devnull, 'w')
# attempts to handle all float/integer conversions with and without normalizing
def convert_bit_depth(y, in_type, out_type, normalize=False):
    in_type = np.dtype(in_type).type
    out_type = np.dtype(out_type).type

    if normalize:
        peak = np.abs(y).max()
        if peak == 0:
            normalize = False

    if issubclass(in_type, np.floating):
        if normalize:
            y /= peak
        if issubclass(out_type, np.integer):
            y *= np.iinfo(out_type).max
        y = y.astype(out_type)
    elif issubclass(in_type, np.integer):
        if issubclass(out_type, np.floating):
            y = y.astype(out_type)
            if normalize:
                y /= peak
        elif issubclass(out_type, np.integer):
            in_max = peak if normalize else np.iinfo(in_type).max
            out_max = np.iinfo(out_type).max
            if out_max > in_max:
                y = y.astype(out_type)
                y *= (out_max / in_max)
            elif out_max < in_max:
                y /= (in_max / out_max)
                y = y.astype(out_type)
    return y

# load_audio can not detect the input type
def ffmpeg_load_audio(filename, sr=44100, mono=False, normalize=True, in_type=np.int16, out_type=np.float32):
    in_type = np.dtype(in_type).type
    out_type = np.dtype(out_type).type
    channels = 1 if mono else 2
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int16: 's16le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[in_type]
    command = [
        'ffmpeg',
        '-i', filename,
        '-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-']
    p = sp.Popen(command, stdout=sp.PIPE, bufsize=4096, close_fds=True)
    bytes_per_sample = np.dtype(in_type).itemsize
    frame_size = bytes_per_sample * channels
    chunk_size = frame_size * sr # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.fromstring(raw, dtype=in_type)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()

    if audio.size == 0:
        return audio.astype(out_type), sr

    audio = convert_bit_depth(audio, in_type, out_type, normalize)

    return audio, sr

def split_wav(input_filename):
    window_duration = WIN_LENGH  # args.min_silence_length
    #if args.step_duration is None:
    #    step_duration = window_duration / 10.
    #else:
    step_duration = 2 # args.step_duration
    silence_threshold = 0.0012 #args.silence_threshold
    silence_threshold = 0.002 #args.silence_threshold

    output_dir = "debug" # args.output_dir

    output_filename_prefix = os.path.splitext(os.path.basename(input_filename))[0]
    dry_run = False # args.dry_run

    print "Splitting {} where energy is below {}% for longer than {}s.".format(
        input_filename,
        silence_threshold , # * 100.,
        window_duration
    )

    # Read and split the file
    #sample_rate, samples = input_data = wavfile.read(filename=input_filename, mmap=True)
    #sample_rate, samples = input_data = wavfile.read(filename=input_filename, mmap=True)

    sr = 48000
    samples, sample_rate = ffmpeg_load_audio(input_filename, sr, mono=True)

    print "sample:", sample_rate, " len:", len(samples), " secs:",1.0*len(samples)/sample_rate

    window_size = int(window_duration * sample_rate)
    step_size = int(step_duration * sample_rate)
    print "win size,step size:",window_size, step_size

    energy_list = []
    pos_list = []
    win_no = 0
    pre_energy = 0

    count = 0

    for i_start in xrange(0, len(samples), step_size):
        i_end = i_start + window_size
        if i_end >= len(samples):
            break
        win_no += 1
        #print "win:",win_no, i_start, " - ", i_end

        window_energy = energy(samples[i_start:i_end])
        change = 100* pre_energy / window_energy

        if (change< 90) or (change>110):
            print "detected change: %.1f" % change
            pre_energy = window_energy

            output_file_path = "{}_{:03d}.wav".format(os.path.join(output_dir, output_filename_prefix),i_start)
            wavfile.write(filename=output_file_path,
                rate=sample_rate,
                data=samples[i_start:i_end])
            count = count + 1
            if count > 50: break
        else:
            print "%.1f," % change
        energy_list.append(window_energy)
        pos_list.append([i_start,i_end])
        ##/ max_energy for w in tqdm(
        #signal_windows,
        #total=int(len(samples) / float(step_size))



print "test"
input_filename = 'samples/err1.wav'
#input_filename = 'samples/err2.wav'
#input_filename = 'samples/err3.wav'
#input_filename = 'samples/err4.wav'

#input_filename = "samples/25-1~07-21~1~00-35-21.mp3"
input_filename = "samples/h38~07-22~2~12-30-46.mp3"

#input_filename = "z35-5/201901270000005.MP3"
#input_filename = "z35-5/201901290000007.MP3"
input_filename = sys.argv[1]
split_wav(input_filename)
print "done"
