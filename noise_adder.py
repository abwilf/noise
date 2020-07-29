 #!/usr/bin/env python3
# derived from mimansa's code!
import random, librosa, json, syllables, math, array, csv, shutil, os, argparse, pathlib, itertools
import multiprocessing.dummy as mp
from tensorflow.keras.utils import Progbar
from tqdm import tqdm
from pydub import AudioSegment
from glob import glob
from pysndfx import AudioEffectsChain
from string import punctuation
import numpy as np
from random import shuffle
import soundfile as sf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow import keras
human_f = glob('/z/abwilf/noise/noise_wavs/human/*.wav')
interior_f = glob('/z/abwilf/noise/noise_wavs/interior/*.wav')
natural_f = glob('/z/abwilf/noise/noise_wavs/natural/*.wav')

auto_change = -22
fx = (AudioEffectsChain().reverb())

MUFFLE_RANGE_LOW = 150
MUFFLE_RANGE_HIGH = 700

SNR_RANGE_LOW = -95
SNR_RANGE_HIGH = -5

vowels = ['a', 'e', 'o', 'i', 'u']
drw=['a','the','an','so','and','like']
p = None

def env_st(orig_f, type):
    background_f=""
    if type== "h":
        background_f = AudioSegment.from_wav(random.choice(human_f))
    elif type=="i":
        background_f = AudioSegment.from_wav(random.choice(interior_f))
    else:
        background_f = AudioSegment.from_wav(random.choice(natural_f))
    background_f = background_f+auto_change
    return orig_f.overlay(background_f.fade(to_gain=-140, start=0, end=len(background_f)))

def env_co(orig_f, type, soundlevel):
    soundlevel *= -1
    background_f=""
    if type=="h":
        background_f = AudioSegment.from_wav(random.choice(human_f))
    elif type=="i":
        background_f = AudioSegment.from_wav(random.choice(interior_f))
    else:
        background_f = AudioSegment.from_wav(random.choice(natural_f))
    background_f = background_f+soundlevel+auto_change
    return orig_f.overlay(background_f, loop=True)

def speed_w(orig_f, updown, duration):
    return orig_f[0:((get_len(orig_f)*1000)/2)-duration]+orig_f[(((get_len(orig_f)*1000)/2)-duration):(((get_len(orig_f)*1000)/2)+duration)].speedup(playback_speed=updown)+orig_f[(((get_len(orig_f)*1000)/2)+duration):]

def fade(orig_f, inout):
    if inout=="in":
        return orig_f.fade(from_gain=-140, start=0, end=len(orig_f))
    else:
        return orig_f.fade(to_gain=-140, start=0, end=len(orig_f))

def dropw(orig_f, utt_json):
    drops=[]
    words = utt_json['words']
    for i in range(0,len(words)):
        word = words[i]['word']
        if words[i]["case"]=="not-found-in-audio":
            continue
        if word in drw:
            drops.append((words[i]['start']*1000, words[i]['end']*1000))

    new_aud = orig_f[:drops[0][0]]
    for i in range(0,len(drops)):
        if i==len(drops)-1:
            new_aud = new_aud+orig_f[drops[i][1]:]
        else:
            new_aud =  new_aud+orig_f[drops[i][1]:drops[i+1][0]]
    return new_aud

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def can_drop_h(word):
    if word[0]=='h' and word[1] in vowels:
        return True
    return False

def can_drop_t(word, next_word):
    if word[-1]=='t' and word[-2] not in vowels and next_word[0] not in vowels:
        return True
    return False

def can_drop_d(word, next_word):
    if word[-2:]=='nd' and next_word[0] not in vowels:
        return True
    return False

def can_drop_ing(word):
    if word[-3:]=='ing':
        return True
    return False

def can_drop_r(word):
    if 'r' in word:
        r_pos = word.rfind('r')
        if r_pos == 0:
            return False
        if word[r_pos-1] in vowels:
            if r_pos!=len(word)-1:
                if word[r_pos+1] not in vowels:
                    return True
                else:
                    return False
            else:
                return True
    return False

def can_drop(word, next_word=""):
    return can_drop_r(word) or can_drop_ing(word) or can_drop_d(word, next_word) or can_drop_t(word, next_word) or can_drop_h(word)

def dropl(orig_f, utt_json):
    words = utt_json['words']
    drops=[]
    words = utt_json['words']
    for i in range(0,len(words)):
        word = words[i]['word']
        if words[i]["case"]=="not-found-in-audio":
            continue
        if can_drop_r(word):
            drops.append((words[i]['end']*1000-words[i]["phones"][-1]["duration"]*800, words[i]['end']*1000))
        elif can_drop_ing(word):
            drops.append((words[i]['end']*1000-words[i]["phones"][-1]["duration"]*500, words[i]['end']*1000))
        elif can_drop_h(word):
            drops.append((words[i]['start']*1000, words[i]['start']*1000+words[i]["phones"][0]["duration"]*900))
        elif i!=len(words)-1:
            next_word = words[i+1]['word']
            if can_drop_d(word, next_word):
                drops.append((words[i]['end']*1000-words[i]["phones"][-1]["duration"]*900, words[i]['end']*1000))
            elif can_drop_t(word, next_word):
                drops.append((words[i]['end']*1000-words[i]["phones"][-1]["duration"]*800, words[i]['end']*1000))
    new_aud = orig_f[:drops[0][0]]
    for i in range(0,len(drops)):
        if i==len(drops)-1:
            new_aud = new_aud+orig_f[drops[i][1]:]
        else:
            new_aud =  new_aud+orig_f[drops[i][1]:drops[i+1][0]]
    return new_aud

def reverb(wav_file_name_in, wav_file_name_out):
    fx(wav_file_name_in, wav_file_name_out)
    y, _ = librosa.load(wav_file_name_out, sr=16000)
    sf.write(wav_file_name_out, y, 16000)

def laugh(orig_f, laugh_f):
    return orig_f+laugh_f

def speed_u(wav_file_name_in, wav_file_name_out, updown):
    y, sr = librosa.load(wav_file_name_in)
    y_effect = librosa.effects.time_stretch(y, updown)
    librosa.output.write_wav(wav_file_name_out, y_effect, sr)


def pitch(wav_file_name_in, wav_file_name_out, updown):
    y, sr = librosa.load(wav_file_name_in)
    if updown=="up":
        n=3
    else:
        n=-3
    y_third = librosa.effects.pitch_shift(y, sr, n_steps=n)
    librosa.output.write_wav(wav_file_name_out, y_third, sr)

def muffle(file_in, file_out, modifier):
    fx_muffle = (AudioEffectsChain().lowpass(modifier))
    fx_muffle(file_in, file_out)
    y, _ = librosa.load(file_out, sr=16000)
    sf.write(file_out, y, 16000)

def get_len(orig_f):
    return orig_f.duration_seconds

def assert_modifier(modifier, option):
    assert modifier != None, "Must pass modifier in for option " + option

def add_noise(file_in, file_out, option, modifier=None):
    '''possible options:
    speedu: speed utterance up or down by <modifier> (e.g. 1.25, .75...)
    fade: modifier: "in" | "out"
    pitch: modifier: "up" | "dn"
    env_st: environmental noise fading in and out.  modifier: 'n'|'h'|'i'
    env_co: environmental noise constant.  modifier: ('n'|'h'|'i', snr).  snr is the signal to noise ratio.  snr > 0 means the original audio will come through more clearly.  snr < 0 means the noise will come through more clearly.
    laugh: modifier: 'f', 'm' - sex of speaker in utterance
    muffle: modifier: rate (700: a little, 350: medium, 150: a lot)
    '''
    if os.path.exists(file_out):
        return
    
    try:
        orig_f = AudioSegment.from_wav(file_in)
    except:
        # print(file_in)
        # exit(1)
        assert False, f'{file_in} cannot be converted'

    if option == 'speedu':
        assert_modifier(modifier, option)
        try:
            modifier = float(modifier)
            speed_u(file_in, file_out, modifier)
        except:
            print("Speed utterance didn't work:", file_in)

    elif option == 'fade':
        assert_modifier(modifier, option)
        try:
            orig_f = fade(orig_f, modifier)
            orig_f = orig_f.set_frame_rate(16000)
            orig_f.export(file_out, format='wav')
        except:
            print("Fade utterance didn't work:", file_in)

    elif option == 'pitch':
        assert_modifier(modifier, option)
        try:
            pitch(file_in, file_out, modifier)
        except:
            print("Pitch didn't work:", file_in)

    elif option == "reverb":
        reverb(file_in,  file_out)

    elif option == 'env_st':
        orig_f = env_st(orig_f, modifier)
        orig_f = orig_f.set_frame_rate(16000)
        orig_f.export(file_out, format="wav")
    
    elif option == 'env_co':
        sound_type, snr = modifier
        snr = float(snr)
        orig_f = env_co(orig_f, sound_type, snr)
        orig_f = orig_f.set_frame_rate(16000)
        orig_f.export(file_out, format='wav')

    elif option == 'muffle':
        modifier = float(modifier)
        muffle(file_in, file_out, modifier)
    
    p.add(1)

def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

def add_noise_dir(in_dir, out_dir, option, modifier=None, overwrite=False):
    if in_dir == out_dir:
        print(f'Cannot add noise to the same directory because filenames remain the same.\nin_dir: {in_dir}\nout_dir: {out_dir}')
        return
    if os.path.exists(out_dir):
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            print(f'\nPath {out_dir} exists and overwrite=False.  Skipping...')
            return

    print(f'\nAdding noise for option {option}, modifier {modifier} to {in_dir} -> {out_dir}...')
    mkdirp(out_dir)
    num_workers = 6
    pool = mp.Pool(num_workers)
    inputs = [(os.path.join(in_dir, elt), os.path.join(out_dir, elt), option, modifier) for elt in list(filter(lambda elt: not os.path.isdir(os.path.join(in_dir, elt)), os.listdir(in_dir)))]
    
    global p
    p = Progbar(len(inputs)) 

    # # testing
    # for input in inputs:
    #     add_noise(*input)
    # exit()

    x = pool.starmap(add_noise, inputs)
    [elt for elt in x] # forces errors to propagate
    pool.close()
    pool.join()
    
def tuple_to_strs(tup):
    return list(map(lambda elt: str(elt), tup))

def parse_opt_mod(a):
    '''
    a = 'env_co__i___-10' -> (env_co, (i, -10))
    a = 'reverb__None' ->  (reverb, None)
    '''
    tot = a.split('___')
    if len(tot) > 1:
        modifier = tot[0].split('__')[-1], tot[1]
    else:
        modifier = tot[0].split('__')[-1]
    option = a.split('__')[0]
    return option, modifier

def opt_mod_to_str(opt, mod):
    s = None
    if type(mod) == tuple:
        s = '__'.join((opt, '___'.join(tuple_to_strs(mod))))

    else:
        combined = tuple_to_strs((opt, mod))
        s = '__'.join((combined))
    return s

def add_noise_dirs(in_dir, out_dir, overwrite=False, options=None):
    a = ['env_co']
    b = ['n', 'h', 'i']
    c = [10, 0, -10]
    packed = list(map(lambda elt: (elt[0], (elt[1], elt[2])), list(itertools.product(a,b,c))))

    if options is None:
        # full options
        # options = [
        #     ('env_st', 'n'),
        #     ('env_st', 'h'),
        #     ('env_st', 'i'),
        #     ('fade', 'in'),
        #     ('fade', 'out'),
        #     ('reverb', None),
        #     ('muffle', 150),
        #     ('muffle', 300),
        #     ('muffle', 550),
        #     *packed,
        # ]

        # some options
        options = [
            ('env_co', ('i', -10)),
            ('env_co', ('n', -10)),
            ('muffle', 150),
            ('reverb', None),
            # ('env_co', ('h', -10)),
            # ('env_st', 'i'),
            # ('env_st', 'n'),
            # ('env_st', 'h'),
        ]

    print('\nAdding noise: total Progress:')
    for opt, mod in tqdm(options):
        add_noise_dir(in_dir, os.path.join(out_dir, opt_mod_to_str(opt, mod)), option=opt, modifier=mod, overwrite=overwrite)
        
if __name__ == '__main__':
    '''usage: python3 noise_adder.py ./orig_wavs ./noisy_wavs -o True'''
    parser = argparse.ArgumentParser(description='Write noisy version of directory containing .wav files')
    parser.add_argument('in_dir', type=str, help='The flattened input directory containing wav files')
    parser.add_argument('out_dir', type=str, help='The directory path that will contain the noisy wav files')
    parser.add_argument('-o', '--overwrite', type=str, default=False, help='Do you wish to force an overwrite of out_dir? True| False')
    
    args = parser.parse_args()

    add_noise_dirs(args.in_dir, args.out_dir, overwrite=args.overwrite)

