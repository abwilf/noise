# Noise Adder
This script adds different types of noise to a directory of wav files, yielding new directories named for the type of noise being added and a "modifier" that applies to the noise addition process (e.g. SNR for additive environmental noise).

## Usage
```
pip install pathlib librosa syllables tqdm pydub pysndfx tensorflow
unzip noise_wavs.zip
python3 noise_adder.py samples noisy_samples --overwrite=True
```

See `add_noise` docstring for details about the options.  You can modify them as needed in `add_noise_dirs`.

## Credits
Large portions of this script were adapted from Mimansa Jaiswal.
