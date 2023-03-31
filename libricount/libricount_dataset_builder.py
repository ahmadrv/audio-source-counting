"""libricount dataset."""

import tensorflow_datasets as tfds
import numpy as np
import dataclasses
import json
from scipy.io.wavfile import read as read_wave
from pathlib import Path


def divisors(n):
    # get factors and their counts
    factors = {}
    nn = n
    i = 2
    while i*i <= nn:
        while nn % i == 0:
            factors[i] = factors.get(i, 0) + 1
            nn //= i
        i += 1
    if nn > 1:
        factors[nn] = factors.get(nn, 0) + 1

    primes = list(factors.keys())

    # generates factors from primes[k:] subset
    def generate(k):
        if k == len(primes):
            yield 1
        else:
            rest = generate(k+1)
            prime = primes[k]
            for factor in rest:
                prime_to_i = 1
                # prime_to_i iterates prime**i values, i being all possible exponents
                for _ in range(factors[prime] + 1):
                    yield factor * prime_to_i
                    prime_to_i *= prime

    # in python3, `yield from generate(0)` would also work
    for factor in generate(0):
        yield factor


@dataclasses.dataclass
class WindowConfig(tfds.core.BuilderConfig):
    window_size: int = 10


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for libricount dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [WindowConfig(name=str(size), window_size=size)
                       for size in [5000, 10, 20, 25, 40, 50, 100, 125, 200, 250,
                       500, 625, 1000, 2500]]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'audio_window': tfds.features.Audio(shape=(80000,)),
                'label': tfds.features.ClassLabel(names=list(map(str, range(11)))),
            }),
            supervised_keys=('audio_window', 'label'),
            homepage='https://zenodo.org/record/1216072',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            'https://zenodo.org/record/1216072/files/LibriCount10-0dB.zip')

        return {
            'train': self._generate_examples(path / 'test'),
        }

    def _generate_examples(self, path: Path):
        """Yields examples."""
        window_size = self.builder_config.window_size
        for file in path.glob('*.wav'):
            # Open file here and cast to nd array
            audio = read_wave(str(file))[1]
            # open json
            meta = json.load(open(file.with_suffix(".json")))
            filename = file.stem
            meta_enc = self._encode_meta(meta)
            for idx in range(int(5000/window_size)):
                w_range = self._get_window_range(window_size, idx)
                yield f"{filename}_{idx}", {
                    'audio_window': self._slice_audio(audio, w_range),
                    'label': self._get_speaker_count(meta_enc, w_range)
                }

    def _get_window_range(self, size, idx):
        return (idx * size * 16, size * (idx + 1) * 16)

    def _slice_audio(self, w, w_range):
        windowed = np.zeros(80000)
        windowed[w_range[0]:w_range[1]] = w[w_range[0]:w_range[1]]
        return windowed

    def _get_speaker_count(self, meta_enc,  w_range):
        if len(meta_enc) == 0:  # in case of zero speakers
            return 0
        vals, counts = np.unique(
            np.sum(meta_enc[:, w_range[0]:w_range[1]], axis=0),
            return_counts=True)
        return np.int64(vals[np.argmax(counts)])

    def _encode_meta(self, meta):
        """Encodes meta dictionary from json file to a one-hot decoded array

        Returns:
            a (80000, #NumSpeakers) array of one-hot encoded values

        Args:
            meta (dict): dictionary of meta json file
        """
        output = []
        for speaker in meta:
            encoded = np.zeros(80000)
            for activity in speaker["activity"]:
                encoded[activity[0]:activity[1]] = 1
            output.append(encoded)

        return np.array(output)
