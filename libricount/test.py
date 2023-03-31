import numpy as np


meta = [
    {"sex": "F", "activity": [[0, 80000]], "speaker_id": 1580},
    {"sex": "F", "activity": [
        [0, 29191],
        [30953, 58981],
        [61703, 64907],
        [68750, 80000]
    ], "speaker_id": 4507},
    {"sex": "F", "activity": [[0, 80000]], "speaker_id": 5683}]


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


def _get_speaker_count(self, meta_enc,  w_range):
    vals, counts = np.unique(
        np.sum(meta_enc[:, w_range[0]:w_range[1]], axis=0),
        return_counts=True)
    return np.int64(vals[np.argmax(counts)])


def _get_window_range(self, size, idx):
    return (idx * size * 16, size * (idx + 1) * 16)
