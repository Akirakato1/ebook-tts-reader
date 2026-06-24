import base64

import numpy as np

from ebook_tts_pipeline.tts.base import GeneratedSentenceAudio
from ebook_tts_pipeline.tts.wsl_worker import decode_audio_item, encode_audio_item


def test_worker_audio_item_round_trips_float32_samples():
    item = GeneratedSentenceAudio(
        sentence_idx=7,
        unit_idx=3,
        role="Narrator",
        speech_type="narration",
        samples=np.array([0.0, 0.25, -0.25], dtype=np.float32),
        sample_rate=24000,
        voice_config_path="/mnt/c/book/voices/narrator.qvp",
    )

    encoded = encode_audio_item(item)

    assert encoded["sentence_idx"] == 7
    assert encoded["unit_idx"] == 3
    assert encoded["dtype"] == "float32"
    assert encoded["shape"] == [3]
    assert base64.b64decode(encoded["samples_b64"])

    decoded = decode_audio_item(encoded)
    assert decoded.sentence_idx == 7
    assert decoded.unit_idx == 3
    assert decoded.role == "Narrator"
    assert decoded.speech_type == "narration"
    assert decoded.sample_rate == 24000
    assert decoded.voice_config_path == "/mnt/c/book/voices/narrator.qvp"
    np.testing.assert_allclose(decoded.samples, item.samples)
