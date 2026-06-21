import numpy as np

from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter


class FakeQwenModel:
    def __init__(self):
        self.voice_design_calls = []
        self.voice_clone_calls = []

    def generate_voice_design(self, text, instruct, language, **kwargs):
        self.voice_design_calls.append({"text": text, "instruct": instruct, "language": language})
        return [np.ones(100, dtype=np.float32) * 0.1], 24000

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        return [{"ref_code": None, "ref_spk_embedding": "embedding", "x_vector_only_mode": True, "icl_mode": False}]

    def generate_voice_clone(self, text, language, voice_clone_prompt, **kwargs):
        self.voice_clone_calls.append({"text": text, "language": language, "prompt": voice_clone_prompt})
        return [np.ones(50, dtype=np.float32), np.ones(25, dtype=np.float32)], 24000


class FakeTorchStore:
    def __init__(self):
        self.saved = {}

    def save(self, value, path):
        self.saved[str(path)] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"qvp")

    def load(self, path, map_location="cpu", weights_only=False):
        return self.saved[str(path)]


def test_qwen_adapter_creates_qvp_once_and_reuses_existing_file(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    adapter = QwenTtsAdapter(model=model, torch_module=torch_store)
    voice_path = tmp_path / "voices" / "elena.qvp"
    voice_record = {
        "voice_profile": {"qwen_instruct": "A soft voice."},
        "voice_identity": {"seed": 42},
    }

    first = adapter.ensure_voice("elena", voice_record, voice_path)
    second = adapter.ensure_voice("elena", voice_record, voice_path)

    assert first == voice_path
    assert second == voice_path
    assert len(model.voice_design_calls) == 1


def test_qwen_adapter_regenerates_existing_qvp_when_forced(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    adapter = QwenTtsAdapter(model=model, torch_module=torch_store)
    voice_path = tmp_path / "voices" / "elena_internal.qvp"
    voice_record = {
        "voice_profile": {"qwen_instruct": "A soft inward voice."},
        "voice_identity": {"seed": 42},
    }

    adapter.ensure_voice("elena_internal", voice_record, voice_path)
    adapter.ensure_voice(
        "elena_internal",
        {**voice_record, "_force_regenerate": True},
        voice_path,
    )

    assert len(model.voice_design_calls) == 2


def test_qwen_adapter_generates_sentence_audio_in_order(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "elena.qvp"
    torch_store.save({"prompt": "saved"}, voice_path)
    adapter = QwenTtsAdapter(model=model, torch_module=torch_store, role_voice_paths={"Elena": voice_path})

    generated = adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Elena", "type": "dialogue", "text": "Hello."},
            {"sentence_idx": 1, "role": "Elena", "type": "dialogue", "text": "Again."},
        ]
    )

    assert [item.sentence_idx for item in generated] == [0, 1]
    assert [len(item.samples) for item in generated] == [50, 25]
