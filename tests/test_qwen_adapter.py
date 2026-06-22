from pathlib import Path

import numpy as np

from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter, QwenTtsRuntime


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
        return [np.ones(10 * (index + 1), dtype=np.float32) for index, _ in enumerate(text)], 24000


class FakeTorchStore:
    def __init__(self):
        self.saved = {}
        self.loads = []

    def save(self, value, path):
        self.saved[str(path)] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"qvp")

    def load(self, path, map_location="cpu", weights_only=False):
        self.loads.append(str(path))
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


def test_qwen_adapter_default_batch_size_matches_ui_default():
    adapter = QwenTtsAdapter(model=FakeQwenModel(), torch_module=FakeTorchStore())

    assert adapter.max_batch_size == 8
    assert adapter.max_block_chars == 600


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
    assert [call["text"] for call in model.voice_clone_calls] == [["Hello. Again."]]
    assert [len(item.samples) for item in generated] == [5, 5]


def test_qwen_adapter_generates_contiguous_same_role_run_as_one_block(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "narrator.qvp"
    torch_store.save({"prompt": "narrator"}, voice_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": voice_path},
        max_batch_size=2,
    )

    generated = adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
            {"sentence_idx": 1, "role": "Narrator", "type": "narration", "text": "Two."},
            {"sentence_idx": 2, "role": "Narrator", "type": "narration", "text": "Three."},
        ]
    )

    assert [call["text"] for call in model.voice_clone_calls] == [["One. Two. Three."]]
    assert [item.sentence_idx for item in generated] == [0, 1, 2]
    assert generated[0].pause_after_ms == 0
    assert generated[1].pause_after_ms == 0
    assert generated[2].pause_after_ms is None


def test_qwen_adapter_caps_contiguous_same_role_generation_blocks(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "narrator.qvp"
    torch_store.save({"prompt": "narrator"}, voice_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": voice_path},
        max_batch_size=8,
        max_block_chars=14,
    )

    generated = adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
            {"sentence_idx": 1, "role": "Narrator", "type": "narration", "text": "Two."},
            {"sentence_idx": 2, "role": "Narrator", "type": "narration", "text": "Three."},
            {"sentence_idx": 3, "role": "Narrator", "type": "narration", "text": "Four."},
        ]
    )

    assert [call["text"] for call in model.voice_clone_calls] == [["One. Two.", "Three. Four."]]
    assert [item.sentence_idx for item in generated] == [0, 1, 2, 3]
    assert generated[0].pause_after_ms == 0
    assert generated[1].pause_after_ms is None
    assert generated[2].pause_after_ms == 0


def test_qwen_adapter_batches_mixed_role_blocks_in_script_order(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    narrator_path = tmp_path / "voices" / "narrator.qvp"
    elena_path = tmp_path / "voices" / "elena.qvp"
    torch_store.save({"prompt": "narrator"}, narrator_path)
    torch_store.save({"prompt": "elena"}, elena_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": narrator_path, "Elena": elena_path},
        max_batch_size=3,
    )

    batches = list(
        adapter.generate_sentence_batches(
            [
                {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
                {"sentence_idx": 1, "role": "Elena", "type": "dialogue", "text": "Hello."},
                {"sentence_idx": 2, "role": "Narrator", "type": "narration", "text": "Two."},
            ]
        )
    )

    assert [call["text"] for call in model.voice_clone_calls] == [["One.", "Hello.", "Two."]]
    assert model.voice_clone_calls[0]["prompt"] == ["narrator", "elena", "narrator"]
    assert [[item.sentence_idx for item in batch] for batch in batches] == [[0, 1, 2]]
    assert torch_store.loads.count(str(narrator_path)) == 1


class RuntimeFakeModel:
    calls = []

    def __init__(self, source):
        self.source = source

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append({"source": source, "kwargs": kwargs})
        return cls(source)

    def generate_voice_design(self, text, instruct, language, **kwargs):
        return [np.ones(10, dtype=np.float32)], 24000

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        return [{"prompt": self.source}]

    def generate_voice_clone(self, text, language, voice_clone_prompt, **kwargs):
        return [np.ones(8, dtype=np.float32) for _ in text], 24000


class RuntimeFakeTorch:
    bfloat16 = "bf16"
    float32 = "fp32"

    class cuda:
        @staticmethod
        def is_available():
            return False

    class backends:
        class mps:
            @staticmethod
            def is_available():
                return False


def test_qwen_runtime_loads_voice_design_and_base_models_from_local_root(tmp_path):
    RuntimeFakeModel.calls = []
    model_root = tmp_path / "models" / "qwen-tts"
    (model_root / "Qwen3-TTS-12Hz-1.7B-Base").mkdir(parents=True)
    (model_root / "Qwen3-TTS-12Hz-1.7B-VoiceDesign").mkdir()
    runtime = QwenTtsRuntime(
        qwen_model_cls=RuntimeFakeModel,
        torch_module=RuntimeFakeTorch,
        model_root=model_root,
        model_choice="1.7B",
        device="cpu",
        precision="bf16",
        attention="eager",
    )

    runtime.generate_voice_design(text="sample", instruct="voice", language="auto")
    runtime.generate_voice_clone(
        text=["hello"],
        language=["auto"],
        voice_clone_prompt=[{"prompt": "x"}],
    )

    assert [Path(call["source"]).name for call in RuntimeFakeModel.calls] == [
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "Qwen3-TTS-12Hz-1.7B-Base",
    ]
    assert RuntimeFakeModel.calls[0]["kwargs"]["device_map"] == "cpu"
    assert RuntimeFakeModel.calls[0]["kwargs"]["dtype"] == "bf16"
    assert RuntimeFakeModel.calls[0]["kwargs"]["attn_implementation"] == "eager"
