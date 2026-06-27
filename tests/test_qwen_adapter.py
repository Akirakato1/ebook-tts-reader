import json
import wave
from pathlib import Path

import numpy as np
import pytest

from ebook_tts_pipeline.tts import qwen_adapter
from ebook_tts_pipeline.tts.qwen_adapter import QwenTtsAdapter, QwenTtsRuntime


class FakeQwenModel:
    def __init__(self):
        self.voice_design_calls = []
        self.voice_prompt_calls = []
        self.voice_clone_calls = []

    def generate_voice_design(self, text, instruct, language, **kwargs):
        self.voice_design_calls.append({"text": text, "instruct": instruct, "language": language})
        return [np.ones(100, dtype=np.float32) * 0.1], 24000

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        self.voice_prompt_calls.append(
            {
                "ref_audio_len": len(ref_audio[0]),
                "sample_rate": ref_audio[1],
                "ref_text": ref_text,
                "x_vector_only_mode": x_vector_only_mode,
            }
        )
        return [{"ref_code": None, "ref_spk_embedding": "embedding", "x_vector_only_mode": True, "icl_mode": False}]

    def generate_voice_clone(self, text, language, voice_clone_prompt, **kwargs):
        self.voice_clone_calls.append(
            {"text": text, "language": language, "prompt": voice_clone_prompt, "kwargs": kwargs}
        )
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


class AdaptiveTorchStore(FakeTorchStore):
    def __init__(self, peak_reserved):
        super().__init__()
        self.cuda = self.FakeCuda(peak_reserved)

    class FakeCuda:
        def __init__(self, peak_reserved):
            self.peak_reserved = peak_reserved

        def is_available(self):
            return True

        def synchronize(self):
            return None

        def reset_peak_memory_stats(self):
            return None

        def memory_allocated(self):
            return 2 * 1024 ** 3

        def memory_reserved(self):
            return 3 * 1024 ** 3

        def max_memory_allocated(self):
            return self.peak_reserved

        def max_memory_reserved(self):
            return self.peak_reserved

        def mem_get_info(self):
            return (4 * 1024 ** 3, 16 * 1024 ** 3)


class CacheCountingTorchStore(FakeTorchStore):
    def __init__(self):
        super().__init__()
        self.cuda = self.FakeCuda()

    class FakeCuda:
        def __init__(self):
            self.empty_cache_calls = 0

        def empty_cache(self):
            self.empty_cache_calls += 1


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


def test_qwen_adapter_has_no_public_batch_or_block_size_controls():
    adapter = QwenTtsAdapter(model=FakeQwenModel(), torch_module=FakeTorchStore())

    assert not hasattr(adapter, "max_batch_size")
    assert not hasattr(adapter, "max_block_chars")


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


def test_qwen_adapter_saves_voice_design_reference_audio_as_preview_sample(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    adapter = QwenTtsAdapter(model=model, torch_module=torch_store)
    voice_path = tmp_path / "voices" / "elena.qvp"
    sample_path = tmp_path / "voices" / "_samples" / "elena.wav"
    sample_text = "Hello, my name is Elena."
    voice_record = {
        "voice_profile": {"qwen_instruct": "A soft voice."},
        "voice_identity": {"seed": 42},
    }

    adapter.ensure_voice(
        "elena",
        voice_record,
        voice_path,
        sample_path=sample_path,
        reference_text=sample_text,
    )

    assert model.voice_design_calls == [
        {"text": sample_text, "instruct": "A soft voice.", "language": "auto"}
    ]
    assert model.voice_prompt_calls[0]["ref_text"] == sample_text
    with wave.open(str(sample_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 24000
        assert wav_file.getnframes() == 100


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


def test_qwen_adapter_prefers_role_id_for_voice_lookup(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    narrator_path = tmp_path / "voices" / "narrator.qvp"
    callie_path = tmp_path / "voices" / "callie_adult.qvp"
    torch_store.save({"prompt": "narrator"}, narrator_path)
    torch_store.save({"prompt": "callie"}, callie_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"narrator": narrator_path, "callie_adult": callie_path},
    )

    adapter.generate_sentences(
        [
            {
                "sentence_idx": 0,
                "role": "Narrator",
                "role_id": "narrator",
                "type": "narration",
                "text": "Narration.",
            },
            {
                "sentence_idx": 1,
                "role": "Callie",
                "role_id": "callie_adult",
                "type": "dialogue",
                "text": "Dialogue.",
            },
        ]
    )

    assert [call["prompt"] for call in model.voice_clone_calls] == [["narrator", "callie"]]


def test_qwen_adapter_speaks_hash_symbol_as_hashtag(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "narrator.qvp"
    torch_store.save({"prompt": "narrator"}, voice_path)
    adapter = QwenTtsAdapter(model=model, torch_module=torch_store, role_voice_paths={"Narrator": voice_path})

    adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "Follow #TeamCallie."},
        ]
    )

    assert [call["text"] for call in model.voice_clone_calls] == [["Follow hashtag TeamCallie."]]


def test_qwen_adapter_generated_audio_records_actual_voice_path(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    narrator_path = tmp_path / "voices" / "narrator.qvp"
    torch_store.save({"prompt": "narrator"}, narrator_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": narrator_path},
    )

    generated = adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
        ]
    )

    assert generated[0].voice_config_path == str(narrator_path)


def test_qwen_adapter_passes_configured_max_new_tokens_to_model(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "narrator.qvp"
    torch_store.save({"prompt": "narrator"}, voice_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": voice_path},
        max_new_tokens=1234,
    )

    adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
        ]
    )

    assert model.voice_clone_calls[0]["kwargs"]["max_new_tokens"] == 1234


def test_qwen_adapter_uses_streaming_text_mode_by_default(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "narrator.qvp"
    torch_store.save({"prompt": "narrator"}, voice_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": voice_path},
    )

    adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
        ]
    )

    assert model.voice_clone_calls[0]["kwargs"]["non_streaming_mode"] is False


def test_qwen_adapter_clears_cuda_cache_on_configured_interval(tmp_path):
    model = FakeQwenModel()
    torch_store = CacheCountingTorchStore()
    narrator_path = tmp_path / "voices" / "narrator.qvp"
    elena_path = tmp_path / "voices" / "elena.qvp"
    torch_store.save({"prompt": "narrator"}, narrator_path)
    torch_store.save({"prompt": "elena"}, elena_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": narrator_path, "Elena": elena_path},
        max_generation_blocks_per_call=1,
        cache_clear_interval=3,
    )

    list(
        adapter.generate_sentence_batches(
            [
                {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
                {"sentence_idx": 1, "role": "Elena", "type": "dialogue", "text": "Two."},
            ]
        )
    )
    assert torch_store.cuda.empty_cache_calls == 0

    list(
        adapter.generate_sentence_batches(
            [
                {"sentence_idx": 2, "role": "Narrator", "type": "narration", "text": "Three."},
            ]
        )
    )
    assert torch_store.cuda.empty_cache_calls == 1


def test_qwen_adapter_uses_instance_generation_block_controls(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "narrator.qvp"
    torch_store.save({"prompt": "narrator"}, voice_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": voice_path},
        max_generation_block_chars=9,
        max_generation_blocks_per_call=1,
    )

    adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
            {"sentence_idx": 1, "role": "Narrator", "type": "narration", "text": "Two."},
            {"sentence_idx": 2, "role": "Narrator", "type": "narration", "text": "Three."},
        ]
    )

    assert [call["text"] for call in model.voice_clone_calls] == [["One. Two."], ["Three."]]


def test_qwen_adapter_zero_generation_block_call_limit_means_unlimited(tmp_path):
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
        max_generation_blocks_per_call=0,
    )

    adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
            {"sentence_idx": 1, "role": "Elena", "type": "dialogue", "text": "Two."},
            {"sentence_idx": 2, "role": "Narrator", "type": "narration", "text": "Three."},
            {"sentence_idx": 3, "role": "Elena", "type": "dialogue", "text": "Four."},
        ]
    )

    assert [call["text"] for call in model.voice_clone_calls] == [["One.", "Two.", "Three.", "Four."]]


def test_qwen_adapter_writes_generation_performance_log(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "narrator.qvp"
    perf_log_path = tmp_path / "perf" / "qwen_generation.jsonl"
    torch_store.save({"prompt": "narrator"}, voice_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": voice_path},
        max_new_tokens=1536,
        performance_log_path=perf_log_path,
    )

    adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
            {"sentence_idx": 1, "role": "Narrator", "type": "narration", "text": "Two."},
        ]
    )

    events = [json.loads(line) for line in perf_log_path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    assert events[0]["batch_size"] == 1
    assert events[0]["max_new_tokens"] == 1536
    assert events[0]["text_chars"] == [9]
    assert events[0]["voice_config_paths"] == [str(voice_path)]
    assert events[0]["sample_counts"] == [10]


def test_qwen_adapter_perf_log_records_section_shape(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    narrator_path = tmp_path / "voices" / "narrator.qvp"
    elena_path = tmp_path / "voices" / "elena.qvp"
    perf_log_path = tmp_path / "perf" / "qwen_generation.jsonl"
    torch_store.save({"prompt": "narrator"}, narrator_path)
    torch_store.save({"prompt": "elena"}, elena_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": narrator_path, "Elena": elena_path},
        performance_log_path=perf_log_path,
    )

    adapter.generate_sentences(
        [
            {
                "sentence_idx": 0,
                "role": "Narrator",
                "type": "narration",
                "text": "One.",
                "_tts_section_idx": 7,
                "_tts_section_char_count": 42,
                "_tts_section_job_count": 3,
            },
            {
                "sentence_idx": 1,
                "role": "Elena",
                "type": "dialogue",
                "text": "Two.",
                "_tts_section_idx": 7,
                "_tts_section_char_count": 42,
                "_tts_section_job_count": 3,
            },
            {
                "sentence_idx": 2,
                "role": "Narrator",
                "type": "narration",
                "text": "Three.",
                "_tts_section_idx": 7,
                "_tts_section_char_count": 42,
                "_tts_section_job_count": 3,
            },
        ]
    )

    event = json.loads(perf_log_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["tts_section_indices"] == [7]
    assert event["tts_section_char_count_max"] == 42
    assert event["tts_section_job_count_sum"] == 3
    assert event["role_switch_count"] == 2
    assert event["unique_voice_count"] == 2
    assert event["max_job_chars"] == 6


def test_qwen_adapter_adapts_block_limit_after_over_target_memory(tmp_path):
    model = FakeQwenModel()
    torch_store = AdaptiveTorchStore(peak_reserved=16 * 1024 ** 3)
    narrator_path = tmp_path / "voices" / "narrator.qvp"
    elena_path = tmp_path / "voices" / "elena.qvp"
    perf_log_path = tmp_path / "perf" / "qwen_generation.jsonl"
    torch_store.save({"prompt": "narrator"}, narrator_path)
    torch_store.save({"prompt": "elena"}, elena_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": narrator_path, "Elena": elena_path},
        max_generation_blocks_per_call=0,
        adaptive_memory_target_bytes=13 * 1024 ** 3,
        performance_log_path=perf_log_path,
    )

    adapter.generate_sentences(
        [
            {"sentence_idx": 0, "role": "Narrator", "type": "narration", "text": "One."},
            {"sentence_idx": 1, "role": "Elena", "type": "dialogue", "text": "Two."},
            {"sentence_idx": 2, "role": "Narrator", "type": "narration", "text": "Three."},
            {"sentence_idx": 3, "role": "Elena", "type": "dialogue", "text": "Four."},
        ]
    )

    event = json.loads(perf_log_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["adaptive_memory_target_bytes"] == 13 * 1024 ** 3
    assert event["adaptive_previous_blocks_per_call"] == 0
    assert event["adaptive_next_blocks_per_call"] == 2
    assert event["adaptive_reason"] == "over_target"
    assert adapter.max_generation_blocks_per_call == 2


def test_qwen_adapter_generates_contiguous_same_role_run_as_one_block(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "narrator.qvp"
    torch_store.save({"prompt": "narrator"}, voice_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": voice_path},
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


def test_qwen_adapter_caps_long_contiguous_same_voice_blocks(tmp_path):
    model = FakeQwenModel()
    torch_store = FakeTorchStore()
    voice_path = tmp_path / "voices" / "narrator.qvp"
    torch_store.save({"prompt": "narrator"}, voice_path)
    adapter = QwenTtsAdapter(
        model=model,
        torch_module=torch_store,
        role_voice_paths={"Narrator": voice_path},
        max_generation_block_chars=14,
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


def test_qwen_adapter_keeps_mixed_voice_blocks_in_one_model_call(tmp_path):
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
    )

    generated_sections = list(
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
    assert [[item.sentence_idx for item in section] for section in generated_sections] == [[0, 1, 2]]
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


def _runtime_model_root(tmp_path: Path) -> Path:
    model_root = tmp_path / "models" / "qwen-tts"
    (model_root / "Qwen3-TTS-12Hz-1.7B-Base").mkdir(parents=True)
    (model_root / "Qwen3-TTS-12Hz-1.7B-VoiceDesign").mkdir()
    return model_root


def test_qwen_runtime_loads_voice_design_and_base_models_from_local_root(tmp_path, capsys):
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
    output = capsys.readouterr().out
    assert "[ebook-tts] qwen_model_resolved" in output
    assert "model_type=VoiceDesign" in output
    assert "model_type=Base" in output
    assert str(model_root / "Qwen3-TTS-12Hz-1.7B-VoiceDesign") in output
    assert str(model_root / "Qwen3-TTS-12Hz-1.7B-Base") in output
    assert RuntimeFakeModel.calls[0]["kwargs"]["device_map"] == "cpu"
    assert RuntimeFakeModel.calls[0]["kwargs"]["dtype"] == "bf16"
    assert RuntimeFakeModel.calls[0]["kwargs"]["attn_implementation"] == "eager"


def test_qwen_runtime_auto_attention_prefers_flash_when_available(tmp_path, monkeypatch):
    RuntimeFakeModel.calls = []
    model_root = _runtime_model_root(tmp_path)

    def fake_find_spec(name):
        return object() if name == "flash_attn" else None

    monkeypatch.setattr(qwen_adapter.importlib.util, "find_spec", fake_find_spec)
    runtime = QwenTtsRuntime(
        qwen_model_cls=RuntimeFakeModel,
        torch_module=RuntimeFakeTorch,
        model_root=model_root,
        device="cpu",
        precision="bf16",
        attention="auto",
    )

    runtime.generate_voice_clone(
        text=["hello"],
        language=["auto"],
        voice_clone_prompt=[{"prompt": "x"}],
    )

    assert RuntimeFakeModel.calls[0]["kwargs"]["attn_implementation"] == "flash_attention_2"


def test_qwen_runtime_auto_attention_falls_back_to_sdpa(tmp_path, monkeypatch):
    RuntimeFakeModel.calls = []
    model_root = _runtime_model_root(tmp_path)
    monkeypatch.setattr(qwen_adapter.importlib.util, "find_spec", lambda name: None)
    runtime = QwenTtsRuntime(
        qwen_model_cls=RuntimeFakeModel,
        torch_module=RuntimeFakeTorch,
        model_root=model_root,
        device="cpu",
        precision="bf16",
        attention="auto",
    )

    runtime.generate_voice_clone(
        text=["hello"],
        language=["auto"],
        voice_clone_prompt=[{"prompt": "x"}],
    )

    assert RuntimeFakeModel.calls[0]["kwargs"]["attn_implementation"] == "sdpa"


def test_qwen_runtime_refuses_to_download_when_local_model_is_missing(tmp_path):
    runtime = QwenTtsRuntime(
        qwen_model_cls=RuntimeFakeModel,
        torch_module=RuntimeFakeTorch,
        model_root=tmp_path / "missing-qwen-models",
        device="cpu",
        precision="bf16",
        attention="eager",
    )

    with pytest.raises(RuntimeError, match="Local Qwen3-TTS model not found"):
        runtime.generate_voice_clone(
            text=["hello"],
            language=["auto"],
            voice_clone_prompt=[{"prompt": "x"}],
        )
