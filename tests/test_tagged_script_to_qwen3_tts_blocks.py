import importlib.util
import numpy as np
import pathlib
import sys
import tempfile
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "tagged_script_to_qwen3_tts_blocks.py"


def load_module():
    spec = importlib.util.spec_from_file_location("tagged_script_to_qwen3_tts_blocks", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TaggedScriptTests(unittest.TestCase):
    def test_parse_tagged_script_keeps_order_and_text(self):
        module = load_module()

        blocks = module.parse_tagged_script(
            """
            [summary]
            本期视频讲了 xxxx

            [settings]
            设置怎么配置等等
            """
        )

        self.assertEqual([block.tag for block in blocks], ["summary", "settings"])
        self.assertEqual(blocks[0].text, "本期视频讲了 xxxx")
        self.assertEqual(blocks[1].text, "设置怎么配置等等")

    def test_parse_tagged_script_rejects_text_before_first_tag(self):
        module = load_module()

        with self.assertRaises(ValueError):
            module.parse_tagged_script("没有标签的开头\n[summary]\n内容")

    def test_parse_tagged_script_keeps_visual_notes_out_of_text(self):
        module = load_module()

        blocks = module.parse_tagged_script(
            """
            [settings]
            (画面：打开设置页)
            第一次使用时，进入左侧的控制中心进行配置。
            (画面：展示 API Key 输入框)
            """
        )

        self.assertEqual(blocks[0].visual_notes, ["画面：打开设置页", "画面：展示 API Key 输入框"])
        self.assertEqual(blocks[0].text, "第一次使用时，进入左侧的控制中心进行配置。")

    def test_safe_stem_preserves_chinese_and_removes_path_chars(self):
        module = load_module()

        self.assertEqual(module.safe_stem("设置/配置:API"), "设置_配置_API")

    def test_clean_caption_removes_boundary_punctuation(self):
        module = load_module()

        self.assertEqual(module.clean_caption("，就可以进入单视频对话"), "就可以进入单视频对话")
        self.assertEqual(module.clean_caption("”、“哪里讲到了通道处理？"), "哪里讲到了通道处理")
        self.assertEqual(module.clean_caption("、“老师提到的材质命名规范有哪些？"), "老师提到的材质命名规范有哪些")

    def test_split_sentences_keeps_complete_sentences(self):
        module = load_module()

        text = "大家好，欢迎来到我的频道。今天介绍 vsummary。"

        self.assertEqual(module.split_sentences(text), ["大家好，欢迎来到我的频道", "今天介绍vsummary"])

    def test_make_local_srt_uses_aligned_times(self):
        module = load_module()

        sentences = [
            module.SentenceAudio(text="第一句。", duration=1.5, start=0.4, end=1.9),
            module.SentenceAudio(text="第二句。", duration=2.5, start=2.2, end=4.7),
        ]
        srt = module.make_local_srt(sentences)

        self.assertIn("00:00:00,400 --> 00:00:01,900", srt)
        self.assertIn("00:00:02,200 --> 00:00:04,700", srt)
        self.assertIn("第一句", srt)
        self.assertIn("第二句", srt)

    def test_offset_srt_times(self):
        module = load_module()

        srt = "1\n00:00:00,000 --> 00:00:01,000\n第一句\n"

        shifted = module.offset_srt_times(srt, 2.5)

        self.assertIn("00:00:02,500 --> 00:00:03,500", shifted)

    def test_apply_config_loads_settings_and_preset(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ref_text = root / "voices" / "tutorial" / "ref.txt"
            ref_text.parent.mkdir(parents=True)
            ref_text.write_text("reference transcript", encoding="utf-8")
            settings = {
                "tts": {
                    "model": "models/qwen",
                    "device": "cuda",
                    "dtype": "bf16",
                    "max_seq_len": 8192,
                    "max_new_tokens": 4096,
                },
                "aligner": {
                    "model": "Qwen/Qwen3-ForcedAligner-0.6B",
                    "device_map": "cuda:0",
                    "dtype": "bf16",
                },
                "paths": {"hf_hub_cache": "cache/hf"},
                "output": {"block_gap_seconds": 0.4, "dry_run_chars_per_second": 6.0},
            }
            preset = {
                "voice": {
                    "ref_audio": "voices/tutorial/ref.wav",
                    "ref_text_file": str(ref_text),
                },
                "style": {
                    "instruct": "tutorial style",
                    "temperature": 0.6,
                    "top_k": 50,
                    "repetition_penalty": 1.1,
                    "greedy": False,
                    "xvec_only": True,
                    "non_streaming_mode": False,
                },
            }
            args = type("Args", (), {"preset": "tutorial"})()

            original_root = module.PROJECT_ROOT
            module.PROJECT_ROOT = root
            try:
                module.apply_config(args, settings, preset)
            finally:
                module.PROJECT_ROOT = original_root

            self.assertEqual(args.model, str((root / "models/qwen").resolve()))
            self.assertEqual(args.ref_audio, str((root / "voices/tutorial/ref.wav").resolve()))
            self.assertEqual(args.ref_text, "reference transcript")
            self.assertEqual(args.temperature, 0.6)
            self.assertEqual(args.repetition_penalty, 1.1)

    def test_load_preset_reads_toml_by_name(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            presets_dir = pathlib.Path(tmp) / "presets"
            presets_dir.mkdir()
            (presets_dir / "tutorial.toml").write_text(
                """
                [voice]
                ref_audio = "voices/tutorial/ref.wav"
                ref_text_file = "voices/tutorial/ref.txt"
                """,
                encoding="utf-8",
            )

            preset = module.load_preset("tutorial", presets_dir)

            self.assertEqual(preset["voice"]["ref_audio"], "voices/tutorial/ref.wav")

    def test_synthesize_text_passes_instruct_to_qwen3(self):
        module = load_module()

        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def generate_voice_clone(self, **kwargs):
                self.kwargs = kwargs
                return [np.zeros(10, dtype=np.float32)], 24000

        args = type(
            "Args",
            (),
            {
                "language": "Chinese",
                "ref_audio": "voice.wav",
                "ref_text": "reference transcript",
                "max_new_tokens": 128,
                "temperature": 0.6,
                "top_k": 50,
                "greedy": False,
                "repetition_penalty": 1.1,
                "xvec_only": True,
                "non_streaming_mode": False,
                "instruct": "用自然、清晰的教程讲解语气朗读。",
            },
        )()
        model = FakeModel()

        audio, sample_rate = module.synthesize_text(model, "测试文本", args)

        self.assertEqual(sample_rate, 24000)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(model.kwargs["text"], "测试文本")
        self.assertEqual(model.kwargs["instruct"], "用自然、清晰的教程讲解语气朗读。")

    def test_build_sentence_cues_groups_aligned_tokens_by_original_sentences(self):
        module = load_module()

        items = [
            {"text": "大", "start_time": 0.1, "end_time": 0.2},
            {"text": "家", "start_time": 0.2, "end_time": 0.3},
            {"text": "好", "start_time": 0.3, "end_time": 0.4},
            {"text": "今", "start_time": 1.0, "end_time": 1.1},
            {"text": "天", "start_time": 1.1, "end_time": 1.2},
        ]

        cues = module.build_sentence_cues(items, "大家好。今天？")

        self.assertEqual([cue.text for cue in cues], ["大家好", "今天"])
        self.assertEqual(cues[0].start, 0.1)
        self.assertEqual(cues[0].end, 0.4)
        self.assertEqual(cues[1].start, 1.0)
        self.assertEqual(cues[1].end, 1.2)

    def test_build_parser_has_only_qwen3_aligner_options(self):
        module = load_module()

        args = module.build_parser().parse_args(["--script", "script.txt", "--output-dir", "out"])

        self.assertFalse(hasattr(args, "backend"))
        self.assertFalse(hasattr(args, "index_model"))
        self.assertFalse(hasattr(args, "ref_audio"))
        self.assertEqual(args.preset, "tutorial")

    def test_dry_run_writes_block_and_full_outputs(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            script = pathlib.Path(tmp) / "script.txt"
            output_dir = pathlib.Path(tmp) / "out"
            script.write_text("[summary]\n大家好。今天介绍 vsummary。\n", encoding="utf-8")
            args = module.build_parser().parse_args(
                [
                    "--script",
                    str(script),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ]
            )
            args.model = "unused"
            args.ref_audio = "voice.wav"
            args.ref_text = "reference transcript"
            args.block_gap_seconds = 0.4
            args.dry_run_chars_per_second = 6.0

            blocks = module.parse_tagged_script(script.read_text(encoding="utf-8"))
            module.write_outputs(blocks, args)

            self.assertTrue((output_dir / "audio_blocks" / "001_summary.wav").exists())
            self.assertTrue((output_dir / "subtitle_blocks" / "001_summary.srt").exists())
            self.assertTrue((output_dir / "audio_full.wav").exists())
            self.assertTrue((output_dir / "SRT_FULL.srt").exists())
            self.assertTrue((output_dir / "blocks.json").exists())


if __name__ == "__main__":
    unittest.main()
