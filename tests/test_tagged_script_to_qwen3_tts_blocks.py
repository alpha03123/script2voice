import importlib.util
import json
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

    def test_srt_for_block_uses_local_zero_based_timeline(self):
        module = load_module()

        sentences = [
            module.SentenceAudio(text="第一句。", duration=1.5),
            module.SentenceAudio(text="第二句。", duration=2.5),
        ]
        srt = module.make_local_srt(sentences, gap_seconds=0.0)

        self.assertIn("00:00:00,000 -->", srt)
        self.assertIn("--> 00:00:04,000", srt)
        self.assertIn("第一句", srt)
        self.assertIn("第二句", srt)

    def test_srt_captions_do_not_end_with_punctuation(self):
        module = load_module()

        sentences = [
            module.SentenceAudio(text="这是字幕，但是后面还有内容。", duration=4.0),
        ]
        srt = module.make_local_srt(sentences, gap_seconds=0.0)
        caption_lines = [
            line
            for line in srt.splitlines()
            if line and not line.isdigit() and "-->" not in line
        ]

        self.assertTrue(caption_lines)
        self.assertTrue(all(not line.endswith(("，", ",", "。", "；", ";", "、")) for line in caption_lines))

    def test_srt_captions_do_not_emit_tiny_or_punctuation_only_lines(self):
        module = load_module()

        text = "需要说明的是，如果你使用外部大语言模型供应商，AI 总结和问答时会把必要的文本内容发送给你配置的模型接口。"
        sentences = [
            module.SentenceAudio(text=text, duration=8.0),
        ]
        srt = module.make_local_srt(sentences, gap_seconds=0.0)
        caption_lines = [
            line
            for line in srt.splitlines()
            if line and not line.isdigit() and "-->" not in line
        ]

        self.assertNotIn("AI", caption_lines)
        self.assertNotIn("，", caption_lines)
        self.assertNotIn(",", caption_lines)
        self.assertTrue(all(len(line) >= 4 for line in caption_lines[:-1]))

    def test_srt_keeps_long_sentence_complete(self):
        module = load_module()

        text = "你是不是也经常这样：网盘里存了几个 T 的硬核教程视频，什么 UE5 全套教程、Python 精讲、AI 绘画课、剪辑课、编程课，收藏夹越来越满"
        sentences = [
            module.SentenceAudio(text=text, duration=10.0),
        ]

        srt = module.make_local_srt(sentences, gap_seconds=0.0, max_caption_chars=24)
        caption_lines = [
            line
            for line in srt.splitlines()
            if line and not line.isdigit() and "-->" not in line
        ]

        self.assertEqual(caption_lines, [text])
        self.assertIn("--> 00:00:10,000", srt)

    def test_srt_keeps_colon_sentence_complete(self):
        module = load_module()

        text = "这就是它最核心的能力：把视频变成可以对话的知识资产。"
        sentences = [
            module.SentenceAudio(text=text, duration=5.0),
        ]

        srt = module.make_local_srt(sentences, gap_seconds=0.0, max_caption_chars=24)
        caption_lines = [
            line
            for line in srt.splitlines()
            if line and not line.isdigit() and "-->" not in line
        ]

        self.assertEqual(caption_lines, ["这就是它最核心的能力：把视频变成可以对话的知识资产"])

    def test_split_sentences_does_not_split_by_caption_length(self):
        module = load_module()

        text = "语音转写部分使用的是 faster-whisper，推荐选择 large-v3-turbo，兼顾速度和识别质量。"

        sentences = module.split_sentences(text, max_chars=24)

        self.assertEqual(sentences, ["语音转写部分使用的是 faster-whisper，推荐选择 large-v3-turbo，兼顾速度和识别质量"])

    def test_clean_caption_removes_leading_punctuation(self):
        module = load_module()

        self.assertEqual(module.clean_caption("，就可以进入单视频对话"), "就可以进入单视频对话")
        self.assertEqual(module.clean_caption("”、“哪里讲到了通道处理"), "哪里讲到了通道处理")
        self.assertEqual(module.clean_caption("”、“帮我总结一下材质系统的学习路线"), "帮我总结一下材质系统的学习路线")

    def test_offset_srt_times(self):
        module = load_module()

        srt = "1\n00:00:00,000 --> 00:00:01,000\n第一句\n"

        shifted = module.offset_srt_times(srt, 2.5)

        self.assertIn("00:00:02,500 --> 00:00:03,500", shifted)

    def test_apply_voice_preset_sets_ref_audio_and_text(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            presets_path = pathlib.Path(tmp) / "voices.json"
            presets_path.write_text(
                json.dumps(
                    {
                        "default": {
                            "ref_audio": "voice.wav",
                            "ref_text": "reference transcript",
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "voice_presets": str(presets_path),
                    "voice_preset": "default",
                    "ref_audio": None,
                    "ref_text": None,
                },
            )()

            module.apply_voice_preset(args)

            self.assertEqual(args.ref_audio, str((presets_path.parent / "voice.wav").resolve()))
            self.assertEqual(args.ref_text, "reference transcript")

    def test_manual_ref_audio_overrides_voice_preset_audio_only(self):
        module = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            presets_path = pathlib.Path(tmp) / "voices.json"
            presets_path.write_text(
                json.dumps(
                    {
                        "default": {
                            "ref_audio": "voice.wav",
                            "ref_text": "reference transcript",
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = type(
                "Args",
                (),
                {
                    "voice_presets": str(presets_path),
                    "voice_preset": "default",
                    "ref_audio": "manual.wav",
                    "ref_text": None,
                },
            )()

            module.apply_voice_preset(args)

            self.assertEqual(args.ref_audio, "manual.wav")
            self.assertEqual(args.ref_text, "reference transcript")

    def test_synthesize_text_passes_instruct_to_tts(self):
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

        module.synthesize_text(model, "测试文本", args)

        self.assertEqual(model.kwargs["instruct"], "用自然、清晰的教程讲解语气朗读。")

    def test_synthesize_text_supports_indextts2_backend(self):
        module = load_module()

        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def infer(self, **kwargs):
                self.kwargs = kwargs
                return (22050, np.zeros((10, 1), dtype=np.int16))

        args = type(
            "Args",
            (),
            {
                "backend": "indextts2",
                "ref_audio": "voice.wav",
                "hf_hub_cache": r"E:\gittools\models\hf_cache",
                "index_emo_vector": "0,0,0,0,0,0,0,0",
                "index_emo_alpha": 1.0,
                "index_use_random": False,
                "index_diffusion_steps": 25,
                "index_cfg_rate": 0.7,
            },
        )()
        model = FakeModel()

        audio, sample_rate = module.synthesize_text(model, "测试文本", args)

        self.assertEqual(sample_rate, 22050)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(model.kwargs["spk_audio_prompt"], "voice.wav")
        self.assertEqual(model.kwargs["diffusion_steps"], 25)
        self.assertEqual(model.kwargs["inference_cfg_rate"], 0.7)

    def test_build_parser_exposes_indextts2_backend(self):
        module = load_module()

        args = module.build_parser().parse_args(
            [
                "--script",
                "script.txt",
                "--output-dir",
                "out",
                "--backend",
                "indextts2",
            ]
        )

        self.assertEqual(args.backend, "indextts2")
        self.assertTrue(args.index_fp16)
        self.assertEqual(args.hf_hub_cache, r"E:\gittools\models\hf_cache")
        self.assertEqual(args.index_diffusion_steps, 25)
        self.assertEqual(args.index_cfg_rate, 0.7)


if __name__ == "__main__":
    unittest.main()
