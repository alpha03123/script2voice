@echo off
set "SCRIPT2VOICE_TTS_MODEL=E:\gittools\models\Qwen3-TTS-12Hz-1.7B-Base"
set "SCRIPT2VOICE_HF_HUB_CACHE=E:\gittools\models\hf_cache"
"E:\conda-envs\ai-cu128\python.exe" "E:\gittools\self\tagged-tts-blocks\tagged_script_to_qwen3_tts_blocks.py" %*
