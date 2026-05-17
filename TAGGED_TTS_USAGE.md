# Tagged TTS Blocks 使用说明

这个工具把带语义标签的口播稿转换成模块化的 TTS 音频和字幕。

适合流程：

```text
口播稿
→ 按 [tag] 切成模块
→ 每个模块内按自然句生成 TTS
→ 每个模块输出一对 wav + srt
→ 额外输出全局 audio_full.wav + SRT_FULL.srt
```

## 输入格式

`[tag]` 必须单独一行，作为切割标记。

`(画面建议)` 必须单独一行，作为剪辑备注，不进入 TTS，不进入 SRT，只写入 `blocks.json`。

示例：

```text
[summary]
(画面：展示项目首页和 Logo)
大家好，今天介绍一个本地 AI 视频知识管理工具。

[settings]
(画面：打开控制中心，展示 API Base URL、模型名称、API Key)
第一次使用时，进入左侧的控制中心进行配置。
```

不会误切这些内容：

```text
正文里出现 [summary] 不会触发切割。
请打开设置（Settings）页面不会被当成画面建议。
函数调用 foo(bar) 也不会被忽略。
```

只有整行是 `[xxx]` 或 `(xxx)` 才生效。

## 输出结构

```text
output/
  audio_blocks/
    001_summary.wav
    002_settings.wav

  subtitle_blocks/
    001_summary.srt
    002_settings.srt

  audio_full.wav
  SRT_FULL.srt
  blocks.json
```

含义：

```text
audio_blocks/*.wav       每个 [tag] 的独立音频
subtitle_blocks/*.srt    每个 [tag] 的独立字幕，时间从 00:00:00,000 开始
audio_full.wav           所有 block 拼接后的完整音频
SRT_FULL.srt             所有 block 字幕按全局时间偏移合并后的完整字幕
blocks.json              模块、画面建议、句子时间、文件路径和时长元数据
```

## 对齐规则

工具不是用文本长度猜字幕时间，而是：

```text
模块内文本
→ 按自然句切分
→ 每句单独生成临时 wav
→ 用每句 wav 的真实时长生成 SRT
→ 拼成模块 wav
```

所以：

```text
audio_blocks/001_xxx.wav 对齐 subtitle_blocks/001_xxx.srt
audio_full.wav 对齐 SRT_FULL.srt
```

不要混用：

```text
audio_full.wav 不要配局部 001_xxx.srt
001_xxx.wav 不要配 SRT_FULL.srt
```

默认静音：

```text
句子之间 0.2 秒
模块之间 0.4 秒
```

这两个静音都会被写进对应 SRT 时间轴，所以不会破坏对齐。

## 常用命令

先 dry-run 检查脚本切分，不加载 TTS 模型：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd `
  --script "E:\video_process\videos\5月16日\vsummary_script.txt" `
  --output-dir "E:\video_process\videos\5月16日\vsummary_tts_blocks" `
  --dry-run
```

正式生成：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd `
  --script "E:\video_process\videos\5月16日\vsummary_script.txt" `
  --output-dir "E:\video_process\videos\5月16日\vsummary_tts_blocks" `
  --device cuda `
  --dtype bf16 `
  --language Chinese
```

使用 IndexTTS2 后端生成：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts-indextts2.cmd `
  --script "E:\video_process\videos\5月16日\vsummary_script.txt" `
  --output-dir "E:\video_process\videos\5月16日\vsummary_indextts2_blocks"
```

IndexTTS2 默认使用实测可用的质量配置：

```text
--index-fp16
--index-diffusion-steps 25
--index-cfg-rate 0.7
```

也可以显式写出来：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts-indextts2.cmd `
  --script "script.txt" `
  --output-dir "out" `
  --index-fp16 `
  --index-diffusion-steps 25 `
  --index-cfg-rate 0.7
```

不建议正式口播使用 `--index-diffusion-steps 8 --index-cfg-rate 0`，速度更快但质量明显下降。

调整字幕最大字符数：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd --script "script.txt" --output-dir "out" --max-caption-chars 28
```

## 语气提示词

默认会给 TTS 传入教程讲解风格提示词：

```text
请用自然、清晰、稳定的中文教程讲解语气朗读。语速适中，情绪克制但有亲和力，重点术语读清楚，不要使用夸张、播音腔或过度营销的语气。
```

临时改成产品介绍风格：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd `
  --script "script.txt" `
  --output-dir "out" `
  --instruct "请用清晰、可信、不过度营销的中文产品介绍语气朗读。语速适中，重点功能读清楚。"
```

关闭提示词：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd --script "script.txt" --output-dir "out" --instruct ""
```

选择音色预设：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd --script "script.txt" --output-dir "out" --voice-preset default
```

临时覆盖参考音频和文本：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd `
  --script "script.txt" `
  --output-dir "out" `
  --ref-audio "E:\voices\my_voice.wav" `
  --ref-text "这段参考音频里实际说的话"
```

调整静音：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd --script "script.txt" --output-dir "out" --sentence-gap-seconds 0.15 --block-gap-seconds 0.5
```

保留每句临时 wav，用于排查：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd --script "script.txt" --output-dir "out" --keep-tmp
```

查看帮助：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd --help
```

## 注册到 PATH

当前已经提供命令包装器：

```text
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd
E:\gittools\self\tagged-tts-blocks\tagged-tts-indextts2.cmd
```

临时加入当前 PowerShell 会话：

```powershell
$env:Path = "E:\gittools\self\tagged-tts-blocks;$env:Path"
tagged-tts --help
```

永久加入用户 PATH：

```powershell
[Environment]::SetEnvironmentVariable(
  "Path",
  "E:\gittools\self\tagged-tts-blocks;" + [Environment]::GetEnvironmentVariable("Path", "User"),
  "User"
)
```

永久修改后，需要重开 PowerShell 才能直接使用：

```powershell
tagged-tts --help
```

## 依赖环境

包装器默认使用：

```text
E:\conda-envs\ai-cu128\python.exe
```

TTS 依赖：

```text
faster-qwen3-tts 已安装到 ai-cu128 虚拟环境
```

默认模型：

```text
E:\gittools\models\Qwen3-TTS-12Hz-1.7B-Base
```

默认参考音频：

```text
E:\gittools\self\tagged-tts-blocks\voice_presets.json
```

默认预设名：

```text
default
```

## 添加音色预设

编辑：

```text
E:\gittools\self\tagged-tts-blocks\voice_presets.json
```

格式：

```json
{
  "default": {
    "ref_audio": "assets/ref_audio.wav",
    "ref_text": "参考音频里实际说的话"
  },
  "my_voice": {
    "ref_audio": "E:/voices/my_voice.wav",
    "ref_text": "我的参考音频里实际说的话"
  }
}
```

使用：

```powershell
E:\gittools\self\tagged-tts-blocks\tagged-tts.cmd --script "script.txt" --output-dir "out" --voice-preset my_voice
```

