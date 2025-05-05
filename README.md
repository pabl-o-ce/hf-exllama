---
title: Exllama
emoji: 😽
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
header: mini
fullWidth: true
license: apache-2.0
short_description: 'Chat: exllama v2'
---

# Exllama Chat 😽

[![Open In Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/pabloce/exllama)
[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A Gradio-based chat interface for ExLlamaV2, featuring Mistral-7B-Instruct-v0.3 and Llama-3-70B-Instruct models. Experience high-performance inference on consumer GPUs with Flash Attention support.

## 🌟 Features

- 🚀 Powered by ExLlamaV2 inference library
- 💨 Flash Attention support for optimized performance
- 🎯 Supports multiple instruction-tuned models:
  - Mistral-7B-Instruct v0.3
  - Meta's Llama-3-70B-Instruct
- ⚡ Dynamic text generation with adjustable parameters
- 🎨 Clean, modern UI with dark mode support

## 🎮 Parameters

Customize your chat experience with these adjustable parameters:

- **System Message**: Set the AI assistant's behavior and context
- **Max Tokens**: Control response length (1-4096)
- **Temperature**: Adjust response creativity (0.1-4.0)
- **Top-p**: Fine-tune response diversity (0.1-1.0)
- **Top-k**: Control vocabulary sampling (0-100)
- **Repetition Penalty**: Prevent repetitive text (0.0-2.0)

## 🛠️ Technical Details

- **Framework**: Gradio 5.5.0
- **Models**: ExLlamaV2-compatible models
- **UI**: Custom-themed interface with Gradio's Soft theme
- **Optimization**: Flash Attention for improved performance

## 🔗 Links

- [Try it on Hugging Face Spaces](https://huggingface.co/spaces/pabloce/exllama)
- [ExLlamaV2 GitHub Repository](https://github.com/turboderp/exllamav2)
- [Join our Discord](https://discord.gg/gmVgCk6X2x)

## 📝 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ExLlamaV2](https://github.com/turboderp/exllamav2) for the core inference library
- [Hugging Face](https://huggingface.co/) for hosting and model distribution
- [Gradio](https://gradio.app/) for the web interface framework

---

Made with ❤️ using ExLlamaV2 and Gradio
