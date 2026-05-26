# AI Content Moderation System

A complete **AI-powered content moderation suite** designed for a Home Beauty & Haircare Services Platform. This system ensures safe user-generated content by detecting toxic text, inappropriate images, and risky videos.

---

## 🎯 Project Overview

This project contains **three independent but integrated** FastAPI services for content safety:

- **Text Moderation** — Detects toxicity, hate speech, insults, threats, etc.
- **Image Moderation** — Detects explicit, nude, or inappropriate images.
- **Video Moderation** — Analyzes videos by extracting key frames and detecting risky content.

Built to protect users and maintain platform quality in a beauty services marketplace.

---

## ✨ Key Features

- **Multi-Modal Analysis**: Text, Image, and Video support
- **Advanced AI Models**:
  - Text: Detoxify (Unbiased) + NLTK + Google Translate
  - Image: NudeNet + YOLOv8x
  - Video: YOLOv8l + Frame Extraction
- **Performance Optimized**:
  - Redis caching (Text)
  - Rate limiting
  - Async processing
- **Smart Decision Making**: Returns `Allowed / Warning / Rejected` with confidence scores
- **Production Ready**: Proper error handling, logging, and temp file cleanup

---

## 🛠 Tech Stack

- **Framework**: FastAPI
- **AI Models**: Detoxify, NudeNet, Ultralytics YOLOv8
- **NLP**: NLTK, langdetect, googletrans
- **Others**: Redis, OpenCV, SlowAPI (Rate Limiting)

---

## 📁 Project Structure
