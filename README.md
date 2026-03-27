---
title: ShopVision
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: test_app.py
pinned: false
---

# ShopVision - E-commerce with YOLOv12 Object Detection

E-commerce platform combining YOLOv12 object detection, product matching, and conversational AI chatbot.

## Features

- Real-time object detection via webcam (4 classes: Baby T-Shirt, Cardigan, Travel Bag, T-Shirt)
- Product matching with Supabase database
- Gemini-powered chatbot for customer support
- Interactive shopping cart and product catalog
- Analytics dashboard with detection metrics

## Tech Stack

- Streamlit (Frontend)
- YOLOv12s (9.3M parameters, 4 classes)
- Google Gemini API (Chatbot)
- Supabase (Database)
- OpenCV (Computer Vision)

## Model Performance

- Training: Tesla T4 GPU, 50 epochs, 4 hours
- Inference: 130ms avg on Intel Core i7-12700H CPU (7-8 FPS)
- Deployment: CPU-optimized for Hugging Face Spaces
