# 🛍️ ShopVision

Computer vision system for visual product search using real-time object detection and conversational interaction.

## 🚀 Demo

👉 **Live Application:** [shopvision-ecommerce.streamlit.app](https://shopvision-ecommerce.streamlit.app/)

*(Add GIF or video demo here)*

---

## 🧠 How it works

**Pipeline:**

1. User uploads image or uses webcam (local only)
2. YOLOv12s detects products in frame
3. Detection is stabilized over multiple frames (10s window for webcam)
4. Detected class is matched to products using confidence + fuzzy similarity
5. Results displayed in Streamlit UI with product recommendations
6. Optional: Gemini API handles user queries via chatbot

---

## ⚙️ Key Features

- **Real-time detection** (~7–8 FPS on CPU, ~130ms inference)
- **Temporal filtering** to reduce false positives (10-second stability window)
- **Product matching** via confidence thresholds + fuzzy similarity
- **Image upload support** for cloud deployment
- **Conversational interface** using Gemini API
- **Shopping cart & catalog** with database integration
- **Analytics dashboard** with detection metrics

---

## 🧪 Technical Details

| Component | Technology |
|-----------|-----------|
| **Model** | YOLOv12s (9.3M parameters) |
| **Dataset** | 1,000+ labeled images (4 categories) |
| **Classes** | Baby T-Shirt, Cardigan, Travel Bag, T-Shirt |
| **Training** | Tesla T4 GPU, 50 epochs, 4 hours |
| **Backend** | Python, OpenCV, Ultralytics |
| **Database** | Supabase (PostgreSQL) |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Cloud |

**Performance:**
- Inference: 130ms avg on Intel Core i7-12700H CPU
- Throughput: 7-8 FPS
- Detection confidence threshold: 0.5

---

## 📦 Setup

```bash
# Clone repository
git clone https://github.com/dridi10331/shopvision-ecommerce.git
cd shopvision-ecommerce

# Install dependencies
pip install -r requirements.txt

# Configure environment (create config/.env)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
GEMINI_API_KEY=your_gemini_api_key

# Run application
streamlit run app_main.py
```

**Cloud Deployment:**
1. Deploy to [Streamlit Cloud](https://share.streamlit.io)
2. Add secrets in dashboard (SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY)

---

## 📁 Project Structure

```
shopvision-ecommerce/
├── app_main.py              # Main application
├── backend/
│   ├── chatbot.py          # Gemini integration
│   └── product_matcher.py  # Matching algorithm
├── models/
│   └── best.pt             # YOLOv12 weights
└── config/
    └── .env                # Environment variables
```

---

## 📌 Limitations & Future Work

**Current Limitations:**
- Webcam detection only works locally (not on cloud)
- Limited to 4 product categories
- Fuzzy matching may produce false positives

**Planned Improvements:**
- Replace fuzzy matching with embeddings (CLIP/DINO)
- Improve FPS using ONNX optimization
- Scale to 20+ product categories
- Add multi-language support
- Implement user authentication

---

## 👤 Author

**Ahmed Omar**  
GitHub: [@dridi10331](https://github.com/dridi10331)

---

## 📄 License

MIT License
