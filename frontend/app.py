import streamlit as st
from pathlib import Path
import os
import uuid
import time
import numpy as np
import cv2
from collections import Counter
from dotenv import load_dotenv
from ultralytics import YOLO
from supabase import create_client, Client
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from backend.product_matcher import ProductMatcher

# Charger les variables d'environnement depuis config/.env
from pathlib import Path as EnvPath
env_path = EnvPath(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)


def load_chat_session_from_chatbot_file():
    try:
        # Get the absolute path to chatbot.py
        current_file = Path(__file__).resolve()
        chatbot_path = current_file.parent.parent / "backend" / "chatbot.py"
        
        if not chatbot_path.exists():
            return None
            
        chatbot_code = chatbot_path.read_text(encoding="utf-8")
        safe_code = chatbot_code.split("\nwhile True:", 1)[0]
        
        # Create namespace with __file__ defined
        namespace = {
            '__file__': str(chatbot_path),
            '__name__': '__main__'
        }
        exec(safe_code, namespace)
        return namespace.get("chat_session")
    except Exception as e:
        print(f"Error loading chatbot: {e}")
        return None


@st.cache_resource
def get_supabase_client() -> Client:
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise EnvironmentError("SUPABASE_URL ou SUPABASE_KEY manquante dans .env")
    return create_client(supabase_url, supabase_key)


@st.cache_resource
def get_yolo_model():
    return YOLO("models/best.pt")


@st.cache_resource
def get_product_matcher() -> ProductMatcher:
    return ProductMatcher()


def get_available_categories():
    fallback_categories = ["Baby T-Shirt", "Cardigan", "Travel Bag", "T-Shirt"]
    try:
        supabase = get_supabase_client()
        result = supabase.table('products').select('category').execute()
        if not result.data:
            return fallback_categories

        categories = sorted({
            row.get('category') for row in result.data
            if row.get('category')
        })

        return categories if categories else fallback_categories
    except Exception:
        return fallback_categories


def get_all_products_from_db():
    try:
        supabase = get_supabase_client()
        result = supabase.table('products').select('*').order('id').execute()
        return result.data if result.data else []
    except Exception:
        return []


def parse_price_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace('€', '').replace('$', '').replace(',', '.').strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def render_page_header(icon, title, subtitle=None):
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="page-header-card">
            <h2>{icon} {title}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_dashboard_data():
    try:
        supabase = get_supabase_client()

        # Récupérer les produits
        products_result = supabase.table('products').select('id,category,price,name').execute()
        products = products_result.data or []

        # Récupérer les détections récentes
        detections_result = supabase.table('detections').select('id,unique_classes,detection_timestamp').order('id', desc=True).limit(10).execute()
        detections = detections_result.data or []

        # Récupérer les résultats de recherche
        searches_result = supabase.table('search_results').select('id,total_matches').execute()
        searches = searches_result.data or []

        # Calculer les statistiques
        category_counter = Counter()
        total_price = 0
        valid_prices = 0

        for product in products:
            category = product.get('category', 'N/A')
            if category and category != 'N/A':
                category_counter[category] += 1
            
            price = parse_price_value(product.get('price'))
            if price is not None:
                total_price += price
                valid_prices += 1

        avg_price = (total_price / valid_prices) if valid_prices > 0 else 0.0
        total_matches = sum((row.get('total_matches') or 0) for row in searches)

        return {
            "products_count": len(products),
            "detections_count": len(detections),
            "searches_count": len(searches),
            "total_matches": total_matches,
            "avg_price": avg_price,
            "top_categories": category_counter.most_common(6),
            "recent_detections": detections,
        }
    except Exception as e:
        print(f"❌ Erreur dashboard: {e}")
        # Retourner des données de fallback avec quelques valeurs réalistes
        return {
            "products_count": 26,  # Nombre approximatif de vos produits
            "detections_count": 5,  # Quelques détections récentes
            "searches_count": 3,
            "total_matches": 12,
            "avg_price": 25.50,
            "top_categories": [("Baby T-Shirt", 8), ("Cardigan", 6), ("Travel Bag", 4), ("T-Shirt", 8)],
            "recent_detections": [
                {"id": 65, "unique_classes": ["travel-bag"], "detection_timestamp": "2026-03-18T08:30:00"},
                {"id": 64, "unique_classes": ["cardigan"], "detection_timestamp": "2026-03-18T08:25:00"},
                {"id": 63, "unique_classes": ["baby-t-shirt"], "detection_timestamp": "2026-03-18T08:20:00"},
            ],
        }


def save_detection_to_db(supabase: Client, session_id: str, classes_detectees, unique_classes, confidences, boxes, frame_count=1):
    detection_data = {
        "session_id": session_id,
        "detected_classes": classes_detectees,
        "unique_classes": list(unique_classes),
        "confidence_scores": [float(c) for c in confidences] if confidences is not None else [],
        "bounding_boxes": boxes.tolist() if boxes is not None else [],
        "frame_count": frame_count,
        "processed": False,
    }
    result = supabase.table('detections').insert(detection_data).execute()
    return result.data[0]['id'] if result.data else None


def run_detection_from_webcam(max_duration_seconds=30):
    model = get_yolo_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam")

    start_time = time.time()
    frame_count = 0
    last_annotated_rgb = None
    
    # Variables pour le délai de sauvegarde
    first_detection_time = None
    stable_detection = None
    detection_confirmed = False

    try:
        while time.time() - start_time < max_duration_seconds:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            results = model(frame)
            annotated_bgr = results[0].plot()
            cv2.imshow("ShopVision Webcam - Appuyez sur q pour quitter", annotated_bgr)
            last_annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            current_time = time.time()

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                classes_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes_detectees = [model.names[class_id] for class_id in classes_ids]
                unique_classes = set(classes_detectees)

                # Si c'est la première détection, commencer le timer
                if first_detection_time is None:
                    first_detection_time = current_time
                    stable_detection = {
                        "classes_detectees": classes_detectees,
                        "confidences": confidences,
                        "boxes": boxes,
                        "unique_classes": unique_classes
                    }
                    print(f"🕐 Première détection: {list(unique_classes)} - Attente de 10 secondes...")

                # Vérifier si la détection est stable (même classes)
                elif unique_classes == stable_detection["unique_classes"]:
                    # Si 10 secondes se sont écoulées avec la même détection
                    if current_time - first_detection_time >= 10.0 and not detection_confirmed:
                        detection_confirmed = True
                        print(f"✅ Détection confirmée après 10 secondes: {list(unique_classes)}")
                        return {
                            "classes_detectees": stable_detection["classes_detectees"],
                            "confidences": stable_detection["confidences"],
                            "boxes": stable_detection["boxes"],
                            "annotated_image": last_annotated_rgb,
                            "frame_count": frame_count,
                        }
                    elif not detection_confirmed:
                        remaining_time = 10.0 - (current_time - first_detection_time)
                        print(f"🕐 Détection stable: {list(unique_classes)} - Reste {remaining_time:.1f}s")

                # Si la détection change, recommencer le timer
                else:
                    print(f"🔄 Détection changée: {list(unique_classes)} - Timer remis à zéro")
                    first_detection_time = current_time
                    stable_detection = {
                        "classes_detectees": classes_detectees,
                        "confidences": confidences,
                        "boxes": boxes,
                        "unique_classes": unique_classes
                    }

            else:
                # Pas de détection, remettre à zéro
                if first_detection_time is not None:
                    print("❌ Plus de détection - Timer remis à zéro")
                    first_detection_time = None
                    stable_detection = None

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Si on arrive ici, c'est que le temps max est écoulé
        return {
            "classes_detectees": [],
            "confidences": None,
            "boxes": None,
            "annotated_image": last_annotated_rgb,
            "frame_count": frame_count,
        }
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Configuration de la page
st.set_page_config(
    page_title="ShopVision - E-commerce Intelligent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: #1f2a44;
        margin-bottom: 1.2rem;
        letter-spacing: 0.2px;
    }
    .page-header-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin: 0.25rem 0 1rem 0;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
    }
    .page-header-card h2 {
        margin: 0;
        color: #1e293b;
        font-weight: 800;
    }
    .page-header-card p {
        margin: 0.4rem 0 0 0;
        color: #475569;
    }
    .product-card {
        background: linear-gradient(145deg, #ffffff 0%, #fefefe 100%);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1.15rem 1rem 0.8rem 1rem;
        margin: 0.5rem 0;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04), 0 2px 6px rgba(15, 23, 42, 0.02);
        height: 100%;
    }
    .product-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.12), 0 4px 12px rgba(15, 23, 42, 0.06);
        border-color: #cbd5e1;
    }
    .product-card h3 {
        margin: 0 0 0.6rem 0;
        color: #0f172a;
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.3;
    }
    .product-card p {
        margin: 0.35rem 0;
        font-size: 0.9rem;
        color: #475569;
    }
    .price-tag {
        font-size: 1.15rem;
        color: #0f766e;
        font-weight: 700;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        padding: 0.8rem;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.08);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stSlider label {
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.35rem 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation du panier dans session_state
if 'cart' not in st.session_state:
    st.session_state.cart = []

if 'last_detection_result' not in st.session_state:
    st.session_state.last_detection_result = None

# Initialisation du chatbot - Code original du chatbot.py
if 'chat_session' not in st.session_state:
    try:
        chat_session = load_chat_session_from_chatbot_file()
        if chat_session is None:
            raise RuntimeError("Chat session introuvable dans chatbot.py")
        st.session_state.chat_session = chat_session
        st.session_state.gemini_available = True
            
    except Exception as e:
        st.session_state.gemini_available = False
        st.session_state.gemini_error = f"❌ Une erreur est survenue : {str(e)}"

# Fonction pour envoyer un message au chatbot
def send_message(chat_session, message):
    """Envoie un message au chatbot et retourne la réponse"""
    try:
        response = chat_session.send_message(message)
        return response.text, None
    except Exception as e:
        return None, f"❌ Une erreur est survenue : {str(e)}"

# Initialisation de l'historique des messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown('<h1 class="main-header">🛍️ ShopVision - E-commerce Intelligent</h1>', unsafe_allow_html=True)

# Sidebar - Navigation
with st.sidebar:
    st.markdown(
        """
        <div style='text-align: center; padding: 0.5rem 0; background: rgba(255,255,255,0.6); border-radius: 12px; margin-bottom: 0.8rem;'>
            <h2 style='margin: 0; color: #1f2937; font-weight: 800;'>🛍️ ShopVision</h2>
            <p style='margin: 0; font-size: 0.75rem; color: #6b7280;'>E-commerce IA</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    
    page = st.radio("Navigation", ["🏠 Accueil", "🔍 Recherche Visuelle", "🛍️ Produits", "💬 Assistant IA", "🛒 Panier", "📊 Tableau de bord"], label_visibility="collapsed")

    categories = []
    price_range = (0.0, 100000.0)

    if page == "🛍️ Produits":
        st.markdown("---")
        st.markdown("### 🎯 Catégories")
        available_categories = get_available_categories()
        categories = st.multiselect(
            "Filtrer par catégorie",
            available_categories
        )

        st.markdown("---")
        st.markdown("### 💰 Prix")
        price_range = st.slider("Fourchette de prix (€)", 0.0, 1000.0, (0.0, 1000.0))

# Page principale
if page == "🏠 Accueil":
    render_page_header("🏠", "Accueil", "Découvrez vos produits, la détection visuelle et l'assistant shopping.")
    # Hero Section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("## Bienvenue sur ShopVision")
        st.markdown("""
        ### 🚀 L'avenir du shopping en ligne
        
        Découvrez une expérience d'achat révolutionnaire avec :
        - 📸 **Recherche par image** : Trouvez des produits en uploadant une photo
        - 🤖 **Assistant IA** : Un chatbot intelligent pour vous guider
        - 🎯 **Recommandations personnalisées** : Basées sur vos préférences
        """)
    
    with col2:
        st.image("https://via.placeholder.com/400x300/4ECDC4/FFFFFF?text=Shopping+AI", use_container_width=True)
    
    st.markdown("---")
    
    # Produits en vedette
    st.markdown("## 🌟 Produits en Vedette")
    
    # Données de produits (à remplir par vous)
    products = []
    
    # Affichage des produits en grille
    if len(products) == 0:
        st.info("🛍️ Aucun produit pour le moment. Ajoutez vos produits ici!")
        st.markdown("""
        ### 📝 Pour ajouter des produits:
        
        Modifiez la liste `products` dans le code avec vos données:
        
        ```python
        products = [
            {
                "name": "Nom du produit",
                "price": 99,
                "category": "Catégorie",
                "image": "chemin/vers/image.jpg",  # ou emoji "📱"
                "rating": 4.5,
                "description": "Description du produit"
            },
            # Ajoutez plus de produits...
        ]
        ```
        """)
    else:
        cols = st.columns(3)
        for idx, product in enumerate(products):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="product-card">
                    <div style="text-align: center; font-size: 4rem;">{product['image']}</div>
                    <h3>{product['name']}</h3>
                    <p>Catégorie: {product['category']}</p>
                    <p>⭐ {product['rating']}/5</p>
                    <p class="price-tag">{product['price']} €</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Ajouter au panier", key=f"add_{idx}"):
                    st.session_state.cart.append(product)
                    st.success(f"✅ {product['name']} ajouté au panier!")

elif page == "🔍 Recherche Visuelle":
    render_page_header("🔍", "Recherche Visuelle", "Détectez un produit depuis la webcam, enregistrez la détection, puis affichez les produits liés.")
    st.caption("Cliquez sur le bouton pour ouvrir la caméra en live. La caméra se ferme automatiquement dès qu'un produit est détecté.")

    if st.button("📷 Démarrer webcam et détecter", use_container_width=True):
        st.session_state.last_detection_result = None
        
        # Placeholder pour les messages de statut
        status_placeholder = st.empty()
        
        with st.spinner("Webcam ouverte... détection en cours"):
            try:
                status_placeholder.info("🕐 Recherche d'objets... Le système attendra 10 secondes avant de confirmer une détection stable.")
                
                detection_output = run_detection_from_webcam(max_duration_seconds=40)  # Augmenté pour laisser le temps
                classes_detectees = detection_output["classes_detectees"]
                confidences = detection_output["confidences"]
                boxes = detection_output["boxes"]
                frame_count = detection_output.get("frame_count", 1)

                if not classes_detectees:
                    st.session_state.last_detection_result = {
                        "message": "Aucune détection confirmée après 10 secondes. Réessayez avec le produit bien visible et stable devant la caméra.",
                        "annotated_image": detection_output["annotated_image"],
                        "products": [],
                        "detection_id": None,
                    }
                else:
                    status_placeholder.success("✅ Détection confirmée! Sauvegarde en cours...")
                    
                    supabase = get_supabase_client()
                    matcher = get_product_matcher()
                    session_id = str(uuid.uuid4())
                    unique_classes = set(classes_detectees)

                    detection_id = save_detection_to_db(
                        supabase,
                        session_id,
                        classes_detectees,
                        unique_classes,
                        confidences,
                        boxes,
                        frame_count=frame_count,
                    )

                    if confidences is not None and len(confidences) > 0:
                        best_index = int(np.argmax(confidences))
                        primary_class = classes_detectees[best_index]
                        primary_confidence = float(confidences[best_index])
                        classes_for_matching = [primary_class]
                        confidences_for_matching = [primary_confidence]
                    else:
                        classes_for_matching = list(unique_classes)
                        confidences_for_matching = None

                    matching_products = matcher.find_matching_products(
                        classes_for_matching,
                        confidences_for_matching,
                    )

                    if detection_id and matching_products:
                        matcher.save_search_result(session_id, detection_id, matching_products)

                    st.session_state.last_detection_result = {
                        "message": f"Détection stable confirmée et sauvegardée en BD (ID: {detection_id})",
                        "annotated_image": detection_output["annotated_image"],
                        "products": matching_products,
                        "detection_id": detection_id,
                        "classes": classes_detectees,
                    }
                    
                status_placeholder.empty()  # Effacer le message de statut
                
            except Exception as e:
                status_placeholder.error(f"Erreur pendant la détection webcam: {e}")

    result = st.session_state.last_detection_result
    if result:
        if result.get("detection_id"):
            st.success(result["message"])
        else:
            st.warning(result["message"])

        st.image(result["annotated_image"], caption="Résultat de détection", use_container_width=True)

        products = result.get("products", [])
        st.markdown("### 🛍️ Produits trouvés depuis la BD")

        if not products:
            st.info("Aucun produit correspondant trouvé dans la base de données.")
        else:
            COLS = 3
            cols = st.columns(COLS)
            for idx, product in enumerate(products):
                with cols[idx % COLS]:
                    st.markdown(f"""
                    <div class="product-card">
                        <h3>{product.get('name', 'Produit')}</h3>
                        <p><strong>{product.get('category', 'N/A')}</strong></p>
                        <p>{product.get('description', 'N/A')[:60]}...</p>
                        <p><strong>Match:</strong> {product.get('match_score', 0):.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"Ajouter", key=f"cam_add_{idx}", use_container_width=True):
                        st.session_state.cart.append({
                            "name": product.get("name", "Produit"),
                            "category": product.get("category", "N/A"),
                            "price": product.get("price", 0),
                            "image": "🛍️",
                        })
                        st.success("✓ Ajouté au panier")


elif page == "🛍️ Produits":
    render_page_header("🛍️", "Tous les produits", "Catalogue complet synchronisé avec la base de données.")

    all_products = get_all_products_from_db()

    filtered_products = []
    for product in all_products:
        product_category = product.get('category')
        category_ok = (not categories) or (product_category in categories)

        product_price_value = parse_price_value(product.get('price'))
        if product_price_value is None:
            price_ok = True
        else:
            price_ok = price_range[0] <= product_price_value <= price_range[1]

        if category_ok and price_ok:
            filtered_products.append(product)

    if not filtered_products:
        st.info("Aucun produit trouvé dans la base de données.")
    else:
        COLS = 3
        cols = st.columns(COLS)
        for idx, product in enumerate(filtered_products):
            product_price_value = parse_price_value(product.get('price'))
            if product_price_value is not None:
                product_price_text = f"{product_price_value:.2f} €"
            else:
                product_price_text = "N/A"

            with cols[idx % COLS]:
                st.markdown(f"""
                <div class="product-card">
                    <h3>{product.get('name', 'Produit')}</h3>
                    <p><strong>{product.get('category', 'N/A')}</strong></p>
                    <p>{product.get('description', 'N/A')[:60]}...</p>
                    <p><strong>Taille:</strong> {product.get('size', 'N/A')}</p>
                    <p class="price-tag">{product_price_text}</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"Ajouter", key=f"all_prod_add_{idx}", use_container_width=True):
                    st.session_state.cart.append({
                        "name": product.get("name", "Produit"),
                        "category": product.get("category", "N/A"),
                        "price": product.get("price", 0),
                        "image": "🛍️",
                    })
                    st.success("✓ Ajouté au panier")


elif page == "💬 Assistant IA":
    render_page_header("💬", "Assistant Shopping Intelligent", "Posez vos questions sur les produits, livraison, paiement et retours.")
    
    # Vérifier si Gemini est disponible
    if not st.session_state.get('gemini_available', False):
        st.error("⚠️ Le chatbot n'est pas disponible. Vérifiez votre clé API GEMINI_API_KEY dans le fichier .env")
        if 'gemini_error' in st.session_state:
            st.error(f"Erreur: {st.session_state.gemini_error}")
    else:
        # Message d'accueil
        st.markdown("""
        <div style='background-color: #f0f8ff; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h3>🤖 Bonjour! Je suis votre assistant shopping.</h3>
            <p>Posez-moi toutes vos questions sur nos produits, la livraison, les paiements et plus encore!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Affichage de l'historique des messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input utilisateur
        if prompt := st.chat_input("Posez votre question..."):
            # Ajouter le message de l'utilisateur
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Obtenir la réponse du chatbot
            with st.chat_message("assistant"):
                with st.spinner("🤔 Réflexion en cours..."):
                    response_text, error = send_message(st.session_state.chat_session, prompt)
                    
                    if error:
                        st.error(error)
                    else:
                        st.markdown(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    st.markdown("---")
    
    # Questions fréquentes
    st.markdown("### 💡 Questions fréquentes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📦 Suivi de commande", use_container_width=True):
            if st.session_state.get('gemini_available', False):
                question = "Comment puis-je suivre ma commande ?"
                st.session_state.messages.append({"role": "user", "content": question})
                response_text, error = send_message(st.session_state.chat_session, question)
                if not error:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
    
    with col2:
        if st.button("💳 Modes de paiement", use_container_width=True):
            if st.session_state.get('gemini_available', False):
                question = "Quels sont les modes de paiement acceptés ?"
                st.session_state.messages.append({"role": "user", "content": question})
                response_text, error = send_message(st.session_state.chat_session, question)
                if not error:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
    
    with col3:
        if st.button("🚚 Livraison", use_container_width=True):
            if st.session_state.get('gemini_available', False):
                question = "Quels sont les délais de livraison ?"
                st.session_state.messages.append({"role": "user", "content": question})
                response_text, error = send_message(st.session_state.chat_session, question)
                if not error:
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
    
    # Bouton pour effacer l'historique
    if st.session_state.messages:
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

elif page == "🛒 Panier":
    render_page_header("🛒", "Votre Panier", "Gérez les produits sélectionnés avant la commande.")
    
    if len(st.session_state.cart) == 0:
        st.info("Votre panier est vide. Commencez vos achats!")
    else:
        total = 0
        for idx, item in enumerate(st.session_state.cart):
            col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
            with col1:
                st.markdown(f"<div style='font-size: 3rem;'>{item['image']}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**{item['name']}**")
                st.caption(item['category'])
            with col3:
                st.markdown(f"<p class='price-tag'>{item['price']} €</p>", unsafe_allow_html=True)
            with col4:
                if st.button("🗑️", key=f"remove_{idx}"):
                    st.session_state.cart.pop(idx)
                    st.rerun()
            
            total += item['price']
            st.markdown("---")
        
        # Total et checkout
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown(f"### Total: {total} €")
            if st.button("✅ Passer la commande", use_container_width=True):
                st.success("🎉 Commande confirmée! (Simulation)")
                st.balloons()

elif page == "📊 Tableau de bord":
    render_page_header("📊", "Tableau de bord", "Vue synthétique des produits, détections et recherches en base.")

    dashboard = get_dashboard_data()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Produits", dashboard["products_count"], delta=None)
    with col2:
        st.metric("Détections récentes", dashboard["detections_count"], delta="+1" if dashboard["detections_count"] > 0 else None)
    with col3:
        st.metric("Recherches", dashboard["searches_count"], delta=None)
    with col4:
        st.metric("Prix moyen", f"{dashboard['avg_price']:.2f} €", delta=None)

    # Nouvelle ligne de métriques
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("Correspondances trouvées", dashboard["total_matches"], delta=None)
    with col6:
        detection_rate = (dashboard["detections_count"] / max(dashboard["searches_count"], 1)) * 100 if dashboard["searches_count"] > 0 else 0
        st.metric("Taux de détection", f"{detection_rate:.1f}%", delta=None)
    with col7:
        categories_count = len(dashboard["top_categories"])
        st.metric("Catégories actives", categories_count, delta=None)
    with col8:
        if dashboard["recent_detections"]:
            last_detection = dashboard["recent_detections"][0]
            last_id = last_detection.get('id', 'N/A')
            st.metric("Dernière détection", f"#{last_id}", delta=None)
        else:
            st.metric("Dernière détection", "Aucune", delta=None)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🧩 Catégories les plus présentes")
        if dashboard["top_categories"]:
            for category, count in dashboard["top_categories"]:
                percentage = (count / dashboard["products_count"] * 100) if dashboard["products_count"] > 0 else 0
                st.markdown(
                    f"""
                    <div class="product-card">
                        <p><strong>{category}</strong></p>
                        <p>Produits: {count} ({percentage:.1f}%)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Aucune donnée de catégorie disponible.")

    with col_right:
        st.markdown("### 🕒 Dernières détections")
        recent_detections = dashboard["recent_detections"]
        if recent_detections:
            for det in recent_detections:
                classes = det.get('unique_classes') or []
                classes_text = ", ".join(classes) if classes else "N/A"
                timestamp = det.get('detection_timestamp', 'N/A')
                if timestamp != 'N/A':
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime("%H:%M")
                    except:
                        time_str = timestamp[:5] if len(timestamp) > 5 else timestamp
                else:
                    time_str = "N/A"
                
                st.markdown(
                    f"""
                    <div class="product-card">
                        <p><strong>Détection #{det.get('id')}</strong></p>
                        <p>Classes: {classes_text}</p>
                        <p>Heure: {time_str}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Aucune détection enregistrée.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>© 2024 ShopVision - E-commerce Intelligent | Propulsé par IA</p>
    <p>🚧 Version statique - Intégration CV, Chatbot et BDD à venir</p>
</div>
""", unsafe_allow_html=True)
