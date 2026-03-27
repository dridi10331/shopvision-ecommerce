from ultralytics import YOLO
import cv2
import time
import os
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client
from product_matcher import ProductMatcher

# Charger les variables d'environnement
load_dotenv()

# Also try loading from config/.env if not found
from pathlib import Path as EnvPath
config_env = EnvPath(__file__).parent.parent / "config" / ".env"
if config_env.exists():
    load_dotenv(config_env)

# Initialiser Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise EnvironmentError("❌ Veuillez définir SUPABASE_URL et SUPABASE_KEY dans le fichier .env")
supabase: Client = create_client(supabase_url, supabase_key)

# Générer un ID de session unique
session_id = str(uuid.uuid4())

# Initialiser le matcher de produits
product_matcher = ProductMatcher()
# Fonction pour sauvegarder les détections
def save_detection_to_db(session_id, classes_detectees, unique_classes, confidences, boxes, frame_count):
    """Sauvegarde les détections dans la base de données"""
    try:
        # Préparer les données pour la BD
        detection_data = {
            "session_id": session_id,
            "detected_classes": classes_detectees,
            "unique_classes": list(unique_classes),
            "confidence_scores": confidences.tolist() if confidences is not None else [],
            "bounding_boxes": boxes.tolist() if boxes is not None else [],
            "frame_count": frame_count,
            "processed": False
        }
        
        # Insérer dans Supabase
        result = supabase.table('detections').insert(detection_data).execute()
        print(f"✅ Détection sauvegardée: ID {result.data[0]['id']}")
        return result.data[0]['id']
        
    except Exception as e:
        print(f"❌ Erreur sauvegarde BD: {e}")
        return None

# Charger le modèle
model = YOLO("models/best.pt")
# Accéder à la webcam
cap = cv2.VideoCapture(0)
# Initialiser le timer et compteur de frames
last_time = time.time()
interval = 1  # Intervalle en secondes
frame_count = 0

print(f"🚀 Session de détection démarrée: {session_id}")
print("Appuyez sur 'q' pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    results = model(frame)
    current_time = time.time()
    
    classes_detectees = []
    confidences = None
    boxes = None
    
    # Récupérer les informations de détection
    if results[0].boxes is not None:
        classes_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes_detectees = [model.names[class_id] for class_id in classes_ids]
    
    # Vérifier l'intervalle de temps
    if current_time - last_time >= interval:
        unique_classes = set(classes_detectees)
        
        print(f"\n📊 Détections à {current_time:.1f}s (Frame {frame_count}):")
        print("Toutes les classes:", classes_detectees)
        print("Classes uniques:", list(unique_classes))
        
        # Sauvegarder en base de données si des objets sont détectés
        if classes_detectees:
            detection_id = save_detection_to_db(
                session_id, 
                classes_detectees, 
                unique_classes, 
                confidences, 
                boxes, 
                frame_count
            )
            print(f"💾 Sauvegardé avec ID: {detection_id}")
            
            # 🔍 Rechercher les produits correspondants
            matching_products = product_matcher.find_matching_products(
                list(unique_classes), 
                confidences
            )
            
            if matching_products:
                print(f"🛍️ {len(matching_products)} produits trouvés:")
                for product in matching_products[:3]:  # Afficher top 3
                    print(f"  - {product['name']} (Score: {product['match_score']:.2f})")
                
                # Sauvegarder les résultats de recherche
                product_matcher.save_search_result(session_id, detection_id, matching_products)
            else:
                print("❌ Aucun produit correspondant trouvé")
        
        # Réinitialiser le timer
        last_time = current_time
    
    # Affichage vidéo
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv11 sur webcam", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"🏁 Session terminée: {session_id}")