from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("models/best.pt")

# Map class IDs to product categories (for e-commerce)
# Temporairement simplifié pour éviter la confusion
CLASS_TO_CATEGORY = {
    0: "Baby T-Shirt",        # Fusionner Baby T-Shirt et T-Shirts
    1: "Cardigan",       # Garder Cardigan
    2: "Travel Bag",     # Garder Travel Bag
    3: "T-Shirt",        # Fusionner avec classe 0
}

CLASS_TO_NAME = {
    0: 'Baby T-Shirt',
    1: 'Cardigan',
    2: 'Travel Bag', 
    3: 'T-Shirt',
}

print("🚀 Démarrage de la détection de produits...")
print("📷 Ouverture de la webcam...")
print("Appuyez sur 'q' pour quitter")

# Afficher les classes du modèle
print(f"\n📋 Classes détectées par le modèle:")
for class_id, class_name in model.names.items():
    print(f"  Classe {class_id}: {class_name}")

print(f"\n🎯 Mapping des classes:")
for class_id, category in CLASS_TO_CATEGORY.items():
    model_class = model.names.get(class_id, "INCONNU")
    print(f"  Classe {class_id}: {model_class} → {category}")

print("\n" + "="*50)

# Accéder à la webcam
cap = cv2.VideoCapture(0)

# Initialiser le timer
last_time = time.time()
interval = 2  # Intervalle en secondes pour afficher les résultats

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Impossible de lire la webcam")
        break
    
    # Faire la détection avec seuil modéré
    results = model(frame, conf=0.6)  # Seuil équilibré
    current_time = time.time()
    
    detected_products = []
    
    # Récupérer les informations de détection
    raw_detections = []  # Pour debug
    if results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls.cpu().numpy()[0])
            confidence = float(box.conf.cpu().numpy()[0])
            
            # Récupérer les coordonnées de la bounding box
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            
            raw_detections.append({
                'class_id': class_id,
                'class_name': model.names.get(class_id, 'unknown'),
                'confidence': confidence,
                'box_size': f"{int(box_width)}x{int(box_height)}",
                'area_percent': box_area / (frame.shape[0] * frame.shape[1])
            })
            
            # Filtrer les détections trop grandes (probablement des personnes)
            frame_area = frame.shape[0] * frame.shape[1]
            if box_area > frame_area * 0.3:  # Si la box fait plus de 30% de l'image
                print(f"⚠️ Détection ignorée (trop grande): {CLASS_TO_NAME.get(class_id)} - Area: {box_area/frame_area:.2%}")
                continue
            
            # Filtre spécial pour les cardigans - exiger une confiance plus élevée
            if class_id == 1 and confidence < 0.8:  # Cardigan = classe 1, seuil à 0.8
                print(f"⚠️ Cardigan ignoré (confiance trop faible): {confidence:.2f}")
                continue
            
            category = CLASS_TO_CATEGORY.get(class_id, f'class_{class_id}')
            name = CLASS_TO_NAME.get(class_id, f'class_{class_id}')
            
            detected_products.append({
                'name': name,
                'category': category,
                'confidence': round(confidence * 100, 1),
                'box_size': f"{int(box_width)}x{int(box_height)}"
            })
    
    # Afficher les résultats toutes les 2 secondes
    if current_time - last_time >= interval:
        print(f"\n📊 Détections à {current_time:.1f}s:")
        
        # Afficher toutes les détections brutes pour debug
        if raw_detections:
            print(f"� Détections brutes ({len(raw_detections)}):")
            for det in raw_detections:
                print(f"  - {det['class_name']} (ID:{det['class_id']}) - {det['confidence']:.2f} - {det['box_size']} - {det['area_percent']:.1%}")
        
        # Afficher les détections finales
        if detected_products:
            print(f"✅ Produits validés ({len(detected_products)}):")
            for product in detected_products:
                print(f"  - {product['name']} ({product['confidence']}% confiance) - Taille: {product['box_size']}")
        else:
            print("❌ Aucun produit détecté après filtrage")
        
        # Réinitialiser le timer
        last_time = current_time
    
    # Afficher la vidéo avec annotations
    annotated_frame = results[0].plot()
    
    # Ajouter overlay avec nombre de produits
    if detected_products:
        cv2.rectangle(annotated_frame, (10, 10), (400, 60), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Produits detectes: {len(detected_products)}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Detection de Produits - Appuyez sur 'q' pour quitter", annotated_frame)
    
    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Nettoyer
cap.release()
cv2.destroyAllWindows()
print("🏁 Détection terminée!")