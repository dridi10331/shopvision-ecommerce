import os
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import List, Dict, Any

load_dotenv()

# Also try loading from config/.env if not found
from pathlib import Path as EnvPath
config_env = EnvPath(__file__).parent.parent / "config" / ".env"
if config_env.exists():
    load_dotenv(config_env)

class ProductMatcher:
    def __init__(self):
        """Initialise le matcher de produits"""
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(supabase_url, supabase_key)
    
    def find_matching_products(self, detected_classes: List[str], confidences: List[float] = None) -> List[Dict[str, Any]]:
        """
        Trouve les produits correspondants aux classes détectées
        
        Args:
            detected_classes: Liste des classes détectées par YOLO
            confidences: Scores de confiance correspondants
            
        Returns:
            Liste des produits correspondants avec scores
        """
        matching_products = []
        
        for i, detected_class in enumerate(detected_classes):
            confidence = confidences[i] if confidences is not None else 1.0
            
            # 1. Trouver le mapping pour cette classe
            mapping = self._get_class_mapping(detected_class, confidence)
            
            if mapping:
                # 2. Rechercher les produits dans cette catégorie
                products = self._search_products_by_category(
                    mapping['product_category'], 
                    mapping['keywords']
                )
                
                # 3. Ajouter le score de correspondance
                for product in products:
                    product['detection_confidence'] = confidence
                    product['detected_class'] = detected_class
                    product['match_score'] = self._calculate_match_score(
                        product, detected_class, confidence
                    )
                
                matching_products.extend(products)
        
        # Trier par score de correspondance
        matching_products.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Supprimer les doublons
        seen_ids = set()
        unique_products = []
        for product in matching_products:
            if product['id'] not in seen_ids:
                seen_ids.add(product['id'])
                unique_products.append(product)
        
        return unique_products
    
    def _get_class_mapping(self, detected_class: str, confidence: float) -> Dict[str, Any]:
        """Récupère le mapping pour une classe détectée"""
        try:
            normalized = detected_class.lower().replace("-", " ").replace("_", " ").strip()
            compact = normalized.replace(" ", "")
            tokens = [token for token in normalized.split(" ") if token]

            result = self.supabase.table('object_product_mapping').select("*").execute()
            mappings = result.data if result.data else []

            eligible_mappings = [m for m in mappings if m.get('confidence_threshold', 0) <= confidence]

            # 1) Correspondance directe (ex: "cardigan")
            for mapping in eligible_mappings:
                mapped = str(mapping.get('detected_class', '')).lower().strip()
                if mapped == detected_class.lower() or mapped == normalized or mapped.replace(" ", "") == compact:
                    return mapping

            # 2) Correspondance partielle via tokens (ex: "travel-bag" -> token "bag")
            scored_candidates = []
            for mapping in eligible_mappings:
                mapped = str(mapping.get('detected_class', '')).lower().replace("-", " ").replace("_", " ").strip()
                mapped_tokens = [token for token in mapped.split(" ") if token]

                score = 0
                for token in tokens:
                    if token in mapped_tokens or token == mapped or token in mapped:
                        score += 1

                if score > 0:
                    scored_candidates.append((score, mapping))

            if scored_candidates:
                scored_candidates.sort(key=lambda item: item[0], reverse=True)
                return scored_candidates[0][1]

            # 3) Fallback si confiance trop basse: essayer une correspondance
            # sans filtre de seuil pour ne pas perdre les produits de catégorie.
            scored_candidates = []
            for mapping in mappings:
                mapped = str(mapping.get('detected_class', '')).lower().replace("-", " ").replace("_", " ").strip()
                mapped_tokens = [token for token in mapped.split(" ") if token]

                score = 0
                if mapped == detected_class.lower() or mapped == normalized or mapped.replace(" ", "") == compact:
                    score += 3
                for token in tokens:
                    if token in mapped_tokens or token == mapped or token in mapped:
                        score += 1

                if score > 0:
                    scored_candidates.append((score, mapping))

            if scored_candidates:
                scored_candidates.sort(key=lambda item: item[0], reverse=True)
                return scored_candidates[0][1]

            return None
            
        except Exception as e:
            print(f"❌ Erreur mapping: {e}")
            return None
    
    def _search_products_by_category(self, category: str, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Recherche les produits par catégorie et mots-clés"""
        try:
            result = self.supabase.table('products').select("*").ilike('category', f'%{category}%').execute()
            products = result.data if result.data else []

            if not products or not keywords:
                return products

            # Ne pas filtrer strictement par keywords (sinon zéro résultat),
            # mais classer les produits de la même catégorie par pertinence.
            def keyword_score(product: Dict[str, Any]) -> int:
                searchable_text = f"{product.get('name', '')} {product.get('description', '')}".lower()
                return sum(1 for keyword in keywords if str(keyword).lower() in searchable_text)

            products.sort(key=keyword_score, reverse=True)
            return products
            
        except Exception as e:
            print(f"❌ Erreur recherche produits: {e}")
            return []
    
    def _calculate_match_score(self, product: Dict[str, Any], detected_class: str, confidence: float) -> float:
        """Calcule un score de correspondance pour un produit"""
        base_score = confidence
        
        # Bonus si le nom du produit contient la classe détectée
        if detected_class.lower() in product.get('name', '').lower():
            base_score += 0.2
        
        # Bonus si la description contient la classe détectée
        if detected_class.lower() in product.get('description', '').lower():
            base_score += 0.1
        
        return min(base_score, 1.0)  # Limiter à 1.0
    
    def save_search_result(self, session_id: str, detection_id: int, products: List[Dict[str, Any]]):
        """Sauvegarde les résultats de recherche"""
        try:
            search_data = {
                "session_id": session_id,
                "detection_id": detection_id,
                "found_products": [p['id'] for p in products],
                "match_scores": [p['match_score'] for p in products],
                "total_matches": len(products)
            }
            
            # Créer la table search_results si elle n'existe pas
            result = self.supabase.table('search_results').insert(search_data).execute()
            return result.data[0]['id']
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde recherche: {e}")
            return None

# Fonction utilitaire pour utilisation facile
def find_products_for_detection(detected_classes: List[str], confidences: List[float] = None) -> List[Dict[str, Any]]:
    """Fonction simple pour trouver des produits"""
    matcher = ProductMatcher()
    return matcher.find_matching_products(detected_classes, confidences)