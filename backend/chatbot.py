import os
from dotenv import load_dotenv
import google.generativeai as genai

# Charger les variables d'environnement depuis .env
load_dotenv()

# Also try loading from config/.env if not found
from pathlib import Path as EnvPath
config_env = EnvPath(__file__).parent.parent / "config" / ".env"
if config_env.exists():
    load_dotenv(config_env)

# Récupération de la clé API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("❌ Veuillez définir la variable d'environnement GEMINI_API_KEY.")
genai.configure(api_key=api_key)

# Configuration de génération
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

system_instruction = """
Q : Quels produits proposez-vous ?
R : Baby t-shirts, cardigans pour enfants, t-shirts classiques et travel bags.

Q : Les vêtements sont-ils adaptés aux bébés ?
R : Oui, tous nos vêtements sont conçus pour le confort des bébés.

Q : Les tissus sont-ils hypoallergéniques ?
R : Oui, nos tissus sont doux et hypoallergéniques.

Q : Les vêtements sont-ils certifiés sans produits chimiques nocifs ?
R : Oui, tous nos produits respectent les normes de sécurité pour enfants.

Q : Quelles tailles proposez-vous pour les t-shirts bébé ?
R : De 0–3 mois jusqu’à 6 ans.

Q : Et pour les cardigans ?
R : Les cardigans suivent les mêmes tailles que les t-shirts.

Q : Comment savoir si la taille est correcte ?
R : Consultez notre guide des tailles en cm sur chaque fiche produit.

Q : Les couleurs sont-elles fidèles aux images ?
R : Nous faisons de notre mieux, mais de légères variations peuvent apparaître selon l’écran.

Q : Puis-je choisir plusieurs couleurs pour un même produit ?
R : Oui, vous pouvez sélectionner la couleur souhaitée avant d’ajouter au panier.

Q : Les produits conviennent-ils aux garçons et aux filles ?
R : Oui, nos designs sont unisexes.

Q : Les t-shirts sont-ils extensibles ?
R : Oui, en coton élastique pour un confort optimal.

Q : Les cardigans sont-ils faciles à enfiler ?
R : Oui, avec boutons ou fermeture facile adaptés aux bébés.

Q : Les vêtements rétrécissent-ils au lavage ?
R : Nos vêtements sont pré-lavés pour minimiser le rétrécissement.

Q : Peut-on mettre les vêtements au sèche-linge ?
R : Nous recommandons un séchage à l’air libre pour préserver la qualité.

Q : Les t-shirts sont-ils imprimés ou brodés ?
R : Selon le modèle, certains sont imprimés et d’autres brodés.

Q : Les travel bags sont-ils adaptés aux voyages courts ?
R : Oui, ils peuvent contenir tous les essentiels pour une sortie ou voyage court.

Q : Peut-on mettre des bouteilles ou biberons dans le travel bag ?
R : Oui, il contient des compartiments pour biberons et accessoires.

Q : Les sacs sont-ils imperméables ?
R : Partiellement imperméables et résistants aux éclaboussures.

Q : Quels matériaux pour les travel bags ?
R : Nylon et polyester de haute qualité.

Q : Les sacs ont-ils des poches pour vêtements et couches ?
R : Oui, plusieurs compartiments sont disponibles.

Q : Les produits sont-ils lavables en machine ?
R : Oui, la plupart des vêtements peuvent être lavés à 30–40°C.

Q : Les t-shirts sont-ils légers pour l’été ?
R : Oui, conçus pour être respirants et confortables.

Q : Les cardigans sont-ils chauds pour l’hiver ?
R : Oui, nos cardigans sont en coton épais ou maille douce selon le modèle.

Q : Avez-vous des motifs ou dessins sur les vêtements ?
R : Oui, nous proposons des designs modernes et colorés pour enfants.

Q : Les travel bags sont-ils adaptés aux poussettes ?
R : Oui, certains modèles ont des attaches pour poussette.

Q : Les vêtements ont-ils des étiquettes scratch pour éviter les irritations ?
R : Oui, toutes les étiquettes sont conçues pour le confort des bébés.

Q : Les produits sont-ils fabriqués localement ?
R : Certains produits sont fabriqués localement, d’autres importés.

Q : Les t-shirts sont-ils à manches longues ou courtes ?
R : Nous proposons les deux modèles selon la saison.

Q : Puis-je acheter des ensembles complets ?
R : Oui, certains produits sont vendus en ensembles assortis.

Q : Les produits sont-ils adaptés aux peaux sensibles ?
R : Oui, tous les tissus sont doux et non irritants.

Q : Quels accessoires sont disponibles avec les sacs de voyage ?
R : Pochettes, compartiments séparés pour couches et accessoires.

Q : Les sacs sont-ils légers à porter ?
R : Oui, conçus pour être pratiques et ergonomiques.

Q : Peut-on commander plusieurs produits en un seul panier ?
R : Oui, ajoutez simplement tous les articles souhaités avant de passer commande.

Q : Les produits sont-ils unisexes ou genre spécifique ?
R : La plupart sont unisexes, certains modèles sont spécifiques garçon/fille.

Q : Avez-vous des promotions sur certains produits ?
R : Oui, consultez notre section “Promotions” pour les offres actuelles.

Q : Peut-on commander un produit épuisé ?
R : Non, il faudra attendre la prochaine réassort.

Q : Les produits sont-ils garantis ?
R : Oui, nous avons une garantie qualité sur défaut de fabrication.

Q : Y a-t-il des nouveautés régulièrement ?
R : Oui, nous ajoutons de nouveaux produits chaque saison.

Q : Les sacs sont-ils compatibles avec un usage quotidien ?
R : Oui, ils sont résistants et pratiques pour tous les jours.

Q : Les produits peuvent-ils se froisser facilement ?
R : Non, nos tissus sont conçus pour résister aux plis.


Q : Livrez-vous partout ?
R : Oui, livraison en Tunisie et à l’international selon la destination.

Q : Quels sont les délais de livraison ?
R : Tunisie : 2–4 jours ouvrables. International : 7–14 jours ouvrables.

Q : Quels sont les frais de livraison ?
R : Calculés automatiquement selon la destination et le poids.

Q : Puis-je suivre ma commande ?
R : Oui, un numéro de suivi est envoyé après expédition.

Q : Que faire si ma commande n’arrive pas ?
R : Contactez le service client avec votre numéro de commande.

Q : Peut-on changer l’adresse de livraison après commande ?
R : Oui, contactez le service client rapidement avant l’expédition.

Q : Livrez-vous le week-end ?
R : Normalement, nous livrons du lundi au samedi.

Q : Puis-je récupérer ma commande en magasin ?
R : Selon disponibilité, contactez-nous pour plus de détails.

Q : Les colis sont-ils emballés de manière sécurisée ?
R : Oui, nous utilisons des emballages adaptés pour protéger les produits.

Q : Les produits fragiles sont-ils protégés ?
R : Oui, tout est emballé pour éviter les dommages.


Q : Quels moyens de paiement acceptez-vous ?
R : Carte bancaire, paiement en ligne sécurisé et paiement à la livraison.

Q : Le paiement est-il sécurisé ?
R : Oui, toutes les transactions sont protégées.

Q : Puis-je payer en plusieurs fois ?
R : Selon la méthode de paiement, certaines options sont disponibles.

Q : Acceptez-vous PayPal ?
R : Oui, sur certaines commandes selon la région.

Q : Puis-je payer à la livraison ?
R : Oui, en Tunisie uniquement.

Q : Les paiements sont-ils confirmés immédiatement ?
R : Oui, dès la validation, vous recevez un email de confirmation.

Q : Que faire en cas d’échec du paiement ?
R : Vérifiez votre compte et réessayez ou contactez le service client.

Q : Puis-je modifier le moyen de paiement après commande ?
R : Non, il faut annuler la commande et en passer une nouvelle.

Q : Les informations bancaires sont-elles sécurisées ?
R : Oui, nous utilisons un protocole SSL pour toutes les transactions.

Q : Recevrai-je une facture pour ma commande ?
R : Oui, une facture est envoyée automatiquement par email.
Q : Puis-je retourner un produit ?
R : Oui, dans les 7 à 14 jours après réception.

Q : Le produit doit-il être neuf pour un retour ?
R : Oui, non porté, non lavé et dans son emballage d’origine.

Q : Qui paie les frais de retour ?
R : Le client, sauf en cas d’erreur de notre part.

Q : Comment effectuer un retour ?
R : Contactez le service client pour recevoir les instructions et l’étiquette de retour.

Q : Puis-je échanger un produit contre une autre taille ?
R : Oui, selon disponibilité.

Q : Puis-je échanger un produit contre un autre modèle ?
R : Oui, sous réserve de stock disponible.

Q : Combien de temps pour recevoir un échange ?
R : Environ 5 à 7 jours après réception du produit retourné.

Q : Les promotions sont-elles valables pour les échanges ?
R : Oui, tant que le produit échangé reste dans la période promotionnelle.

Q : Puis-je retourner un produit acheté en promotion ?
R : Oui, les conditions de retour sont les mêmes.

Q : Que faire si le produit reçu est endommagé ?
R : Contactez le service client immédiatement pour un remplacement gratuit.
Q : Quelle est la capacité d’un travel bag ?
R : Il peut contenir vêtements, couches, biberons et accessoires essentiels.

Q : Les sacs sont-ils imperméables ?
R : Oui, partiellement imperméables et résistants aux éclaboussures.

Q : Les sacs sont-ils légers ?
R : Oui, conçus pour être pratiques à porter.

Q : Les sacs ont-ils plusieurs compartiments ?
R : Oui, pour organiser vêtements, couches et accessoires.

Q : Peut-on attacher le sac à une poussette ?
R : Certains modèles sont équipés de fixations pour poussette.

Q : Les sacs peuvent-ils contenir des objets électroniques ?
R : Oui, ils peuvent contenir tablettes et accessoires légers.

Q : Les sacs sont-ils adaptés pour voyager en avion ?
R : Oui, ils sont conformes aux dimensions des bagages cabine pour bébé.

Q : Les sacs ont-ils des poches pour bouteilles et biberons ?
R : Oui, plusieurs poches sont prévues pour les biberons.

Q : Les sacs sont-ils faciles à nettoyer ?
R : Oui, ils se nettoient avec un chiffon humide ou un lavage doux.

Q : Les sacs sont-ils résistants à l’usure ?
R : Oui, conçus pour durer et résister à un usage quotidien.
Q : Comment contacter le service client ?
R : Par email, téléphone ou via le formulaire de contact sur le site.

Q : Quels sont les horaires du service client ?
R : Du lundi au samedi, de 9h à 18h.

Q : Combien de temps pour obtenir une réponse du service client ?
R : En général, sous 24 heures ouvrables.

Q : Puis-je poser une question sur un produit avant de commander ?
R : Oui, le service client est là pour vous conseiller.

Q : Puis-je demander un conseil pour choisir la taille ?
R : Oui, nous pouvons vous guider selon l’âge et les mensurations.

Q : Que faire si je n’ai pas reçu ma commande ?
R : Contactez le service client avec votre numéro de commande.

Q : Puis-je annuler une commande ?
R : Oui, avant l’expédition uniquement.

Q : Puis-je modifier une commande après validation ?
R : Non, il faut annuler et passer une nouvelle commande.

Q : Puis-je obtenir une facture ou reçu pour ma commande ?
R : Oui, envoyée automatiquement par email.

Q : Le service client peut-il m’aider pour un retour ?
R : Oui, nous guidons chaque étape pour un retour ou échange facile.
Q : Comment contacter le service client ?
R : Par email, téléphone ou via le formulaire de contact sur le site.

Q : Quels sont les horaires du service client ?
R : Du lundi au samedi, de 9h à 18h.

Q : Combien de temps pour obtenir une réponse du service client ?
R : En général, sous 24 heures ouvrables.

Q : Puis-je poser une question sur un produit avant de commander ?
R : Oui, le service client est là pour vous conseiller.

Q : Puis-je demander un conseil pour choisir la taille ?
R : Oui, nous pouvons vous guider selon l’âge et les mensurations.

Q : Que faire si je n’ai pas reçu ma commande ?
R : Contactez le service client avec votre numéro de commande.

Q : Puis-je annuler une commande ?
R : Oui, avant l’expédition uniquement.

Q : Puis-je modifier une commande après validation ?
R : Non, il faut annuler et passer une nouvelle commande.

Q : Puis-je obtenir une facture ou reçu pour ma commande ?
R : Oui, envoyée automatiquement par email.

Q : Le service client peut-il m’aider pour un retour ?
R : Oui, nous guidons chaque étape pour un retour ou échange facile.
"""


model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction,
    safety_settings={  
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE", 
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
    },
)
chat_session = model.start_chat(history=[
    {"role": "user", "parts": ["bonjour"]},
    {"role": "model", "parts": [
    "Bonjour 👋 ! Je suis là pour répondre à toutes vos questions sur nos produits et services.\nComment puis-je vous aider aujourd'hui ?"
    ]}
])
while True:
    user_input = input("Vous : ")
    if user_input.strip().lower() in ["quit", "exit", "q"]:
        print("👋 Merci pour votre participation. À bientôt !")
        break
    try:
        response = chat_session.send_message(user_input)
        print(f"Assistant AI: {response.text}\n")
    except Exception as e:
        print(" Une erreur est survenue :", str(e))

