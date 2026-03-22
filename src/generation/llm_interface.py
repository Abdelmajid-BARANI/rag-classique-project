"""
LLM Interface Module
Interface pour interagir avec Llama 3.1:8b via Ollama
"""
import requests
import json
import time
from typing import List, Dict, Optional
from loguru import logger


MAX_RETRIES = 2
RETRY_DELAY = 2  # secondes


class OllamaLLM:
    """Classe pour interagir avec Ollama"""
    
    def __init__(self, model: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        """
        Initialise l'interface Ollama
        
        Args:
            model: Nom du modèle Ollama à utiliser
            host: URL de l'API Ollama
        """
        self.model = model
        self.host = host
        self.api_url = f"{host}/api/generate"
        self.chat_url = f"{host}/api/chat"
        self.is_connected = False
        
        # Vérifier que Ollama est accessible
        self._check_connection()
        logger.info(f"Interface Ollama initialisée: {model} @ {host}")
    
    def _check_connection(self):
        """Vérifie que le serveur Ollama est accessible"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                self.is_connected = True
                # Vérifier si le modèle est disponible
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if self.model not in model_names:
                    logger.warning(
                        f"Le modèle '{self.model}' n'est pas téléchargé. "
                        f"Modèles disponibles: {model_names}. "
                        f"Exécutez: ollama pull {self.model}"
                    )
                else:
                    logger.success(f"Connexion à Ollama établie, modèle '{self.model}' disponible")
            else:
                logger.warning(f"Ollama accessible mais code de réponse: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Impossible de se connecter à Ollama sur {self.host}. "
                "Assurez-vous qu'Ollama est installé et en cours d'exécution."
            )
        except Exception as e:
            logger.error(f"Erreur lors de la vérification Ollama: {e}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        timeout: int = 600
    ) -> str:
        """
        Génère une réponse avec le modèle
        
        Args:
            prompt: Prompt à envoyer au modèle
            temperature: Température de génération
            max_tokens: Nombre maximum de tokens
            stream: Activer le streaming
            timeout: Timeout en secondes (défaut: 600)
            
        Returns:
            Réponse générée par le modèle
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "keep_alive": "10m",
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            # Tentatives avec retry
            last_error = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = requests.post(self.api_url, json=payload, timeout=timeout)
                    response.raise_for_status()
                    result = response.json()
                    return result.get("response", "")
                except requests.exceptions.ConnectionError as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        logger.warning(f"Tentative {attempt + 1} échouée, nouvelle tentative dans {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                    continue
                except requests.exceptions.Timeout:
                    last_error = TimeoutError(f"Timeout Ollama ({timeout}s)")
                    if attempt < MAX_RETRIES:
                        logger.warning(f"Timeout ({timeout}s), tentative {attempt + 1}/{MAX_RETRIES + 1}...")
                        time.sleep(RETRY_DELAY)
                        continue
                    logger.error(f"Timeout lors de la requête à Ollama (>{timeout}s) après {MAX_RETRIES + 1} tentatives")
                    raise
            
            # Si toutes les tentatives échouent
            raise ConnectionError(f"Impossible de contacter Ollama après {MAX_RETRIES + 1} tentatives: {last_error}")
            
        except requests.exceptions.Timeout:
            raise
        except ConnectionError:
            raise
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context_chunks: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict:
        """
        Génère une réponse en utilisant des chunks de contexte (RAG)
        
        Args:
            query: Question de l'utilisateur
            context_chunks: Liste des chunks pertinents
            temperature: Température de génération
            max_tokens: Nombre maximum de tokens
            
        Returns:
            Dictionnaire contenant la réponse et les métadonnées
        """
        # Construire le contexte à partir des chunks
        context_text = "\n\n".join([
            f"[Document {i+1}]: {chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Construire le prompt
        prompt = self._build_rag_prompt(query, context_text)
        
        logger.info(f"Génération de réponse pour la requête: {query[:50]}...")
        
        # Générer la réponse
        response = self.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "query": query,
            "answer": response,
            "context_chunks": context_chunks,
            "num_chunks_used": len(context_chunks)
        }
    
    def _build_rag_prompt(self, query: str, context: str) -> str:
        """
        Construit un prompt RAG
        
        Args:
            query: Question
            context: Contexte récupéré
            
        Returns:
            Prompt formaté
        """
        prompt = f"""Tu es un assistant juridique spécialisé en droit fiscal français. Réponds UNIQUEMENT à partir des documents fournis ci-dessous.

RÈGLES STRICTES :
- Commence TOUJOURS par répondre directement à la question posée en une phrase courte.
- Ensuite, développe avec les détails et citations des documents sources (ex: "Selon [Document X], Article Y...").
- Si un article cité est ABROGÉ, signale-le clairement avec sa date d'abrogation. Si aucun article en vigueur ne le remplace dans le contexte, précise : « Cet article est abrogé et aucun texte de remplacement n'est présent dans les documents fournis. »
- Ne répète pas deux fois la même information. Si plusieurs documents disent la même chose, cite le plus pertinent une seule fois.
- Si l'information ne se trouve PAS dans le contexte, dis simplement : « L'information demandée n'est pas présente dans les documents fournis. »
- Ne refuse JAMAIS de répondre. Ce sont des documents juridiques publics.
- Sois complet et ne tronque jamais ta réponse.

CONTEXTE :
{context}

QUESTION : {query}

RÉPONSE :"""
        
        return prompt
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Interface de chat avec historique
        
        Args:
            messages: Liste de messages au format [{"role": "user/assistant", "content": "..."}]
            temperature: Température de génération
            
        Returns:
            Réponse du modèle
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            response = requests.post(self.chat_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except Exception as e:
            logger.error(f"Erreur lors du chat: {e}")
            raise


if __name__ == "__main__":
    # Test de l'interface
    llm = OllamaLLM()
    response = llm.generate("Bonjour, comment vas-tu?", temperature=0.7, max_tokens=100)
    print(f"Réponse: {response}")
