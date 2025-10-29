import json
import hashlib
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from falkordb import Graph
from typing import Iterator, Optional, Dict, Tuple
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.steps.qa_step import QAStep
from graphrag_sdk.steps.stream_qa_step import StreamingQAStep
from graphrag_sdk.model_config import KnowledgeGraphModelConfig
from graphrag_sdk.steps.graph_query_step import GraphQueryGenerationStep

CYPHER_ERROR_RES = "Sorry, I could not find the answer to your question"

class ChatSession:
    # Class-level ontology cache for all instances
    _ontology_cache: Dict[int, str] = {}
    _cache_lock = threading.Lock()
    _max_ontology_cache_size = 100
    """
    Represents a chat session with a Knowledge Graph.

    Args:
        model_config (KnowledgeGraphModelConfig): The model configuration to use.
        ontology (Ontology): The ontology to use.
        graph (Graph): The graph to query.

    Examples:
        >>> from graphrag_sdk import KnowledgeGraph, Orchestrator
        >>> from graphrag_sdk.ontology import Ontology
        >>> from graphrag_sdk.model_config import KnowledgeGraphModelConfig
        >>> model_config = KnowledgeGraphModelConfig.with_model(model)
        >>> kg = KnowledgeGraph("test_kg", model_config, ontology)
        >>> chat_session = kg.start_chat()
        >>> chat_session.send_message("What is the capital of France?")
    """

    def __init__(self, model_config: KnowledgeGraphModelConfig, ontology: Ontology, graph: Graph,
                cypher_system_instruction: str, qa_system_instruction: str,
                cypher_gen_prompt: str, qa_prompt: str, cypher_gen_prompt_history: str):
        """
        Initializes a new ChatSession object.

        Args:
            model_config (KnowledgeGraphModelConfig): The model configuration.
            ontology (Ontology): The ontology object.
            graph (Graph): The graph object.

        Attributes:
            model_config (KnowledgeGraphModelConfig): The model configuration.
            ontology (Ontology): The ontology object.
            graph (Graph): The graph object.
            cypher_chat_session (CypherChatSession): The Cypher chat session object.
            qa_chat_session (QAChatSession): The QA chat session object.
        """
        self.model_config = model_config
        self.graph = graph
        self.ontology = ontology
        
        # Query result caching
        self._query_cache: Dict[str, Tuple[str, float]] = {}
        self._cache_ttl = 3600  # 1 hour cache duration
        self._max_cache_size = 1000
        
        # Parallel execution
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Filter the ontology to remove unique and required attributes that are not needed for Q&A. 
        ontology_prompt = self.clean_ontology_for_prompt(ontology)
                
        cypher_system_instruction = cypher_system_instruction.format(ontology=ontology_prompt)
        
        self.cypher_prompt = cypher_gen_prompt
        self.qa_prompt = qa_prompt
        self.cypher_prompt_with_history = cypher_gen_prompt_history
        
        self.cypher_chat_session = model_config.cypher_generation.start_chat(
                cypher_system_instruction
            )
        self.qa_chat_session = model_config.qa.start_chat(
                qa_system_instruction
            )
        self.last_complete_response = {
            "question": None, 
            "response": None, 
            "context": None, 
            "cypher": None
            }
        
        # Metadata to store additional information about the chat session (currently only last query execution time)
        self.metadata = {"last_query_execution_time": None}
        
    def _get_cache_key(self, message: str, cypher: str) -> str:
        """Generate MD5 hash cache key from message and cypher"""
        content = f"{message}:{cypher}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cache entry is within TTL"""
        return time.time() - timestamp < self._cache_ttl
    
    def _get_cached_response(self, message: str, cypher: str) -> Optional[str]:
        """Retrieve cached response if valid"""
        cache_key = self._get_cache_key(message, cypher)
        if cache_key in self._query_cache:
            response, timestamp = self._query_cache[cache_key]
            if self._is_cache_valid(timestamp):
                return response
            else:
                del self._query_cache[cache_key]
        return None
    
    def _cache_response(self, message: str, cypher: str, response: str):
        """Cache the response with timestamp"""
        cache_key = self._get_cache_key(message, cypher)
        
        # Remove oldest entries if cache is full
        if len(self._query_cache) >= self._max_cache_size:
            oldest_key = min(self._query_cache.keys(), 
                           key=lambda k: self._query_cache[k][1])
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = (response, time.time())

    async def send_message_async(self, message: str) -> dict:
        """
        Async version of send_message.
        
        Args:
            message (str): The message to send.
            
        Returns:
            dict: The response dictionary.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.send_message, message)
    
    def _generate_cypher_optimized(self, message: str) -> tuple:
        """
        Optimized cypher generation with parallel validation.
        
        Args:
            message (str): The message to generate cypher for.
            
        Returns:
            tuple: A tuple containing (context, cypher)
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit cypher generation
            cypher_future = executor.submit(self._generate_cypher_query, message)
            
            # Wait for cypher generation
            context, cypher = cypher_future.result()
            
            return (context, cypher)

    def _generate_cypher_query(self, message: str) -> tuple:
        """
        Generate a Cypher query for the given message.
        
        Args:
            message (str): The message to generate a query for.
            
        Returns:
            tuple: A tuple containing (context, cypher)
        """
        cypher_step = GraphQueryGenerationStep(
            graph=self.graph,
            chat_session=self.cypher_chat_session,
            ontology=self.ontology,
            last_answer=self.last_complete_response["response"],
            cypher_prompt=self.cypher_prompt,
            cypher_prompt_with_history=self.cypher_prompt_with_history
        )

        (context, cypher, query_execution_time) = cypher_step.run(message)
        self.metadata["last_query_execution_time"] = query_execution_time
        
        return (context, cypher)

    def send_message(self, message: str) -> dict:
        """
        Sends a message to the chat session.

        Args:
            message (str): The message to send.

        Returns:
            dict: The response to the message in the following format:
                    {"question": message, 
                    "response": answer, 
                    "context": context, 
                    "cypher": cypher}
        """
        (context, cypher) = self._generate_cypher_query(message)

        # If the cypher is empty, return an error message
        if not cypher or len(cypher) == 0:
            self.last_complete_response = {
                "question": message,
                "response": CYPHER_ERROR_RES,
                "context": None,
                "cypher": None
            }
            return self.last_complete_response
        
        # Check cache for existing response
        cached_answer = self._get_cached_response(message, cypher)
        if cached_answer:
            self.last_complete_response = {
                "question": message, 
                "response": cached_answer, 
                "context": context, 
                "cypher": cypher
            }
            return self.last_complete_response
        
        qa_step = QAStep(
            chat_session=self.qa_chat_session,
            qa_prompt=self.qa_prompt,
        )

        answer = qa_step.run(message, cypher, context)
        
        # Cache the response
        self._cache_response(message, cypher, answer)

        self.last_complete_response = {
            "question": message, 
            "response": answer, 
            "context": context, 
            "cypher": cypher
        }
        
        return self.last_complete_response
    
    def send_message_stream(self, message: str) -> Iterator[str]:

        """
        Sends a message to the chat session and streams the response.

        Args:
            message (str): The message to send.

        Yields:
            str: Chunks of the response as they're generated.
        """
        (context, cypher) = self._generate_cypher_query(message)

        if not cypher or len(cypher) == 0:
            # Stream the error message for consistency with successful responses
            yield CYPHER_ERROR_RES
            
            self.last_complete_response = {
                "question": message,
                "response": CYPHER_ERROR_RES,
                "context": None,
                "cypher": None
            }
            return

        qa_step = StreamingQAStep(
            chat_session=self.qa_chat_session,
            qa_prompt=self.qa_prompt,
        )

        # Yield chunks of the response as they're generated
        for chunk in qa_step.run(message, cypher, context):
            yield chunk

        # Set the last answer using chat history to ensure we have the complete response
        self.last_complete_response = {
            "question": message, 
            "response": qa_step.chat_session.get_chat_history()[-1]['content'], 
            "context": context, 
            "cypher": cypher
        }
        
    def clean_ontology_for_prompt(self, ontology: Ontology) -> str:
        """
        Cached version of ontology cleaning that removes 'unique' and 'required' keys.

        Args:
            ontology (Ontology): The ontology to clean and transform.

        Returns:
            str: The cleaned ontology as a JSON string.
        """
        cache_key = hash(str(ontology.to_json()))
        
        with self._cache_lock:
            if cache_key not in self._ontology_cache:
                self._ontology_cache[cache_key] = self._process_ontology(ontology)
                
                # Manage cache size
                if len(self._ontology_cache) > self._max_ontology_cache_size:
                    # Remove oldest entries (simple strategy - remove first 20%)
                    keys_to_remove = list(self._ontology_cache.keys())[:self._max_ontology_cache_size // 5]
                    for key in keys_to_remove:
                        del self._ontology_cache[key]
            
            return self._ontology_cache[cache_key]
    
    def _process_ontology(self, ontology: Ontology) -> str:
        """
        Original ontology processing logic.

        Args:
            ontology (Ontology): The ontology to process.

        Returns:
            str: The processed ontology as JSON string.
        """
        # Convert the ontology object to a JSON.
        ontology_dict = ontology.to_json()
        
        # Remove unique and required attributes from the ontology.
        for entity in ontology_dict["entities"]:
            for attribute in entity["attributes"]:
                del attribute['unique']
                del attribute['required']
        
        for relation in ontology_dict["relations"]:
            for attribute in relation["attributes"]:
                del attribute['unique']
                del attribute['required']
        
        # Return the transformed ontology as a JSON string
        return json.dumps(ontology_dict)