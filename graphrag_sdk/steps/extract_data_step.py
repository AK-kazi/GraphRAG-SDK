import os
import time
import json
import logging
import gc
import psutil
from tqdm import tqdm
from uuid import uuid4
from falkordb import Graph
from threading import Lock
from typing import Optional, List, Dict, Any, Iterator, Tuple
from dataclasses import dataclass
from graphrag_sdk.steps.Step import Step
from graphrag_sdk.document import Document
from ratelimit import limits, sleep_and_retry
from graphrag_sdk.source import AbstractSource
from concurrent.futures import Future, ThreadPoolExecutor
from graphrag_sdk.helpers import extract_json, map_dict_to_cypher_properties
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.models import (
    GenerativeModel,
    GenerativeModelChatSession,
    GenerationResponse,
    FinishReason,
)
from graphrag_sdk.fixtures.prompts import (
    EXTRACT_DATA_SYSTEM,
    EXTRACT_DATA_PROMPT,
    FIX_JSON_PROMPT,
    COMPLETE_DATA_EXTRACTION,
)

RENDER_STEP_SIZE = 0.5

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class BatchOperation:
    operation_type: str  # 'entity' or 'relation'
    data: Dict[str, Any]
    timestamp: float

@dataclass
class MemoryConfig:
    max_memory_mb: int = 2048
    memory_check_interval: int = 10
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    chunk_size_reduction: float = 0.5  # Reduce chunk size by 50% under memory pressure


class ExtractDataStep(Step):
    """
    Extract Data Step
    """

    def __init__(
        self,
        sources: list[AbstractSource],
        ontology: Ontology,
        model: GenerativeModel,
        graph: Graph,
        config: Optional[dict] = None,
        hide_progress: Optional[bool] = False,
        enable_batch_operations: bool = False,
        batch_size: int = 1000,
        enable_memory_monitoring: bool = False,
        memory_config: Optional[MemoryConfig] = None,
    ) -> None:
        """
        Initialize the ExtractDataStep.
        
        Args:
            sources (list[AbstractSource]): List of data sources to process.
            ontology (Ontology): The ontology associated with the knowledge graph.
            model (GenerativeModel): The generative model used for data extraction.
            graph (Graph): The FalkorDB graph instance.
            config (Optional[dict]): Configuration options for the step.
            hide_progress (Optional[bool]): Flag to hide progress bar. Defaults to False.
            enable_batch_operations (bool): Enable batch entity/relation creation.
            batch_size (int): Size of batches for entity/relation creation.
        """
        self.sources = sources
        self.ontology = ontology
        self.config = config
        if config is None:
            self.config = {
                "max_workers": 16,
                "max_input_tokens": 500000,
                "max_output_tokens": 8192,
            }
        else:
            self.config = config
        self.model = model
        self.graph = graph
        self.hide_progress = hide_progress
        self.enable_batch_operations = enable_batch_operations
        self.batch_size = batch_size
        self.enable_memory_monitoring = enable_memory_monitoring
        self.memory_config = memory_config or MemoryConfig()        self.process_files = 0
        self.counter_lock = Lock()
        
        # Batch buffers
        self.entity_batch = []
        self.relation_batch = []
        
        # Performance tracking
        self.created_entities = 0
        self.created_relations = 0
        self.processed_count = 0
        self.memory_warnings = 0        
        if not os.path.exists("logs"):
            os.makedirs("logs")

    def _create_chat(self) -> GenerativeModelChatSession:
        return self.model.start_chat(EXTRACT_DATA_SYSTEM.replace("#ONTOLOGY", str(self.ontology.to_json())))

    def run(self, instructions: Optional[str] = None):
        """
        Run the data extraction process.
        
        Args:
            instructions (Optional[str]): Optional additional instructions for data extraction.
        """
        if self.enable_memory_monitoring:
            return self._run_memory_aware(instructions)
        else:
            return self._run_traditional(instructions)

    def _run_memory_aware(self, instructions: Optional[str] = None):
        """Memory-aware version of run method"""
        print(f"Starting memory-aware extraction with max {self.memory_config.max_memory_mb} MB")
        
        try:
            # Process documents in streaming fashion
            documents_processed = 0
            
            with tqdm(desc="Processing Documents", disable=self.hide_progress) as pbar:
                if self.enable_memory_monitoring:
                    for document, source_instruction in self._get_documents_stream():
                        try:
                            self._process_document_with_memory_check(document, source_instruction, instructions)
                            documents_processed += 1
                            pbar.update(1)
                            
                        except Exception as e:
                            print(f"Error processing document: {e}")
                            continue
                else:
                    # Fallback to traditional processing
                    return self._run_traditional(instructions)
            
            # Flush any remaining batches
            if self.enable_batch_operations:
                self.flush_all_batches()
            
            print(f"Processing completed. Documents: {documents_processed}, "
                  f"Entities: {self.created_entities}, Relations: {self.created_relations}")
            print(f"Memory warnings encountered: {self.memory_warnings}")
            
            return []
            
        except Exception as e:
            print(f"Fatal error in extraction: {e}")
            # Try to save any pending data
            if self.enable_batch_operations:
                self.flush_all_batches()
            raise

    def _run_traditional(self, instructions: Optional[str] = None):
        """Traditional processing method for backward compatibility"""
        # Each task is represented by a tuple containing:
        #   1. A Future object (the asynchronous processing task)
        #   2. A string (the ID of the document being processed)
        tasks: list[tuple[Future, str]] = []
        
        # Collect documents from all sources
        documents = [
            (document, source.instruction)
            for source in self.sources
            for document in source.load()
            if document.not_empty()
            ]
        
        with tqdm(total=len(documents), desc="Process Documents", disable=self.hide_progress) as pbar:
            with ThreadPoolExecutor(max_workers=self.config["max_workers"]) as executor:
                
                # Concurrency document processing
                for document, instructions in documents:
                    task_id = "extract_data_step_" + str(uuid4())
                    task = executor.submit(
                        self._process_document,
                        task_id,
                        self._create_chat(),
                        document,
                        self.ontology,
                        self.graph,
                        instructions,
                    )
                    tasks.append((task, document.id))
                    
                # Wait for all tasks to be completed
                while any(task[0].running() or not task[0].done() for task in tasks):
                    time.sleep(RENDER_STEP_SIZE)
                    with self.counter_lock:
                        pbar.n = self.process_files
                    pbar.refresh()

        # Collect failed documents
        failed_documents = [task[1] for task in tasks if task[0].exception()]
        
        # Flush any remaining batches if batch operations are enabled
        if self.enable_batch_operations:
            self.flush_all_batches()
            print(f"Batch operations completed. Entities created: {self.created_entities}, Relations created: {self.created_relations}")
        
        return failed_documents

    def _process_document_with_memory_check(self, document: Document, source_instruction: str, instructions: Optional[str] = None):
        """Process document with memory monitoring"""
        try:
            # Check memory before processing
            if self.enable_memory_monitoring and self._check_memory_usage():
                self._handle_memory_pressure()
            
            # Process the document using existing logic
            task_id = "extract_data_step_" + str(uuid4())
            self._process_document(
                task_id,
                self._create_chat(),
                document,
                self.ontology,
                self.graph,
                source_instruction,
                instructions,
            )
            self.processed_count += 1
            
            # Periodic cleanup
            if self.processed_count % 50 == 0:
                gc.collect()
            
        except MemoryError:
            print(f"Memory error processing document. Implementing emergency cleanup...")
            self._emergency_memory_cleanup()
            raise

    def _process_document(
        self,
        task_id: str,
        chat_session: GenerativeModelChatSession,
        document: Document,
        ontology: Ontology,
        graph: Graph,
        source_instructions: Optional[str] = "",
        instructions: Optional[str] = "",
        retries: Optional[int] = 1,
    ):
        try:
            """
            Process a single source document and extract entities and relations.
            
            Args:
                task_id (str): The unique ID for the task.
                chat_session (GenerativeModelChatSession): The chat session for the extraction.
                document (Document): The document to process.
                ontology (Ontology): The ontology associated with the graph.
                graph (Graph): The FalkorDB graph instance.
                source_instructions (Optional[str]): Instructions specific to the source.
                instructions (Optional[str]): Additional instructions.
                retries (Optional[int]): Number of times to retry if the model stops unexpectedly.
            """
            _task_logger = logging.getLogger(task_id)
            _task_logger.setLevel(logging.DEBUG)

            fh = logging.FileHandler(f"logs/{task_id}.log")
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
            )
            fh.setLevel(logging.DEBUG)

            _task_logger.addHandler(fh)

            logger.debug(f"Processing task: {task_id}")
            _task_logger.debug(f"Processing task: {task_id}")
                            
            text = document.content[: self.config["max_input_tokens"]]
            user_message = EXTRACT_DATA_PROMPT.format(
                text=text,
                instructions="\n".join(
                    [
                        source_instructions if source_instructions is not None else "",
                        instructions if instructions is not None else "",
                    ]
                ),
                max_tokens=self.config["max_output_tokens"],
                ontology=str(ontology.to_json()),
            )

            _task_logger.debug("User message: " + user_message.replace("\n", " "))

            responses: list[GenerationResponse] = []
            response_idx = 0

            responses.append(self._call_model(chat_session, user_message))

            _task_logger.debug(f"Model response: {responses[response_idx].text}")

            while responses[response_idx].finish_reason == FinishReason.MAX_TOKENS and response_idx < retries:
                _task_logger.debug("Asking model to continue")
                response_idx += 1
                responses.append(self._call_model(chat_session, COMPLETE_DATA_EXTRACTION))
                _task_logger.debug(
                    f"Model response after continue: {responses[response_idx].text}"
                )

            if responses[response_idx].finish_reason != FinishReason.STOP:
                _task_logger.debug(
                    f"Model stopped unexpectedly: {responses[response_idx].finish_reason}"
                )
                raise Exception(
                    f"Model stopped unexpectedly: {responses[response_idx].finish_reason}"
                )

            # Full json response is in the last response
            last_respond = responses[-1].text

            try:
                data = json.loads(extract_json(last_respond))
            except Exception as e:
                _task_logger.debug(f"Error extracting JSON: {e}")
                _task_logger.debug(f"Prompting model to fix JSON")
                json_fix_response = self._call_model(
                    self._create_chat(),
                    FIX_JSON_PROMPT.format(json=last_respond, error=str(e)),
                )
                data = json.loads(extract_json(json_fix_response.text))
                _task_logger.debug(f"Fixed JSON: {data}")

            if "entities" not in data or "relations" not in data:
                _task_logger.debug(
                    f"Invalid data format. Missing entities or relations. {data}"
                )
                raise Exception(
                    f"Invalid data format. Missing 'entities' or 'relations' in JSON."
                )
            # Process entities
            for entity in data["entities"]:
                try:
                    if self.enable_batch_operations:
                        self._add_to_entity_batch(entity)
                    else:
                        self._create_entity(graph, entity, ontology)
                except Exception as e:
                    _task_logger.error(f"Error creating entity: {e}")
                    continue

            # Process relations
            for relation in data["relations"]:
                try:
                    if self.enable_batch_operations:
                        self._add_to_relation_batch(relation)
                    else:
                        self._create_relation(graph, relation, ontology)
                except Exception as e:
                    _task_logger.error(f"Error creating relation: {e}")
                    continue
            
        except Exception as e:
            logger.exception(f"Task id: {task_id} failed - {e}")
            raise e
        finally:
            with self.counter_lock:
                self.process_files += 1

    def _create_entity(self, graph: Graph, args: dict, ontology: Ontology) -> None:
        """
        Create an entity in the graph based on the extracted data.
        
        Args:
            graph (Graph): The graph instance to create the entity in.
            args (dict): The entity data extracted from the source.
            ontology (Ontology): The ontology to validate the entity type.
        """
        # Get unique attributes from entity
        entity = ontology.get_entity_with_label(args["label"])
        if entity is None:
            print(f"Entity with label {args['label']} not found in ontology")
            return None
        unique_attributes_schema = [attr for attr in entity.attributes if attr.unique]
        unique_attributes = {
            attr.name: (
                args["attributes"][attr.name] if attr.name in args["attributes"] else ""
            )
            for attr in unique_attributes_schema
        }
        unique_attributes_text = map_dict_to_cypher_properties(unique_attributes)
        non_unique_attributes = {
            attr.name: args["attributes"][attr.name]
            for attr in entity.attributes
            if not attr.unique and attr.name in args["attributes"]
        }
        non_unique_attributes_text = map_dict_to_cypher_properties(
            non_unique_attributes
        )
        set_statement = (
            f"SET n += {non_unique_attributes_text}"
            if len(non_unique_attributes.keys()) > 0
            else ""
        )
        query = f"MERGE (n:{args['label']} {unique_attributes_text}) {set_statement}"
        logger.debug(f"Query: {query}")
        result = graph.query(query)
        return result

    def _create_relation(self, graph: Graph, args: dict, ontology: Ontology) -> None:
        """
        Create a relation in the graph based on the extracted data.
        
        Args:
            graph (Graph): The graph instance to create the relation in.
            args (dict): The relation data extracted from the source.
            ontology (Ontology): The ontology to validate the relation type.
        """
        relations = ontology.get_relations_with_label(args["label"])
        if len(relations) == 0:
            print(f"Relations with label {args['label']} not found in ontology")
            return None
        source_unique_attributes = (
            args["source"]["attributes"]
            if "source" in args and "attributes" in args["source"]
            else {}
        )
        source_unique_attributes_text = map_dict_to_cypher_properties(
            source_unique_attributes
        )

        target_unique_attributes = (
            args["target"]["attributes"]
            if "target" in args and "attributes" in args["target"]
            else {}
        )
        target_unique_attributes_text = map_dict_to_cypher_properties(
            target_unique_attributes
        )

        relation_attributes = (
            map_dict_to_cypher_properties(args["attributes"])
            if "attributes" in args
            else {}
        )
        set_statement = (
            f"SET r += {relation_attributes}"
            if "attributes" in args
            and len(
                args["attributes"]
                if isinstance(args["attributes"], list)
                else args["attributes"].keys()
            )
            > 0
            else ""
        )
        query = f"MATCH (s:{args['source']['label']} {source_unique_attributes_text}) MATCH (d:{args['target']['label']} {target_unique_attributes_text}) MERGE (s)-[r:{args['label']}]->(d) {set_statement}"
        logger.debug(f"Query: {query}")
        result = graph.query(query)
        return result

    @sleep_and_retry
    @limits(calls=15, period=60)
    def _call_model(
        self,
        chat_session: GenerativeModelChatSession,
        prompt: str,
        retry: int = 6,
    ) -> GenerationResponse:
        """
        Call the generative model with rate limiting and retries.
        
        Args:
            chat_session (GenerativeModelChatSession): The chat session for interacting with the model.
            prompt (str): The prompt to send to the model.
            retry (Optional[int]): Number of retries in case of quota exceeded or errors.
        
        Returns:
            GenerationResponse: The model's response.
        
        Raises:
            Exception: If an error occurs after exhausting retries.
        """
        try:
            return chat_session.send_message(prompt)
        except Exception as e:
            # If exception is caused by quota exceeded, wait 10 seconds and try again for 6 times
            if "Quota exceeded" in str(e) and retry > 0:
                time.sleep(10)
                retry -= 1
                return self._call_model(chat_session, prompt, retry)
            else:
                if retry == 0:
                    logger.error("Quota exceeded")
                raise e
        """
        Initialize the ExtractDataStep.
        
        Args:
            sources (list[AbstractSource]): List of data sources to process.
            ontology (Ontology): The ontology associated with the knowledge graph.
            model (GenerativeModel): The generative model used for data extraction.
            graph (Graph): The FalkorDB graph instance.
            config (Optional[dict]): Configuration options for the step.
            hide_progress (Optional[bool]): Flag to hide progress bar. Defaults to False.
            enable_batch_operations (bool): Enable batch entity/relation creation.
            batch_size (int): Size of batches for entity/relation creation.
        """
        self.sources = sources
        self.ontology = ontology
        self.config = config
        if config is None:
            self.config = {
                "max_workers": 16,
                "max_input_tokens": 500000,
                "max_output_tokens": 8192,
            }
        else:
            self.config = config
        self.model = model
        self.graph = graph
        self.hide_progress = hide_progress
        self.enable_batch_operations = enable_batch_operations
        self.batch_size = batch_size
        self.enable_memory_monitoring = enable_memory_monitoring
        self.memory_config = memory_config or MemoryConfig()        self.process_files = 0
        self.counter_lock = Lock()
        
        # Batch buffers
        self.entity_batch = []
        self.relation_batch = []
        
        # Performance tracking
        self.created_entities = 0
        self.created_relations = 0
        self.processed_count = 0
        self.memory_warnings = 0        
        if not os.path.exists("logs"):
            os.makedirs("logs")

    def _create_entity_batch_query(self, entities: List[Dict]) -> str:
        """Generate optimized batch entity creation query"""
        # Group entities by type for more efficient queries
        entities_by_type = {}
        for entity in entities:
            entity_type = entity['label']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)
        
        queries = []
        for entity_type, type_entities in entities_by_type.items():
            # Extract unique and non-unique properties
            unique_props = self._get_unique_properties(type_entities[0]['attributes'])
            non_unique_props = self._get_non_unique_properties(type_entities[0]['attributes'])
            
            query = f"""
            UNWIND $entities_{entity_type.lower()} AS entity
            MERGE (n:{entity_type} {{{unique_props}}})
            SET n += entity.non_unique_props
            """
            queries.append(query)
        
        return "\n".join(queries)

    def _create_relation_batch_query(self, relations: List[Dict]) -> str:
        """Generate optimized batch relation creation query"""
        # Group relations by type
        relations_by_type = {}
        for relation in relations:
            rel_type = relation['label']
            if rel_type not in relations_by_type:
                relations_by_type[rel_type] = []
            relations_by_type[rel_type].append(relation)
        
        queries = []
        for rel_type, type_relations in relations_by_type.items():
            query = f"""
            UNWIND $relations_{rel_type.lower()} AS rel
            MATCH (source:{{rel.source.label}} {{{{rel.source.unique_props}}}})
            MATCH (target:{{rel.target.label}} {{{{rel.target.unique_props}}}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r += rel.properties
            """
            queries.append(query)
        
        return "\n".join(queries)

    def _get_unique_properties(self, attributes: Dict) -> str:
        """Extract unique properties from attributes"""
        entity = self.ontology.get_entity_with_label(attributes.get('label', ''))
        if entity is None:
            return ""
        
        unique_attrs = [attr for attr in entity.attributes if attr.unique]
        unique_props = {}
        for attr in unique_attrs:
            if attr.name in attributes:
                unique_props[attr.name] = attributes[attr.name]
        
        return map_dict_to_cypher_properties(unique_props)

    def _get_non_unique_properties(self, attributes: Dict) -> str:
        """Extract non-unique properties from attributes"""
        entity = self.ontology.get_entity_with_label(attributes.get('label', ''))
        if entity is None:
            return ""
        
        non_unique_attrs = [attr for attr in entity.attributes if not attr.unique]
        non_unique_props = {}
        for attr in non_unique_attrs:
            if attr.name in attributes:
                non_unique_props[attr.name] = attributes[attr.name]
        
        return map_dict_to_cypher_properties(non_unique_props)

    def _flush_entity_batch(self):
        """Flush entity batch to database"""
        if not self.entity_batch:
            return
        
        try:
            # Prepare batch data
            entities_by_type = {}
            for entity in self.entity_batch:
                entity_type = entity['label']
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                
                # Split attributes into unique and non-unique
                unique_props = {}
                non_unique_props = {}
                for attr_name, attr_value in entity['attributes'].items():
                    if self._is_unique_attribute(entity_type, attr_name):
                        unique_props[attr_name] = attr_value
                    else:
                        non_unique_props[attr_name] = attr_value
                
                entities_by_type[entity_type].append({
                    'unique_props': unique_props,
                    'non_unique_props': non_unique_props
                })
            
            # Execute batch query
            query = self._create_entity_batch_query(self.entity_batch)
            params = {}
            
            for entity_type, type_entities in entities_by_type.items():
                params[f"entities_{entity_type.lower()}"] = type_entities
            
            result = self.graph.query(query, params)
            self.created_entities += len(self.entity_batch)
            self.entity_batch.clear()
            
            return result
            
        except Exception as e:
            print(f"Error in batch entity creation: {e}")
            # Fallback to individual creation
            self._fallback_entity_creation()

    def _flush_relation_batch(self):
        """Flush relation batch to database"""
        if not self.relation_batch:
            return
        
        try:
            # Prepare batch data
            relations_by_type = {}
            for relation in self.relation_batch:
                rel_type = relation['label']
                if rel_type not in relations_by_type:
                    relations_by_type[rel_type] = []
                
                # Prepare source and target unique properties
                source_unique = self._extract_unique_properties(
                    relation['source']['label'], 
                    relation['source']['attributes']
                )
                target_unique = self._extract_unique_properties(
                    relation['target']['label'], 
                    relation['target']['attributes']
                )
                
                relations_by_type[rel_type].append({
                    'source': {
                        'label': relation['source']['label'],
                        'unique_props': source_unique
                    },
                    'target': {
                        'label': relation['target']['label'],
                        'unique_props': target_unique
                    },
                    'properties': relation.get('attributes', {})
                })
            
            # Execute batch query
            query = self._create_relation_batch_query(self.relation_batch)
            params = {}
            
            for rel_type, type_relations in relations_by_type.items():
                params[f"relations_{rel_type.lower()}"] = type_relations
            
            result = self.graph.query(query, params)
            self.created_relations += len(self.relation_batch)
            self.relation_batch.clear()
            
            return result
            
        except Exception as e:
            print(f"Error in batch relation creation: {e}")
            # Fallback to individual creation
            self._fallback_relation_creation()

    def _fallback_entity_creation(self):
        """Fallback to individual entity creation"""
        for entity in self.entity_batch:
            try:
                self._create_entity(self.graph, entity, self.ontology)
                self.created_entities += 1
            except Exception as e:
                print(f"Error creating entity {entity}: {e}")
        
        self.entity_batch.clear()

    def _fallback_relation_creation(self):
        """Fallback to individual relation creation"""
        for relation in self.relation_batch:
            try:
                self._create_relation(self.graph, relation, self.ontology)
                self.created_relations += 1
            except Exception as e:
                print(f"Error creating relation {relation}: {e}")
        
        self.relation_batch.clear()

    def _is_unique_attribute(self, entity_type: str, attr_name: str) -> bool:
        """Check if an attribute is unique for a given entity type"""
        entity = self.ontology.get_entity_with_label(entity_type)
        if entity is None:
            return False
        
        for attr in entity.attributes:
            if attr.name == attr_name:
                return attr.unique
        
        return False

    def _extract_unique_properties(self, entity_type: str, attributes: Dict) -> Dict:
        """Extract unique properties for entity matching"""
        entity = self.ontology.get_entity_with_label(entity_type)
        if entity is None:
            return {}
        
        unique_props = {}
        for attr in entity.attributes:
            if attr.unique and attr.name in attributes:
                unique_props[attr.name] = attributes[attr.name]
        
        return unique_props

    def _add_to_entity_batch(self, entity_data: Dict):
        """Add entity to batch, flush if needed"""
        self.entity_batch.append(entity_data)
        if len(self.entity_batch) >= self.batch_size:
            self._flush_entity_batch()

    def _add_to_relation_batch(self, relation_data: Dict):
        """Add relation to batch, flush if needed"""
        self.relation_batch.append(relation_data)
        if len(self.relation_batch) >= self.batch_size:
            self._flush_relation_batch()

    def flush_all_batches(self):
        """Flush all remaining batches"""
        self._flush_entity_batch()
        self._flush_relation_batch()

    def _check_memory_usage(self) -> bool:
        """Check if memory usage exceeds threshold"""
        memory = psutil.virtual_memory()
        usage_percent = memory.percent / 100.0
        usage_mb = memory.used / (1024 * 1024)
        
        # Log memory usage periodically
        if self.processed_count % self.memory_config.memory_check_interval == 0:
            print(f"Memory usage: {usage_percent:.1%} ({usage_mb:.0f} MB)")
        
        # Check if we're over the threshold
        if usage_mb > self.memory_config.max_memory_mb:
            self.memory_warnings += 1
            return True
        
        # Check if we should trigger garbage collection
        if usage_percent > self.memory_config.gc_threshold:
            gc.collect()
            print(f"Triggered garbage collection at {usage_percent:.1%} memory usage")
        
        return False

    def _get_documents_stream(self) -> Iterator[Tuple[Document, str]]:
        """Stream documents instead of loading all at once"""
        for source in self.sources:
            try:
                for document in source.load():
                    if document.not_empty():
                        yield document, source.instruction
                        
                        # Check memory after each document
                        if self.enable_memory_monitoring and self._check_memory_usage():
                            self._handle_memory_pressure()
                            
            except Exception as e:
                print(f"Error loading from source {source}: {e}")
                continue

    def _handle_memory_pressure(self):
        """Handle memory pressure by reducing processing load"""
        print(f"Memory pressure detected! Implementing mitigation strategies...")
        
        # Force garbage collection
        gc.collect()
        
        # Flush any pending batches to free memory
        if self.enable_batch_operations:
            self.flush_all_batches()
        
        # Reduce chunk size for future documents
        if hasattr(self, 'current_chunk_size'):
            self.current_chunk_size = int(
                self.current_chunk_size * self.memory_config.chunk_size_reduction
            )
            print(f"Reduced chunk size to {self.current_chunk_size}")
        
        # Wait for memory to be freed
        time.sleep(1)

    def _emergency_memory_cleanup(self):
        """Emergency cleanup when memory is critically low"""
        print("EMERGENCY: Performing aggressive memory cleanup...")
        
        # Flush all batches immediately
        if self.enable_batch_operations:
            self.flush_all_batches()
        
        # Force multiple garbage collection cycles
        for _ in range(3):
            gc.collect()
            time.sleep(0.1)
        
        # Clear any caches
        if hasattr(self, '_cache'):
            self._cache.clear()
        
        print("Emergency cleanup completed")
