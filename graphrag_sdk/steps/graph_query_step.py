import logging
import time
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from falkordb import Graph
from typing import Optional
from graphrag_sdk.steps.Step import Step
from graphrag_sdk.ontology import Ontology
from graphrag_sdk.models import (
    GenerativeModelChatSession,
)
from graphrag_sdk.helpers import (
    extract_cypher,
    validate_cypher,
    stringify_falkordb_response,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GraphQueryGenerationStep(Step):
    """
    Graph Query Step
    """

    def __init__(
        self,
        graph: Graph,
        ontology: Ontology,
        chat_session: GenerativeModelChatSession,
        config: Optional[dict] = None,
        last_answer: Optional[str] = None,
        cypher_prompt: Optional[str] = None,
        cypher_prompt_with_history: Optional[str] = None,
    ) -> None:
        """
        Initializes the GraphQueryGenerationStep object.
        
        Args:
            graph (Graph): The graph object to query.
            ontology (Ontology): The ontology object.
            chat_session (GenerativeModelChatSession): The chat session object.
            config (Optional[dict]): The configuration object.
            last_answer (Optional[str]): The last answer.
            cypher_prompt (Optional[str]): The Cypher prompt.
            cypher_prompt_with_history (Optional[str]): The Cypher prompt with history.
        """
        self.ontology = ontology
        self.config = config or {}
        self.graph = graph
        self.chat_session = chat_session
        self.last_answer = last_answer
        self.cypher_prompt = cypher_prompt
        self.cypher_prompt_with_history = cypher_prompt_with_history
        
        # Retry configuration
        self.base_delay = 0.1
        self.max_delay = 2.0
        self.jitter_factor = 0.1

    def run(self, question: str, retries: Optional[int] = 10) -> tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Run the step to generate and validate a Cypher query with enhanced retry logic.
        
        Args:
            question (str): The question being asked to generate the query.
            retries (Optional[int]): Number of retries allowed in case of errors.
            
        Returns:
            tuple[Optional[str], Optional[str], Optional[int]]: The context, the generated Cypher query and the query execution time.
        """
        cypher = ""
        error = None
        for i in range(retries):
            try:
                cypher_prompt = (
                    (self.cypher_prompt.format(question=question) 
                    if self.last_answer is None
                    else self.cypher_prompt_with_history.format(question=question, last_answer=self.last_answer))
                )   
                logger.debug(f"Cypher Prompt: {cypher_prompt}")
                cypher_statement_response = self.chat_session.send_message(
                    cypher_prompt,
                )
                logger.debug(f"Cypher Statement Response: {cypher_statement_response}")
                cypher = extract_cypher(cypher_statement_response.text)
                logger.debug(f"Cypher: {cypher}")

                if not cypher or len(cypher) == 0:
                    return (None, None, None)

                # Parallel validation and execution
                with ThreadPoolExecutor(max_workers=2) as executor:
                    validation_future = executor.submit(validate_cypher, cypher, self.ontology)
                    validation_errors = validation_future.result()
                    
                    if validation_errors:
                        raise Exception("\n".join(validation_errors))
                    
                    # Execute query
                    query_result = self.graph.query(cypher)
                    result_set = query_result.result_set
                    execution_time = query_result.run_time_ms
                    context = stringify_falkordb_response(result_set)
                    logger.debug(f"Context: {context}")
                    logger.debug(f"Context size: {len(result_set)}")
                    logger.debug(f"Context characters: {len(str(context))}")

                    return (context, cypher, execution_time)
                    
            except Exception as e:
                logger.debug(f"Error: {e}")
                error = e
                if hasattr(self.chat_session, 'delete_last_message'):
                    self.chat_session.delete_last_message()

                if i == retries - 1:
                    logger.error(f"Failed after {retries} retries: {e}")
        raise Exception("Failed to generate Cypher query: " + str(error))

    async def run_async(self, question: str, retries: Optional[int] = 10) -> tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Async version of run with parallel validation.
        
        Args:
            question (str): The question being asked to generate the query.
            retries (Optional[int]): Number of retries allowed in case of errors.
            
        Returns:
            tuple[Optional[str], Optional[str], Optional[int]]: The context, the generated Cypher query and the query execution time.
        """
        async def _run_with_retry():
            for i in range(retries):
                try:
                    # Generate cypher
                    cypher_prompt = (
                        (self.cypher_prompt.format(question=question) 
                        if self.last_answer is None
                        else self.cypher_prompt_with_history.format(question=question, last_answer=self.last_answer))
                    )
                    
                    # Run cypher generation in executor
                    loop = asyncio.get_event_loop()
                    cypher_statement_response = await loop.run_in_executor(
                        None, self.chat_session.send_message, cypher_prompt
                    )
                    cypher = extract_cypher(cypher_statement_response.text)
                    
                    if not cypher:
                        return (None, None, None)
                    
                    # Parallel validation and execution
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        validation_future = executor.submit(validate_cypher, cypher, self.ontology)
                        validation_errors = validation_future.result()
                        
                        if validation_errors:
                            raise Exception("\n".join(validation_errors))
                        
                        # Execute query
                        query_result = self.graph.query(cypher)
                        result_set = query_result.result_set
                        execution_time = query_result.run_time_ms
                        context = stringify_falkordb_response(result_set)
                        
                        return (context, cypher, execution_time)
                        
                except Exception as e:
                    if i == retries - 1:
                        raise
                    await asyncio.sleep(2 ** i)  # Exponential backoff
        
        return await _run_with_retry()
