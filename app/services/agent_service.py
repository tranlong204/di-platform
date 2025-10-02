"""
Agent orchestration service for multi-step workflows
"""

import json
import time
from typing import List, Dict, Any, Optional, Callable
from sqlalchemy.orm import Session
from loguru import logger

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage

from app.core.config import settings
from app.services.rag_service import RAGQueryService
from app.services.vector_store import VectorStoreService
from app.models import Agent, AgentExecution


class AgentOrchestrationService:
    """Service for orchestrating multi-step agent workflows"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.1
        )
        self.rag_service = RAGQueryService()
        self.vector_store = VectorStoreService()
        
        # Available tools for agents
        self.tools = self._create_tools()
        
        # Agent prompt template
        self.agent_prompt = self._create_agent_prompt()
    
    def _create_tools(self) -> List[Tool]:
        """Create available tools for agents"""
        return [
            Tool(
                name="document_search",
                description="Search for information across all documents in the knowledge base",
                func=self._search_documents
            ),
            Tool(
                name="multi_document_search",
                description="Search for information across specific documents",
                func=self._search_specific_documents
            ),
            Tool(
                name="document_summary",
                description="Get a summary of a specific document",
                func=self._get_document_summary
            ),
            Tool(
                name="cross_document_analysis",
                description="Analyze information across multiple documents",
                func=self._cross_document_analysis
            ),
            Tool(
                name="fact_checking",
                description="Verify facts by searching multiple sources",
                func=self._fact_checking
            ),
            Tool(
                name="trend_analysis",
                description="Analyze trends across multiple documents",
                func=self._trend_analysis
            )
        ]
    
    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """Create agent prompt template"""
        system_message = """You are an advanced AI research assistant with access to a comprehensive document knowledge base. 
        You can perform complex multi-step research workflows to answer questions thoroughly.
        
        Available capabilities:
        - Search across all documents or specific documents
        - Cross-document analysis and comparison
        - Fact-checking and verification
        - Trend analysis across multiple sources
        - Document summarization
        
        When answering complex questions:
        1. Break down the question into sub-questions if needed
        2. Use multiple tools to gather comprehensive information
        3. Synthesize findings from multiple sources
        4. Provide well-reasoned conclusions with citations
        
        Always be thorough and cite your sources."""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    async def create_agent(
        self, 
        name: str, 
        description: str, 
        workflow_config: Dict[str, Any],
        db: Session
    ) -> Agent:
        """Create a new agent"""
        try:
            agent = Agent(
                name=name,
                description=description,
                workflow_config=workflow_config,
                tools=[tool.name for tool in self.tools]
            )
            
            db.add(agent)
            db.commit()
            db.refresh(agent)
            
            logger.info(f"Created agent: {name}")
            return agent
            
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            db.rollback()
            raise
    
    async def execute_agent_workflow(
        self, 
        agent_id: int, 
        query: str, 
        db: Session
    ) -> Dict[str, Any]:
        """Execute an agent workflow"""
        start_time = time.time()
        
        try:
            # Get agent configuration
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Create agent executor
            agent_executor = create_openai_tools_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.agent_prompt
            )
            
            executor = AgentExecutor(
                agent=agent_executor,
                tools=self.tools,
                verbose=True,
                max_iterations=10,
                early_stopping_method="generate"
            )
            
            # Execute the workflow
            execution_steps = []
            
            def log_step(step: Dict[str, Any]):
                execution_steps.append(step)
                logger.info(f"Agent step: {step}")
            
            # Add step logging
            original_run = executor.run
            
            def logged_run(input_data):
                result = original_run(input_data)
                log_step({
                    "input": input_data,
                    "output": result,
                    "timestamp": time.time()
                })
                return result
            
            executor.run = logged_run
            
            # Execute the agent
            result = executor.run(query)
            
            execution_time = time.time() - start_time
            
            # Save execution record
            execution = AgentExecution(
                agent_id=agent_id,
                query_id=None,  # Will be linked if query is saved
                execution_steps=execution_steps,
                final_result=result,
                execution_time=execution_time,
                success=True
            )
            
            db.add(execution)
            db.commit()
            db.refresh(execution)
            
            logger.info(f"Agent workflow completed in {execution_time:.2f}s")
            
            return {
                'execution_id': execution.id,
                'agent_id': agent_id,
                'query': query,
                'result': result,
                'execution_steps': execution_steps,
                'execution_time': execution_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error executing agent workflow: {str(e)}")
            
            # Save failed execution
            execution = AgentExecution(
                agent_id=agent_id,
                query_id=None,
                execution_steps=[],
                final_result=None,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            db.add(execution)
            db.commit()
            
            raise
    
    # Tool implementations
    def _search_documents(self, query: str) -> str:
        """Search across all documents"""
        try:
            # This would be called synchronously by the agent
            # In a real implementation, you'd need to handle async/sync conversion
            return f"Searching documents for: {query}\n[This would return actual search results]"
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    def _search_specific_documents(self, query_and_docs: str) -> str:
        """Search specific documents"""
        try:
            # Parse query and document IDs from input
            parts = query_and_docs.split("|")
            query = parts[0]
            doc_ids = [int(x.strip()) for x in parts[1].split(",")] if len(parts) > 1 else []
            
            return f"Searching documents {doc_ids} for: {query}\n[This would return actual search results]"
        except Exception as e:
            return f"Error searching specific documents: {str(e)}"
    
    def _get_document_summary(self, document_id: str) -> str:
        """Get document summary"""
        try:
            doc_id = int(document_id)
            return f"Summary of document {doc_id}:\n[This would return actual document summary]"
        except Exception as e:
            return f"Error getting document summary: {str(e)}"
    
    def _cross_document_analysis(self, analysis_query: str) -> str:
        """Cross-document analysis"""
        try:
            return f"Cross-document analysis for: {analysis_query}\n[This would return analysis results]"
        except Exception as e:
            return f"Error in cross-document analysis: {str(e)}"
    
    def _fact_checking(self, fact_query: str) -> str:
        """Fact checking across sources"""
        try:
            return f"Fact checking: {fact_query}\n[This would return verification results]"
        except Exception as e:
            return f"Error in fact checking: {str(e)}"
    
    def _trend_analysis(self, trend_query: str) -> str:
        """Trend analysis across documents"""
        try:
            return f"Trend analysis for: {trend_query}\n[This would return trend analysis]"
        except Exception as e:
            return f"Error in trend analysis: {str(e)}"
    
    async def get_agent_executions(
        self, 
        agent_id: int, 
        db: Session, 
        limit: int = 50
    ) -> List[AgentExecution]:
        """Get agent execution history"""
        return db.query(AgentExecution).filter(
            AgentExecution.agent_id == agent_id
        ).order_by(AgentExecution.created_at.desc()).limit(limit).all()
    
    async def get_all_agents(self, db: Session) -> List[Agent]:
        """Get all agents"""
        return db.query(Agent).filter(Agent.active == True).all()
    
    async def update_agent(
        self, 
        agent_id: int, 
        updates: Dict[str, Any], 
        db: Session
    ) -> bool:
        """Update agent configuration"""
        try:
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if not agent:
                return False
            
            for key, value in updates.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            
            db.commit()
            logger.info(f"Updated agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating agent: {str(e)}")
            db.rollback()
            return False
