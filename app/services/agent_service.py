"""
Agent orchestration service for multi-step workflows
"""

import json
import time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from loguru import logger

from openai import OpenAI
from app.core.config import settings
from app.services.rag_service import RAGQueryService
from app.services.vector_store import VectorStoreService
from app.models import Agent, AgentExecution


class AgentOrchestrationService:
    """Service for orchestrating multi-step agent workflows"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.rag_service = RAGQueryService()
        self.vector_store = VectorStoreService()
    
    async def execute_agent_workflow(
        self, 
        agent_id: int, 
        query_id: int, 
        db: Session
    ) -> Dict[str, Any]:
        """Execute an agent workflow"""
        start_time = time.time()
        
        try:
            # Get agent configuration
            agent = db.query(Agent).filter(Agent.id == agent_id).first()
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Get query
            query = db.query(Query).filter(Query.id == query_id).first()
            if not query:
                raise ValueError(f"Query {query_id} not found")
            
            # Execute workflow steps
            workflow_config = agent.workflow_config
            steps = workflow_config.get('steps', [])
            
            execution_steps = []
            final_result = ""
            
            for step in steps:
                step_result = await self._execute_step(step, query.query_text, db)
                execution_steps.append({
                    'step_name': step.get('name', 'unknown'),
                    'result': step_result,
                    'timestamp': time.time()
                })
                
                # Use step result for next step if needed
                if step.get('use_result_for_next', False):
                    query.query_text = step_result
            
            # Generate final result
            final_result = await self._generate_final_result(
                query.query_text, execution_steps, agent
            )
            
            # Save execution record
            execution = AgentExecution(
                agent_id=agent_id,
                query_id=query_id,
                execution_steps=execution_steps,
                final_result=final_result,
                execution_time=time.time() - start_time,
                success=True
            )
            
            db.add(execution)
            db.commit()
            db.refresh(execution)
            
            return {
                'execution_id': execution.id,
                'agent_id': agent_id,
                'query_id': query_id,
                'final_result': final_result,
                'execution_steps': execution_steps,
                'execution_time': time.time() - start_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error executing agent workflow: {str(e)}")
            
            # Save failed execution
            execution = AgentExecution(
                agent_id=agent_id,
                query_id=query_id,
                execution_steps=[],
                final_result="",
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            db.add(execution)
            db.commit()
            
            raise
    
    async def _execute_step(self, step: Dict[str, Any], query: str, db: Session) -> str:
        """Execute a single workflow step"""
        step_type = step.get('type', 'rag')
        
        if step_type == 'rag':
            # Use RAG service
            result = await self.rag_service.process_query(query, db)
            return result.get('response', '')
        
        elif step_type == 'search':
            # Vector search
            docs = await self.vector_store.search_similar_documents(query, k=5)
            return f"Found {len(docs)} relevant documents"
        
        elif step_type == 'summarize':
            # Summarize using OpenAI
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": "Summarize the following text concisely:"},
                    {"role": "user", "content": query}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        
        else:
            return f"Unknown step type: {step_type}"
    
    async def _generate_final_result(
        self, 
        query: str, 
        steps: List[Dict[str, Any]], 
        agent: Agent
    ) -> str:
        """Generate final result from workflow steps"""
        try:
            # Prepare context from steps
            context = "Workflow Steps:\n"
            for i, step in enumerate(steps, 1):
                context += f"{i}. {step['step_name']}: {step['result']}\n"
            
            # Generate final response
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": f"You are executing a workflow for: {agent.description}. Use the workflow results to provide a comprehensive answer."},
                    {"role": "user", "content": f"Original Query: {query}\n\n{context}\n\nProvide a final comprehensive answer based on the workflow results."}
                ],
                max_tokens=settings.max_tokens,
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating final result: {str(e)}")
            return "I encountered an error while generating the final result. Please try again."
    
    async def create_agent(
        self, 
        name: str, 
        description: str, 
        workflow_config: Dict[str, Any], 
        db: Session
    ) -> Agent:
        """Create a new agent"""
        agent = Agent(
            name=name,
            description=description,
            workflow_config=workflow_config,
            active=True
        )
        
        db.add(agent)
        db.commit()
        db.refresh(agent)
        
        return agent
    
    async def get_agent(self, agent_id: int, db: Session) -> Optional[Agent]:
        """Get agent by ID"""
        return db.query(Agent).filter(Agent.id == agent_id).first()
    
    async def list_agents(self, db: Session, active_only: bool = True) -> List[Agent]:
        """List all agents"""
        query = db.query(Agent)
        if active_only:
            query = query.filter(Agent.active == True)
        return query.all()
    
    async def update_agent(
        self, 
        agent_id: int, 
        updates: Dict[str, Any], 
        db: Session
    ) -> Optional[Agent]:
        """Update an agent"""
        agent = await self.get_agent(agent_id, db)
        if not agent:
            return None
        
        for key, value in updates.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
        
        db.commit()
        db.refresh(agent)
        return agent
    
    async def delete_agent(self, agent_id: int, db: Session) -> bool:
        """Delete an agent"""
        agent = await self.get_agent(agent_id, db)
        if not agent:
            return False
        
        db.delete(agent)
        db.commit()
        return True