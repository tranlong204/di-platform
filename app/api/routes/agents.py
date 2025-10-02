"""
API routes for agent management and execution
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from loguru import logger

from app.core.database import get_db
from app.services.agent_service import AgentOrchestrationService
from app.models import Agent, AgentExecution


router = APIRouter()


class AgentCreateRequest(BaseModel):
    name: str
    description: str
    workflow_config: Dict[str, Any]


class AgentResponse(BaseModel):
    id: int
    name: str
    description: str
    workflow_config: Dict[str, Any]
    tools: List[str]
    active: bool
    created_at: str


class AgentExecutionRequest(BaseModel):
    query: str


class AgentExecutionResponse(BaseModel):
    execution_id: int
    agent_id: int
    query: str
    result: str
    execution_steps: List[Dict[str, Any]]
    execution_time: float
    success: bool


class AgentListResponse(BaseModel):
    agents: List[AgentResponse]
    total: int


@router.post("/", response_model=AgentResponse)
async def create_agent(
    request: AgentCreateRequest,
    db: Session = Depends(get_db)
):
    """Create a new agent"""
    try:
        agent_service = AgentOrchestrationService()
        agent = await agent_service.create_agent(
            name=request.name,
            description=request.description,
            workflow_config=request.workflow_config,
            db=db
        )
        
        return AgentResponse(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            workflow_config=agent.workflow_config,
            tools=agent.tools,
            active=agent.active,
            created_at=agent.created_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    db: Session = Depends(get_db)
):
    """List all active agents"""
    try:
        agent_service = AgentOrchestrationService()
        agents = await agent_service.get_all_agents(db)
        
        agent_responses = [
            AgentResponse(
                id=agent.id,
                name=agent.name,
                description=agent.description,
                workflow_config=agent.workflow_config,
                tools=agent.tools,
                active=agent.active,
                created_at=agent.created_at.isoformat()
            )
            for agent in agents
        ]
        
        return AgentListResponse(
            agents=agent_responses,
            total=len(agent_responses)
        )
        
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """Get agent by ID"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            workflow_config=agent.workflow_config,
            tools=agent.tools,
            active=agent.active,
            created_at=agent.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/execute", response_model=AgentExecutionResponse)
async def execute_agent_workflow(
    agent_id: int,
    request: AgentExecutionRequest,
    db: Session = Depends(get_db)
):
    """Execute an agent workflow"""
    try:
        agent_service = AgentOrchestrationService()
        result = await agent_service.execute_agent_workflow(
            agent_id=agent_id,
            query=request.query,
            db=db
        )
        
        return AgentExecutionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error executing agent workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/executions")
async def get_agent_executions(
    agent_id: int,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get agent execution history"""
    try:
        agent_service = AgentOrchestrationService()
        executions = await agent_service.get_agent_executions(agent_id, db, limit)
        
        execution_data = [
            {
                "id": execution.id,
                "query_id": execution.query_id,
                "execution_steps": execution.execution_steps,
                "final_result": execution.final_result,
                "execution_time": execution.execution_time,
                "success": execution.success,
                "error_message": execution.error_message,
                "created_at": execution.created_at.isoformat()
            }
            for execution in executions
        ]
        
        return {
            "agent_id": agent_id,
            "executions": execution_data,
            "total": len(execution_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting agent executions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{agent_id}")
async def update_agent(
    agent_id: int,
    updates: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Update agent configuration"""
    try:
        agent_service = AgentOrchestrationService()
        success = await agent_service.update_agent(agent_id, updates, db)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {"message": "Agent updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: int,
    db: Session = Depends(get_db)
):
    """Delete an agent"""
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Soft delete by setting active to False
        agent.active = False
        db.commit()
        
        return {"message": "Agent deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
