from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import Swarm
from autogen_agentchat.base import Handoff
from autogen_ext.models.ollama import OllamaChatCompletionClient

from tools import *

def create_agents_for_group_chat() -> Swarm:
    """
    Create a group chat with agents for the given task.
    """
    # Create the Client
    model_client = OllamaChatCompletionClient(
        model="llama3.1:8b",
        endpoint="http://localhost:11434",
        model_info={
            "json_output": True,
            "function_calling": True,
            "vision": True,
            "family": "unknown",
        },
    )


    # Data Gatherer Agent
    data_gatherer_prompt = """
    You are an Economic Data Gatherer Agent. 
    Collect the latest relevant economic news, data, and public sentiment for the query. 
    Use the serper_web_search and scrape_website tools to get accurate, up-to-date information.
    Return your findings as JSON with fields: 'source', 'date', 'headline', 'summary', 'key_data_points'.
    Once you have gather the data you should send the data in a message and complete the task.
    """

    data_gatherer_agent = AssistantAgent(
        name="DataGathererAgent",
        model_client = model_client,    
        system_message=data_gatherer_prompt,
        tools=[serper_web_search, scrape_website],
        handoffs=["economic_modeler_agent"]
    )

    # Economic Modeler Agent
    economic_modeler_prompt = """
    You are an Economic Modeler Agent. Use economic theory and the data provided to build and calibrate quantitative economic models. Provide equations, assumptions, calibration, and interpretations in structured JSON.
    """

    economic_modeler_agent = AssistantAgent(
        name="EconomicModelerAgent",
        model_client = model_client,    
        system_message=economic_modeler_prompt
    )

    # Data Analyzer Agent
    data_analyzer_prompt = """
    You are a Data Analyzer Agent specializing in economic data. Analyze the datasets and economic indicators provided, identifying trends, anomalies, and policy impacts. Return JSON with 'summary', 'key_insights', 'recommendations'.
    """

    data_analyzer_agent = AssistantAgent(
        name="DataAnalyzerAgent",
        model_client = model_client,    
        system_message=data_analyzer_prompt
    )

    # Synthesis Agent (Swarm Coordinator)
    synthesis_agent_prompt = """
    You are the Synthesis Agent. Combine inputs from multiple economic agents into a coherent and comprehensive economic report. Remove contradictions and duplicate findings. Provide JSON with 'executive_summary', 'consensus_findings', 'open_questions', and 'final_recommendations'.
    """

    synthesis_agent = AssistantAgent(
        name="SynthesisAgent",
        model_client = model_client,    
        system_message=synthesis_agent_prompt
    )

    # Create swarm with orchestration instructions
    economic_swarm = Swarm(
        [
            data_gatherer_agent,
            economic_modeler_agent,
            data_analyzer_agent,
            synthesis_agent,
        ],
        termination_condition=MaxMessageTermination(max_messages=50)
    )

    return economic_swarm