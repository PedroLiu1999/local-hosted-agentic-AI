from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import Swarm,RoundRobinGroupChat,SelectorGroupChat
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential

from tools import *

def create_agents_for_group_chat() -> Swarm:
    """
    Create a group chat with agents for the given task.
    """
    # Create the Client
    model_client = OllamaChatCompletionClient(
        model="qwen3:8b",
        host="http://localhost:11434",
        model_info={
            "json_output": True,
            "function_calling": True,
            "vision": True,
            "family": "unknown",
        },
    )

    # # Create the Client
    # model_client = AzureAIChatCompletionClient(
    #     model="gpt-4o-mini",
    #     endpoint="https://models.inference.ai.azure.com",
    #     # To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
    #     credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN")),
    #     model_info={
    #         "json_output": True,
    #         "function_calling": True,
    #         "vision": True,
    #         "family": "unknown",
    #     },
    # )
    


    # Data Gatherer Agent
    data_gatherer_prompt = """
    You are an Economic Data Gatherer Agent. 
    Collect the latest relevant economic news, data, and public sentiment for the query. 
    Use the serper_web_search and scrape_website tools to get accurate, up-to-date information.
    Return your findings as JSON with fields: 'source', 'date', 'headline', 'summary', 'key_data_points'.
    """

    data_gatherer_agent = AssistantAgent(
        name="DataGathererAgent",
        model_client = model_client,    
        system_message=data_gatherer_prompt,
        tools=[serper_web_search, scrape_website],
        handoffs=["EconomicModelerAgent","DataAnalyzerAgent"]
    )

    # Economic Modeler Agent
    economic_modeler_prompt = """
    You are an Economic Modeler Agent. 
    Use economic theory and the data provided to build and calibrate quantitative economic models. 
    Provide equations, assumptions, calibration, and interpretations in structured JSON.
    """

    economic_modeler_agent = AssistantAgent(
        name="EconomicModelerAgent",
        model_client = model_client,    
        system_message=economic_modeler_prompt,
        handoffs=["DataAnalyzerAgent"]
    )

    # Data Analyzer Agent
    data_analyzer_prompt = """
    You are a Data Analyzer Agent specializing in economic data. 
    Analyze the datasets and economic indicators provided, identifying trends, anomalies, and policy impacts. Return JSON with 'summary', 'key_insights', 'recommendations'.
    """

    data_analyzer_agent = AssistantAgent(
        name="DataAnalyzerAgent",
        model_client = model_client,    
        system_message=data_analyzer_prompt,
        handoffs=["SynthesisAgent"]
    )

    # Synthesis Agent (Swarm Coordinator)
    synthesis_agent_prompt = """
    You are the Synthesis Agent. Combine inputs from multiple economic agents into a coherent and comprehensive economic report. Remove contradictions and duplicate findings. Provide JSON with 'executive_summary', 'consensus_findings', 'open_questions', and 'final_recommendations'.
    If the task is complete return TERMINATE
    """

    synthesis_agent = AssistantAgent(
        name="SynthesisAgent",
        model_client = model_client,    
        system_message=synthesis_agent_prompt,
        
    )


    selector_prompt = """Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """

        # Create swarm with orchestration instructions
    economic_swarm = SelectorGroupChat(
        [
            data_gatherer_agent,
            economic_modeler_agent,
            data_analyzer_agent,
            synthesis_agent,
        ],
        termination_condition=MaxMessageTermination(max_messages=50)|TextMentionTermination("TERMINATE"),
        model_client=model_client,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True

    )

    return economic_swarm