import os
import asyncio
from typing import AsyncGenerator, Annotated, TypedDict, List

# --- Tus Imports de A2A y Agent Stack ---
from a2a.types import AgentSkill, Message
from a2a.utils.message import get_message_text
from agentstack_sdk.server import Server
from agentstack_sdk.server.context import RunContext
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.a2a.extensions import AgentDetail, AgentDetailTool

# --- LangGraph & LangChain Imports ---
from langgraph.graph import StateGraph, END
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_ollama import ChatOllama

# --- 1. Definición del Estado ---
class AgentState(TypedDict):
    query: str
    trends: List[str]
    context_reports: List[str]
    final_report: str

# --- 2. Definición de Nodos (Lógica del Agente) ---

# Herramienta de búsqueda y LLM con Ollama
search_wrapper = DuckDuckGoSearchAPIWrapper()
# Usamos el modelo granite4 solicitado
llm = ChatOllama(model="granite4:tiny-h") 

async def analyze_and_get_trends(state: AgentState):
    """Paso 1, 2 y 3: Identifica país, construye URL y extrae el top 5 de trends24.in"""
    query = state['query']
    
    # 1. Analizar si hay un país mencionado
    country_prompt = (
        "Identify if a specific country is mentioned in the following user query. "
        "Return only the country name in English (lowercase), or 'global' if no country is specified. "
        "Return ONLY the word, no explanation. "
        "Query: " + query
    )
    country_res = await llm.ainvoke(country_prompt)
    country = country_res.content.strip().lower().replace("'", "").replace('"', "")
    
    # 2. Construir la URL de trends24.in
    base_url = "https://trends24.in/"
    url = f"{base_url}{country}/" if country != "global" else base_url
    
    print(f"--- [LangGraph] Buscando tendencias para el país: {country} en {url} ---")
    
    # 3. Obtener el top 5 de tendencias
    search_query = f"site:trends24.in top trending topics {country if country != 'global' else ''}"
    results = search_wrapper.results(search_query, max_results=5)
    
    results_text = "\n".join([r.get('snippet', '') for r in results])
    extract_prompt = (
        f"Based on these search results for {url}, extract EXACTLY the TOP 5 trending topics for today. "
        "Return them as a comma-separated list, nothing else. "
        f"Results:\n{results_text}"
    )
    trends_res = await llm.ainvoke(extract_prompt)
    trends = [t.strip() for t in trends_res.content.split(",") if t.strip()]
    
    return {"trends": trends[:5]}

async def research_trends_context(state: AgentState):
    """Paso 4: Busca el 'por qué' de cada tendencia y captura la URL fuente"""
    reports = []
    for trend in state['trends']:
        print(f"--- [LangGraph] Investigando contexto para: {trend} ---")
        context_query = f"why is {trend} trending news today source article"
        search_results = search_wrapper.results(context_query, max_results=3)
        
        context_prompt = (
            f"Explain briefly why '{trend}' is trending. Use the search results below. "
            "You MUST include at least one source URL from the results. "
            "Format: Friendly explanation followed by [Source: URL]\n\n"
            f"Search Results:\n{search_results}"
        )
        context_res = await llm.ainvoke(context_prompt)
        reports.append(context_res.content.strip())
        
    return {"context_reports": reports}

async def synthesize_report(state: AgentState):
    """Paso 5: Genera el reporte final amistoso, dinámico y con emojis"""
    reports_text = "\n\n".join(state['context_reports'])
    
    final_prompt = (
        "Synthesize the following trend analysis into a single, coherent, and friendly report. "
        "You are an insightful X trends analyst. Use emojis (💡, 📰, 🚀, etc.) to make it engaging. "
        "Ensure each of the 5 trends has its explanation and MUST include the source URL. "
        "Greet the user as a friend.\n\n"
        f"Context Reports:\n{reports_text}"
    )
    final_res = await llm.ainvoke(final_prompt)
    return {"final_report": final_res.content.strip()}

# --- 3. Construcción del Grafo ---

workflow = StateGraph(AgentState)

workflow.add_node("get_trends", analyze_and_get_trends)
workflow.add_node("research_context", research_trends_context)
workflow.add_node("synthesize", synthesize_report)

workflow.set_entry_point("get_trends")
workflow.add_edge("get_trends", "research_context")
workflow.add_edge("research_context", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()

# --- 4. Configuración de Agent Stack ---

AGENT_DETAIL = AgentDetail(
    interaction_mode="multi-turn",
    user_greeting="¡Hola! Soy tu analista amigo de tendencias en X (ex-Twitter). Pregúntame sobre cualquier país.",
    version="1.2.0",
    framework="LangGraph + Ollama",
    author={"name": "Edgar Bruney"},
    tools=[
        AgentDetailTool(name="DuckDuckGo", description="Búsqueda web para tendencias en tiempo real."),
        AgentDetailTool(name="Ollama (DeepSeek)", description="Cerebro local para análisis dinámico.")
    ],
)

AGENT_SKILLS = [
    AgentSkill(
        id="x-trends-ollama",
        name="X Trends Ollama Specialist",
        description="Analiza tendencias de X por país usando modelos locales de Ollama.",
        examples=["¿Qué es tendencia en México?", "Dime las tendencias globales."],
        tags=["social-media", "trends", "x", "analysis", "ollama", "local-llm"]
    )
]

server = Server()

# --- 5. Handler del Servidor ---

@server.agent(name="X Trends Agent LangGraph", detail=AGENT_DETAIL, skills=AGENT_SKILLS)
async def langgraph_trends_agent(input: Message, context: RunContext) -> AsyncGenerator[AgentMessage, None]:
    user_query = get_message_text(input)
    print(f"--- [LangGraph] Recibiendo query: '{user_query}' ---")

    # Ejecutar el grafo
    initial_state = {"query": user_query, "trends": [], "context_reports": [], "final_report": ""}
    result = await app.ainvoke(initial_state)

    yield AgentMessage(text=result["final_report"])

def run():
    print("Starting Ollama-based X Trends Agent server...")
    server.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", 8002)))

if __name__ == "__main__":
    run()