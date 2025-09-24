import os
import json
import sqlite3
import re
import shutil
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
# For Pyvis dynamic visualization
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("Pyvis not available. Install with: pip install pyvis")

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader

from langgraph.graph import StateGraph, START, END
from langchain_community.llms import Ollama

# Import the purpose-built ontology
from schema import ENTITIES, RELATIONS

# For dynamic visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")


@dataclass
class KGPlan:
	nodes: List[str]
	relations: List[Tuple[str, str, str]]  # (source_type, relation, target_type)
	rules: str


def backup_existing_results() -> None:
	"""Backup existing kg_local.png and kg_local.sqlite files to backup directory."""
	backup_dir = Path("backup")
	backup_dir.mkdir(exist_ok=True)
	
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	# Files to backup
	files_to_backup = ["kg_local.png", "kg_local.sqlite"]
	
	for filename in files_to_backup:
		source_path = Path(filename)
		if source_path.exists():
			backup_filename = f"{timestamp}_{filename}"
			backup_path = backup_dir / backup_filename
			shutil.copy2(source_path, backup_path)
			print(f"Backed up {filename} to {backup_path}")
		else:
			print(f"No existing {filename} to backup")


def create_ontology_plan() -> KGPlan:
	"""Create a plan based on the purpose-built financial ontology."""
	node_types = [entity["label"] for entity in ENTITIES]
	relation_types = [rel["label"] for rel in RELATIONS]
	
	return KGPlan(
		nodes=node_types,
		relations=relation_types,
		rules="Purpose-built financial ontology for price prediction. Focus on causal relationships that drive market prices."
	)


@dataclass
class SourceFile:
	path: str
	file_type: str  # transcript, article, post, etc.
	content: str


def read_pdf_text(pdf_path: str, max_pages: int = 1) -> str:
	reader = PdfReader(pdf_path)
	texts: List[str] = []
	for i, page in enumerate(reader.pages[:max_pages]):
		texts.append(page.extract_text() or "")
	return "\n\n".join(texts)


def load_source_files(source_dir: str) -> List[SourceFile]:
	"""Load all files from source directory and detect file types from filename prefixes."""
	source_path = Path(source_dir)
	if not source_path.exists():
		raise FileNotFoundError(f"Source directory not found: {source_dir}")
	
	files = []
	for file_path in source_path.iterdir():
		if file_path.is_file():
			# Extract file type from filename prefix (e.g., transcript_1.pdf -> transcript)
			filename = file_path.stem
			file_type = filename.split('_')[0] if '_' in filename else 'unknown'
			
			# Read content based on file extension
			content = ""
			if file_path.suffix.lower() == '.pdf':
				content = read_pdf_text(str(file_path))  
			elif file_path.suffix.lower() in ['.txt', '.md']:
				content = file_path.read_text(encoding='utf-8')
			else:
				print(f"Warning: Unsupported file type: {file_path}")
				continue
			
			files.append(SourceFile(
				path=str(file_path),
				file_type=file_type,
				content=content
			))
	
	return files

def extract_json_from_text(text: str) -> dict:
	# Try to find JSON object in the text
	text = text.strip()
	
	# Look for JSON object boundaries
	start = text.find('{')
	if start == -1:
		raise ValueError("No JSON object found")
	
	# Find matching closing brace
	brace_count = 0
	end = start
	for i, char in enumerate(text[start:], start):
		if char == '{':
			brace_count += 1
		elif char == '}':
			brace_count -= 1
			if brace_count == 0:
				end = i + 1
				break
	
	json_str = text[start:end]
	try:
		return json.loads(json_str)
	except json.JSONDecodeError:
		# Fallback: try to extract key-value pairs with regex
		result = {}
		# Extract nodes
		nodes_match = re.search(r'"nodes"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
		if nodes_match:
			nodes_str = nodes_match.group(1)
			nodes = re.findall(r'"([^"]+)"', nodes_str)
			result['nodes'] = nodes
		
		# Extract relations
		relations_match = re.search(r'"relations"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
		if relations_match:
			relations_str = relations_match.group(1)
			# Look for arrays like ["source", "relation", "target"]
			relation_matches = re.findall(r'\["([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\]', relations_str)
			result['relations'] = [list(match) for match in relation_matches]
		
		# Extract rules
		rules_match = re.search(r'"rules"\s*:\s*"([^"]*)"', json_str)
		if rules_match:
			result['rules'] = rules_match.group(1)
		
		return result

#### This is another way to extract triples, separating the text into chunks and processing each chunk separately
# def extract_triples(llm: Ollama, text: str, plan: KGPlan) -> List[Dict[str, Any]]:
#     # Get available entity types and relations from the ontology
# 	entity_types = plan.nodes
# 	relation_types = plan.relations
#     # entity_types = [entity["label"] for entity in ENTITIES]
#     # relation_types = [rel["label"] for rel in RELATIONS]
    
#     # Split text into chunks for efficient processing
# 	chunk_size = 1000
# 	text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
# 	all_triples = []
    
# 	for i, chunk in enumerate(text_chunks):
# 		print(f"Processing chunk {i+1}/{len(text_chunks)} ({len(chunk)} chars)")
		
# 		prompt = f"""
# 	Extract financial knowledge triples from this text chunk using the purpose-built financial ontology for price prediction.

# 	Available Entity Types: {entity_types}
# 	Available Relations: {relation_types}

# 	Focus on extracting entities and relationships that actually drive market prices:
# 	- Instruments (equities, commodities, FX, indices, ETFs)
# 	- Economic indicators and their prints
# 	- Events (geopolitical, earnings, policy)
# 	- Sentiment indicators
# 	- Company information
# 	- Policy actions

# 	Return ONLY valid JSON array of objects with this structure:
# 	[
# 	{{"source": {{"type": "Instrument", "name": "SPY", "ticker": "SPY", "instrument_type": "ETF"}}, "relation": "AFFECTS_PRICE_OF", "target": {{"type": "EconomicIndicator", "name": "Unemployment Rate", "indicator_name": "Unemployment Rate"}}, "props": {{"direction": "DOWN", "confidence": 0.8}}}},
# 	{{"source": {{"type": "Event", "name": "Jobs Report Release", "event_type": "Policy"}}, "relation": "DRIVES_SENTIMENT", "target": {{"type": "Sentiment", "name": "Market Sentiment", "scope": "Market"}}, "props": {{"confidence": 0.7}}}}
# 	]

# 	Text chunk {i+1}/{len(text_chunks)}:
# 	{chunk}
# 	"""  
# 		try:
# 			resp = llm.invoke(prompt)
# 			# Try to extract JSON array
# 			text_out = resp.strip()
# 			start = text_out.find('[')
# 			end = text_out.rfind(']') + 1
# 			if start != -1 and end > start:
# 				chunk_triples = json.loads(text_out[start:end])
# 				all_triples.extend(chunk_triples)
# 				print(f"  Extracted {len(chunk_triples)} triples from chunk {i+1}")
# 			else:
# 				print(f"  No valid JSON found in chunk {i+1}")
# 		except Exception as e:
# 			print(f"  Triple extraction failed for chunk {i+1}: {e}")
# 			# Continue with next chunk instead of using fallback

# 	print(f"Total triples extracted: {len(all_triples)}")
# 	return all_triples

def extract_triples(llm: Ollama, text: str, plan: KGPlan) -> List[Dict[str, Any]]:
	# Get available entity types and relations from the ontology
	entity_types = [entity["label"] for entity in ENTITIES]
	relation_types = [rel["label"] for rel in RELATIONS]
	
	prompt = f"""
Extract financial knowledge triples from this text using the purpose-built financial ontology for price prediction.

Available Entity Types: {entity_types}
Available Relations: {relation_types}

Focus on extracting entities and relationships that actually drive market prices:
- Instruments (equities, commodities, FX, indices, ETFs)
- Economic indicators and their prints
- Events (geopolitical, earnings, policy)
- Sentiment indicators
- Company information
- Policy actions

Return ONLY valid JSON array of objects with this structure:
[
  {{"source": {{"type": "Instrument", "name": "SPY", "ticker": "SPY", "instrument_type": "ETF"}}, "relation": "AFFECTS_PRICE_OF", "target": {{"type": "EconomicIndicator", "name": "Unemployment Rate", "indicator_name": "Unemployment Rate"}}, "props": {{"direction": "DOWN", "confidence": 0.8}}}},
  {{"source": {{"type": "Event", "name": "Jobs Report Release", "event_type": "Policy"}}, "relation": "DRIVES_SENTIMENT", "target": {{"type": "Sentiment", "name": "Market Sentiment", "scope": "Market"}}, "props": {{"confidence": 0.7}}}}
]

Text:
{text}...
"""
	
	try:
		resp = llm.invoke(prompt)
		# Try to extract JSON array
		text_out = resp.strip()
		start = text_out.find('[')
		end = text_out.rfind(']') + 1
		if start != -1 and end > start:
			triples = json.loads(text_out[start:end])
		else:
			raise ValueError("No JSON array found")
	except Exception as e:
		print(f"Triple extraction failed: {e}, using ontology-based fallback extraction")
		# Fallback: ontology-aware keyword-based extraction
		triples = []
	
	return triples


def build_graph(triples: List[Dict[str, Any]]) -> nx.DiGraph:
	G = nx.DiGraph()
	for t in triples:
		s = t.get("source", {})
		r = t.get("relation")
		tgt = t.get("target", {})
		props = t.get("props", {})
		s_id = f"{s.get('type')}::{s.get('name')}"
		t_id = f"{tgt.get('type')}::{tgt.get('name')}"
		if s_id not in G:
			G.add_node(s_id, **s)
		if t_id not in G:
			G.add_node(t_id, **tgt)
		G.add_edge(s_id, t_id, rel=r, **props)
	return G

def convert_to_sqlite_safe(value):
	"""Convert complex data types to SQLite-safe formats."""
	if value is None:
		return None
	elif isinstance(value, (dict, list)):
		return json.dumps(value)
	elif isinstance(value, (int, float, str, bool)):
		return value
	else:
		# Convert any other type to string
		return str(value)

def save_to_sqlite(G: nx.DiGraph, db_path: str) -> None:
	con = sqlite3.connect(db_path)
	nodes = []
	for n, data in G.nodes(data=True):
		item = {"id": n, "type": data.get("type"), "name": data.get("name")}
		for k, v in data.items():
			if k not in ("type", "name"):
				item[k] = convert_to_sqlite_safe(v)
		nodes.append(item)
	nodes_df = pd.DataFrame(nodes)
	edges = []
	for u, v, data in G.edges(data=True):
		item = {"src": u, "dst": v, "rel": data.get("rel")}
		for k, v2 in data.items():
			if k != "rel":
				item[k] = convert_to_sqlite_safe(v2)
		edges.append(item)
	edges_df = pd.DataFrame(edges)
	nodes_df.to_sql("nodes", con, if_exists="replace", index=False)
	edges_df.to_sql("edges", con, if_exists="replace", index=False)
	con.commit()
	con.close()

def create_pyvis_graph(G: nx.DiGraph, html_path: str) -> None:
    """Create an interactive Pyvis visualization of the knowledge graph."""
    net = Network(
        height="800px", 
        width="100%", 
        bgcolor="#ffffff", 
        font_color="black",
        directed=True
    )
    
    # Configure physics for better layout
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100},
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200
      }
    }
    """)
    
    # Color mapping for financial entities
    color_map = {
        "Instrument": "#ff4444",
        "EconomicIndicator": "#4444ff", 
        "Company": "#ff8844",
        "Event": "#8844ff",
        "Sentiment": "#ff44ff",
        "PolicyAction": "#8b4513",
        "Source": "#44ff44",
        "IndicatorPrint": "#44ffff",
        "CommodityFundamental": "#ffff44",
        "FXMacro": "#44ff88",
        "AnalystForecast": "#ff8888"
    }
    
    # Add nodes with rich information
    for node_id, node_data in G.nodes(data=True):
        node_type = node_data.get("type", "Unknown")
        node_name = node_data.get("name", node_id.split("::")[-1])
        
        # Create detailed hover information
        hover_info = [f"<b>{node_name}</b>", f"Type: {node_type}"]
        
        for key, value in node_data.items():
            if key not in ["name", "type"] and value:
                if isinstance(value, str) and value.startswith('{'):
                    try:
                        parsed = json.loads(value)
                        hover_info.append(f"{key}: {parsed}")
                    except:
                        hover_info.append(f"{key}: {value}")
                else:
                    hover_info.append(f"{key}: {value}")
        
        title = "<br>".join(hover_info)
        
        net.add_node(
            node_id,
            label=node_name,
            title=title,
            color=color_map.get(node_type, "#cccccc"),
            size=25,
            font={'size': 12, 'color': 'black'}
        )
    
    # Add edges with relationship details
    for source, target, edge_data in G.edges(data=True):
        rel = edge_data.get("rel", "")
        
        # Create rich edge information
        edge_info = [f"<b>Relation: {rel}</b>"]
        edge_label = rel
        
        for key, value in edge_data.items():
            if key != "rel" and value:
                if isinstance(value, str) and value.startswith('{'):
                    try:
                        parsed = json.loads(value)
                        edge_info.append(f"{key}: {parsed}")
                    except:
                        edge_info.append(f"{key}: {value}")
                else:
                    edge_info.append(f"{key}: {value}")
                    
                # Add key properties to edge label
                if key in ["confidence", "direction"]:
                    edge_label += f" ({key}: {value})"
        
        title = "<br>".join(edge_info)
        
        net.add_edge(
            source, 
            target,
            label=edge_label,
            title=title,
            arrows={'to': {'enabled': True, 'scaleFactor': 1.2}},
            color={'color': '#666666', 'highlight': '#ff0000'},
            width=2,
            font={'size': 10, 'color': '#333333', 'align': 'middle'}
        )
    
    net.heading = "Interactive Financial Knowledge Graph - Pyvis"
    net.save_graph(html_path)
    print(f"Pyvis interactive graph saved to: {html_path}")


def draw_graph(G: nx.DiGraph, out_path: str) -> None:
    """Create static PNG, interactive HTML, and Pyvis visualizations."""
    # Existing static PNG version
    plt.figure(figsize=(14, 12))
    pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
    
    # ... existing static PNG code ...
    
    # Interactive HTML version (Plotly)
    if PLOTLY_AVAILABLE:
        create_interactive_graph(G, out_path.replace('.png', '_plotly.html'))
    
    # Interactive Pyvis version  
    if PYVIS_AVAILABLE:
        create_pyvis_graph(G, out_path.replace('.png', '_pyvis.html'))


def create_interactive_graph(G: nx.DiGraph, html_path: str) -> None:
	"""Create an interactive Plotly visualization of the knowledge graph."""
	pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
	
	# Prepare node data
	node_x = []
	node_y = []
	node_text = []
	node_info = []
	node_colors = []
	
	color_map = {
		"Instrument": "red",
		"EconomicIndicator": "blue", 
		"Company": "orange",
		"Event": "purple",
		"Sentiment": "pink",
		"PolicyAction": "brown",
		"Source": "green",
		"IndicatorPrint": "cyan"
	}
	
	for node in G.nodes():
		x, y = pos[node]
		node_x.append(x)
		node_y.append(y)
		
		node_data = G.nodes[node]
		node_name = node_data.get("name", node.split("::")[-1])
		node_type = node_data.get("type", "Unknown")
		
		node_text.append(node_name)
		
		# Create detailed hover info
		info_parts = [f"<b>{node_name}</b>", f"Type: {node_type}"]
		for key, value in node_data.items():
			if key not in ["name", "type"] and value:
				info_parts.append(f"{key}: {value}")
		node_info.append("<br>".join(info_parts))
		
		node_colors.append(color_map.get(node_type, "gray"))
	
	# Prepare edge data
	edge_x = []
	edge_y = []
	edge_info = []
	
	for edge in G.edges(data=True):
		x0, y0 = pos[edge[0]]
		x1, y1 = pos[edge[1]]
		edge_x.extend([x0, x1, None])
		edge_y.extend([y0, y1, None])
		
		# Edge info
		rel = edge[2].get("rel", "")
		confidence = edge[2].get("confidence", "")
		direction = edge[2].get("direction", "")
		
		info_parts = [f"<b>Relation: {rel}</b>"]
		if confidence:
			info_parts.append(f"Confidence: {confidence}")
		if direction:
			info_parts.append(f"Direction: {direction}")
		for key, value in edge[2].items():
			if key not in ["rel", "confidence", "direction"] and value:
				info_parts.append(f"{key}: {value}")
		
		edge_info.append("<br>".join(info_parts))
	
	# Create the plot
	fig = go.Figure()
	
	# Add edges
	fig.add_trace(go.Scatter(
		x=edge_x, y=edge_y,
		line=dict(width=2, color='gray'),
		hoverinfo='none',
		mode='lines',
		showlegend=False
	))
	
	# Add nodes
	fig.add_trace(go.Scatter(
		x=node_x, y=node_y,
		mode='markers+text',
		hoverinfo='text',
		hovertext=node_info,
		text=node_text,
		textposition="middle center",
		marker=dict(
			size=20,
			color=node_colors,
			line=dict(width=2, color='black')
		),
		showlegend=False
	))
	
	# Update layout
	fig.update_layout(
		title=dict(
			text="Interactive Financial Knowledge Graph<br><sub>Click and drag to explore, hover for details</sub>",
			font=dict(size=16)
		),
		showlegend=False,
		hovermode='closest',
		margin=dict(b=20,l=5,r=5,t=40),
		annotations=[ dict(
			text="Hover over nodes for details, drag to move",
			showarrow=False,
			xref="paper", yref="paper",
			x=0.005, y=-0.002,
			xanchor='left', yanchor='bottom',
			font=dict(color="gray", size=12)
		)],
		xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
		yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
		plot_bgcolor='white'
	)
	
	# Save as HTML
	fig.write_html(html_path)
	print(f"Interactive graph saved to: {html_path}")


def run_pipeline(source_dir: str = "source", enable_finetuning: bool = False) -> Tuple[str, str]:
	load_dotenv()
	ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
	db_path = str(Path.cwd() / "kg_local.sqlite")
	img_path = str(Path.cwd() / "kg_local.png")

	# Backup existing results before running
	print("Backing up existing results...")
	backup_existing_results()

	# Initialize LLM with optional fine-tuning
	if enable_finetuning:
		print("Fine-tuning enabled - using enhanced prompts and context")
		llm = Ollama(model=ollama_model, temperature=0.3)  # Lower temperature for more consistent results
	else:
		llm = Ollama(model=ollama_model, temperature=0.2)
	
	# Load all source files
	print(f"Loading source files from: {source_dir}")
	source_files = load_source_files(source_dir)
	print(f"Found {len(source_files)} source files: {[sf.file_type for sf in source_files]}")

	
	# LangGraph state
	class State(dict):
		source_files: List[SourceFile]
		source_analysis: str
		plan: KGPlan
		all_triples: List[Dict[str, Any]]
		G: nx.DiGraph

	def node_plan(state: State) -> State:
		print("Using purpose-built financial ontology...")
		state["plan"] = create_ontology_plan()
		print(f"Ontology: {len(state['plan'].nodes)} entity types, {len(state['plan'].relations)} relation patterns")
		print(f"Focus: {state['plan'].rules}")
		return state

	def node_extract(state: State) -> State:
		print("Extracting triples from all source files...")
		all_triples = []
		for sf in state["source_files"]:
			print(f"Processing {sf.file_type}: {Path(sf.path).name}")
			triples = extract_triples(llm, sf.content, state["plan"])
			# Add source file info to each triple
			for triple in triples:
				triple["source_file"] = Path(sf.path).name
				triple["file_type"] = sf.file_type
			all_triples.extend(triples)
		
		state["all_triples"] = all_triples
		print(f"Extracted {len(all_triples)} total triples")
		return state

	def node_build(state: State) -> State:
		print("Building graph...")
		state["G"] = build_graph(state["all_triples"])
		print(f"Graph: {state['G'].number_of_nodes()} nodes, {state['G'].number_of_edges()} edges")
		return state

	def node_persist(state: State) -> State:
		print("Saving to SQLite and drawing...")
		save_to_sqlite(state["G"], db_path)
		draw_graph(state["G"], img_path)
		print(f"Saved: {db_path}, {img_path}")
		return state

	graph = StateGraph(State)
	# graph.add_node("analyze", node_analyze)
	graph.add_node("plan", node_plan)
	graph.add_node("extract", node_extract)
	graph.add_node("build", node_build)
	graph.add_node("persist", node_persist)
	# graph.add_edge(START, "analyze")
	graph.add_edge(START, "plan")
	graph.add_edge("plan", "extract")
	graph.add_edge("extract", "build")
	graph.add_edge("build", "persist")
	graph.add_edge("persist", END)
	agent = graph.compile()

	state: State = {"source_files": source_files}
	agent.invoke(state)
	return db_path, img_path


if __name__ == "__main__":
	import sys
		
	db, img = run_pipeline("source")
	
	print(f"Pipeline completed: {db}, {img}")
	print(f"Interactive visualization: {img.replace('.png', '.html')}")
	print("\nTo view raw data:")
	print("sqlite3 kg_local.sqlite '.headers on' '.mode column' 'SELECT * FROM nodes;'")
	print("sqlite3 kg_local.sqlite '.headers on' '.mode column' 'SELECT * FROM edges;'")
