# Financial Knowledge Graph Builder

A purpose-built system for extracting financial knowledge graphs from unstructured documents using local LLMs and a specialized financial ontology. Designed specifically for price prediction by focusing on entities and relationships that actually drive market prices.

## Features

- **Purpose-built Financial Ontology**: Uses `schema.py` with entities and relations specifically designed for price prediction
- **Dynamic Interactive Visualization**: Both static PNG and interactive HTML graphs with hover details
- **Local LLM Support**: Works with Ollama models (llama3.2, etc.)
- **Automatic Backup**: Preserves previous results with timestamp-based backups
- **SQLite Storage**: Raw data accessible via SQL queries
- **Multi-format Support**: PDF, TXT, MD files with file-type detection

## Quick Start

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:
```bash
# Ollama model (default: llama3.2)
OLLAMA_MODEL=llama3.2

```

### 3. Add Source Files

Place your financial documents in the `source/` directory with proper naming:
```bash
# File naming convention: type_number.extension
source/transcript_1.pdf      # Market discussion transcript
source/article_1.pdf         # Financial news article  
source/post_1.txt           # Social media post
source/report_1.md          # Analyst report
```

### 4. Run Pipeline

```bash
# Open Ollama server
ollama serve  

# Basic run
python langgraph_local_kg.py

```

## Output Files

- **`kg_local.sqlite`**: Raw graph data (nodes and edges tables)
- **`kg_local.png`**: Static visualization
- **`kg_local.html`**: Interactive visualization (if Plotly installed)
- **`backup/`**: Timestamped backups of previous runs

## Viewing Results

### Interactive Visualization

Open `kg_local.html` in your browser for:
- **Hover details**: Click on nodes/edges for full information
- **Drag to explore**: Move nodes around
- **Zoom and pan**: Navigate large graphs
- **Color-coded entities**: Different colors for each entity type

### Raw Data Queries

```bash
# View all nodes
sqlite3 kg_local.sqlite '.headers on' '.mode column' 'SELECT * FROM nodes;'

# View all edges  
sqlite3 kg_local.sqlite '.headers on' '.mode column' 'SELECT * FROM edges;'

# Find specific relationships
sqlite3 kg_local.sqlite '.headers on' '.mode column' 'SELECT * FROM edges WHERE rel="AFFECTS_PRICE_OF";'

# Count by entity type
sqlite3 kg_local.sqlite 'SELECT type, COUNT(*) FROM nodes GROUP BY type;'
```

### Python Data Access

```python
import sqlite3
import pandas as pd

# Load data into pandas
con = sqlite3.connect('kg_local.sqlite')
nodes_df = pd.read_sql_query("SELECT * FROM nodes", con)
edges_df = pd.read_sql_query("SELECT * FROM edges", con)
con.close()

# Analyze the graph
print(f"Total entities: {len(nodes_df)}")
print(f"Total relationships: {len(edges_df)}")
print("\nEntity types:")
print(nodes_df['type'].value_counts())
```

## Financial Ontology

The system uses a purpose-built ontology (`schema.py`) with entities specifically designed for price prediction:

### Entity Types
- **Instrument**: Tradable assets (equity, commodity, FX, index, ETF)
- **Company**: Issuer or operating entity
- **EconomicIndicator**: Macro time series (GDP, inflation, employment)
- **Event**: Discrete drivers (geopolitical, earnings, policy)
- **Sentiment**: Market mood indicators
- **PolicyAction**: Central bank decisions
- **CommodityFundamental**: Supply/demand metrics
- **Source**: Document provenance

### Relationship Types
- **AFFECTS_PRICE_OF**: Causal price impact with direction/confidence
- **HAS_EXPOSURE_TO**: Company/instrument exposure to commodities/countries
- **DRIVES_SENTIMENT**: Events/indicators influencing market mood
- **REPORTS_ON**: Source documents reporting on entities
- **COINTEGRATED_WITH**: Statistical long-run linkages

## Advanced Usage

### Custom Entity Extraction

Modify `extract_triples()` in `langgraph_local_kg.py` to:
- Add domain-specific entity recognition
- Implement custom relationship patterns
- Enhance confidence scoring

### Graph Analysis

```python
import networkx as nx

# Load graph from SQLite
G = nx.DiGraph()
# ... load nodes and edges ...

# Analyze graph structure
print(f"Graph density: {nx.density(G)}")
print(f"Average clustering: {nx.average_clustering(G)}")

# Find influential nodes
centrality = nx.degree_centrality(G)
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Most connected entities:", top_nodes)
```

### Batch Processing

```python
# Process multiple source directories
source_dirs = ["source_q1", "source_q2", "source_q3"]
for source_dir in source_dirs:
    db, img = run_pipeline(source_dir)
    print(f"Processed {source_dir}: {db}")
```

## Troubleshooting

### Common Issues

1. **Ollama not running**:
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

2. **Plotly not available**:
   ```bash
   pip install plotly
   ```

3. **Empty graph results**:
   - Check source files are readable
   - Verify file naming convention
   - Try with `--finetune` flag

4. **Memory issues**:
   - Reduce source file size
   - Use smaller LLM model
   - Process files individually

### Performance Tips

- Use smaller models (llama3.2:1b) for faster processing
- Limit source file size (first 2 pages for analysis)
- Enable fine-tuning for better accuracy
- Use SSD storage for faster SQLite access

### Illustrations

![A Graph Result](pictures/result_example.png)


## File Structure

```
KG_project/
├── langgraph_local_kg.py    # Main pipeline
├── schema.py               # Financial ontology
├── requirements.txt        # Dependencies
├── README.md              # This file
├── source/                # Input documents
│   ├── transcript_1.pdf
│   └── article_1.pdf
├── backup/                # Timestamped backups
│   ├── 20250920_043524_kg_local.png
│   └── 20250920_043524_kg_local.sqlite
├── kg_local.sqlite        # Current results
├── kg_local.png          # Static visualization
└── kg_local.html         # Interactive visualization
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## License

MIT License - see LICENSE file for details.
