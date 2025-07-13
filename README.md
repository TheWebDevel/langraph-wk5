# Multi-Agent Support System

A sophisticated multi-agent system built with LangGraph that provides intelligent support for IT and Finance queries with automatic routing, tool usage, and intelligent fallback mechanisms.

## Demo Videos

### Application Demo
[📹 **Demo Video: Multi-Agent Support System in Action**]([https://example.com/demo-video](https://drive.google.com/file/d/1itqcria8onsFJxgib_Otcm8dyPhIdbh7/view?usp=sharing))
*Watch the system handle various queries, from internal FAQ lookups to web search fallbacks*

### Testing Demo

https://github.com/user-attachments/assets/8aed5a3c-0b09-4906-b87c-7eaac2355a82


https://github.com/user-attachments/assets/801167a2-cbea-453a-9cd0-47a5359967da



## Architecture
<img width="1659" height="1482" alt="Mermaid Diagram Jul 12 2025" src="https://github.com/user-attachments/assets/56ecd8fc-aa09-4788-bb59-6286257606e4" />

### 6-Agent System
1. **Supervisor Agent** - Entry point, initial query analysis
2. **Decider Agent** - Classifies queries as IT, Finance, or Chat
3. **IT Agent** - Handles IT-related queries with intelligent fallback
4. **Finance Agent** - Handles Finance-related queries with intelligent fallback
5. **Chat Agent** - Handles casual conversation and greetings
6. **Call Tool Agent** - Executes vector search and web search tools

### Tools
- **Vector Search** - Semantic search through internal FAQ data
- **Web Search** - Current information via Tavily API
- **Intelligent Fallback** - Automatic fallback from internal to web search

### Message Flow
```
User Query → Supervisor → Decider → IT/Finance/Chat Agent → Final Answer
```

## Key Features

- **Intelligent Fallback Logic**: First tries internal vector search, then falls back to web search when content isn't relevant
- **Relevance Detection**: Advanced algorithms to determine if internal content matches the query
- **Automatic Routing**: Smart classification and routing to specialist agents
- **Tool Integration**: Seamless vector search and web search integration
- **Robust Error Handling**: Comprehensive error handling with graceful fallbacks
- **Debug Logging**: Detailed logging for system transparency and troubleshooting

## Setup

### Prerequisites
- Python 3.8+
- AWS Bedrock access with Claude 3 Sonnet
- Tavily API key (optional but recommended for full functionality)

### Installation
```bash
# Clone the repository

# Install dependencies
pip install -r requirements.txt

# Initialize the vector database (OPTIONAL)
python initialize_db.py
```

### Vector Database Initialization
The system uses persistent vector storage for efficient FAQ searching:

1. **On First Run**: The `initialize_db.py` script will:
   - Clear any existing vector database
   - Load and chunk FAQ documents from `data/`
   - Create embeddings using SentenceTransformers
   - Build a FAISS index for fast similarity search
   - Save everything to `vector_db/` directory

2. **On Application Start**: The system will:
   - Load the existing vector database from disk
   - Only embed new queries (not the entire corpus)
   - Provide fast, efficient search results

3. **To Rebuild Database**: Run `python initialize_db.py` again to clear and rebuild

### Environment Variables
Create a `.env` file in the project root:
```bash
# AWS Bedrock Configuration (REQUIRED)
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1

# Claude 3 Sonnet Model Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# Tavily Search API Key (OPTIONAL - for web search)
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

### Streamlit Interface
```bash
# Make sure vector database is initialized first (OPTIONAL)
python initialize_db.py

# Start the web interface
streamlit run streamlit_app.py
```
Access the web interface at `http://localhost:8501`

### Command Line Testing
```bash
python test_agents.py
```
Runs comprehensive unit tests with LLM-as-judge evaluation

### Vector Database Management
```bash
# Initialize/rebuild the vector database
python initialize_db.py

# Test the vector store functionality
python test_vector_store.py

# Check database status
ls -la vector_db/
```

## Example Queries

### IT Queries (Internal FAQ)
- "How do I set up VPN?"
- "What software is approved for use?"
- "How do I request a new laptop?"

### IT Queries (Web Search Fallback)
- "What are the latest cybersecurity threats?"
- "How do I protect against ransomware attacks?"
- "What's the best antivirus software for 2025?"

### Finance Queries (Internal FAQ)
- "How do I file a reimbursement request?"
- "When is payroll processed?"
- "Where can I find budget reports?"

### Finance Queries (Web Search Fallback)
- "What are the current tax updates for the new regime in India?"
- "What are the current tax deduction limits for 2025?"
- "How do I calculate depreciation for business equipment?"

### Chat Queries
- "Hello, how are you?"
- "What can you help me with?"
- "Tell me a joke"

## How It Works

### Fallback Logic
1. **Internal Search**: First attempts to find relevant information in internal FAQ data
2. **Relevance Check**: Uses advanced algorithms to determine if internal content matches the query
3. **Web Search Fallback**: If internal content isn't relevant, automatically searches the web
4. **Response Generation**: Provides comprehensive answers with proper sourcing

### Agent Flow
- **Supervisor**: Analyzes and prepares the query
- **Decider**: Classifies as IT, Finance, or Chat
- **Specialist Agent**: Handles the query with appropriate tools
- **Final Answer**: Formats response with agent flow and data source information

## Data Sources

- `data/it_faq.txt` - IT department FAQ and guidelines
- `data/finance_faq.txt` - Finance department FAQ and guidelines
- `data/example_queries.txt` - Test queries for system validation
- `vector_db/` - Persistent vector database (auto-generated)

## System Requirements

- AWS Bedrock access with Claude 3 Sonnet
- Tavily API key (for web search functionality)
- Internet connection for web search
- Sufficient memory for vector embeddings (FAISS)

## Dependencies

Key dependencies include:
- `langgraph>=0.2.0` - Multi-agent workflow framework
- `langchain>=0.2.0` - LLM integration
- `boto3>=1.34.0` - AWS Bedrock integration
- `faiss-cpu>=1.7.4` - Vector similarity search
- `sentence-transformers>=2.2.0` - Text embeddings
- `streamlit>=1.28.0` - Web interface
- `langchain-tavily>=0.1.0` - Web search integration

## Troubleshooting

### Common Issues

1. **AWS Credentials Error**
   - Verify AWS credentials in `.env` file
   - Ensure Bedrock access is granted
   - Check AWS region configuration

2. **Tavily API Key Missing**
   - Add TAVILY_API_KEY to `.env` file
   - System will still work for internal queries without web search

3. **Import Errors**
   - Run: `pip install -r requirements.txt`
   - Ensure Python 3.8+ is installed

4. **Vector Search Issues**
   - Check that FAQ data files exist in `data/` directory
   - Run `python initialize_db.py` to rebuild the vector database
   - Verify sufficient memory for embeddings
   - Check that `vector_db/` directory exists and contains files

### Debug Mode
The system includes comprehensive debug logging. Check console output for:
- Internal search results
- Relevance detection decisions
- Fallback logic execution
- Web search results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with `python test_agents.py`
5. Submit a pull request

## License

This project is licensed under the MIT License.
