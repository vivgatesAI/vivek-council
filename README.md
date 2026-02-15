# 🏛️ Vivek Council

A modern LLM Council application powered by Venice AI. Multiple AI models collaborate to answer your hardest questions through structured debate and synthesis.

Based on the [llm-council](https://github.com/karpathy/llm-council) concept, but reimagined as a single-file application with a modern design and Venice API integration.

![Vivek Council](./screenshot.png)

## ✨ Features

- **First Opinions**: All council members provide initial responses to your query
- **Cross Review**: Each model reviews and evaluates others' responses (anonymized)
- **Final Synthesis**: Chairman model produces a polished final answer
- **Modern UI**: Sleek dark theme with tabbed interface to explore all perspectives
- **Real-time Progress**: Watch the council work through stages
- **Easy Deployment**: One-click deploy to Railway

## 🚀 Quick Start

### Local Development

1. **Clone and install**
   ```bash
   git clone https://github.com/vivgatesAI/vivek-council.git
   cd vivek-council
   pip install -r requirements.txt
   ```

2. **Set your Venice API key**
   ```bash
   cp .env.example .env
   # Edit .env and add your VENICE_API_KEY
   ```

3. **Run the app**
   ```bash
   python main.py
   ```

4. **Open http://localhost:8000**

### Deploy to Railway

1. **Fork this repository** to your GitHub account

2. **Create a new project on Railway**
   - Go to [railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your forked repository

3. **Add environment variable**
   - In Railway dashboard, go to your project
   - Click "Variables" tab
   - Add `VENICE_API_KEY` with your Venice API key (get it at [venice.ai/settings/api](https://venice.ai/settings/api))

4. **Deploy!**
   - Railway will automatically deploy from your main branch
   - Click the generated URL to access your council

## ⚙️ Configuration

### Council Models

Edit `main.py` to customize your council:

```python
class Config:
    COUNCIL_MODELS: List[str] = [
        "claude-opus-4-6",    # Best overall quality
        "deepseek-v3.2",       # Efficient reasoning
        "grok-41-fast",        # Fast agent work
        "kimi-k2-5",           # Strong reasoning
    ]
    
    CHAIRMAN_MODEL: str = "claude-opus-4-6"
```

### Available Models

| Model | Description |
|-------|-------------|
| `claude-opus-4-6` | Claude Opus 4.6 - Best quality |
| `claude-sonnet-45` | Claude Sonnet 4.5 - Balanced |
| `deepseek-v3.2` | DeepSeek V3.2 - Efficient |
| `grok-41-fast` | Grok 4.1 Fast - Quick |
| `kimi-k2-5` | Kimi K2.5 - Reasoning |
| `openai-gpt-52` | GPT-5.2 - Frontier |
| `llama-3.3-70b` | Llama 3.3 70B - Open |
| `venice-uncensored` | Venice Uncensored - Creative |

## 📖 How It Works

### Stage 1: First Opinions
Your question is sent to all council models simultaneously. Each provides their own response based on their training and capabilities.

### Stage 2: Cross Review
Each model receives the other responses (anonymized so they don't know which model wrote what). They evaluate accuracy, insight, and clarity, then provide their own improved answer.

### Stage 3: Final Synthesis
The Chairman model reviews all opinions and critiques to produce a final, polished response that incorporates the best elements from all perspectives.

## 🛠️ Development

### Project Structure

```
vivek-council/
├── main.py           # Single-file FastAPI application
├── templates/
│   └── index.html    # Embedded frontend template
├── static/
│   └── style.css     # Modern dark theme
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container build
├── railway.json      # Railway config
└── .env.example      # Environment template
```

### Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Vanilla JS + CSS (embedded)
- **API**: Venice AI
- **Deployment**: Railway

## 🔒 Privacy

- Your conversations are stored locally in `data/conversations/`
- No data is sent to third parties (only Venice API)
- Venice offers private inference on many models

## 📝 License

MIT License - feel free to use and modify!

---

Built with ⚡️ by Vivek
