# GAIA Evaluator - Quick Start Guide

Evaluate Manus AI agents on the GAIA benchmark using pure REST API.

## Setup

### 1. Install Dependencies
```bash
pip install requests pandas python-dotenv tqdm datasets huggingface_hub
```

### 2. Get API Keys
You'll need:
- **Manus API Key**: Get from [Manus Dashboard](https://manus.ai)
- **HuggingFace Token** (optional): Get from [HuggingFace Settings](https://huggingface.co/settings/tokens)

### 3. Configure Environment
Create a `.env` file:
```env
MANUS_DEV_KEY=your-manus-api-key
HF_TOKEN=your-hf-token  # Optional
```

## Usage

### Basic Evaluation
```python
from gaia_manus_rest import GAIAEvaluator

# Simple evaluation (3 questions)
evaluator = GAIAEvaluator(
    manus_agent_profile="manus-1.5-lite",
    levels=[1],
    max_questions=3
)
results = evaluator.run()
```

### Full Level Evaluation (Parallel)
```python
evaluator = GAIAEvaluator(
    manus_agent_profile="manus-1.5-lite",
    levels=[1],
    max_workers=5,           # Run 5 tasks in parallel
    batch_submission=True,   # Fastest mode
    max_questions=None       # Evaluate all questions
)
results = evaluator.run()
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `manus_agent_profile` | `"manus-1.5-lite"` | Model to use |
| `levels` | `[1]` | GAIA difficulty levels (1-3) |
| `max_questions` | `None` | Limit questions (None = all) |
| `max_workers` | `1` | Parallel workers (1 = sequential) |
| `batch_submission` | `False` | Submit all at once (fastest) |
| `delete_after_evaluation` | `True` | Auto-delete completed tasks |
| `output_dir` | `"./gaia_results"` | Results directory |

## Execution Modes

**Sequential (Safest)**
```python
evaluator = GAIAEvaluator(max_workers=1, max_questions=10)
```

**Parallel (Balanced)**
```python
evaluator = GAIAEvaluator(max_workers=5, max_questions=50)
```

**Batch (Default)**
```python
evaluator = GAIAEvaluator(
    max_workers=10, 
    batch_submission=True, 
    max_questions=100
)
```

## Output

Results are saved as CSV:
```
./gaia_results/gaia_manus_{profile}_{timestamp}.csv
```

Key columns:
- `question` - The GAIA question
- `ground_truth` - Correct answer
- `final_answer` - Extracted model answer
- `response` - Full model response
- `status` - Task status
- `credit_usage` - API credits used

## Troubleshooting
**Tasks timing out?** → Increase timeout
```python
evaluator = GAIAEvaluator(max_poll_time=600)
```

**Missing final answers?** → Check response format includes `FINAL ANSWER:`
