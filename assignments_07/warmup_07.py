import json
from dotenv import load_dotenv
from openai import OpenAI
import os
from datetime import datetime
from scipy import stats
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from smolagents import ToolCallingAgent, OpenAIServerModel, tool, CodeAgent

if load_dotenv():
    print('Successfully loaded environment variables from .env')

client = OpenAI()
api_key = os.getenv("OPENAI_API_KEY")
# --- Tool Definitions and the ReAct Loop ---

#Q1

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

tools = [
    {
        'type': 'function',
        'function': {
            'name': 'celsius_to_fahrenheit',
            'description': 'Convert a Celsius temperature to Fahrenheit and return it as a formatted string.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'celsius': {
                        'type': 'number',
                        'description':'Temperature in Celsius.'
                    }
                },
                'required': ['celsius'],
             },
        },
    }
]

print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


#Q2
print("\n Q2")
def get_current_time() -> str:
    '''Return the current local time as a formatted string.'''
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


tools_q2 = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    }
]

def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''
    
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': user_prompt},
    ]

   
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools_q2,        
        tool_choice='auto',    
    )

    first_message = first_response.choices[0].message

    messages.append({
        'role': 'assistant',
        'content': first_message.content,
        'tool_calls': first_message.tool_calls,
    })

    
    if first_message.tool_calls:
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'name': function_name,
                'content': tool_result,
            })

        
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        return second_response.choices[0].message.content or ''
    
   
    return first_message.content or ''

result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(result)

# Prediction was correct! The tool was not called —
# the agent answered directly from its internal knowledge.
# Only 1 API call was made.


#Q3
print("\n Q3")
tools_q3 = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Returns the current local time as a string.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
            },
        },
    },

    {
        'type': 'function',
        'function': {
            'name': 'celsius_to_fahrenheit',
            'description': 'Convert a Celsius temperature to Fahrenheit and return it as a formatted string.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'celsius': {
                        'type': 'number',
                        'description':'Temperature in Celsius.'
                    }
                },
                'required': ['celsius'],
             },
        },
    }
]


def run_agent_q3(user_prompt: str) -> str:
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': user_prompt},
    ]

    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools_q3,
        tool_choice='auto',
    )

    first_message = first_response.choices[0].message
    messages.append({
        'role': 'assistant',
        'content': first_message.content,
        'tool_calls': first_message.tool_calls,
    })

    if first_message.tool_calls:
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or '{}')

            if function_name == 'get_current_time':
                tool_result = get_current_time()
            elif function_name == 'celsius_to_fahrenheit':
                tool_result = celsius_to_fahrenheit(tool_args['celsius'])
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            messages.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'name': function_name,
                'content': tool_result,
            })

        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        return second_response.choices[0].message.content or ''

    return first_message.content or ''


response_a = run_agent_q3("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

response_b = run_agent_q3("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# Response A comment: celsius_to_fahrenheit tool was called because
# user asked for temperature conversion

# Response B comment: no tool was called because
# this is a conceptual question, agent answered directly


#Q4

print("\n Q4")
RESOURCES_DIR = Path("assignments_07/resources")

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted([
            p.name for p in self.resources_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv"
        ])

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {"error": f"No CSV loaded. Try: load_csv '{example}'."}
        return None

    def list_csv_files(self):
        files = self._available_csv_files()
        return {"files": files}

    def load_csv(self, filename: str):
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename
        if not path.exists():
            return {"error": f"Could not find '{filename}'."}
        self.df = pd.read_csv(path)
        self.csv_name = filename
        return {"message": f"Loaded {filename}.", 
                "columns": self.df.columns.tolist()}

    def get_columns(self):
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns=None):
        error = self._ensure_loaded()
        if error:
            return error
        data = self.df if columns is None else self.df[columns]
        return data.describe(include="all").transpose().round(3).to_dict()

    def describe_column(self, column: str):
        error = self._ensure_loaded()
        if error:
            return error
        if column not in self.df.columns:
            return {"error": f"'{column}' not found."}
        return {k: round(v, 3) if isinstance(v, float) else v
                for k, v in self.df[column].describe().to_dict().items()}

    def plot_data(self, y: str, x: str = None, plot_type: str = "line"):
        error = self._ensure_loaded()
        if error:
            return error
        if y not in self.df.columns:
            return f"Error: column '{y}' not found."
        if x:
            self.df.plot(x=x, y=y, kind=plot_type)
        else:
            self.df[y].plot(kind="line")
        plt.show()
        return f"Plotted {y} vs {x or 'row index'}."

    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return {"error": f"Column not found. Available: {self.df.columns.tolist()}"}
    
        pearson_r, p_value = stats.pearsonr(self.df[col1], self.df[col2])
        return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(pearson_r, 4),
        "p_value": round(p_value, 4)
        }

tools_schema = [
    {"type": "function", "function": {
        "name": "list_csv_files",
        "description": "List available CSV files in resources/.",
    }},
    {"type": "function", "function": {
        "name": "load_csv",
        "description": "Load a CSV file from resources/.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "CSV filename."}
        }, "required": ["filename"]},
    }},
    {"type": "function", "function": {
        "name": "get_columns",
        "description": "Get column names of loaded CSV.",
    }},
    {"type": "function", "function": {
        "name": "summarize_columns",
        "description": "Show summary statistics for columns.",
        "parameters": {"type": "object", "properties": {
            "columns": {"type": "array", "items": {"type": "string"}}
        }},
    }},
    {"type": "function", "function": {
        "name": "describe_column",
        "description": "Show statistics for a single column.",
        "parameters": {"type": "object", "properties": {
            "column": {"type": "string", "description": "Column name."}
        }, "required": ["column"]},
    }},
    {"type": "function", "function": {
        "name": "plot_data",
        "description": "Plot data from the active CSV.",
        "parameters": {"type": "object", "properties": {
            "y": {"type": "string"},
            "x": {"type": "string"},
            "plot_type": {"type": "string", "enum": ["scatter", "line"]},
        }, "required": ["y"]},
    }},
    {"type": "function", "function": {
        "name": "compute_correlation",
        "description": "Compute the Pearson correlation between two columns.",
        "parameters": {"type": "object", "properties": {
            "col1": {"type": "string", "description": "First column name."},
            "col2": {"type": "string", "description": "Second column name."},
        }, "required": ["col1", "col2"]},
    }},
]

csv_backend = CsvManager(RESOURCES_DIR)  # ← создаём объект

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }

    for loop_idx in range(max_tool_rounds):
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        if not msg.tool_calls:
            return msg.content

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")
            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    result = {"error": f"Tool '{name}' failed: {e}"}

            messages.append(observe_tool_result(tool_call.id, result))

    return "I hit the tool-round limit. Try a simpler request."


# Q5
print("\n Q5")
SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first. "
    "Keep answers short and student-friendly."
)

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh."
)
print(result)


#Q6
print("\n Q6")
print(json.dumps(messages, indent=2, default=str))

# Each role in the messages list represents a step in the ReAct loop:
# - "system"    → system prompt, instructions for the agent
# - "user"      → the user's request
# - "assistant" → model response
# - "tool"      → result of the tool call


#Q7
print("\n Q7")
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two columns.

    Args:
        col1: First column name.
        col2: Second column name.

    Returns:
        Dict with col1, col2, pearson_r, and p_value.
    """
    return csv_backend.compute_correlation(col1, col2)

print(compute_correlation.description)

# Smolagents reads the docstring and type hints automatically.
# In Q4 we wrote the full JSON schema by hand (name, description,
# parameters, properties). With @tool we only need type hints
# and a docstring — much less code, same result.


#Q8

print("\n Q8")
model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini",
)

@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.

    Returns:
        A dict with a files list.
    """
    return csv_backend.list_csv_files()

@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/.

    Args:
        filename: CSV filename in resources/.

    Returns:
        A dict with status and column names.
    """
    return csv_backend.load_csv(filename)

@tool
def get_columns() -> list:
    """Return column names for the currently loaded CSV.

    Returns:
        A list of column names.
    """
    return csv_backend.get_columns()

@tool
def summarize_columns(columns: list = None) -> dict:
    """Return summary stats for selected columns or all columns.

    Args:
        columns: Column names to summarize. If None, summarizes all.

    Returns:
        A dict of summary statistics.
    """
    return csv_backend.summarize_columns(columns)

@tool
def describe_column(column: str) -> dict:
    """Describe a single column with basic stats.

    Args:
        column: The name of the column to describe.

    Returns:
        A dict of basic stats.
    """
    return csv_backend.describe_column(column)

@tool
def plot_data(y: str, x: str = None, plot_type: str = "line") -> str:
    """Plot data from the active CSV.

    Args:
        y: Column name for y-axis.
        x: Column name for x-axis. If None, use row index.
        plot_type: Type of plot — line or scatter.

    Returns:
        A success message string.
    """
    return csv_backend.plot_data(y=y, x=x, plot_type=plot_type)

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions="You are a data assistant. Use tools for data work. Do not guess."
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions="You are a data assistant. Use tools for simple tasks. Write code only when tools are not enough.",
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

print("\n--- Tool Agent ---")
response_tool = tool_agent.run(prompt)
print(response_tool)

print("\n--- Code Agent ---")
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_backend})
print(response_code)

# ToolCallingAgent: plotted correctly but lied about green dots —
# plot_data has no color parameter. Classic hallucination.
#
# CodeAgent: tried to write matplotlib code with color='green'
# but struggled to access the DataFrame directly.
#
# Use ToolCallingAgent for simple tasks, CodeAgent when
# custom styling or code is needed.

#Q9
# Q1: ToolCallingAgent is better for simple, predictable tasks like
# loading and summarizing a CSV. The tools already cover everything
# needed — no custom code required, less risk of errors.

# Q2: CodeAgent can generate and run arbitrary Python code.
# This means it could accidentally run harmful commands on your system.
# ToolCallingAgent only calls predefined tools — much safer.