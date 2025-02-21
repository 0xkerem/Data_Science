from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

@tool
def get_income_statement(ticker: str) -> dict:
    """
    A tool that fetches the income statement of a US stock in dictionary format.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        A dictionary containing the income statement data, or an error message if the data could not be retrieved.
    """
    import re
    import json
    import time
    import random
    from bs4 import BeautifulSoup
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }
    
    url = f"https://stockanalysis.com/stocks/{ticker}/financials/"
    
    # Sleep for a short randomized duration to mimic human browsing behavior
    time.sleep(random.uniform(0.5, 2))
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        script_tags = soup.find_all('script')
        
        for script in script_tags:
            if script.string and 'const data' in script.string:
                # Look for the financial data in the JavaScript code
                match = re.search(r'financialData:\s*(\{.*?\})', script.string, re.DOTALL)
                if match:
                    financial_data_str = match.group(1)
                    # Add quotes around JavaScript object keys
                    financial_data_str = re.sub(r'(\w+):', r'"\1":', financial_data_str)
                    # Handle negative decimals (e.g., -.123 → -0.123)
                    financial_data_str = re.sub(r'(-)\.(\d+)', r'\g<1>0.\2', financial_data_str)
                    # Handle positive decimals (e.g., .123 → 0.123)
                    financial_data_str = re.sub(r'(?<=[:, \[ ])\.(\d+)', r'0.\1', financial_data_str)
                    # Parse the corrected JSON string into a Python dictionary
                    financial_data = json.loads(financial_data_str)
                    return financial_data
        return {"error": "Income statement data not found"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON parsing error: {e}"}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, DuckDuckGoSearchTool(), get_current_time_in_timezone, get_income_statement], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()