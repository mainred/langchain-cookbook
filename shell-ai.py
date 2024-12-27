# shell-ai generate Linux commands from user-input linux-related questions, or common answers for other questions
# steps:
# - train a fine-tuned model from `shell-ai-traning-data.jsonl` on Azure OpenAI Service and create a deployment out of it
# - specify the deployment and auth info in the .env
# - add command_not_found_handler to capture command not found issue, and execute the ai script to give answers
# command_not_found_handler() {
#     python /home/azureuser/llm/langchain/langchain-cookbook/shell-ai.py "$@"
#     if [ $? = 0 ]; then
#             return 0
#     else
#             return 127
#     fi
# }
from dotenv import load_dotenv
env_path = "/home/azureuser/llm/langchain/langchain-cookbook/.env"
load_dotenv(env_path)

import os
import sys

from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
)

if len(sys.argv) != 2:
    sys.exit(1)
question = sys.argv[1]
print(model.invoke(question).content)
