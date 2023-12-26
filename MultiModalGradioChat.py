import urllib.request
import json
import os
import ssl
from dotenv import load_dotenv
from IPython.display import Image, display
import PIL.Image
import gradio as gr
import base64
import time
import os
import tqdm as notebook_tqdm
from tqdm import tqdm
import requests

  
load_dotenv()  

#set max chat history to keep
max_items = 16

# Initialize an empty list for the conversation history with max len
conversation_history = []


#OPEN AI KEY
GPT4V_KEY = os.getenv("AZURE_OPENAI_API_KEY")


#Set Headers
headers = {
            "Content-Type": "application/json",
            "api-key": GPT4V_KEY,
         }



# Add the system message to the conversation history , Cutomize the System message to your needs
system_message = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "You are an AI assistant that helps people find information.Make the URL as markdown hyper links for rendering as hyperlinks example [Gradio Website][1][1]: https://www.gradio.app/"
        }
    ]
}
#Maintain the conversation_history 
conversation_history.append(system_message)

#RAG Pattern Index details where Images are embedded for Retrieval
dataSources = [
            {
            "type": "AzureCognitiveSearch",
            "parameters": {
                "endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
                "key": os.getenv("AZURE_SEARCH_KEY"),
                "indexName":os.getenv("AZURE_SEARCH_INDEX_NAME"),
                "fieldsMapping": {
                                "vectorFields": "image_vector"
                            },
            }
            }
        ]


#current_dir the avatar image needs to be placed in current_dir
current_dir = os.path.abspath(os.getcwd())

#to keep limited chat history
def keep_latest_n_items(history, n):
    # Keep only the latest n items
    history = history[-n:]
    return history

#to handle like and dislike for chat responses from LLM, boilerplate code to be expanded
def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


# Image to Base 64 Converter
def convertImageToBase64(image_path):
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')

# Function that takes User Inputs and displays it on ChatUI and also maintains the history for chatcompletion
def buildHistoryForUiAndChatCompletion(history,txt,img):
    #if user enters only text
    if not img:
        history += [(txt,None)]
        # Add the user message to the conversation history
        user_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": txt
                    }
                ]
            }
        conversation_history.append(user_message)
        return history
    #if user enters image and text
    base64 = convertImageToBase64(img)
    data_url = f"data:image/jpeg;base64,{base64}"
    history += [(f"{txt} ![]({data_url})", None)]
    # Add the user message to the conversation history
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64}"
                }
                },
            {
                "type": "text",
                "text": txt
            }
        ]
    }
    #append the user message to chat api pay load
    conversation_history.append(user_message)
    return history

# Function that takes User Inputs, generates Response and displays on Chat UI
def call_AzureOpenAI_Vision_RAG_API(history,text,img):
    body = {
        "dataSources": dataSources,
        "messages": conversation_history,
        "max_tokens": 800,
        "temperature": 0,
        "top_p": 1
    }
    
    GPT4V_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")+"openai/deployments/"+os.getenv("AZURE_OPENAI_VISIONGPT_DEPLOYMENT_NAME")+"/extensions/chat/completions?api-version="+os.getenv("AZURE_OPENAI_API_VERSION")

    #post the API request
    response = requests.post(GPT4V_ENDPOINT, headers=headers, json=body)
    #print(response.json())
    #get llm reponse
    content = response.json()['choices'][0]['message']['content']
 
    #llm response added to history
    assistant_message = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": content
            }
        ]
    }
    conversation_history.append(assistant_message)
    history += [(None,content)]
    #conversation_history = keep_latest_n_items(conversation_history, 10)
    return history 
    

# Interface Code
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="red")) as app:
    with gr.Row():
        image_box = gr.Image(type="filepath")
        chatbot = gr.Chatbot(
            scale = 2,
            height=750,
            bubble_full_width=False,
            #the avv.png is the bot avatar image this needs to be present else comment
            avatar_images=(None, (os.path.join(os.path.dirname(current_dir), "avv.png")))
        )
    text_box = gr.Textbox(
            placeholder="Enter text and press enter",
            container=False,
        )
    
    #btnupd = gr.UploadButton("📁", file_types=["image", "video", "audio"],type="filepath")
    btn = gr.Button("Submit")
    clicked = btn.click(buildHistoryForUiAndChatCompletion,
                        [chatbot,text_box,image_box],
                        chatbot
                        ).then(call_AzureOpenAI_Vision_RAG_API,
                                [chatbot,text_box,image_box],
                                chatbot
                                )
    chatbot.like(print_like_dislike, None, None)
    
app.queue()
app.launch(debug=True)
