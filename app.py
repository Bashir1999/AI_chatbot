import gradio as gr
from huggingface_hub import InferenceClient

client = InferenceClient("mattshumer/Reflection-Llama-3.1-70B",
    token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

#meta-llama/Meta-Llama-3.1-8B-Instruct
#HuggingFaceH4/zephyr-7b-beta

PERSONALITIES = {
    "Friendly": "You are a friendly and helpful assistant.",
    "Professional": "You are a professional and concise assistant.",
    "Humorous": "You are a witty and humorous assistant.",
    "Empathetic": "You are a compassionate and empathetic assistant."
}

def respond(message, history, personality):
    system_message = PERSONALITIES[personality]
    messages = [{"role": "system", "content": system_message}]
    

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})
    
    messages.append({"role": "user", "content": message})


    response = client.chat_completion(messages, max_tokens=1024)
    bot_message = response["choices"][0]["message"]["content"]
    

    history.append((message, bot_message))
    
    return history, ""


def generate_fun_fact(history):
    message = "Give me a fun fact."
    system_message = "You are a helpful assistant that shares fun facts when asked."
    messages = [{"role": "system", "content": system_message}]
    

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})
    

    messages.append({"role": "user", "content": message})
    

    response = client.chat_completion(messages, max_tokens=256)
    fun_fact = response["choices"][0]["message"]["content"]
    
    history.append((message, fun_fact))
    
    return history

def generate_daily_challenge(history):
    message = "Give me a daily challenge."
    system_message = "You are a helpful assistant that gives fun or motivational daily challenges."
    messages = [{"role": "system", "content": system_message}]

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(messages, max_tokens=256)
    challenge = response["choices"][0]["message"]["content"]

    history.append((message, challenge))

    return history


def generate_inspiration(history):
    message = "Give me an inspirational quote or motivational message."
    system_message = "You are a helpful assistant that provides inspiring or motivational quotes when asked."
    messages = [{"role": "system", "content": system_message}]
    

    for user_message, bot_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": bot_message})
    

    messages.append({"role": "user", "content": message})
    

    response = client.chat_completion(messages, max_tokens=256)
    inspiration = response["choices"][0]["message"]["content"]
    
    history.append((message, inspiration))
    
    return history


def clear_conversation():
    return [], ""


with gr.Blocks(css="""
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    .gradio-container {
        font-family: 'Poppins', sans-serif;
        max-width: 800px;
        margin: 2rem auto;
        padding: 2rem;
        background-color: #f5f5f5;
        border-radius: 20px;
        box-shadow: 0px 4px 30px rgba(0, 0, 0, 0.1);
    }
    .gr-button {
        background-color: #007BFF;
        color: black;
        font-weight: bold;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .gr-button:hover {
        background-color: #0056b3;
    }
    .gr-clear-button {
        background-color: #DC3545;
        color: black;
        font-weight: bold;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .gr-clear-button:hover {
        background-color: #c82333;
    }
    .gr-funfact-button, .gr-inspireme-button {
        background-color: #28A745;
        color: black;
        font-weight: bold;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .gr-funfact-button:hover, .gr-inspireme-button:hover {
        background-color: #218838;
    }
    .gr-textbox, .gr-chatbox {
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 1rem;
        font-size: 1rem;
    }
    .gr-chatbox .user {
        color: #007BFF;
        font-weight: bold;
        margin-bottom: 0.3rem;
    }
    .gr-chatbox .assistant {
        color: #333;
        background-color: #e9f5ff;
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 0.3rem;
    }
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1.5rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
""") as demo:


    gr.Markdown("<div class='title'> 🤖 AI Chatbot 🤖</div>")
    gr.Markdown("<div class='subtitle'>Choose a personality, get inspired, or learn something new!</div>")


    personality = gr.Radio(choices=["Friendly", "Professional", "Humorous", "Empathetic"], 
                           label="Select a Personality", value="Friendly")


    chatbot = gr.Chatbot(label="Chatbot", height=400)


    message = gr.Textbox(placeholder="Type your message here...", label="Your Input")


    history = gr.State(value=[])


    send_btn = gr.Button("Send Message")


    clear_btn = gr.Button("Clear", elem_classes="gr-clear-button")

    fun_fact_btn = gr.Button("Fun Fact", elem_classes="gr-funfact-button")

    inspire_me_btn = gr.Button("Inspire Me", elem_classes="gr-inspireme-button")

    challenge_btn = gr.Button("Daily Challenge", elem_classes="gr-challenge-button")



    send_btn.click(fn=respond, inputs=[message, history, personality], outputs=[chatbot, message])

    clear_btn.click(fn=clear_conversation, outputs=[chatbot, message])

    fun_fact_btn.click(fn=generate_fun_fact, inputs=history, outputs=chatbot)
    challenge_btn.click(fn=generate_daily_challenge, inputs=history, outputs=chatbot)

    inspire_me_btn.click(fn=generate_inspiration, inputs=history, outputs=chatbot)

if __name__ == "__main__":
    demo.launch()
