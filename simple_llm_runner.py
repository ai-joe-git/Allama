import sys
from simple_llm_tool.model import GGUFModel
from simple_llm_tool.prompt_engineering import few_shot_prompt

def main():
    if len(sys.argv) < 3:
        print("Usage: python simple_llm_runner.py <model_path> <mode>")
        sys.exit(1)

    model_path = sys.argv[1]
    mode = sys.argv[2]

    print(f"Loading model from {model_path}...")
    model = GGUFModel(model_path, quantization="int8")
    print("Model loaded successfully.")

    if mode == "generate":
        generate_mode(model)
    elif mode == "chat":
        chat_mode(model)
    else:
        print("Invalid mode. Use 'generate' or 'chat'.")
        sys.exit(1)

def generate_mode(model):
    print("Enter your prompt (or 'quit' to exit):")
    while True:
        prompt = input("> ")
        if prompt.lower() == 'quit':
            break
        
        print("Generating...")
        for token in model.generate_stream(prompt, max_tokens=100):
            print(token, end='', flush=True)
        print("\n")

def chat_mode(model):
    print("Chat mode. Enter your messages (or 'quit' to exit):")
    context = ""
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        context += f"\nHuman: {user_input}\nAssistant:"
        print("Assistant:", end=' ', flush=True)
        
        full_response = ""
        for token in model.generate_stream(context, max_tokens=200):
            print(token, end='', flush=True)
            full_response += token
        
        print("\n")
        context += full_response

if __name__ == "__main__":
    main()
