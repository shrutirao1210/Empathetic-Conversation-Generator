import torch
import os
from config import ROBERTA_MODEL_NAME, BART_MODEL_NAME, NUM_EMOTIONS, MAX_LEN, DEVICE, LABEL_TO_EMOTION, CONFIDENCE_THRESHOLD
from dataset import get_tokenizers
from model import DecoupledEmpatheticModel

def load_model(checkpoint_path):
    print("Loading decoupled architecture and weights...")
    rob_tokenizer, bart_tokenizer = get_tokenizers()
    
    model = DecoupledEmpatheticModel(ROBERTA_MODEL_NAME, BART_MODEL_NAME, NUM_EMOTIONS)
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=DEVICE))
        print("Decoupled model weights loaded successfully.")
    else:
        print(f"Watch out: Checkpoint not found at {checkpoint_path}! Using random weights.")
        
    model.to(DEVICE)
    model.eval() 
    return model, rob_tokenizer, bart_tokenizer

def chat(model, rob_tokenizer, bart_tokenizer):
    print("\n--- Decoupled Empathetic Chatbot ---")
    print("Type 'quit' or 'exit' to stop.")
    # Hold conversational buffer memory (Bug 6 Fix)
    conversation_history = []
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            conversation_history.append(f"User: {user_input}")
            full_context = " </s> ".join(conversation_history[-6:])
                
            rob_inputs = rob_tokenizer(user_input, max_length=MAX_LEN, truncation=True, return_tensors='pt').to(DEVICE)
            bart_inputs = bart_tokenizer(full_context, max_length=MAX_LEN, truncation=True, return_tensors='pt').to(DEVICE)
            
            # Use generate wrapper to pass inputs into our model replacing the normal logic
            generated_ids, predicted_class, max_prob = model.generate_response(
                rob_input_ids=rob_inputs['input_ids'],
                rob_attention_mask=rob_inputs['attention_mask'],
                bart_input_ids=bart_inputs['input_ids'],
                bart_attention_mask=bart_inputs['attention_mask'],
                max_length=128,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            emotion_idx = predicted_class.item()
            prob = max_prob.item()
            is_fallback_active = prob < CONFIDENCE_THRESHOLD
            
            detected_emotion_str = LABEL_TO_EMOTION[emotion_idx]
            
            if is_fallback_active:
                print(f"*[Confidence low ({prob:.2f}): Fell back to NEUTRAL emotion routing]*")
            else:
                print(f"*[Perceived User Emotion: {detected_emotion_str} (Conf: {prob:.2f})]*")
            
            response = bart_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Bot: {response.strip()}")
            
            conversation_history.append(f"Bot: {response.strip()}")
            # Keep list bounded — the [-6:] slice handles context window,
            # but we also trim the list itself to avoid unbounded memory growth
            if len(conversation_history) > 12:
                conversation_history = conversation_history[-12:]
                
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    check_path = "checkpoints/best_decoupled_model.pt"
    model, rob_tok, bart_tok = load_model(check_path)
    chat(model, rob_tok, bart_tok)
