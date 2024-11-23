# from transformers import pipeline

# # Initialize the text-generation pipeline
# pipe = pipeline("text-generation", model="medalpaca/medalpaca-13b")

# def lungs_precautions(disease):
#     # Convert the input to string if necessary
#     prom = str(disease)
    
#     # Create the complete prompt with the disease inserted
#     prompt_2 = f"Imagine you are a healthcare professional providing precautionary measures to a patient diagnosed with {prom}. Please list detailed and practical steps the patient should follow to manage their condition effectively and prevent complications. Include lifestyle changes, dietary recommendations, medication guidelines, and any other relevant advice."
    
#     # Generate the output using the pipeline
#     result = pipe(prompt_2, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, temperature=0.7, top_p=0.9)
    
#     # Extract the generated text
#     generated_text = result[0]['generated_text']
    
#     # Remove the prompt text from the generated description
#     description = generated_text[len(prompt_2):].strip()
    
#     # Return only the generated response
#     return description

# # Example usage:
# disease = "chronic obstructive pulmonary disease (COPD)"
# precautions = lungs_precautions(disease)
# print(precautions)


from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Xenova/gpt-3.5-turbo")
model = AutoModelForCausalLM.from_pretrained("Xenova/gpt-3.5-turbo")

def lungs_precautions(disease):
    # Convert the input to string if necessary
    prom = str(disease)
    
    # Create the complete prompt with the disease inserted
    prompt_2 = f"Imagine you are a healthcare professional providing precautionary measures to a patient diagnosed with {prom}. Please list detailed and practical steps the patient should follow to manage their condition effectively and prevent complications. Include lifestyle changes, dietary recommendations, medication guidelines, and any other relevant advice."
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt_2, return_tensors="pt")
    
    # Generate the output with adjusted parameters
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, temperature=0.7, top_p=0.9)
    
    # Decode the output, skipping special tokens
    description = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt text from the generated description
    result = description[len(prompt_2):].strip()
    
    # Return only the generated response
    return result

# Example usage:
disease = "chronic obstructive pulmonary disease (COPD)"
precautions = lungs_precautions(disease)
print(precautions)

