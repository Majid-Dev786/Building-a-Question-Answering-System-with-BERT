import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Example dataset for testing
context = "The capital city of France is Paris. It is known for its famous landmarks such as the Eiffel Tower and Louvre Museum."
questions = [
    "What is the capital of France?",
    "What are some famous landmarks in Paris?"
]
answers = [
    "Paris",
    "Eiffel Tower and Louvre Museum"
]

# Tokenize the context and questions
inputs = tokenizer(context, questions, padding=True, truncation=True, return_tensors="tf")

# Predict answers for the questions
outputs = model(inputs)
start_logits, end_logits = outputs.start_logits, outputs.end_logits

# Decode the predicted answer
start_index = tf.argmax(start_logits, axis=1).numpy()[0]
end_index = tf.argmax(end_logits, axis=1).numpy()[0]
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))

# Print the predicted answers
for i, question in enumerate(questions):
    print(f"Question: {question}")
    print(f"Predicted Answer: {answer}")
    print(f"True Answer: {answers[i]}")
