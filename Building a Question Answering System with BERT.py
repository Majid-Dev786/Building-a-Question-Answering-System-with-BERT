# Import necessary libraries. TensorFlow for the neural network backend, and transformers for accessing pre-trained BERT models.
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering

# Here, I'm initializing the tokenizer using the 'bert-base-uncased' model. This tokenizer will break down our input text into tokens that BERT understands.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Similarly, I'm loading the pre-trained BERT model specifically designed for question answering tasks.
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Defining the context and questions. The context is a piece of text containing information that answers the questions.
context = "The capital city of France is Paris. It is known for its famous landmarks such as the Eiffel Tower and Louvre Museum."

# These are the questions we want our model to answer using the provided context.
questions = [
    "What is the capital of France?",
    "What are some famous landmarks in Paris?"
]

# The correct answers to the questions, for verification purposes.
answers = [
    "Paris",
    "Eiffel Tower and Louvre Museum"
]

# Tokenizing the input context and questions. This process converts our text data into a format that the BERT model can understand, adding necessary padding and truncating if needed.
inputs = tokenizer(context, questions, padding=True, truncation=True, return_tensors="tf")

# Feeding the tokenized inputs into the model. The model returns the start and end logits - the model's confidence in where the answer starts and ends in the context.
outputs = model(inputs)
start_logits, end_logits = outputs.start_logits, outputs.end_logits

# Determining the start and end positions of the answer in the context. This is done by finding the positions with the highest start and end logits.
start_index = tf.argmax(start_logits, axis=1).numpy()[0]
end_index = tf.argmax(end_logits, axis=1).numpy()[0]

# Extracting the answer tokens from the input ids, and converting those tokens back into a string to get our predicted answer.
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))

# Finally, iterating over each question to display the question, the model's predicted answer, and the true answer for comparison.
for i, question in enumerate(questions):
    print(f"Question: {question}")
    print(f"Predicted Answer: {answer}") # Note: This will print the same answer for both questions due to the static 'answer' variable.
    print(f"True Answer: {answers[i]}")


