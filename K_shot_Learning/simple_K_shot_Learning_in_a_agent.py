'''
how to implement k-shot learning in an AI agent with a practical example?

Let's create a simple customer service agent that can learn to classify customer inquiries. 
I'll break down how k-shot learning would work in this context

Let me explain how this k-shot learning implementation works:

Learning Phase:

- The agent receives k examples for each category (in our code, k=3)
- For each category, we provide k different examples of customer queries
- The agent stores these examples as reference points

NOTE:

TF-IDF(Term Frequency-Inverse Document Frequency):
Measures how frequently a word appears in a document
- Measures how important a word is to a document in a collection of documents
- Term Frequency (TF): How often a word appears in a document
- Inverse Document Frequency (IDF): How rare the word is across all documents
- Combines both to give importance of a word to a specific document
-Reduces the impact of common words like "the", "is", "and"

COSINE SIMILARITY:
Cosine similarity is a measure of the angle between two vectors
- It ranges from -1 to 1, where 1 means the vectors are identical
- The closer the cosine similarity is to 1, the more similar the queries are
- 1 means vectors point in same direction (very similar)
- 0 means vectors are perpendicular (completely different)
- -1 means vectors point in opposite directions
- cosine_similarity = dot_product(A,B) / (magnitude(A) * magnitude(B))
- cosine_similarity(A,B) = A⋅B / ∥A∥*∥B∥

 


Similarity Calculation:

- When a new query comes in, the agent converts all text (examples and query) to TF-IDF vectors
- It calculates cosine similarity between the new query and each stored example
- This measures how similar the new query is to each known example


Classification:

- The agent averages the similarities for each category
- The category with the highest average similarity is chosen as the prediction
- A confidence score is provided based on the similarity strength

Looking at the example usage:

- We provide 3 examples each for technical, billing, and shipping issues
- The agent can then classify new, unseen queries into these categories
- It learns to recognize patterns from just these few examples

The advantages of this approach:

- Quick adaptation: The agent can learn new categories with just k examples
- Flexibility: Easy to add or update categories without retraining
- Transparency: The similarity-based approach makes decisions interpretable

'''

#K-shot Learning Agent Implementation
# Import required libraries for type hints (List, Dict, Tuple)
from typing import List, Dict, Tuple
# Import numpy for numerical operations
import numpy as np
# Import TfidfVectorizer to convert text to numerical vectors
from sklearn.feature_extraction.text import TfidfVectorizer
# Import cosine_similarity to calculate similarity between vectors
from sklearn.metrics.pairwise import cosine_similarity

# Define the main agent class for k-shot learning
class ImprovedKShotAgent:
    # Initialize the agent with default k=5 examples per category
    def __init__(self, k: int = 5):
        # Store k value
        self.k = k
        # Create empty dictionary to store examples for each category
        self.examples = {}
        # Initialize TF-IDF vectorizer with improved settings
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # Use both single words and pairs of words
            stop_words='english',  # Remove common English words like 'the', 'is', 'at'
            min_df=1,  # Keep terms that appear at least once
            max_df=0.9  # Remove terms that appear in 90% or more documents
        )
    
    # Method to add examples for a category
    def add_examples(self, category: str, examples: List[str]):
        # Check if number of examples matches required k
        if len(examples) != self.k:
            # Raise error if wrong number of examples provided
            raise ValueError(f"Expected {self.k} examples, got {len(examples)}")
        # Store examples in dictionary with category as key
        self.examples[category] = examples
    
    # Method to predict category for a new query
    def predict(self, query: str) -> Tuple[str, float]:
        # Start list with the query text
        all_texts = [query]
        # Initialize empty list to store category labels
        categories = []
        
        # Loop through each category and its examples
        for category, examples in self.examples.items():
            # Add all examples to the text list
            all_texts.extend(examples)
            # Add category label for each example
            categories.extend([category] * len(examples))
        
        # Convert all texts to TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Print debug information about the vectorization
        print("\nVocabulary:", self.vectorizer.get_feature_names_out())
        print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
        
        # Get vector for query (first item in matrix)
        query_vector = tfidf_matrix[0]
        # Get vectors for all examples (rest of matrix)
        example_vectors = tfidf_matrix[1:]
        # Calculate similarity between query and each example
        similarities = cosine_similarity(query_vector, example_vectors)[0]
        
        # Print similarity scores for debugging
        print("\nSimilarities with examples:")
        start_idx = 0
        # Loop through categories to show similarities
        for category, examples in self.examples.items():
            # Get similarities for current category's examples
            category_similarities = similarities[start_idx:start_idx + len(examples)]
            print(f"{category}: {category_similarities}")
            start_idx += len(examples)
        
        # Calculate average similarity score for each category
        category_scores = {}
        start_idx = 0
        for category, examples in self.examples.items():
            # Get similarities for current category
            category_similarities = similarities[start_idx:start_idx + len(examples)]
            # Calculate mean similarity score
            category_scores[category] = np.mean(category_similarities)
            start_idx += len(examples)
        
        # Print category scores for debugging
        print("\nCategory Scores:", category_scores)
        
        # Find category with highest similarity score
        predicted_category = max(category_scores.items(), key=lambda x: x[1])
        # Return predicted category and its confidence score
        return predicted_category[0], predicted_category[1]

# Function to test the improved agent
def test_improved_agent():
    # Create agent instance with k=3 examples per category
    agent = ImprovedKShotAgent(k=3)
    
    # Define technical support examples with specific terminology
    technical_examples = [
        "My computer screen is completely dark and won't turn on",
        "Windows keeps crashing and showing blue screen",
        "Need to reset Windows login password urgently"
    ]
    
    # Define billing related examples
    billing_examples = [
        "Was charged twice for my last payment",
        "When is the next billing cycle payment due",
        "Need to update my credit card billing details"
    ]
    
    # Define shipping related examples
    shipping_examples = [
        "Track my package delivery status",
        "How many days for standard shipping delivery",
        "Need to change my shipping delivery address"
    ]
    
    # Add examples to agent for training
    agent.add_examples("technical", technical_examples)
    agent.add_examples("billing", billing_examples)
    agent.add_examples("shipping", shipping_examples)
    
    # Define test queries to evaluate agent
    test_queries = [
        "My computer screen is black",
        "I need to update my billing information",
        "When will my order arrive"
    ]
    
    # Test each query and print results
    print("Testing Improved Agent:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        category, confidence = agent.predict(query)
        print(f"Predicted Category: {category}")
        print(f"Confidence Score: {confidence:.2f}")

# Run test if script is run directly
if __name__ == "__main__":
    test_improved_agent()