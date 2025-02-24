from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class ChatPromptingAgent:
    def __init__(self, k: int = 5):
        """
        Initialize the chat prompting agent
        Args:
            k (int): Number of examples per operation type
        """
        self.k = k
        self.examples = {}
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.9
        )
        
        # Define patterns for extracting DSL from chat responses
        self.dsl_patterns = {
            'add': r'add\(\d+,\s*\d+\)',
            'subtract': r'subtract\(\d+,\s*\d+\)',
            'multiply': r'multiply\(\d+,\s*\d+\)',
            'divide': r'divide\(\d+,\s*\d+\)'
        }
    
    def add_examples(self, operation: str, examples: List[Tuple[str, str, str]]):
        """
        Add examples for an operation type
        Args:
            operation (str): Operation name (add, subtract, etc.)
            examples (List[Tuple]): List of (natural language, chat response, DSL) triples
        """
        if len(examples) != self.k:
            raise ValueError(f"Expected {self.k} examples, got {len(examples)}")
        if operation not in self.dsl_patterns:
            raise ValueError(f"Invalid operation: {operation}")
        self.examples[operation] = examples
    
    def extract_dsl(self, operation: str, chat_response: str) -> str:
        """Extract DSL from chat response"""
        pattern = self.dsl_patterns.get(operation)
        match = re.search(pattern, chat_response)
        return match.group(0) if match else ""
    
    def generate_chat_response(self, operation: str, numbers: List[str]) -> str:
        """Generate a chat-style response"""
        templates = {
            'add': "Let me help you add those numbers. The operation would be: add({}, {})",
            'multiply': "I'll multiply those numbers for you. Here's the operation: multiply({}, {})"
        }
        template = templates.get(operation, "Here's the operation: {}({}, {})")
        return template.format(numbers[0], numbers[1])
    
    def predict(self, query: str) -> Tuple[str, str, str, float]:
        """
        Predict DSL for natural language query
        Returns:
            Tuple[str, str, str, float]: (operation, chat response, DSL output, confidence)
        """
        # Extract all natural language examples
        all_texts = [query]
        operations = []
        chat_responses = []
        dsl_examples = []
        
        for operation, examples in self.examples.items():
            for nl, chat, dsl in examples:
                all_texts.append(nl)
                operations.append(operation)
                chat_responses.append(chat)
                dsl_examples.append(dsl)
        
        # Convert to TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Calculate similarities
        query_vector = tfidf_matrix[0]
        example_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(query_vector, example_vectors)[0]
        
        # Find most similar example
        best_idx = np.argmax(similarities)
        operation = operations[best_idx]
        
        # Extract numbers from query
        numbers = re.findall(r'\d+', query)
        if len(numbers) < 2:
            numbers = ['0', '0']  # Default values
            
        # Generate chat response
        chat_response = self.generate_chat_response(operation, numbers)
        
        # Extract DSL from chat response
        dsl_output = self.extract_dsl(operation, chat_response)
            
        return operation, chat_response, dsl_output, similarities[best_idx]

def test_chat_agent():
    """Test the chat prompting agent"""
    agent = ChatPromptingAgent(k=2)
    
    # Add examples for each operation
    add_examples = [
        ("add 5 and 3", 
         "Let me help you add those numbers. The operation would be: add(5, 3)", 
         "add(5, 3)"),
        ("what is 2 plus 4", 
         "I'll help you add 2 and 4. Here's the operation: add(2, 4)", 
         "add(2, 4)")
    ]
    
    multiply_examples = [
        ("multiply 6 and 7", 
         "I'll multiply those numbers for you. Here's the operation: multiply(6, 7)", 
         "multiply(6, 7)"),
        ("what is 3 times 9", 
         "Let me multiply 3 and 9. The operation is: multiply(3, 9)", 
         "multiply(3, 9)")
    ]
    
    agent.add_examples("add", add_examples)
    agent.add_examples("multiply", multiply_examples)
    
    # Test queries
    test_queries = [
        "add 10 and 20",
        "what is 8 times 4",
        "multiply 15 and 3"
    ]
    
    print("Testing Chat Prompting Agent:")
    for query in test_queries:
        operation, chat, dsl, confidence = agent.predict(query)
        print(f"\nQuery: {query}")
        print(f"Operation: {operation}")
        print(f"Chat Response: {chat}")
        print(f"DSL Output: {dsl}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    test_chat_agent()