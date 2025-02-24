from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class StructuredDecodingAgent:
    def __init__(self, k: int = 5):
        """
        Initialize the structured decoding agent
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
        
        # Define valid DSL operations and their patterns
        self.valid_operations = {
            'add': r'add\(\d+,\s*\d+\)',
            'subtract': r'subtract\(\d+,\s*\d+\)',
            'multiply': r'multiply\(\d+,\s*\d+\)',
            'divide': r'divide\(\d+,\s*\d+\)'
        }
    
    def add_examples(self, operation: str, examples: List[Tuple[str, str]]):
        """
        Add examples for an operation type
        Args:
            operation (str): Operation name (add, subtract, etc.)
            examples (List[Tuple]): List of (natural language, DSL) pairs
        """
        if len(examples) != self.k:
            raise ValueError(f"Expected {self.k} examples, got {len(examples)}")
        if operation not in self.valid_operations:
            raise ValueError(f"Invalid operation: {operation}")
        self.examples[operation] = examples
    
    def validate_dsl(self, operation: str, dsl: str) -> bool:
        """Check if DSL matches expected pattern"""
        pattern = self.valid_operations.get(operation)
        return bool(re.match(pattern, dsl))
    
    def predict(self, query: str) -> Tuple[str, str, float]:
        """
        Predict DSL for natural language query
        Returns:
            Tuple[str, str, float]: (operation, DSL output, confidence)
        """
        # Extract all natural language examples
        all_texts = [query]
        operations = []
        nl_examples = []
        dsl_examples = []
        
        for operation, examples in self.examples.items():
            for nl, dsl in examples:
                all_texts.append(nl)
                operations.append(operation)
                nl_examples.append(nl)
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
        example_dsl = dsl_examples[best_idx]
        
        # Extract numbers from query
        numbers = re.findall(r'\d+', query)
        if len(numbers) >= 2:
            # Replace numbers in example DSL with numbers from query
            dsl_output = re.sub(r'\d+', lambda x: numbers.pop(0), example_dsl, count=2)
        else:
            dsl_output = example_dsl
            
        return operation, dsl_output, similarities[best_idx]

def test_structured_agent():
    """Test the structured decoding agent"""
    agent = StructuredDecodingAgent(k=2)
    
    # Add examples for each operation
    add_examples = [
        ("add 5 and 3", "add(5, 3)"),
        ("what is 2 plus 4", "add(2, 4)")
    ]
    
    multiply_examples = [
        ("multiply 6 and 7", "multiply(6, 7)"),
        ("what is 3 times 9", "multiply(3, 9)")
    ]
    
    agent.add_examples("add", add_examples)
    agent.add_examples("multiply", multiply_examples)
    
    # Test queries
    test_queries = [
        "add 10 and 20",
        "what is 8 times 4",
        "multiply 15 and 3"
    ]
    
    print("Testing Structured Decoding Agent:")
    for query in test_queries:
        operation, dsl, confidence = agent.predict(query)
        print(f"\nQuery: {query}")
        print(f"Operation: {operation}")
        print(f"DSL Output: {dsl}")
        print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    test_structured_agent()