"""
DSL Converter for K-shot Learning Examples
This module converts natural language queries to a structured DSL format
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Union

class DSLConverter:
    """
    Converts natural language queries to Domain Specific Language (DSL) commands
    Supports various query types: factual, mathematical, classification, reasoning, etc.
    """
    
    def __init__(self):
        self.operation_patterns = {
            # Factual query patterns
            'factual': r'what is the (capital|population|area|language) of ([a-zA-Z\s]+)',
            
            # Math operation patterns
            'add': r'(\d+)\s*\+\s*(\d+)|add\s+(\d+)\s+and\s+(\d+)|sum\s+of\s+(\d+)\s+and\s+(\d+)',
            'subtract': r'(\d+)\s*\-\s*(\d+)|subtract\s+(\d+)\s+from\s+(\d+)|difference\s+between\s+(\d+)\s+and\s+(\d+)',
            'multiply': r'(\d+)\s*\*\s*(\d+)|multiply\s+(\d+)\s+by\s+(\d+)|product\s+of\s+(\d+)\s+and\s+(\d+)',
            'divide': r'(\d+)\s*\/\s*(\d+)|divide\s+(\d+)\s+by\s+(\d+)|quotient\s+of\s+(\d+)\s+and\s+(\d+)',
            
            # Classification patterns
            'classify': r'classify\s+this\s+(email|text|message|document)',
            
            # Reasoning patterns
            'logical_reasoning': r'is\s+([a-zA-Z\s]+)\s+(taller|shorter|older|younger|faster|slower)\s+than\s+([a-zA-Z\s]+)',
            
            # Creative tasks
            'creative': r'write\s+a\s+(poem|story|slogan|essay)',
            
            # Summarization
            'summarize': r'summarize'
        }
    
    def detect_operation(self, query: str) -> str:
        """
        Detect the operation type from a natural language query
        Args:
            query: Natural language query
        Returns:
            Operation type as string
        """
        query = query.lower()
        
        for operation, pattern in self.operation_patterns.items():
            if re.search(pattern, query):
                return operation
        
        # Default to general query if no specific pattern matches
        return "general_query"
    
    def convert_to_dsl(self, query: str) -> str:
        """
        Convert natural language query to DSL format
        Args:
            query: Natural language query
        Returns:
            DSL representation of the query
        """
        operation = self.detect_operation(query)
        
        # Handle different operation types
        if operation == 'factual':
            match = re.search(self.operation_patterns['factual'], query.lower())
            if match:
                attribute, entity = match.groups()
                return f"get_{attribute}({entity.strip()})"
        
        elif operation == 'add':
            # Try different add patterns
            match = re.search(r'add\s+(\d+)\s+and\s+(\d+)', query.lower())
            if match:
                num1, num2 = match.groups()
                return f"add({num1}, {num2})"
            
            match = re.search(r'sum\s+of\s+(\d+)\s+and\s+(\d+)', query.lower())
            if match:
                num1, num2 = match.groups()
                return f"add({num1}, {num2})"
                
            match = re.search(r'(\d+)\s*\+\s*(\d+)', query.lower())
            if match:
                num1, num2 = match.groups()
                return f"add({num1}, {num2})"
        
        elif operation == 'multiply':
            # Try different multiply patterns
            match = re.search(r'multiply\s+(\d+)\s+by\s+(\d+)', query.lower())
            if match:
                num1, num2 = match.groups()
                return f"multiply({num1}, {num2})"
            
            match = re.search(r'product\s+of\s+(\d+)\s+and\s+(\d+)', query.lower())
            if match:
                num1, num2 = match.groups()
                return f"multiply({num1}, {num2})"
                
            match = re.search(r'(\d+)\s*\*\s*(\d+)', query.lower())
            if match:
                num1, num2 = match.groups()
                return f"multiply({num1}, {num2})"
        
        elif operation == 'classify':
            match = re.search(self.operation_patterns['classify'], query.lower())
            if match:
                content_type = match.group(1)
                content = query.split(f"classify this {content_type}:")[-1].strip()
                return f"classify({content_type}, \"{content}\")"
        
        elif operation == 'summarize':
            # Extract the text to summarize
            text = query.split("summarize the following paragraph:")[-1].strip()
            # Truncate for DSL representation
            short_text = text[:50] + "..." if len(text) > 50 else text
            return f"summarize(\"{short_text}\")"
        
        # Default DSL for general queries
        return f"query(\"{query}\")"
    
    def convert_examples_to_dsl(self, examples_file: str) -> Dict:
        """
        Convert all examples in a JSON file to DSL format
        Args:
            examples_file: Path to JSON file with examples
        Returns:
            Dictionary with original examples and their DSL representations
        """
        with open(examples_file, 'r') as f:
            data = json.load(f)
        
        result = {"test_cases": []}
        
        for test_case in data["test_cases"]:
            dsl_examples = []
            
            # Convert main query
            test_case["dsl"] = self.convert_to_dsl(test_case["query"])
            
            # Convert examples
            for example in test_case.get("examples", []):
                dsl = self.convert_to_dsl(example["query"])
                dsl_examples.append({
                    "query": example["query"],
                    "response": example["response"],
                    "dsl": dsl
                })
            
            test_case["dsl_examples"] = dsl_examples
            result["test_cases"].append(test_case)
        
        return result

def main():
    """Example usage of the DSL converter"""
    converter = DSLConverter()
    
    # Example queries
    queries = [
        "What is the capital of France?",
        "Add 5 and 3",
        "Multiply 6 by 7",
        "Classify this email: CONGRATULATIONS! You've won a FREE iPhone!",
        "Summarize the following paragraph: Photosynthesis is the process by which green plants use sunlight to synthesize foods."
    ]
    
    print("DSL Conversion Examples:")
    print("-" * 50)
    for query in queries:
        dsl = converter.convert_to_dsl(query)
        print(f"Query: {query}")
        print(f"DSL: {dsl}")
        print("-" * 50)
    
    # Convert examples from file
    try:
        result = converter.convert_examples_to_dsl("A79/NL to DSL/Agents/llm_handler/K-shot Examples for Agent Testing")
        print("\nConverted all examples to DSL format.")
        
        # Print first example with DSL
        first_case = result["test_cases"][0]
        print(f"\nExample conversion for: {first_case['id']}")
        print(f"Query: {first_case['query']}")
        print(f"DSL: {first_case['dsl']}")
        
        if first_case.get("dsl_examples"):
            print("\nFirst example with DSL:")
            example = first_case["dsl_examples"][0]
            print(f"Query: {example['query']}")
            print(f"DSL: {example['dsl']}")
    except Exception as e:
        print(f"Error processing examples file: {e}")

if __name__ == "__main__":
    main() 