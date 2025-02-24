from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import sys
import os
import time

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Agents.Structured_Decoding_Agent import StructuredDecodingAgent
from Agents.Chat_Prompting_Agent import ChatPromptingAgent

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

def compare_agents():
    """Compare Structured Decoding vs Chat Prompting approaches"""
    
    # Initialize both agents
    structured_agent = StructuredDecodingAgent(k=2)
    chat_agent = ChatPromptingAgent(k=2)
    
    # Add same examples to both agents
    add_examples_structured = [
        ("add 5 and 3", "add(5, 3)"),
        ("what is 2 plus 4", "add(2, 4)")
    ]
    
    multiply_examples_structured = [
        ("multiply 6 and 7", "multiply(6, 7)"),
        ("what is 3 times 9", "multiply(3, 9)")
    ]
    
    add_examples_chat = [
        ("add 5 and 3", 
         "Let me help you add those numbers. The operation would be: add(5, 3)", 
         "add(5, 3)"),
        ("what is 2 plus 4", 
         "I'll help you add 2 and 4. Here's the operation: add(2, 4)", 
         "add(2, 4)")
    ]
    
    multiply_examples_chat = [
        ("multiply 6 and 7", 
         "I'll multiply those numbers for you. Here's the operation: multiply(6, 7)", 
         "multiply(6, 7)"),
        ("what is 3 times 9", 
         "Let me multiply 3 and 9. The operation is: multiply(3, 9)", 
         "multiply(3, 9)")
    ]
    
    # Add examples to agents
    structured_agent.add_examples("add", add_examples_structured)
    structured_agent.add_examples("multiply", multiply_examples_structured)
    chat_agent.add_examples("add", add_examples_chat)
    chat_agent.add_examples("multiply", multiply_examples_chat)
    
    # Test queries with varying complexity
    test_queries = [
        "add 10 and 20",
        "what is 15 plus 25",
        "can you add 30 and 40",
        "multiply 8 and 12",
        "what is the product of 5 and 7",
        "calculate 15 times 3"
    ]
    
    # Performance metrics
    metrics = {
        'structured': {'correct': 0, 'confidence': [], 'time': []},
        'chat': {'correct': 0, 'confidence': [], 'time': []}
    }
    
    print("\n=== Comparing Structured Decoding vs Chat Prompting ===\n")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        # Test Structured Agent
        start_time = time.time()
        op_struct, dsl_struct, conf_struct = structured_agent.predict(query)
        struct_time = time.time() - start_time
        
        # Test Chat Agent
        start_time = time.time()
        op_chat, chat_resp, dsl_chat, conf_chat = chat_agent.predict(query)
        chat_time = time.time() - start_time
        
        # Update metrics
        metrics['structured']['time'].append(struct_time)
        metrics['chat']['time'].append(chat_time)
        metrics['structured']['confidence'].append(conf_struct)
        metrics['chat']['confidence'].append(conf_chat)
        
        # Check correctness (assuming same DSL format is correct)
        if dsl_struct == dsl_chat:
            metrics['structured']['correct'] += 1
            metrics['chat']['correct'] += 1
        
        # Print results for this query
        print("\nStructured Decoding:")
        print(f"DSL Output: {dsl_struct}")
        print(f"Confidence: {conf_struct:.2f}")
        print(f"Time: {struct_time:.3f}s")
        
        print("\nChat Prompting:")
        print(f"Chat Response: {chat_resp}")
        print(f"DSL Output: {dsl_chat}")
        print(f"Confidence: {conf_chat:.2f}")
        print(f"Time: {chat_time:.3f}s")
    
    # Print summary with more detailed metrics
    print("\n=== Performance Summary ===")
    for agent_type in ['structured', 'chat']:
        print(f"\n{agent_type.capitalize()} Agent:")
        print(f"Accuracy: {metrics[agent_type]['correct']/len(test_queries)*100:.1f}%")
        print(f"Avg Confidence: {np.mean(metrics[agent_type]['confidence']):.2f}")
        print(f"Avg Response Time: {np.mean(metrics[agent_type]['time']):.3f}s")
        print(f"Max Response Time: {max(metrics[agent_type]['time']):.3f}s")
        print(f"Min Response Time: {min(metrics[agent_type]['time']):.3f}s")

if __name__ == "__main__":
    compare_agents()