import time
import json
import statistics
import argparse
import os
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from agents import StandardLLMAgent, StructuredLLMAgent

# Set environment variable for mocking
os.environ["BENCHMARK_MOCK"] = "true"

class Benchmark:
    def __init__(
        self,
        test_cases: List[Dict[str, Any]],
        model_name: str = "gpt-4",
        num_runs: int = 3,
        output_file: str = "benchmark_results.json"
    ):
        self.test_cases = test_cases
        self.model_name = model_name
        self.num_runs = num_runs
        self.output_file = output_file
        
        # Initialize agents
        system_message = "You are a helpful assistant that thinks step by step."
        self.standard_agent = StandardLLMAgent(model_name=model_name, system_message=system_message)
        self.structured_agent = StructuredLLMAgent(model_name=model_name, system_message=system_message)
    
    def run_single_test(self, agent_type: str, query: str) -> Tuple[str, float]:
        start_time = time.time()
        
        if agent_type == "standard":
            response = self.standard_agent.run(query)
        elif agent_type == "structured":
            response = self.structured_agent.run(query)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        elapsed_time = time.time() - start_time
        return response, elapsed_time
    
    def run_benchmark(self) -> Dict[str, Any]:
        results = {
            "metadata": {
                "model_name": self.model_name,
                "num_runs": self.num_runs,
                "test_cases": len(self.test_cases),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "tests": []
        }
        
        for test_case in self.test_cases:
            test_id = test_case["id"]
            query = test_case["query"]
            expected_answer = test_case.get("expected_answer", None)
            
            print(f"Running test case {test_id}: {query[:50]}...")
            
            test_result = {
                "id": test_id,
                "query": query,
                "expected_answer": expected_answer,
                "agents": {
                    "standard": {"runs": []},
                    "structured": {"runs": []}
                }
            }
            
            # Run each agent multiple times
            for _ in range(self.num_runs):
                # Standard agent
                standard_response, standard_time = self.run_single_test("standard", query)
                test_result["agents"]["standard"]["runs"].append({
                    "response": standard_response,
                    "time": standard_time
                })
                
                # Structured agent
                structured_response, structured_time = self.run_single_test("structured", query)
                test_result["agents"]["structured"]["runs"].append({
                    "response": structured_response,
                    "time": structured_time
                })
            
            # Calculate average times
            test_result["agents"]["standard"]["avg_time"] = statistics.mean(
                [run["time"] for run in test_result["agents"]["standard"]["runs"]]
            )
            test_result["agents"]["structured"]["avg_time"] = statistics.mean(
                [run["time"] for run in test_result["agents"]["structured"]["runs"]]
            )
            
            results["tests"].append(test_result)
        
        # Calculate overall stats
        all_standard_times = [
            run["time"] 
            for test in results["tests"] 
            for run in test["agents"]["standard"]["runs"]
        ]
        
        all_structured_times = [
            run["time"] 
            for test in results["tests"] 
            for run in test["agents"]["structured"]["runs"]
        ]
        
        results["summary"] = {
            "standard_agent": {
                "avg_time": statistics.mean(all_standard_times),
                "min_time": min(all_standard_times),
                "max_time": max(all_standard_times),
                "std_dev": statistics.stdev(all_standard_times) if len(all_standard_times) > 1 else 0
            },
            "structured_agent": {
                "avg_time": statistics.mean(all_structured_times),
                "min_time": min(all_structured_times),
                "max_time": max(all_structured_times),
                "std_dev": statistics.stdev(all_structured_times) if len(all_structured_times) > 1 else 0
            }
        }
        
        # Save results to file
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def visualize_results(self, results: Dict[str, Any]):
        """Create visualizations for benchmark results"""
        
        # Prepare data for plotting
        test_ids = [test["id"] for test in results["tests"]]
        standard_times = [test["agents"]["standard"]["avg_time"] for test in results["tests"]]
        structured_times = [test["agents"]["structured"]["avg_time"] for test in results["tests"]]
        
        # Bar chart comparing average times per test case
        x = np.arange(len(test_ids))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, standard_times, width, label='Standard Agent')
        ax.bar(x + width/2, structured_times, width, label='Structured Agent')
        
        ax.set_ylabel('Average Time (seconds)')
        ax.set_title('Average Response Time by Test Case')
        ax.set_xticks(x)
        ax.set_xticklabels(test_ids)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('response_times_by_test.png')
        
        # Summary bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        
        summary = results["summary"]
        agents = ['Standard Agent', 'Structured Agent']
        avg_times = [summary["standard_agent"]["avg_time"], summary["structured_agent"]["avg_time"]]
        
        ax.bar(agents, avg_times)
        ax.set_ylabel('Average Time (seconds)')
        ax.set_title('Overall Average Response Time by Agent Type')
        
        for i, v in enumerate(avg_times):
            ax.text(i, v + 0.05, f"{v:.2f}s", ha='center')
        
        plt.tight_layout()
        plt.savefig('overall_avg_times.png')
        
        print(f"Visualizations saved to response_times_by_test.png and overall_avg_times.png")

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM agents")
    parser.add_argument("--model", type=str, default="gpt-4", help="Model name to use")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test case")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for results")
    args = parser.parse_args()
    
    # Load test cases
    with open("test_cases.json", "r") as f:
        test_cases = json.load(f)
    
    # Run benchmark
    benchmark = Benchmark(
        test_cases=test_cases,
        model_name=args.model,
        num_runs=args.runs,
        output_file=args.output
    )
    
    results = benchmark.run_benchmark()
    benchmark.visualize_results(results)
    
    print(f"Benchmark completed. Results saved to {args.output}")
    print("\nSummary:")
    print(f"Standard Agent - Avg Time: {results['summary']['standard_agent']['avg_time']:.2f}s")
    print(f"Structured Agent - Avg Time: {results['summary']['structured_agent']['avg_time']:.2f}s")

if __name__ == "__main__":
    main()