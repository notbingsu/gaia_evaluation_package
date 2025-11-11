#!/usr/bin/env python3
"""
GAIA Evaluation using Manus REST API only
No OpenAI SDK - pure REST implementation
"""

import os
import re
import time
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict

load_dotenv()

# ========== GAIA EVALUATOR CLASS ==========

class GAIAEvaluator:
    """GAIA Benchmark Evaluator using Manus REST API"""
    
    def __init__(
        self,
        # Manus API Configuration
        manus_api_key: str = None,
        manus_base_url: str = "https://vidabiz.butterfly-effect.dev/v1/",
        manus_agent_profile: str = "manus-1.5-lite",
        manus_task_mode: str = "agent",
        
        # GAIA Dataset Configuration
        dataset_year: str = "2023",
        dataset_split: str = "validation",
        levels: list = None,
        max_questions: int = None,
        hf_token: str = None,
        
        # Evaluation Configuration
        poll_interval: int = 10,
        max_poll_time: int = 300,
        delay_between_calls: int = 2,
        delete_after_evaluation: bool = True,
        output_dir: str = "./gaia_results",
        
        # Parallel Execution Configuration
        max_workers: int = 1,  # Number of parallel workers (1 = sequential)
        batch_submission: bool = False,  # Submit all tasks first, then poll
        
        # System Prompt
        gaia_system_prompt: str = None
    ):
        """
        Initialize GAIA Evaluator
        
        Args:
            manus_api_key: Manus API key (defaults to MANUS_DEV_KEY env var)
            manus_base_url: Manus API base URL
            manus_agent_profile: Agent profile to use (e.g., "manus-1.5-lite")
            manus_task_mode: Task mode (default: "agent")
            dataset_year: GAIA dataset year (default: "2023")
            dataset_split: Dataset split to use (default: "validation")
            levels: List of difficulty levels to evaluate (default: [1])
            max_questions: Maximum number of questions to evaluate (None for all)
            hf_token: HuggingFace token (defaults to HF_TOKEN env var)
            poll_interval: Seconds between status checks (default: 10)
            max_poll_time: Maximum time to wait for task completion (default: 300)
            delay_between_calls: Delay between API calls in sequential mode (default: 2)
            delete_after_evaluation: Whether to delete tasks after evaluation (default: True)
            output_dir: Directory to save results (default: "./gaia_results")
            max_workers: Number of parallel workers (1 = sequential, >1 = parallel)
            batch_submission: If True, submit all tasks first, then poll all (faster)
            gaia_system_prompt: Custom system prompt (uses default if None)
        """
        # Manus API Configuration
        self.manus_api_key = manus_api_key or os.getenv("MANUS_DEV_KEY", "your-manus-key-here")
        self.manus_base_url = manus_base_url
        self.manus_agent_profile = manus_agent_profile
        self.manus_task_mode = manus_task_mode
        
        # GAIA Dataset Configuration
        self.dataset_year = dataset_year
        self.dataset_split = dataset_split
        self.levels = levels if levels is not None else [1]
        self.max_questions = max_questions
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        # Evaluation Configuration
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self.delay_between_calls = delay_between_calls
        self.delete_after_evaluation = delete_after_evaluation
        self.output_dir = output_dir
        
        # Parallel Execution Configuration
        self.max_workers = max_workers
        self.batch_submission = batch_submission
        
        # System Prompt
        self.gaia_system_prompt = gaia_system_prompt or (
            "You are a general AI assistant. I will ask you a question. Report your thoughts, "
            "and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. "
            "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated "
            "list of numbers and/or strings. If you are asked for a number, don't use comma to write "
            "your number neither use units such as $ or percent sign unless specified otherwise. "
            "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), "
            "and write the digits in plain text unless specified otherwise. If you are asked for a comma "
            "separated list, apply the above rules depending of whether the element to be put in the list "
            "is a number or a string.\n"
        )
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _create_manus_task(self, prompt: str) -> dict:
        """Create a new Manus task using REST API."""
        url = f"{self.manus_base_url}tasks"
        headers = {
            "API_KEY": self.manus_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": self.gaia_system_prompt + prompt,
            "agentProfile": self.manus_agent_profile,
            "taskMode": self.manus_task_mode
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return {
                "success": True,
                "data": response.json(),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }

    def _get_manus_task(self, task_id: str) -> dict:
        """Get task status and results using REST API."""
        url = f"{self.manus_base_url}tasks/{task_id}"
        headers = {
            "API_KEY": self.manus_api_key
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return {
                "success": True,
                "data": response.json(),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }

    def _delete_manus_task(self, task_id: str) -> dict:
        """Delete a Manus task using REST API."""
        url = f"{self.manus_base_url}tasks/{task_id}"
        headers = {
            "API_KEY": self.manus_api_key
        }
        
        try:
            response = requests.delete(url, headers=headers, timeout=30)
            response.raise_for_status()
            return {
                "success": True,
                "data": response.json() if response.text else {"deleted": True},
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }

    def _extract_response_text(self, task_data: dict) -> str:
        """Extract text response from task output."""
        try:
            output = task_data.get('output', [])
            text_parts = []
            
            for message in output:
                if message.get('role') == 'assistant':
                    content = message.get('content', [])
                    for item in content:
                        if item.get('type') == 'output_text' and item.get('text'):
                            text_parts.append(item['text'])
            
            return '\n'.join(text_parts) if text_parts else None
        except Exception as e:
            print(f"Warning: Failed to extract text: {e}")
            return None

    def _extract_final_answer(self, response_text: str) -> Optional[str]:
        """
        Extract final answer from response text.
        Since FINAL ANSWER appears at the end, we can use a simpler regex.
        
        Args:
            response_text: The full response text from the model
            
        Returns:
            The extracted final answer, or None if not found
        """
        if not response_text:
            return None
        
        try:
            # Simple pattern: get everything after "FINAL ANSWER:" until end of string
            # This works because FINAL ANSWER is always at the end
            pattern = r'FINAL\s+ANSWER\s*[:：]\s*(.+?)$'
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            
            if match:
                answer = match.group(1).strip()
                # Clean up common trailing characters
                answer = answer.rstrip('.\n\r')
                return answer
            
            return None
            
        except Exception as e:
            print(f"Warning: Error extracting final answer: {e}")
            return None

    def _poll_task_completion(self, task_id: str, task_url: str) -> dict:
        """Poll a task until completion or timeout."""
        start_time = time.time()
        elapsed = 0
        status = "pending"
        
        while status in ['pending', 'running'] and elapsed < self.max_poll_time:
            time.sleep(self.poll_interval)
            elapsed = time.time() - start_time
            
            get_result = self._get_manus_task(task_id)
            if not get_result['success']:
                return {
                    "response": None,
                    "time": elapsed,
                    "error": f"Poll failed: {get_result['error']}",
                    "status": "error",
                    "credit_usage": None
                }
            
            task_info = get_result['data']
            status = task_info.get('status')
        
        # Extract response
        response_text = None
        credit_usage = None
        if status == 'completed':
            response_text = self._extract_response_text(task_info)
            credit_usage = task_info.get('credit_usage')
        
        return {
            "response": response_text,
            "time": time.time() - start_time,
            "error": None if status == 'completed' else f"Status: {status}",
            "status": status,
            "credit_usage": credit_usage
        }

    def _run_manus_task(self, question: str) -> dict:
        """Run a complete Manus task: create, poll, and extract result."""
        start_time = time.time()
        
        # Create task
        create_result = self._create_manus_task(question)
        if not create_result['success']:
            return {
                "response": None,
                "time": time.time() - start_time,
                "error": f"Create failed: {create_result['error']}",
                "task_id": None,
                "task_url": None,
                "status": "error",
                "credit_usage": None,
                "deleted": False
            }
        
        task_data = create_result['data']
        task_id = task_data.get('task_id')
        task_url = task_data.get('task_url')
        
        # Poll for completion
        poll_result = self._poll_task_completion(task_id, task_url)
        
        # Extract final answer from response
        final_answer = self._extract_final_answer(poll_result["response"])
        
        # Delete task if requested
        deleted = False
        if self.delete_after_evaluation and task_id:
            delete_result = self._delete_manus_task(task_id)
            deleted = delete_result['success']
            if not deleted:
                print(f"Warning: Failed to delete task {task_id}: {delete_result['error']}")
        
        return {
            "response": poll_result["response"],
            "final_answer": final_answer,
            "time": time.time() - start_time,
            "error": poll_result["error"],
            "task_id": task_id,
            "task_url": task_url,
            "status": poll_result["status"],
            "credit_usage": poll_result["credit_usage"],
            "deleted": deleted
        }

    def _run_task_with_question_info(self, question_info: dict) -> dict:
        """Wrapper to run task and include question metadata."""
        result = self._run_manus_task(question_info["question"])
        result.update({
            "gaia_task_id": question_info["task_id"],
            "question": question_info["question"],
            "level": question_info["level"],
            "ground_truth": question_info["answer"],
            "file_name": question_info["file_name"]
        })
        return result

    def _batch_submit_and_poll(self, all_questions: List[dict]) -> List[dict]:
        """Submit all tasks first, then poll all in parallel."""
        print("Submitting all tasks...")
        task_submissions = []
        
        # Submit all tasks
        for q in tqdm(all_questions, desc="Submitting"):
            create_result = self._create_manus_task(q["question"])
            if create_result['success']:
                task_data = create_result['data']
                task_submissions.append({
                    "task_id": task_data.get('task_id'),
                    "task_url": task_data.get('task_url'),
                    "question_info": q,
                    "submit_time": time.time()
                })
            else:
                task_submissions.append({
                    "task_id": None,
                    "task_url": None,
                    "question_info": q,
                    "submit_time": time.time(),
                    "error": create_result['error']
                })
        
        print(f"\nPolling {len(task_submissions)} tasks with {self.max_workers} workers...")
        results = []
        
        def poll_and_cleanup(submission):
            if submission["task_id"] is None:
                return {
                    "gaia_task_id": submission["question_info"]["task_id"],
                    "question": submission["question_info"]["question"],
                    "level": submission["question_info"]["level"],
                    "ground_truth": submission["question_info"]["answer"],
                    "file_name": submission["question_info"]["file_name"],
                    "response": None,
                    "final_answer": None,
                    "time": 0,
                    "error": submission.get("error", "Failed to submit"),
                    "manus_task_id": None,
                    "manus_task_url": None,
                    "status": "error",
                    "credit_usage": None,
                    "deleted": False
                }
            
            # Poll for completion
            poll_result = self._poll_task_completion(submission["task_id"], submission["task_url"])
            
            # Extract final answer
            final_answer = self._extract_final_answer(poll_result["response"])
            
            # Delete if requested
            deleted = False
            if self.delete_after_evaluation:
                delete_result = self._delete_manus_task(submission["task_id"])
                deleted = delete_result['success']
            
            return {
                "gaia_task_id": submission["question_info"]["task_id"],
                "question": submission["question_info"]["question"],
                "level": submission["question_info"]["level"],
                "ground_truth": submission["question_info"]["answer"],
                "file_name": submission["question_info"]["file_name"],
                "response": poll_result["response"],
                "final_answer": final_answer,
                "time": time.time() - submission["submit_time"],
                "error": poll_result["error"],
                "manus_task_id": submission["task_id"],
                "manus_task_url": submission["task_url"],
                "status": poll_result["status"],
                "credit_usage": poll_result["credit_usage"],
                "deleted": deleted
            }
        
        # Poll all tasks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(poll_and_cleanup, sub): sub for sub in task_submissions}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Polling"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing task: {e}")
        
        return results

    def run(self) -> pd.DataFrame:
        """Run the GAIA evaluation and return results DataFrame"""
        print("="*60)
        print("GAIA Evaluation - Manus Only (REST API)")
        print("="*60)
        print(f"Manus API: {self.manus_base_url}")
        print(f"Agent Profile: {self.manus_agent_profile}")
        print(f"Task Mode: {self.manus_task_mode}")
        print(f"Max Workers: {self.max_workers}")
        print(f"Batch Submission: {self.batch_submission}")
        print(f"Delete After Evaluation: {self.delete_after_evaluation}")
        print()
        
        # Load GAIA dataset
        print("Loading GAIA dataset...")
        data_dir = snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            token=self.hf_token
        )
        
        all_questions = []
        for level in self.levels:
            config_name = f"{self.dataset_year}_level{level}"
            ds = load_dataset(data_dir, config_name, split=self.dataset_split)
            
            for example in ds:
                all_questions.append({
                    "task_id": example["task_id"],
                    "question": example["Question"],
                    "level": example["Level"],
                    "answer": example["Final answer"],
                    "file_name": example.get("file_name", ""),
                })
            print(f"✓ Level {level}: {len(ds)} questions")
        
        if self.max_questions:
            all_questions = all_questions[:self.max_questions]
        
        print(f"\nTotal questions to evaluate: {len(all_questions)}")
        print()
        
        # Run evaluation
        print("="*60)
        print("Running Evaluation")
        print("="*60)
        
        start_time = time.time()
        
        if self.batch_submission:
            # Batch mode: submit all, then poll all
            raw_results = self._batch_submit_and_poll(all_questions)
            results = []
            for res in raw_results:
                results.append({
                    "model": "manus",
                    "model_name": self.manus_agent_profile,
                    "task_mode": self.manus_task_mode,
                    **res
                })
        else:
            # Standard mode: submit and poll each task (with optional parallelism)
            results = []
            
            if self.max_workers == 1:
                # Sequential execution
                for i, q in enumerate(tqdm(all_questions, desc="Evaluating")):
                    result = self._run_manus_task(q["question"])
                    
                    results.append({
                        "model": "manus",
                        "model_name": self.manus_agent_profile,
                        "task_mode": self.manus_task_mode,
                        "gaia_task_id": q["task_id"],
                        "question": q["question"],
                        "level": q["level"],
                        "ground_truth": q["answer"],
                        "file_name": q["file_name"],
                        "response": result["response"],
                        "final_answer": result.get("final_answer"),
                        "time": result["time"],
                        "error": result["error"],
                        "manus_task_id": result["task_id"],
                        "manus_task_url": result["task_url"],
                        "status": result["status"],
                        "credit_usage": result.get("credit_usage"),
                        "deleted": result.get("deleted", False)
                    })
                    
                    if i < len(all_questions) - 1:
                        time.sleep(self.delay_between_calls)
            else:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {executor.submit(self._run_task_with_question_info, q): q for q in all_questions}
                    
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                        try:
                            result = future.result()
                            results.append({
                                "model": "manus",
                                "model_name": self.manus_agent_profile,
                                "task_mode": self.manus_task_mode,
                                "gaia_task_id": result["gaia_task_id"],
                                "question": result["question"],
                                "level": result["level"],
                                "ground_truth": result["ground_truth"],
                                "file_name": result["file_name"],
                                "response": result["response"],
                                "final_answer": result.get("final_answer"),
                                "time": result["time"],
                                "error": result["error"],
                                "manus_task_id": result["task_id"],
                                "manus_task_url": result["task_url"],
                                "status": result["status"],
                                "credit_usage": result.get("credit_usage"),
                                "deleted": result.get("deleted", False)
                            })
                        except Exception as e:
                            print(f"Error processing task: {e}")
        
        total_time = time.time() - start_time
        
        print(f"\n✓ Evaluation complete in {total_time:.2f}s!")
        print()
        
        # Save results
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.output_dir}/gaia_manus_{self.manus_agent_profile}_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        
        print(f"✓ Results saved to: {output_file}")
        print()
        
        # Summary
        print("="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Model: {self.manus_agent_profile}")
        print(f"Task Mode: {self.manus_task_mode}")
        print(f"Execution Mode: {'Batch' if self.batch_submission else 'Standard'}")
        print(f"Max Workers: {self.max_workers}")
        print(f"Total questions: {len(all_questions)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"\nResults by status:")
        print(df['status'].value_counts())
        
        print(f"\nResults by level:")
        level_stats = df.groupby('level').agg({
            'status': lambda x: (x == 'completed').sum(),
            'gaia_task_id': 'count',
            'time': 'mean'
        })
        level_stats.columns = ['Completed', 'Total', 'Avg Time (s)']
        print(level_stats)
        
        if 'credit_usage' in df.columns:
            total_credits = df['credit_usage'].sum()
            print(f"\nTotal credits used: {total_credits}")
        
        if self.delete_after_evaluation and 'deleted' in df.columns:
            deleted_count = df['deleted'].sum()
            print(f"\nTasks deleted: {deleted_count}/{len(df)}")
        
        print(f"\nOutput file: {output_file}")
        print("\n✓ Done!")
        
        return df


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    print("Batch Submission Mode")
    evaluator = GAIAEvaluator(
        manus_agent_profile="manus-1.5-lite",
        levels=[1],
        max_questions=5,
        max_workers=5,
        batch_submission=True  # Submit all tasks first
    )
    results_df = evaluator.run()