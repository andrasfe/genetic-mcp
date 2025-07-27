"""Worker pool management for parallel LLM processing."""

import asyncio
import time
import uuid
from typing import Any

from .llm_client import MultiModelClient
from .logging_config import log_error, log_operation, log_performance, setup_logging
from .models import Idea, Worker, WorkerStatus

logger = setup_logging(component="worker_pool")


class WorkerPool:
    """Manages a pool of LLM workers for parallel idea generation."""

    def __init__(self, llm_client: MultiModelClient, max_workers: int = 20):
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.workers: dict[str, Worker] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start the worker pool."""
        if self._running:
            return

        log_operation(logger, "START_WORKER_POOL", max_workers=self.max_workers)

        self._running = True
        available_models = self.llm_client.get_available_models()

        # Create workers distributed across available models
        for i in range(self.max_workers):
            model = available_models[i % len(available_models)]
            worker = Worker(
                id=str(uuid.uuid4()),
                model=model,
                status=WorkerStatus.IDLE
            )
            self.workers[worker.id] = worker

            # Start worker task
            task = asyncio.create_task(self._worker_loop(worker))
            self._worker_tasks.append(task)

        logger.info(f"Started worker pool with {self.max_workers} workers across {len(available_models)} models")

    async def stop(self) -> None:
        """Stop the worker pool."""
        self._running = False

        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()

        # Wait for all tasks to complete
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        self.workers.clear()

        logger.info("Stopped worker pool")

    async def submit_task(self, prompt: str, system_prompt: str | None = None,
                         metadata: dict[str, Any] | None = None) -> str:
        """Submit a task to the worker pool."""
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "metadata": metadata or {}
        }

        await self.task_queue.put(task)
        return task_id

    async def get_result(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Get a result from the result queue."""
        try:
            if timeout:
                return await asyncio.wait_for(self.result_queue.get(), timeout)
            else:
                return await self.result_queue.get()
        except asyncio.TimeoutError:
            return None

    async def _worker_loop(self, worker: Worker) -> None:
        """Main loop for a worker."""
        while self._running:
            try:
                # Get task from queue (wait up to 1 second)
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)

                # Update worker status
                worker.status = WorkerStatus.WORKING
                worker.current_task = task["id"]

                logger.debug(f"Worker {worker.id} ({worker.model}) processing task {task['id']}")
                start_time = time.time()

                # Generate idea
                try:
                    content = await self.llm_client.generate(
                        prompt=task["prompt"],
                        client_name=worker.model,
                        system_prompt=task["system_prompt"],
                        temperature=0.8,
                        max_tokens=1500
                    )

                    # Create idea
                    idea = Idea(
                        id=task["id"],
                        content=content,
                        metadata={
                            "worker_id": worker.id,
                            "model": worker.model,
                            **task["metadata"]
                        }
                    )

                    # Put result in queue
                    await self.result_queue.put({
                        "task_id": task["id"],
                        "idea": idea,
                        "worker_id": worker.id,
                        "success": True
                    })

                    worker.completed_tasks += 1

                    log_performance(logger, "WORKER_TASK", time.time() - start_time,
                                   worker_id=worker.id,
                                   model=worker.model,
                                   task_id=task["id"],
                                   status="success")

                except Exception as e:
                    log_error(logger, "WORKER_TASK", e,
                             worker_id=worker.id,
                             model=worker.model,
                             task_id=task["id"])

                    await self.result_queue.put({
                        "task_id": task["id"],
                        "error": str(e),
                        "worker_id": worker.id,
                        "success": False
                    })
                    worker.failed_tasks += 1

                finally:
                    # Update worker status
                    worker.status = WorkerStatus.IDLE
                    worker.current_task = None

            except asyncio.TimeoutError:
                # No task available, continue
                continue
            except asyncio.CancelledError:
                # Worker is being stopped
                break
            except Exception as e:
                logger.error(f"Unexpected error in worker {worker.id}: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing

    def get_active_workers(self) -> list[Worker]:
        """Get list of currently active workers."""
        return [w for w in self.workers.values() if w.status == WorkerStatus.WORKING]

    def get_worker_stats(self) -> dict[str, Any]:
        """Get statistics about the worker pool."""
        active_count = len(self.get_active_workers())
        total_completed = sum(w.completed_tasks for w in self.workers.values())
        total_failed = sum(w.failed_tasks for w in self.workers.values())

        return {
            "total_workers": len(self.workers),
            "active_workers": active_count,
            "idle_workers": len(self.workers) - active_count,
            "total_completed_tasks": total_completed,
            "total_failed_tasks": total_failed,
            "task_queue_size": self.task_queue.qsize(),
            "result_queue_size": self.result_queue.qsize()
        }


class IdeaGenerator:
    """Generates ideas using the worker pool."""

    def __init__(self, worker_pool: WorkerPool, llm_client: MultiModelClient):
        self.worker_pool = worker_pool
        self.llm_client = llm_client

    async def generate_initial_population(self, prompt: str, size: int,
                                        system_prompt: str | None = None) -> list[Idea]:
        """Generate initial population of ideas."""
        start_time = time.time()
        log_operation(logger, "GENERATE_INITIAL_POPULATION",
                      prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                      size=size)

        # Submit all tasks
        task_ids = []
        for i in range(size):
            variation_prompt = self._create_variation_prompt(prompt, i, size)
            task_id = await self.worker_pool.submit_task(
                variation_prompt,
                system_prompt,
                {"generation": 0, "index": i}
            )
            task_ids.append(task_id)

        # Collect results
        ideas = []
        collected = 0
        timeout_count = 0
        max_timeouts = size * 2  # Allow some failures

        while collected < size and timeout_count < max_timeouts:
            result = await self.worker_pool.get_result(timeout=30.0)

            if result and result["success"]:
                ideas.append(result["idea"])
                collected += 1
            else:
                timeout_count += 1
                if result and not result["success"]:
                    logger.warning(f"Task failed: {result.get('error', 'Unknown error')}")

        if len(ideas) < size:
            logger.warning(f"Only generated {len(ideas)} out of {size} requested ideas")

        log_performance(logger, "GENERATE_INITIAL_POPULATION", time.time() - start_time,
                       requested=size,
                       generated=len(ideas),
                       status="completed")

        return ideas

    async def generate_from_parents(self, parent_contents: list[str],
                                   prompt: str, count: int) -> list[Idea]:
        """Generate new ideas based on parent ideas."""
        system_prompt = (
            "You are a creative AI that builds upon and combines existing ideas. "
            "Given parent ideas, create new variations that combine their best elements "
            "while adding novel insights."
        )

        ideas = []
        for i in range(count):
            # Create prompt combining parent ideas
            parent_summary = "\n---\n".join(parent_contents[:2])  # Use up to 2 parents
            combined_prompt = (
                f"Original request: {prompt}\n\n"
                f"Parent ideas to build upon:\n{parent_summary}\n\n"
                f"Create a new idea that combines and improves upon these parent ideas."
            )

            await self.worker_pool.submit_task(
                combined_prompt,
                system_prompt,
                {"generation": "evolved", "index": i}
            )

            result = await self.worker_pool.get_result(timeout=30.0)
            if result and result["success"]:
                ideas.append(result["idea"])

        return ideas

    def _create_variation_prompt(self, base_prompt: str, index: int, total: int) -> str:
        """Create variation of prompt to encourage diversity."""
        variations = [
            f"{base_prompt}\n\nFocus on practical implementation details.",
            f"{base_prompt}\n\nEmphasize innovative and unconventional approaches.",
            f"{base_prompt}\n\nConsider scalability and long-term implications.",
            f"{base_prompt}\n\nPrioritize user experience and accessibility.",
            f"{base_prompt}\n\nExplore technical challenges and solutions.",
            f"{base_prompt}\n\nThink about cost-effectiveness and resource efficiency.",
            f"{base_prompt}\n\nConsider environmental and social impact.",
            f"{base_prompt}\n\nFocus on integration with existing systems.",
            f"{base_prompt}\n\nEmphasize security and privacy considerations.",
            f"{base_prompt}\n\nExplore cutting-edge technologies and methodologies."
        ]

        return variations[index % len(variations)]
