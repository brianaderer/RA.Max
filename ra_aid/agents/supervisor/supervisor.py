"""Supervisor - The expert-driven orchestrator.

The Supervisor is the ONLY component that:
- Creates and owns the task list
- Delegates work to workers
- Makes all decisions
- Knows when done

This is THE ONLY LOOP in the system.
"""

import json
import logging
import re
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ra_aid.agents.supervisor.models import Task, TaskType, TaskStatus
from ra_aid.agents.supervisor.workers import Researcher, Implementor, Verifier
from ra_aid.console.formatting import cpm, console_panel

logger = logging.getLogger(__name__)


PLANNING_PROMPT = """You are an expert software architect planning a task.

Given a user request, create a MINIMAL plan to accomplish it.

Task types:
- RESEARCH: Gather specific information (always do this FIRST)
- IMPLEMENT: Make specific code changes (after research)
- VERIFY: Run tests or check that changes work (after implementation)

Rules:
1. Research BEFORE implementing - you need context first
2. Be SPECIFIC - "find where X is defined" not "understand the codebase"
3. Be MINIMAL - don't over-plan, fewer tasks is better
4. Order by dependencies - research before implement, implement before verify

Output your plan as a JSON array of tasks:
[
  {"type": "RESEARCH", "description": "Find where X is defined"},
  {"type": "RESEARCH", "description": "Check existing patterns for Y"},
  {"type": "IMPLEMENT", "description": "Add Z to file A"},
  {"type": "VERIFY", "description": "Run tests to verify changes"}
]

ONLY output the JSON array, nothing else."""


class Supervisor:
    """
    The expert-driven orchestrator.

    Uses the best model for planning/decisions.
    Delegates execution to cheaper worker models.
    """

    def __init__(self, expert_model, worker_model):
        """
        Initialize the supervisor.

        Args:
            expert_model: The expert/planning model (e.g., claude-opus, gpt-4)
            worker_model: The worker model (e.g., claude-haiku, gpt-3.5)
        """
        self.expert_model = expert_model
        self.worker_model = worker_model

        # Workers
        self.researcher = Researcher(worker_model)
        self.implementor = Implementor(worker_model)
        self.verifier = Verifier(worker_model)

        # State
        self.tasks: List[Task] = []
        self.context: str = ""  # Accumulated research findings
        self.completed_task_ids: set = set()

    def run(self, user_request: str) -> str:
        """
        Main entry point. THE ONLY LOOP IN THE SYSTEM.

        Args:
            user_request: What the user wants to accomplish

        Returns:
            Summary of what was done
        """
        console_panel(f"User Request: {user_request}", title="Supervisor", border_style="blue bold")

        # === PHASE 1: PLAN ===
        console_panel("Creating plan...", title="Phase 1: Planning", border_style="yellow")
        self.tasks = self._create_plan(user_request)
        self._print_plan()

        if not self.tasks:
            return "Could not create a plan for this request."

        # === PHASE 2: EXECUTE ===
        console_panel("Executing tasks...", title="Phase 2: Execution", border_style="green")

        while True:
            task = self._get_next_task()
            if task is None:
                break  # All done!

            task.mark_in_progress()
            self._print_task_start(task)

            try:
                result = self._execute_task(task)
                task.mark_completed(result)
                self.completed_task_ids.add(task.id)
                self._print_task_complete(task)

            except Exception as e:
                logger.error(f"Task failed: {e}")
                task.mark_failed(str(e))
                self._print_task_failed(task)

                # Ask expert what to do
                if not self._handle_failure(task):
                    break  # Expert says abort

        # === PHASE 3: DONE ===
        console_panel("Generating summary...", title="Phase 3: Complete", border_style="blue")
        return self._create_summary()

    def _create_plan(self, user_request: str) -> List[Task]:
        """Use expert model to create task list."""
        messages = [
            SystemMessage(content=PLANNING_PROMPT),
            HumanMessage(content=f"User request: {user_request}")
        ]

        response = self.expert_model.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)

        return self._parse_plan(content)

    def _parse_plan(self, content: str) -> List[Task]:
        """Parse JSON plan from expert response."""
        try:
            # Extract JSON array from response
            # Handle potential markdown code blocks
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)

            plan_data = json.loads(content)

            tasks = []
            for i, item in enumerate(plan_data):
                task_type = TaskType[item["type"].upper()]
                tasks.append(Task(
                    id=i,
                    type=task_type,
                    description=item["description"],
                ))

            return tasks

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse plan: {e}")
            logger.error(f"Content was: {content}")
            return []

    def _get_next_task(self) -> Optional[Task]:
        """Get next pending task with satisfied dependencies."""
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                if task.is_ready(self.completed_task_ids):
                    return task
        return None

    def _execute_task(self, task: Task) -> str:
        """Delegate task to appropriate worker."""
        if task.type == TaskType.RESEARCH:
            result = self.researcher.execute(task, self.context)
            # Accumulate research findings
            self.context += f"\n\n## Research: {task.description}\n{result}"
            return result

        elif task.type == TaskType.IMPLEMENT:
            return self.implementor.execute(task, self.context)

        elif task.type == TaskType.VERIFY:
            return self.verifier.execute(task, self.context)

        else:
            raise ValueError(f"Unknown task type: {task.type}")

    def _handle_failure(self, task: Task) -> bool:
        """
        Ask expert what to do about failed task.

        Returns:
            True to continue, False to abort
        """
        messages = [
            SystemMessage(content="You are an expert deciding how to handle a failed task."),
            HumanMessage(content=f"""
Task failed: {task.description}
Error: {task.error}

Options:
1. SKIP - Mark as skipped and continue with remaining tasks
2. ABORT - Stop execution entirely

Respond with just SKIP or ABORT.""")
        ]

        response = self.expert_model.invoke(messages)
        content = response.content.upper() if isinstance(response.content, str) else str(response.content).upper()

        if "ABORT" in content:
            return False

        # Default to skip and continue
        task.mark_skipped(f"Skipped due to error: {task.error}")
        self.completed_task_ids.add(task.id)  # Allow dependents to proceed
        return True

    def _create_summary(self) -> str:
        """Create final summary of what was done."""
        completed = [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
        failed = [t for t in self.tasks if t.status == TaskStatus.FAILED]
        skipped = [t for t in self.tasks if t.status == TaskStatus.SKIPPED]

        lines = [
            f"## Summary",
            f"Completed: {len(completed)}/{len(self.tasks)} tasks",
            ""
        ]

        if completed:
            lines.append("### Completed")
            for t in completed:
                lines.append(f"- [{t.type.value}] {t.description}")

        if skipped:
            lines.append("\n### Skipped")
            for t in skipped:
                lines.append(f"- [{t.type.value}] {t.description}")

        if failed:
            lines.append("\n### Failed")
            for t in failed:
                lines.append(f"- [{t.type.value}] {t.description}: {t.error}")

        summary = "\n".join(lines)
        cpm(summary, title="Results", border_style="green bold")
        return summary

    def _print_plan(self) -> None:
        """Print the task plan."""
        if not self.tasks:
            console_panel("No tasks created", border_style="red")
            return

        lines = []
        for task in self.tasks:
            lines.append(f"{task.id + 1}. [{task.type.value.upper()}] {task.description}")

        cpm("\n".join(lines), title=f"Plan ({len(self.tasks)} tasks)", border_style="yellow")

    def _print_task_start(self, task: Task) -> None:
        """Print task starting."""
        console_panel(
            f"[{task.type.value.upper()}] {task.description}",
            title=f"Task {task.id + 1}/{len(self.tasks)}",
            border_style="cyan"
        )

    def _print_task_complete(self, task: Task) -> None:
        """Print task completed."""
        result_preview = task.result[:200] + "..." if len(task.result) > 200 else task.result
        console_panel(result_preview, title=f"Task {task.id + 1} Complete", border_style="green")

    def _print_task_failed(self, task: Task) -> None:
        """Print task failed."""
        console_panel(task.error or "Unknown error", title=f"Task {task.id + 1} Failed", border_style="red")


def run_supervisor(user_request: str, expert_model, worker_model) -> str:
    """
    Entry point for supervisor mode.

    Runs in a loop, maintaining context across requests.

    Args:
        user_request: Initial request (can be empty to prompt user)
        expert_model: Model for planning/decisions
        worker_model: Model for worker execution

    Returns:
        Summary of session
    """
    from ra_aid.tools.human import ask_human

    supervisor = Supervisor(expert_model, worker_model)

    # If no initial request, prompt for one
    if not user_request or not user_request.strip():
        user_request = ask_human.invoke({"question": "What would you like help with?"})

    # Main session loop - keep going until user exits
    while True:
        # Run the current request
        result = supervisor.run(user_request)

        # Ask for next request
        console_panel("Ready for next task. Type 'exit' or 'quit' to end session.", border_style="blue")
        next_request = ask_human.invoke({"question": "What's next?"})

        # Check for exit
        if next_request.lower().strip() in ['exit', 'quit', 'q', 'done', 'bye']:
            console_panel("Session ended. Context preserved.", title="Goodbye", border_style="green")
            break

        # Continue with next request (context is preserved in supervisor)
        user_request = next_request

    return "Session completed."
