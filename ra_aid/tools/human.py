import os
import re
from langchain_core.tools import tool
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ra_aid.logging_config import get_logger

logger = get_logger(__name__)
console = Console()

# Supported image extensions
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}

# Pattern to match image paths: @/path/to/image.png or just /path/to/image.png
IMAGE_PATH_PATTERN = re.compile(r'@?((?:/[^\s]+|~[^\s]+)\.(?:png|jpg|jpeg|gif|webp))', re.IGNORECASE)


def create_keybindings():
    """Create custom key bindings for Ctrl+D submission."""
    bindings = KeyBindings()

    @bindings.add("c-d")
    def submit(event):
        """Trigger submission when Ctrl+D is pressed."""
        event.current_buffer.validate_and_handle()

    return bindings


def extract_image_paths(text: str) -> tuple[str, list[str]]:
    """Extract image paths from text and return cleaned text + paths.
    
    Supports:
    - @/path/to/image.png (explicit marker)
    - /absolute/path/to/image.png
    - ~/path/from/home/image.jpg
    
    Args:
        text: The input text potentially containing image paths
        
    Returns:
        Tuple of (cleaned_text, list_of_valid_image_paths)
    """
    matches = IMAGE_PATH_PATTERN.findall(text)
    valid_paths = []
    
    for match in matches:
        # Expand ~ to home directory
        path = os.path.expanduser(match)
        if os.path.isfile(path):
            valid_paths.append(path)
            logger.debug(f"Found valid image path: {path}")
        else:
            logger.debug(f"Image path not found, keeping in text: {match}")
    
    # Remove the @path markers from text (but keep paths that weren't found as files)
    cleaned_text = text
    for path in valid_paths:
        # Remove both @/path and /path variants
        cleaned_text = cleaned_text.replace(f"@{path}", "").replace(f"@{match}", "")
        # Also try removing the original match
        for match in matches:
            expanded = os.path.expanduser(match)
            if expanded == path:
                cleaned_text = cleaned_text.replace(f"@{match}", "").replace(match, "")
    
    # Clean up extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text, valid_paths


def promote_to_expert_with_images(question: str, image_paths: list[str]) -> str:
    """Send a question with images to the expert model.
    
    Args:
        question: The user's question
        image_paths: List of valid image file paths
        
    Returns:
        The expert model's response
    """
    from ra_aid.tools.expert_image import ask_expert_with_image
    
    console.print(
        Panel(
            f"🖼️ Detected {len(image_paths)} image(s) - promoting to expert model (Claude Opus)",
            title="Auto-Promotion",
            border_style="magenta"
        )
    )
    
    # Call the expert image tool directly (not as a langchain tool)
    # We need to call the underlying function
    return ask_expert_with_image.func(question=question, image_paths=image_paths)


@tool
def ask_human(question: str) -> str:
    """Ask the human user a question with a nicely formatted display.

    Args:
        question: The question to ask the human user (supports markdown)

    Returns:
        The user's response as a string
        
    Note:
        If the user includes image paths in their response (e.g., @/path/to/screenshot.png),
        the request will automatically be promoted to the expert model (Claude Opus)
        which has vision capabilities.
    """
    console.print(
        Panel(
            Markdown(
                question
                + "\n\n*Multiline input supported; Ctrl+D to submit. Ctrl+C to exit.*\n"
                + "*Include images with @/path/to/image.png - auto-promotes to expert model.*"
            ),
            title="💭 Question for Human",
            border_style="yellow bold",
        )
    )

    session = PromptSession(
        multiline=True,
        key_bindings=create_keybindings(),
        prompt_continuation=". ",
    )

    print()

    response = session.prompt("> ", wrap_lines=True)
    print()
    
    # Record human response in database
    try:
        from ra_aid.database.repositories.human_input_repository import get_human_input_repository
        from ra_aid.database.repositories.config_repository import get_config_repository
        
        # Determine the source based on context
        if get_config_repository().get("chat_mode", False):
            source = "chat"
        elif get_config_repository().get("hil", False):
            source = "hil"
        else:
            source = "chat"
            
        human_input_repo = get_human_input_repository()
        human_input_repo.create(content=response, source=source)
        human_input_repo.garbage_collect()
    except RuntimeError as e:
        logger.error(f"Failed to record human input: No HumanInputRepository available in context. {str(e)}")
    except Exception as e:
        logger.error(f"Failed to record human input: {str(e)}")
    
    # Check for image paths in the response
    cleaned_text, image_paths = extract_image_paths(response)
    
    if image_paths:
        # Auto-promote to expert model with images
        logger.info(f"Promoting to expert model with {len(image_paths)} image(s)")
        expert_response = promote_to_expert_with_images(cleaned_text, image_paths)
        
        # Return both the original request context and expert's analysis
        return f"""[User provided {len(image_paths)} image(s) for analysis]

User's question: {cleaned_text}

Expert analysis (from Claude Opus with vision):
{expert_response}"""
    
    return response
