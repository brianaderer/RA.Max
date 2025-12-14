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

# Patterns to match image paths:
# 1. Quoted paths (handles spaces): @"/path/to/my image.png" or @'/path/to/image.png'
# 2. Unquoted paths (no spaces): @/path/to/image.png
IMAGE_PATH_PATTERNS = [
    # Double-quoted paths: @"/path with spaces/image.png"
    re.compile(r'@"([^"]+\.(?:png|jpg|jpeg|gif|webp))"', re.IGNORECASE),
    # Single-quoted paths: @'/path with spaces/image.png'
    re.compile(r"@'([^']+\.(?:png|jpg|jpeg|gif|webp))'", re.IGNORECASE),
    # Unquoted paths (no spaces): @/path/to/image.png
    re.compile(r'@((?:/[^\s]+|~/[^\s]+)\.(?:png|jpg|jpeg|gif|webp))', re.IGNORECASE),
]


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
    - @"/path/with spaces/image.png" (quoted for spaces)
    - @'/path/with spaces/image.png' (single quotes)
    - @/path/to/image.png (no spaces)
    - ~/path/from/home/image.jpg
    
    Args:
        text: The input text potentially containing image paths
        
    Returns:
        Tuple of (cleaned_text, list_of_valid_image_paths)
    """
    valid_paths = []
    cleaned_text = text
    
    # Track what we've already matched to avoid duplicates
    matched_paths = set()
    
    for pattern in IMAGE_PATH_PATTERNS:
        for match in pattern.finditer(text):
            path = match.group(1)
            full_match = match.group(0)
            
            # Expand ~ to home directory
            expanded_path = os.path.expanduser(path)
            
            if expanded_path in matched_paths:
                continue
                
            if os.path.isfile(expanded_path):
                valid_paths.append(expanded_path)
                matched_paths.add(expanded_path)
                # Remove the full match from text
                cleaned_text = cleaned_text.replace(full_match, "")
                logger.debug(f"Found valid image path: {expanded_path}")
            else:
                logger.debug(f"Image path not found: {expanded_path}")
    
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
            f"🖼️ Detected {len(image_paths)} image(s) - promoting to expert model (Claude Opus)\n"
            + "\n".join(f"  • {p}" for p in image_paths),
            title="Auto-Promotion",
            border_style="magenta"
        )
    )
    
    # Call the expert image tool directly (not as a langchain tool)
    return ask_expert_with_image.func(question=question, image_paths=image_paths)


@tool
def ask_human(question: str) -> str:
    """Ask the human user a question with a nicely formatted display.

    Args:
        question: The question to ask the human user (supports markdown)

    Returns:
        The user's response as a string
        
    Note:
        If the user includes image paths in their response, the request will 
        automatically be promoted to the expert model (Claude Opus) which has 
        vision capabilities.
        
        For paths with spaces, use quotes: @"/path/with spaces/image.png"
    """
    console.print(
        Panel(
            Markdown(
                question
                + "\n\n*Multiline input supported; Ctrl+D to submit. Ctrl+C to exit.*\n"
                + '*Include images: @"/path/to/image.png" (use quotes for spaces)*'
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
