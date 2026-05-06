import logging
from typing import Any, Dict, Optional

from preping.appworld.exploration.base import BaseExplorerAgent


logger = logging.getLogger(__name__)


class RandomExplorerAgent(BaseExplorerAgent):
    """Random exploration agent for AppWorld."""

    DEFAULT_TEMPERATURE = 1.0

    def __init__(
        self,
        model_name: str,
        key: str,
        env,
        task_config: Dict[str, Any],
        llm_cache: Optional[Dict] = None,
        debug_mode: bool = False,
        exp_config: Optional[Any] = None,
        max_steps: int = 50,
        temperature: float = None,
        **kwargs,
    ):
        temp = temperature if temperature is not None else self.DEFAULT_TEMPERATURE

        super().__init__(
            model_name=model_name,
            key=key,
            env=env,
            task_config=task_config,
            llm_cache=llm_cache,
            debug_mode=debug_mode,
            exp_config=exp_config,
            max_steps=max_steps,
            temperature=temp,
            **kwargs,
        )

        logger.info("RandomExplorerAgent initialized with temperature=%s, max_steps=%s", temp, max_steps)

    def get_exploration_prompt(self) -> str:
        """Get random exploration focused prompt."""
        return """You are an AI assistant randomly exploring the AppWorld environment.

Your goal is to DISCOVER and TEST various APIs in a random, exploratory manner.
You have access to a Python REPL environment where you can execute code.

EXPLORATION STRATEGY:
1. Try different apps randomly - don't stick to one app too long
2. Test various API endpoints with different parameters
3. Be curious! Try unexpected combinations
4. Observe outputs carefully - note formats, errors, edge cases
5. When you feel you've explored enough, call: apis.supervisor.complete_task()

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.

```python
print(apis.api_docs.show_app_descriptions())
```

# To get the list of apis under any app listed above, e.g. spotify
```python
print(apis.api_docs.show_api_descriptions(app_name='spotify'))
```

# To get the specification of a particular api, e.g. spotify app's login api
```python
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
```
Write Python code to explore. Be random and adventurous.

You are in a fresh exploration session. Try to discover new APIs and functionalities.

Begin by checking what apps are available, then randomly pick one to explore.
Write Python code to interact with the environment.

**Key instructions**:
(1) Make sure to end code blocks with ``` followed by a newline(\n).

(2) Remember you can use the variables in your code in subsequent code blocks.

(3) Remember that the email addresses, access tokens and variables (e.g. spotify_password) in the example above are not valid anymore.

(4) You can use the "supervisor" app to get information about my accounts and use the "phone" app to get information about friends and family.

(5) Always look at API specifications (using apis.api_docs.show_api_doc) before calling an API.

(6) Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.

(7) Many APIs return items in "pages". Make sure to run through all the pages by looping over `page_index`.

(8) Once you have completed the task, make sure to call apis.supervisor.complete_task(). If the task asked for some information, return it as the answer argument, i.e. call apis.supervisor.complete_task(answer=<answer>). Many tasks do not require an answer, so in those cases, just call apis.supervisor.complete_task() i.e. do not pass any argument.
"""
