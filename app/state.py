from typing import TypedDict, List, Any, Optional

class GraphState(TypedDict, total=False):
	messages: List[Any]
	# You can add more fields like 'context' or 'results' later
