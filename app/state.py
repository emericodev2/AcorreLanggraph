from typing import TypedDict, List, Any

class GraphState(TypedDict, total=False):
	messages: List[Any]
	# You can add more fields like 'context' or 'results' later
