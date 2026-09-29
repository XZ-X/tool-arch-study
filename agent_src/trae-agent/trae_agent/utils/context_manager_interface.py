from abc import ABC, abstractmethod
from openai.types.responses import (
    FunctionToolParam,
    ResponseFunctionToolCallParam,
    ResponseInputParam,
    ResponseOutputItem,
)
from openai.types.chat import ChatCompletionMessageParam
from typing import List, TypeAlias, Union

MessageBlockEntryType: TypeAlias = Union[ResponseOutputItem|ResponseInputParam|ChatCompletionMessageParam]

class ContextManager(ABC):

    @abstractmethod
    async def add_message_block(self, messages: List[MessageBlockEntryType]) -> None:
        ...
    
    @abstractmethod
    async def get_all_message_blocks(self) -> List[List[MessageBlockEntryType]]:
        ...

    @abstractmethod
    async def get_current_context(self) -> List[MessageBlockEntryType]:
        ...

