"""Shared MessagePool for inter-agent communication.

Replaces file-based communication with a publish-subscribe message pool
inspired by MetaGPT's Global Message Pool pattern.

Agents publish typed messages and subscribe to message types they
care about. The pool handles routing, deduplication, and querying.

Usage:
    pool = MessagePool()
    pool.subscribe("generator", ["evaluation_report", "reflection_report"])
    pool.publish(msg)
    inbox = pool.get_for("generator")  # returns unread messages
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schemas import AgentMessage


class MessagePool:
    """Shared publish-subscribe message pool.

    - All messages are immutable after publish
    - Agents subscribe to message types, not specific senders
    - Each agent has a read cursor; get_for() returns unread messages
    - Messages persist for the experiment lifetime
    """

    def __init__(self):
        self._messages: list[AgentMessage] = []
        self._subscriptions: dict[str, set[str]] = defaultdict(set)
        self._cursors: dict[str, int] = defaultdict(int)  # agent → index of last read

    def subscribe(self, agent_name: str, message_types: list[str]) -> None:
        """Register an agent's interest in specific message types.

        Args:
            agent_name: Name of the subscribing agent.
            message_types: List of message_type strings to subscribe to.
        """
        for mt in message_types:
            self._subscriptions[agent_name].add(mt)

    def unsubscribe(self, agent_name: str, message_types: list[str]) -> None:
        """Remove an agent's subscription to specific message types."""
        if agent_name in self._subscriptions:
            for mt in message_types:
                self._subscriptions[agent_name].discard(mt)

    def publish(self, message: AgentMessage) -> None:
        """Publish a message to the pool.

        The message is available immediately to all subscribers of its type.
        """
        self._messages.append(message)

    def get_for(self, agent_name: str, mark_read: bool = True) -> list[AgentMessage]:
        """Return unread messages matching the agent's subscriptions.

        Args:
            agent_name: Name of the agent requesting messages.
            mark_read: If True, advance the read cursor (default).
                       Set to False for peek/re-read.

        Returns:
            List of AgentMessage objects, ordered by publish time.
        """
        subscribed_types = self._subscriptions.get(agent_name, set())
        if not subscribed_types:
            return []

        cursor = self._cursors.get(agent_name, 0)
        unread = []

        for i in range(cursor, len(self._messages)):
            msg = self._messages[i]
            if msg.message_type in subscribed_types:
                unread.append(msg)

        if mark_read:
            self._cursors[agent_name] = len(self._messages)

        return unread

    def query(
        self,
        message_type: str | None = None,
        sender: str | None = None,
        round_num: int | None = None,
        parent_id: str | None = None,
        max_results: int = 50,
    ) -> list[AgentMessage]:
        """Query messages by filter criteria.

        Args:
            message_type: Filter by message type.
            sender: Filter by sender name.
            round_num: Filter by round number.
            parent_id: Filter by parent message ID.
            max_results: Maximum number of results to return.

        Returns:
            List of matching AgentMessage objects.
        """
        results = []
        for msg in self._messages:
            if message_type and msg.message_type != message_type:
                continue
            if sender and msg.sender != sender:
                continue
            if round_num is not None and msg.round_num != round_num:
                continue
            if parent_id and msg.parent_id != parent_id:
                continue
            results.append(msg)
            if len(results) >= max_results:
                break
        return results

    def get_round_messages(self, round_num: int) -> list[AgentMessage]:
        """Get all messages for a specific round."""
        return self.query(round_num=round_num)

    def latest_by_type(self, message_type: str) -> AgentMessage | None:
        """Get the most recent message of a given type."""
        for msg in reversed(self._messages):
            if msg.message_type == message_type:
                return msg
        return None

    def reset_cursor(self, agent_name: str) -> None:
        """Reset an agent's read cursor to the beginning."""
        self._cursors[agent_name] = 0

    def clear(self) -> None:
        """Remove all messages and reset all cursors."""
        self._messages.clear()
        self._cursors.clear()

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def all_messages(self) -> list[AgentMessage]:
        return list(self._messages)
