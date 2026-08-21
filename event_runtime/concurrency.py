"""Concurrency primitives: Command, ACK, Lease, FencingToken.

This module implements the concurrency consistency mechanisms required for
safe multi-event handling with version validation, acknowledgment, lease
management, and fencing tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CommandStatus(str, Enum):
    """Status of an assignment command."""
    PROPOSED = "PROPOSED"
    VALIDATED = "VALIDATED"
    COMMITTED = "COMMITTED"
    ACKED = "ACKED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    ACK_TIMEOUT = "ACK_TIMEOUT"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class ACKType(str, Enum):
    """Type of acknowledgment."""
    ACCEPTED = "ACCEPTED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"


@dataclass
class AssignmentCommand:
    """Command to assign a UAV to a region."""
    command_id: str
    uav_id: str
    region_id: str
    graph_version: int
    action_version: int
    fencing_token: int
    created_at: float
    expires_at: float
    status: CommandStatus = CommandStatus.PROPOSED
    ack_received_at: float | None = None
    ack_type: ACKType | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid_at(self, now: float, current_graph_version: int) -> bool:
        """Check if command is still valid at given time and version."""
        if self.status in {CommandStatus.REVOKED, CommandStatus.EXPIRED, CommandStatus.REJECTED}:
            return False
        if now > self.expires_at:
            return False
        if current_graph_version > self.graph_version:
            return False
        return True

    def validate(self, current_graph_version: int) -> bool:
        """Validate command against current graph version."""
        if current_graph_version > self.graph_version:
            self.status = CommandStatus.REJECTED
            return False
        self.status = CommandStatus.VALIDATED
        return True

    def commit(self) -> None:
        """Commit the command."""
        if self.status != CommandStatus.VALIDATED:
            raise ValueError(f"Cannot commit command in status {self.status}")
        self.status = CommandStatus.COMMITTED

    def receive_ack(self, ack_type: ACKType, at: float) -> None:
        """Receive acknowledgment."""
        if self.status not in {CommandStatus.COMMITTED, CommandStatus.ACK_TIMEOUT}:
            raise ValueError(f"Cannot receive ACK for command in status {self.status}")
        self.ack_type = ack_type
        self.ack_received_at = at
        if ack_type == ACKType.REJECTED:
            self.status = CommandStatus.REJECTED
        else:
            self.status = CommandStatus.ACKED

    def revoke(self, new_fencing_token: int, at: float) -> None:
        """Revoke command with higher fencing token."""
        if new_fencing_token <= self.fencing_token:
            raise ValueError("New fencing token must be higher")
        self.status = CommandStatus.REVOKED

    def expire(self, at: float) -> None:
        """Expire command."""
        if self.status in {CommandStatus.COMPLETED, CommandStatus.REVOKED}:
            return
        self.status = CommandStatus.EXPIRED


@dataclass
class ACK:
    """Acknowledgment for a command."""
    command_id: str
    uav_id: str
    ack_type: ACKType
    received_at: float
    fencing_token: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssignmentLease:
    """Lease for a UAV-region assignment."""
    lease_id: str
    uav_id: str
    region_id: str
    fencing_token: int
    granted_at: float
    expires_at: float
    renew_interval: float = 1.5
    status: str = "ACTIVE"
    last_renewed_at: float | None = None

    def is_valid_at(self, now: float) -> bool:
        """Check if lease is valid at given time."""
        if self.status != "ACTIVE":
            return False
        if now > self.expires_at:
            return False
        return True

    def renew(self, now: float) -> bool:
        """Renew the lease."""
        if not self.is_valid_at(now):
            return False
        self.last_renewed_at = now
        self.expires_at = now + self.renew_interval * 3
        return True

    def revoke(self) -> None:
        """Revoke the lease."""
        self.status = "REVOKED"

    def expire(self) -> None:
        """Expire the lease."""
        self.status = "EXPIRED"


@dataclass
class FencingToken:
    """Fencing token for exclusive task access."""
    token: int
    region_id: str
    granted_at: float
    granted_to: str
    reason: str = ""

    def __lt__(self, other: FencingToken) -> bool:
        return self.token < other.token

    def __le__(self, other: FencingToken) -> bool:
        return self.token <= other.token


class ConcurrencyManager:
    """Manages commands, ACKs, leases, and fencing tokens."""

    def __init__(self) -> None:
        self.commands: dict[str, AssignmentCommand] = {}
        self.leases: dict[str, AssignmentLease] = {}
        self.fencing_tokens: dict[str, FencingToken] = {}
        self._next_fencing_token: int = 0

    def next_fencing_token(self) -> int:
        """Generate next fencing token."""
        self._next_fencing_token += 1
        return self._next_fencing_token

    def create_command(
        self,
        command_id: str,
        uav_id: str,
        region_id: str,
        graph_version: int,
        action_version: int,
        ttl: float = 0.5,
        now: float = 0.0,
    ) -> AssignmentCommand:
        """Create a new assignment command."""
        fencing_token = self.next_fencing_token()
        command = AssignmentCommand(
            command_id=command_id,
            uav_id=uav_id,
            region_id=region_id,
            graph_version=graph_version,
            action_version=action_version,
            fencing_token=fencing_token,
            created_at=now,
            expires_at=now + ttl,
        )
        self.commands[command_id] = command
        return command

    def validate_command(self, command_id: str, current_graph_version: int) -> bool:
        """Validate a command against current graph version."""
        command = self.commands.get(command_id)
        if command is None:
            return False
        return command.validate(current_graph_version)

    def commit_command(self, command_id: str) -> None:
        """Commit a command."""
        command = self.commands.get(command_id)
        if command is None:
            raise ValueError(f"Command {command_id} not found")
        command.commit()

    def receive_ack(self, command_id: str, ack: ACK) -> None:
        """Receive acknowledgment for a command."""
        command = self.commands.get(command_id)
        if command is None:
            raise ValueError(f"Command {command_id} not found")
        command.receive_ack(ack.ack_type, ack.received_at)

    def revoke_command(self, command_id: str, new_fencing_token: int, at: float) -> None:
        """Revoke a command with higher fencing token."""
        command = self.commands.get(command_id)
        if command is None:
            raise ValueError(f"Command {command_id} not found")
        command.revoke(new_fencing_token, at)

    def expire_commands(self, at: float) -> list[str]:
        """Expire commands that have passed their TTL."""
        expired = []
        for command_id, command in self.commands.items():
            if command.status not in {CommandStatus.COMPLETED, CommandStatus.REVOKED}:
                if at > command.expires_at:
                    command.expire(at)
                    expired.append(command_id)
        return expired

    def create_lease(
        self,
        lease_id: str,
        uav_id: str,
        region_id: str,
        fencing_token: int,
        now: float,
        ttl: float = 5.0,
    ) -> AssignmentLease:
        """Create a new assignment lease."""
        lease = AssignmentLease(
            lease_id=lease_id,
            uav_id=uav_id,
            region_id=region_id,
            fencing_token=fencing_token,
            granted_at=now,
            expires_at=now + ttl,
        )
        self.leases[lease_id] = lease
        return lease

    def get_valid_lease(self, region_id: str, at: float) -> AssignmentLease | None:
        """Get valid lease for a region at given time."""
        for lease in self.leases.values():
            if lease.region_id == region_id and lease.is_valid_at(at):
                return lease
        return None

    def get_valid_holder_count(self, region_id: str, at: float) -> int:
        """Count valid lease holders for a region."""
        count = 0
        for lease in self.leases.values():
            if lease.region_id == region_id and lease.is_valid_at(at):
                count += 1
        return count

    def revoke_lease(self, lease_id: str) -> None:
        """Revoke a lease."""
        lease = self.leases.get(lease_id)
        if lease is not None:
            lease.revoke()

    def expire_leases(self, at: float) -> list[str]:
        """Expire leases that have passed their TTL."""
        expired = []
        for lease_id, lease in self.leases.items():
            if lease.status == "ACTIVE" and at > lease.expires_at:
                lease.expire()
                expired.append(lease_id)
        return expired

    def grant_fencing_token(
        self,
        region_id: str,
        granted_to: str,
        at: float,
        reason: str = "",
    ) -> FencingToken:
        """Grant a new fencing token for a region."""
        token_value = self.next_fencing_token()
        token = FencingToken(
            token=token_value,
            region_id=region_id,
            granted_at=at,
            granted_to=granted_to,
            reason=reason,
        )
        self.fencing_tokens[region_id] = token
        return token

    def get_current_fencing_token(self, region_id: str) -> FencingToken | None:
        """Get current fencing token for a region."""
        return self.fencing_tokens.get(region_id)

    def reject_stale_action(
        self,
        command: AssignmentCommand,
        current_graph_version: int,
    ) -> bool:
        """Reject stale action if graph version has changed."""
        if current_graph_version > command.graph_version:
            command.status = CommandStatus.REJECTED
            return True
        return False

    def validate_assignment_invariant(
        self,
        region_id: str,
        at: float,
    ) -> bool:
        """Validate that exclusive task has at most one valid holder."""
        return self.get_valid_holder_count(region_id, at) <= 1

    def cleanup_expired(self, at: float) -> dict[str, list[str]]:
        """Cleanup all expired commands and leases."""
        return {
            "expired_commands": self.expire_commands(at),
            "expired_leases": self.expire_leases(at),
        }
