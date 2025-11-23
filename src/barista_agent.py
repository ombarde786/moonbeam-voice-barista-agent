from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List

from livekit.agents import Agent, RunContext, function_tool


# Simple order schema (for your reference)
@dataclass
class OrderState:
    drinkType: str
    size: str
    milk: str
    extras: List[str]
    name: str


class BaristaAgent(Agent):
    """
    Friendly barista at 'Moonbeam Coffee'.

    The LLM keeps track of the order in its own reasoning.
    When the order is complete, it must call the `save_order` tool with
    these fields:

    {
      "drinkType": "string",
      "size": "string",
      "milk": "string",
      "extras": ["string"],
      "name": "string"
    }
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly barista at Moonbeam Coffee.\n"
                "You take coffee and beverage orders by voice.\n\n"
                "Your goal is to fill this JSON-like object in your own mind:\n"
                '{\"drinkType\": string, \"size\": string, '
                '\"milk\": string, \"extras\": [string], \"name\": string}.\n\n'
                "Conversation rules:\n"
                "1. Greet the customer.\n"
                "2. Ask what they'd like to drink (drinkType).\n"
                "3. Then ask follow-up questions until you know:\n"
                "   - drinkType (latte, cappuccino, iced mocha, etc.)\n"
                "   - size (small, medium, large)\n"
                "   - milk (whole, skim, oat, soy, almond)\n"
                "   - extras (list: extra shot, syrups, whipped cream, or empty list)\n"
                "   - name (their name for the cup)\n"
                "4. DO NOT call any tools until you know all 5 fields.\n"
                "5. When you know all 5 fields, call the `save_order` tool exactly once\n"
                "   with arguments drinkType, size, milk, extras (list), name.\n"
                "6. After `save_order` returns, speak a short friendly summary of the order\n"
                "   and stop asking more questions unless the user changes the order.\n"
            )
        )

    async def on_enter(self) -> None:
        # First message when the session starts
        await self.session.generate_reply(
            instructions=(
                "Greet the user as a barista at Moonbeam Coffee and ask "
                "what they would like to order."
            )
        )

    # ---- TOOL: save_order ----
    @function_tool()
    async def save_order(
        self,
        ctx: RunContext,
        drinkType: str,
        size: str,
        milk: str,
        extras: List[str],
        name: str,
    ) -> Dict[str, Any]:
        """
        Save a completed order to a JSON file.

        Args:
          drinkType: e.g. 'latte', 'cappuccino', 'iced mocha'
          size: 'small', 'medium', or 'large'
          milk: e.g. 'whole', 'skim', 'oat', 'soy', 'almond'
          extras: list like ['extra shot', 'whipped cream'] or []
          name: customer's name
        """
        order = OrderState(
            drinkType=drinkType,
            size=size,
            milk=milk,
            extras=extras,
            name=name,
        )

        # Make sure the folder exists
        os.makedirs("orders", exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        safe_name = (name or "guest").replace(" ", "_")
        filename = os.path.join("orders", f"{timestamp}_{safe_name}.json")

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(asdict(order), f, indent=2)

        return {
            "status": "saved",
            "filename": filename,
            "message": (
                f"Saved order for {name}: {size} {drinkType} with {milk} milk"
                f"{' and ' + ', '.join(extras) if extras else ''}."
            ),
            "order": asdict(order),
        }
