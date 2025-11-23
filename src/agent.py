from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, room_io
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from barista_agent import BaristaAgent


# Load LIVEKIT_URL / API_KEY / API_SECRET from .env.local
load_dotenv(".env.local")

server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    """
    This is the main voice session for your agent.
    It wires:
    - STT (AssemblyAI)
    - LLM (OpenAI GPT-4.1-mini)
    - TTS (Cartesia Sonic-3)
    - VAD + turn detection
    to your BaristaAgent.
    """
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=BaristaAgent(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # First barista message: greet + ask for order
    await session.generate_reply(
        instructions=(
            "Greet the user as a coffee shop barista at Moonbeam Coffee. "
            "Ask what they'd like to order."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
