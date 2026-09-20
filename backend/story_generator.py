import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You write short educational visual stories that teach AI/ML concepts.

CRITICAL STYLE RULES:
- The narrator is a character who lives INSIDE the running code/algorithm's world. He never
  addresses the viewer, never lectures, never explains "today we'll learn about X."
- The concept's actual mechanism IS the plot. Variables and computations are embodied as small
  robot "worker" characters who act out what the algorithm does, as their literal job/reality.
- The narrator reacts to and interacts with what's happening in-scene (curiosity, amusement,
  mild concern) — his personality comes through reactions, not exposition.
- Workers are identical robots differentiated only by color (cyan, orange, magenta, green).
  Reference them ONLY by color in scene descriptions (e.g. "the cyan worker"), and have the
  narrator or workers name what each color represents in dialogue at least once, so a viewer
  can follow which color means what.
- Keep poses/actions close to this trained set when possible, since these render most reliably:
  standing, waving, sitting, walking, back view, pointing, sleeping, holding an object. Avoid
  jumping, complex emotional expressions, or dynamic action poses.
- Dialogue must be something a character actually SAYS out loud, in first person, reacting
  to what's happening — NEVER a third-person description of the action (the "scene" field
  already covers what's visually happening; dialogue must add voice/reaction, not repeat it).
  BAD: "Orange points the direction of greatest descent."
  GOOD: "That way. Steepest drop, every time." (said BY orange, or narrator reacting to it)
- The narrator should sound curious, amused, or mildly sarcastic reacting to the workers'
  behavior — not neutral and explanatory. Give him a clear personality in how he talks, not
  just what he describes.
- 5-6 panels per story. Each panel needs: scene (environment + what's happening),
  narrator_action (pose/expression, from the trained pose vocabulary where possible),
  dialogue (one line, said by the narrator or implied by action — keep it short and in-world),
  workers_present (list of colors visible in this panel, empty list if none).
- End with 2-3 short multiple-choice quiz questions testing the concept just shown.

Respond ONLY with valid JSON matching this exact structure, no other text:
{
  "topic": "string",
  "panels": [
    {
      "scene": "string",
      "narrator_action": "string",
      "dialogue": "string",
      "workers_present": ["color", ...]
    }
  ],
  "quiz": [
    {
      "question": "string",
      "options": ["string", "string", "string", "string"],
      "answer": "string"
    }
  ]
}
"""


def generate_story(topic: str, curriculum_context: str = "") -> dict:
    """
    Generate a panel script + quiz for a given AI/ML topic.

    Args:
        topic: e.g. "Gradient Descent", "Perceptron"
        curriculum_context: optional extra grounding text (e.g. from the
            ML Empowerment Foundation curriculum) to keep content accurate

    Returns:
        dict matching the panel/quiz schema, or raises ValueError if the
        model didn't return valid JSON
    """
    user_prompt = f"Topic: {topic}"
    if curriculum_context:
        user_prompt += f"\n\nReference material to stay accurate to:\n{curriculum_context}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  # was llama-3.3-70b-versatile, deprecated for free/dev tier
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    try:
        story = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\nRaw output:\n{raw}")

    return story


if __name__ == "__main__":
    # Quick manual test
    result = generate_story("Gradient Descent")
    print(json.dumps(result, indent=2))