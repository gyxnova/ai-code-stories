import gradio as gr
from story_generator import generate_story
from image_generator import generate_panel


def generate_story_only(topic: str):
    """Fast call: just the script + quiz, no image generation.
    Useful if the frontend wants to show the script while images load."""
    story = generate_story(topic)
    return story


def generate_full_story(topic: str):
    """Full pipeline: script + rendered panel images + quiz."""
    story = generate_story(topic)

    images = []
    for i, panel in enumerate(story["panels"]):
        img = generate_panel(panel, seed=1000 + i)
        images.append(img)

    return images, story["quiz"], story


with gr.Blocks(title="AI Code Stories") as demo:
    gr.Markdown("# AI Code Stories\nType an AI/ML topic to generate a visual story.")

    with gr.Row():
        topic_input = gr.Textbox(label="Topic", placeholder="e.g. Gradient Descent")
        generate_btn = gr.Button("Generate Story", variant="primary")

    gallery = gr.Gallery(label="Story Panels", columns=3)
    quiz_output = gr.JSON(label="Quiz")
    script_output = gr.JSON(label="Full Script (debug)")

    generate_btn.click(
        fn=generate_full_story,
        inputs=[topic_input],
        outputs=[gallery, quiz_output, script_output],
        api_name="generate_full_story",
    )

    # Separate lightweight endpoint your frontend can call independently
    story_only_btn = gr.Button("Script Only (debug)", visible=False)
    story_only_btn.click(
        fn=generate_story_only,
        inputs=[topic_input],
        outputs=[script_output],
        api_name="generate_story_only",
    )

demo.launch()