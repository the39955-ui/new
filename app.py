"""
Storyboard -> Blender Prompt Generator (asset-aware, single Gemini call)

Upload a storyboard, list the assets you've ALREADY imported in Blender (Bobo, Zia, Bus...),
and Gemini produces one standalone prompt per segment. Paste each into Claude (Blender MCP).
Claude is told to use your assets as-is and to match the scene's theme to them.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
import os
import re
import time
import uuid

import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="Storyboard → Blender Prompts", page_icon="🎬", layout="wide")

DEFAULT_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

ss = st.session_state
ss.setdefault("api_key", os.environ.get("GEMINI_API_KEY", ""))
ss.setdefault("models", DEFAULT_MODELS)
ss.setdefault("plan", None)
ss.setdefault("assets_text", "")
ss.setdefault("env_text", "")
ss.setdefault("segment_prompts", [])
ss.setdefault("final_prompt", "")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🔑 Gemini API key")
    key_input = st.text_input(
        "API key",
        type="password",
        placeholder="Key saved (hidden). Paste to replace." if ss.api_key else "Paste key, then Save",
    )
    c1, c2 = st.columns(2)
    if c1.button("Save key", use_container_width=True):
        if key_input.strip():
            ss.api_key = key_input.strip()
            st.success("Saved for this session.")
        else:
            st.warning("Enter a key first.")
    if c2.button("Clear key", use_container_width=True):
        ss.api_key = ""
    st.caption("✅ Key kept in session memory only." if ss.api_key else "No key saved yet.")

    st.divider()
    st.header("🧠 Model")
    if st.button("↻ Refresh model list from API", use_container_width=True, disabled=not ss.api_key):
        try:
            client = genai.Client(api_key=ss.api_key)
            found = [
                m.name.replace("models/", "")
                for m in client.models.list()
                if "generateContent" in (getattr(m, "supported_actions", None) or [])
                and "gemini" in m.name
            ]
            if found:
                ss.models = sorted(set(found), reverse=True)
                st.success(f"Loaded {len(found)} models.")
            else:
                st.warning("No models returned.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not fetch models: {e}")
    model = st.selectbox("Gemini multimodal model", ss.models)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.1)

    st.divider()
    st.header("⚙️ Scene overrides")
    fps = st.number_input("Frame rate (fps)", 12, 60, 24)
    render_engine = st.selectbox("Render engine", ["EEVEE", "Cycles"])
    prompt_size = st.radio(
        "Prompt size",
        ["3 small parts per segment (resume-friendly)", "1 prompt per segment"],
        index=0,
        help="Small parts use less of a free account's limit and can be resumed in any chat or account.",
    )
    split_parts = prompt_size.startswith("3")

# --------------------------------------------------------------------------- #
# Prompts for Gemini (ONE call does everything)
# --------------------------------------------------------------------------- #
SYSTEM = """You are a technical director converting an animation storyboard into Blender-ready scene specs.
Another AI (Claude, with a Blender MCP connector) will read your output and write bpy scripts.

STORYBOARD READING
- Read ALL text and visuals: title, panel numbers, timecodes ("0:35 - 0:55" -> seconds), captions, characters,
  props, environment, camera angles, actions.

EXISTING ASSETS (most important)
- The user may list assets ALREADY imported in their Blender scene (names as shown in the Outliner).
- Map every storyboard character / vehicle / prop to the closest listed asset and put its EXACT listed name
  in "existing_asset". If nothing matches, set "existing_asset" to null.
- Existing assets are READ-ONLY. For them give ONLY object-level location and rotation_deg and animation of
  location / rotation (or existing pose bones / shape keys if the user says they exist). NEVER give scale,
  materials, colours, redesign or "appearance" changes for them. Do not describe how they look.
- The scene's look must follow the user's assets, not generic defaults: in "visual_theme_notes" say that the
  environment, props, lighting and world should be built to match the style of the existing assets
  (inspect their materials and palette). Use the storyboard only for layout, action and mood.
- For things with no existing asset, give a short appearance and an import_strategy
  (Sketchfab / Poly Haven / Hyper3D / primitive stand-in).

ENVIRONMENT RULE
- The user message says whether an environment is ALREADY provided in the Blender scene.
- If provided: use it as-is. Return "environment.objects": [] and set "environment.setting" to its name.
  Write "keep as is" for lighting and world. "environment_changes" stays empty unless the action strictly
  needs a small extra prop. Never restyle, rebuild or add a second environment.
- If NOT provided: create a simple environment that fits the scene AND matches the visual theme of the
  existing character / vehicle assets (palette, shading style, level of detail), not a generic look.

SCENE-WISE INPUT
- The input may be scene cards (thumbnail + title + time range + lyrics/dialogue + animation notes) instead of
  one storyboard sheet. Treat each card as one segment, in order. The thumbnail shows the intended layout; the
  text gives timing and action. If a time range is missing, estimate it from the scene order.
- If lyrics or dialogue are given, copy them into "lyrics_or_dialogue" and plan mouth movement (singing or
  talking) and expressions that follow the lines and their rhythm.

FACIAL PERFORMANCE (required for every character that appears on screen)
- Analyse each segment's mood, action, caption and lyrics/dialogue, then give each character an "emotion_arc" and an
  "expressions" timeline that fits the scene (e.g. surprise, laughter, delight, cheeky, sleepy, scared).
- Each expression entry: time_s, frame, emotion, intensity (0 to 1), and short details for eyes, brows
  and mouth. Include eye direction (what the character looks at), blinks, and mouth motion for talking,
  singing, laughing, chewing, licking or sneezing. Use 2 to 6 entries per character per segment.
- Expressions must start a few frames BEFORE the action they react to (anticipation) and hold after it.
- These are performance targets only. Never redesign a character's face. For existing assets, the facial
  animation must use whatever controls the asset already has (shape keys, face bones, custom properties).

SPEC RULES
- Blender: right-handed, Z-up, metres, degrees (Euler XYZ), camera looks down local -Z.
- Coordinates are layout targets; the real asset sizes are unknown to you, so keep relationships
  (distances, facing, framing) rather than absolute precision.
- Define the persistent environment ONCE at top level. Per segment list only "environment_changes".
- Times are seconds from the start of the whole animation, plus frame numbers at the given fps.
- Be concise: no filler, no repeated information, keyframes only where motion happens.
- Return ONLY strict JSON: double-quoted keys, no comments, no trailing commas, no extra keys."""

SCHEMA = """{
  "project_title": str,
  "visual_theme_notes": str,
  "total_duration_s": number,
  "characters": [{"name": str, "kind": "character"|"vehicle"|"prop", "existing_asset": str|null, "appearance": str|null, "import_strategy": str|null}],
  "environment": {"setting": str, "objects": [{"name": str, "type": str, "location": [x,y,z], "rotation_deg": [x,y,z], "scale": [x,y,z]}], "lighting": str, "world_background": str},
  "segments": [{
    "number": int,
    "title": str,
    "time_range": {"start_s": number, "end_s": number, "start_frame": int, "end_frame": int},
    "summary": str,
    "lyrics_or_dialogue": str|null,
    "environment_changes": [str],
    "characters": [{"name": str, "placement": {"location": [x,y,z], "rotation_deg": [x,y,z]}, "pose_note": str, "emotion_arc": str, "expressions": [{"time_s": number, "frame": int, "emotion": str, "intensity": number, "eyes": str, "brows": str, "mouth": str}]}],
    "camera": {"shot_type": str, "location": [x,y,z], "rotation_deg": [x,y,z], "focal_length_mm": number, "dof": {"enabled": bool, "focus_target": str, "f_stop": number}, "movement": str},
    "animation": [{"target": str, "action": str, "keyframes": [{"time_s": number, "frame": int, "property": str, "value": str}], "easing": str}],
    "claude_instructions": str,
    "end_state": str
  }]
}"""

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_parts(files):
    return [
        types.Part.from_bytes(data=f.getvalue(), mime_type=f.type or "image/png")
        for f in files or []
    ]


def parse_json(text):
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))


def call_gemini(client, contents):
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        temperature=temperature,
        response_mime_type="application/json",
    )
    return client.models.generate_content(model=model, contents=contents, config=cfg).text


def generate_plan(client, contents):
    """One call normally. A single retry only if the JSON is broken or the request errors."""
    raw = ""
    for attempt in range(2):
        try:
            raw = call_gemini(client, contents)
            return parse_json(raw), raw
        except json.JSONDecodeError:
            contents = [*contents, "Your previous reply was not valid JSON. Return strict JSON only."]
        except Exception:  # noqa: BLE001
            if attempt == 1:
                raise
            time.sleep(3)
    return None, raw


def run_pipeline(source_contents, asset_text, env_text, asset_files, notes):
    """source_contents = storyboard images OR scene cards, already turned into Gemini content parts."""
    client = genai.Client(api_key=ss.api_key)
    contents = [*source_contents]
    if asset_files:
        contents += ["Screenshots of the assets already imported in my Blender scene:", *to_parts(asset_files)]

    contents.append(
        f"""FPS: {fps} | Render engine: {render_engine}

Assets ALREADY imported in my Blender scene (exact Outliner names):
{asset_text.strip() or 'None. Nothing is imported yet; suggest import strategies.'}

Environment ALREADY in my Blender scene:
{env_text.strip() or 'NONE provided. Create one that fits the scene and matches the theme of the assets above.'}
{('Extra notes: ' + notes.strip()) if notes.strip() else ''}

Return JSON in exactly this shape:
{SCHEMA}"""
    )

    with st.spinner("Gemini is planning every segment (1 call)…"):
        plan, raw = generate_plan(client, contents)

    if plan is None:
        st.error("Gemini returned invalid JSON twice. Try again or switch to a stronger model.")
        st.code(raw)
        return
    ss.plan = plan
    ss.assets_text = asset_text
    ss.env_text = env_text
    build_segment_prompts()


# --------------------------------------------------------------------------- #
# Per-segment prompt for Claude
# --------------------------------------------------------------------------- #
ASSET_RULES = """## My assets already in the Blender scene (READ-ONLY)
These objects are ALREADY imported and finished. Treat their design as final:
- Do NOT import, duplicate, remodel, retexture, recolour, re-material, add or remove modifiers,
  edit meshes / rigs / shape keys, change origins, or change scale or dimensions.
- Only change object-level location and rotation (and use existing pose bones / shape keys / actions if
  they exist). Never apply or bake transforms in a way that alters the mesh.
- Before building, inspect the scene ONCE and COMPACTLY (see Tool output limits): dimensions, lowest point,
  forward axis, and face-related controls of my listed assets only. Use the REAL measured sizes, not guesses. Put each asset's lowest
  point on the ground, face it correctly, and treat the coordinates in the spec as layout targets that you
  adjust to the real sizes while keeping the same distances, facing and framing.
- THEME: anything new you add (props, effects, and an environment ONLY if I did not provide one) must match
  the look of these assets (materials, palette, shading style such as toon vs PBR, outline, level of detail).
  The storyboard / scene cards give layout, action and mood only.
- If a needed asset is not in my scene, use a simple stand-in or the import strategy given, and say so.

My asset list:
{assets}
"""



FACE_RULES = """## Facial expressions and performance (required)
Animate every on-screen character's face so it matches the scene. Use the `expressions` timeline in the spec
(emotion, intensity 0 to 1, eyes, brows, mouth) as the performance plan.
- In your single scene inspection, also list each character's face-related controls only (filtered names, see
  Tool output limits): shape keys, face / jaw / eye / brow bones, custom properties, drivers. Match the spec's emotions to the
  CLOSEST controls that already exist (smile, brow raise, eyes wide, blink, jaw open, mouth shapes, eye look).
- Animate only VALUES with keyframes (shape key values, bone rotations, custom property values). Never create,
  rename, edit or delete shape keys, bones, drivers or materials on my assets.
- Start each expression 3 to 6 frames before the action it reacts to, ease in, hold, then settle back to neutral.
- Add natural life: a blink every 3 to 5 seconds, eyes looking at the thing they react to, small head
  tilts, and mouth motion for talking, singing, laughing, chewing, licking or sneezing (oscillate the jaw or
  mouth shapes if there are no speech shapes).
- If a character has no facial controls at all, animate head and eye bones or object-level head motion only,
  and tell me exactly what was missing.
- In your report, list per character which controls you used for expressions.
"""

ENV_PROVIDED = """My environment is already in the Blender scene: {env}
- Use it as-is. Do NOT add, remove, rebuild, restyle, retexture or relight it, and do not add a second
  environment or extra backdrop objects.
- Allowed: place characters, vehicles and the camera inside it (inspect it once to find ground level, paths
  and usable spots), and add only small temporary props or effects the action strictly needs, in the same style.
- Keep its existing lighting and world. Change them only if the shot is unreadable, and as little as possible."""


OUTPUT_LIMITS = """## Tool output limits (important)
Tool results over 1 MB fail, so keep EVERY result small:
- Never dump the whole scene, mesh data, vertices, node trees, or full bone / shape-key lists.
- Inspect ONLY the objects named above (match by name). Print one compact line per object:
  name, type, dimensions, lowest Z, rotation. Nothing else per object.
- For facial controls, print only names that look face-related (smile, blink, brow, eye, jaw, mouth, cheek,
  tongue, lip, viseme), at most 60 names per character. For materials print names only.
- Make every script print at most about 100 lines or 20 KB. If there is more, summarise it with counts.
- Viewport screenshot: at most one, small (max 800 px wide).
- Do not read keyframes back to verify. Confirm with counts (for example, keyframes inserted per object).
"""


def vec_safe(d, key, default):
    v = d.get(key)
    return v if v is not None else default


ASSET_RULES_SHORT = """## My assets (READ-ONLY)
{assets}
Use them exactly as they are: never remodel, retexture, recolour, re-material, rescale, or edit rigs, shape keys,
modifiers or origins. Only object-level location / rotation and the values of controls they already have.
Use measured real sizes, not guesses. Reuse objects already in the file by name."""

LOG_RULES = """## Progress log (lets any chat or account resume this work)
- Keep a Blender text datablock named BUILD_LOG (create it if missing). Read it first and print at most its last 40 lines.
- If `{code}` is already logged as DONE, tell me and stop. {dep}
- When finished, append ONE short line: `{code} DONE | frames A-B | objects created or reused (names only) | fallbacks`.
- Then save the file with bpy.ops.wm.save_mainfile() if it has a path."""


def fence(obj):
    return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```"


def build_segment_prompts():
    plan = ss.plan or {}
    segs = plan.get("segments", [])
    n = len(segs)
    cmap = {c.get("name"): c for c in plan.get("characters", [])}
    env = plan.get("environment", {})
    env_names = [o.get("name") for o in env.get("objects", []) if o.get("name")]
    assets = ss.assets_text.strip()
    env_given = bool(ss.env_text.strip())
    out = []

    for i, seg in enumerate(segs, 1):
        title = seg.get("title", "Untitled")
        tr = seg.get("time_range", {})
        s_s, e_s = tr.get("start_s"), tr.get("end_s")
        start_f = tr.get("start_frame", int(round((s_s or 0) * fps)))
        end_f = tr.get("end_frame", int(round((e_s or 0) * fps)))
        prev_end = segs[i - 2].get("end_state", "") if i > 1 else ""
        seg_chars = seg.get("characters", [])
        cam = seg.get("camera", {})

        def who():
            L = ["## Who is in this segment"]
            for c in seg_chars:
                info = cmap.get(c.get("name"), {})
                if info.get("existing_asset"):
                    L.append(f"- **{c.get('name')}** → my existing asset `{info['existing_asset']}` (do not modify it)")
                else:
                    how = info.get("import_strategy") or "simple stand-in shape"
                    L.append(f"- **{c.get('name')}** → NOT in my scene. {info.get('appearance') or ''} Build as: {how}")
            return L

        def layout_block():
            L = ["## Scene state"]
            if i == 1:
                L.append(
                    f"- First segment. Use the open Blender file and KEEP my imported assets untouched. "
                    f"Remove only default cube/light/camera if present. FPS {fps}, engine {render_engine}, "
                    f"frame start 0, frame end {end_f}."
                )
            else:
                L.append(f"- Segments 1 to {i - 1} are already built in the open file. Do NOT delete or rebuild them.")
                L.append(f"- Add this segment on frames {start_f} to {end_f} and extend the frame end to {end_f}.")
                if prev_end:
                    L.append(f"- Previous segment ended with: {prev_end}")
            L.append("")
            if assets:
                L.append(ASSET_RULES.format(assets=assets))
            L.append("## Requirements")
            L.append(f"- FPS {fps} | Engine {render_engine} | Units metres, Z-up | Rotations in degrees (Euler XYZ)")
            if plan.get("visual_theme_notes"):
                L.append(f"- Theme: {plan['visual_theme_notes']}")
            L.append("- Name every object clearly.\n")
            if plan.get("project_title"):
                L.append(f"## Project: {plan['project_title']}")
            L += who()
            L.append("\n## Environment")
            if env_given:
                L.append(ENV_PROVIDED.format(env=ss.env_text.strip()))
                for ch in seg.get("environment_changes", []):
                    L.append(f"- Small addition needed: {ch}")
            elif i == 1:
                L.append("No environment was provided. Create this one now, matching my assets' look and the scene:")
                L.append(fence(env))
            else:
                if env_names:
                    L.append("Already built in earlier segments: " + ", ".join(env_names) + ".")
                for ch in seg.get("environment_changes", []):
                    L.append(f"- Change: {ch}")
            L.append("\n## Layout spec (place everything at the start of this segment)")
            if seg.get("summary"):
                L.append(seg["summary"])
            layout = {
                "time_range": tr,
                "characters": [
                    {"name": c.get("name"), "placement": c.get("placement"), "pose_note": c.get("pose_note")}
                    for c in seg_chars
                ],
                "camera_start": {k: cam.get(k) for k in ("shot_type", "location", "rotation_deg", "focal_length_mm", "dof")},
            }
            L.append(fence(layout))
            return L

        def motion_block(short_assets):
            L = []
            if short_assets and assets:
                L.append(ASSET_RULES_SHORT.format(assets=assets))
                L.append("")
            L += who()
            L.append("\n## Motion spec (body, objects, camera)")
            if seg.get("claude_instructions"):
                L.append(f"**Instructions:** {seg['claude_instructions']}")
            L.append(fence({"time_range": tr, "animation": seg.get("animation", []), "camera_movement": cam.get("movement")}))
            if seg.get("end_state"):
                L.append(f"End of segment should look like: {seg['end_state']}")
            L.append("- Use `keyframe_insert` with the exact frames given.")
            return L

        def faces_block(short_assets):
            L = []
            if short_assets and assets:
                L.append(ASSET_RULES_SHORT.format(assets=assets))
                L.append("")
            L += who()
            L.append("")
            L.append(FACE_RULES)
            if seg.get("lyrics_or_dialogue"):
                L.append("**Lyrics / dialogue (sync mouth and expressions to these lines):**\n" + seg["lyrics_or_dialogue"] + "\n")
            L.append("## Expression plan")
            L.append(fence({
                "time_range": tr,
                "characters": [
                    {"name": c.get("name"), "emotion_arc": c.get("emotion_arc"), "expressions": c.get("expressions", [])}
                    for c in seg_chars
                ],
            }))
            return L

        intro = (
            "Use the Blender MCP connector. Keep tool calls minimal: one COMPACT inspection, then ONE bpy "
            "script that does this part, then one quick check. Keep every tool result small (limits below).\n"
        )
        done_layout = "Run the script, then report in 3 lines what was created or reused and any fallbacks. Then stop and wait for the next part."
        done_motion = done_layout
        done_faces = (
            "Run the script, then report in 3 lines which facial controls drove each character's expressions, "
            "and any fallbacks. Then stop and wait for the next part."
        )

        parts = []
        if split_parts:
            total = 3
            prev_seg_dep = (
                f"If the previous segment (S{i - 1}P3) is not logged DONE, tell me and stop." if i > 1 else ""
            )
            specs = [
                ("Layout", 1, layout_block(), done_layout, prev_seg_dep, False),
                ("Motion", 2, motion_block(True), done_motion, f"If S{i}P1 is not logged DONE, tell me and stop.", False),
                ("Faces", 3, faces_block(True), done_faces, f"If S{i}P2 is not logged DONE, tell me and stop.", False),
            ]
            for label, k, blocks, done, dep, _ in specs:
                code = f"S{i}P{k}"
                L = [f"# TASK: Segment {i} of {n} ({title}, {s_s}s to {e_s}s) · Part {k} of {total}: {label}\n", intro]
                L.append(LOG_RULES.format(code=code, dep=dep) + "\n")
                L += blocks
                L.append("\n" + OUTPUT_LIMITS)
                L.append("## Deliverable\n" + done)
                parts.append({"label": label, "text": "\n".join(L)})
        else:
            code = f"S{i}"
            dep = f"If the previous segment (S{i - 1}) is not logged DONE, tell me and stop." if i > 1 else ""
            L = [f"# TASK: Build Segment {i} of {n}: {title} ({s_s}s to {e_s}s)\n", intro]
            L.append(LOG_RULES.format(code=code, dep=dep) + "\n")
            L += layout_block()
            L.append("")
            L += motion_block(False)
            L.append("")
            L += faces_block(False)
            L.append("\n" + OUTPUT_LIMITS)
            L.append("## Deliverable\n" + done_faces)
            parts.append({"label": "Full segment", "text": "\n".join(L)})

        out.append({"title": title, "parts": parts})

    ss.segment_prompts = out
    ss.final_prompt = "\n\n---\n\n".join(pt["text"] for seg in out for pt in seg["parts"])


# --------------------------------------------------------------------------- #
# Shared UI pieces
# --------------------------------------------------------------------------- #
IMG_TYPES = ["png", "jpg", "jpeg", "webp"]


def shared_inputs(prefix):
    asset_text = st.text_area(
        "Assets already imported in my Blender scene",
        placeholder="One per line, exact Outliner name, with what it is. Example:\nBobo (boy character, rigged)\nZia (girl character, rigged)\nBus (school bus vehicle)",
        height=110,
        key=f"{prefix}_assets",
        help="Claude uses these exactly as they are and matches anything new to their theme.",
    )
    env_text = st.text_area(
        "Environment already in my Blender scene (optional)",
        placeholder="Exact Outliner name, e.g. Playground_Env. Leave empty and Claude will build one that matches your assets.",
        height=70,
        key=f"{prefix}_env",
        help="If filled, Claude uses it as-is and adds no environment. If empty, it creates one matching your assets.",
    )
    with st.expander("Optional: asset screenshots and notes"):
        asset_files = st.file_uploader(
            "Screenshots of your imported assets (helps Gemini match the theme)",
            type=IMG_TYPES, accept_multiple_files=True, key=f"{prefix}_assets_img",
        )
        notes = st.text_area("Extra notes", placeholder="Anything else Claude should know.", key=f"{prefix}_notes")
    return asset_text, env_text, asset_files, notes


def render_outputs():
    if ss.plan:
        with st.expander("What Gemini planned (raw JSON)", expanded=False):
            st.json(ss.plan, expanded=1)

    if ss.segment_prompts:
        st.subheader("Copy one part at a time into Claude")
        st.caption(
            "Every part is small and standalone. Progress is saved in a BUILD_LOG inside your .blend file, so if "
            "your account runs out: save the .blend, switch account, open the same file, and paste the next part. "
            "Claude skips anything already done."
        )
        tabs = st.tabs([f"{i}. {sg['title']}" for i, sg in enumerate(ss.segment_prompts, 1)])
        for i, (tab, sg) in enumerate(zip(tabs, ss.segment_prompts), 1):
            with tab:
                total = len(sg["parts"])
                for k, part in enumerate(sg["parts"], 1):
                    st.markdown(f"**Part {k} of {total} · {part['label']}**  ·  ≈ {len(part['text']) // 4:,} tokens")
                    st.code(part["text"], language="markdown", wrap_lines=True)
                    st.download_button(
                        f"⬇️ Download part {k} (.md)",
                        part["text"],
                        file_name=f"segment_{i}_part_{k}_{part['label'].lower().replace(' ', '_')}.md",
                        mime="text/markdown",
                        key=f"dl_{i}_{k}",
                    )
        with st.expander("Everything in one file"):
            st.download_button(
                "⬇️ Download all parts (.md)",
                ss.final_prompt,
                file_name="blender_prompts_all_parts.md",
                mime="text/markdown",
                key="dl_all",
            )


def new_scene():
    return {"id": uuid.uuid4().hex[:8], "title": "", "time": "", "lyrics": "", "animation": ""}


ss.setdefault("scenes", [new_scene()])


# --------------------------------------------------------------------------- #
# Page 1: storyboard image
# --------------------------------------------------------------------------- #
def page_storyboard():
    st.title("🖼️ Storyboard image → Blender prompts")
    st.caption("Upload one storyboard image. Gemini reads it and plans every segment.")
    story_files = st.file_uploader("1 · Storyboard image(s)", type=IMG_TYPES, accept_multiple_files=True, key="sb_files")
    if story_files:
        st.image([f.getvalue() for f in story_files], width=420)
    asset_text, env_text, asset_files, notes = shared_inputs("sb")

    if not ss.api_key:
        st.info("Save your Gemini API key in the sidebar first.")
    if st.button("✨ Generate (1 Gemini call)", type="primary", disabled=not (ss.api_key and story_files), key="sb_go"):
        try:
            run_pipeline(["Storyboard image(s):", *to_parts(story_files)], asset_text, env_text, asset_files, notes)
        except Exception as e:  # noqa: BLE001
            st.error(f"Generation failed: {e}")
    render_outputs()


# --------------------------------------------------------------------------- #
# Page 2: scene-wise input
# --------------------------------------------------------------------------- #
def page_scenes():
    st.title("🎞️ Scene-wise input → Blender prompts")
    st.caption("Add one card per scene: thumbnail, title, time, lyrics or dialogue, and the animation.")

    imgs = {}
    for i, sc in enumerate(ss.scenes, 1):
        with st.container(border=True):
            c_img, c_txt = st.columns([1, 3])
            with c_img:
                f = st.file_uploader("Thumbnail", type=IMG_TYPES, key=f"sf_{sc['id']}", label_visibility="collapsed")
                if f:
                    st.image(f.getvalue(), use_container_width=True)
                imgs[sc["id"]] = f
            with c_txt:
                r1, r2 = st.columns([3, 1])
                sc["title"] = r1.text_input(f"Scene {i} title", sc["title"], key=f"st_{sc['id']}", placeholder="Playground fun")
                sc["time"] = r2.text_input("Time", sc["time"], key=f"sm_{sc['id']}", placeholder="4:00–5:00")
                sc["lyrics"] = st.text_area(
                    "Lyrics / dialogue (optional)", sc["lyrics"], key=f"sl_{sc['id']}", height=100,
                    placeholder="Clap your hands and tap your toes,\nWiggle your fingers, touch your nose!",
                )
                sc["animation"] = st.text_area(
                    "Animation", sc["animation"], key=f"sa_{sc['id']}", height=70,
                    placeholder="The bus parks safely; Bobo and Zia dance at the playground.",
                )
                if len(ss.scenes) > 1 and st.button("Remove scene", key=f"sr_{sc['id']}"):
                    ss.scenes.remove(sc)
                    st.rerun()

    if st.button("＋ Add scene", key="add_scene"):
        ss.scenes.append(new_scene())
        st.rerun()

    with st.expander("Or paste all scenes as text"):
        pasted = st.text_area(
            "Paste scenes in any readable format",
            height=160,
            key="sc_pasted",
            placeholder="Scene 5: Playground fun (4:00–5:00)\nClap your hands and tap your toes...\nAnimation: The bus parks safely; Bobo and Zia dance at the playground.",
        )

    st.divider()
    asset_text, env_text, asset_files, notes = shared_inputs("sc")

    has_input = any(sc["title"] or sc["lyrics"] or sc["animation"] or imgs.get(sc["id"]) for sc in ss.scenes) or pasted.strip()
    if not ss.api_key:
        st.info("Save your Gemini API key in the sidebar first.")
    if st.button("✨ Generate (1 Gemini call)", type="primary", disabled=not (ss.api_key and has_input), key="sc_go"):
        contents = ["Scene cards, in order:"]
        for i, sc in enumerate(ss.scenes, 1):
            if not (sc["title"] or sc["lyrics"] or sc["animation"] or imgs.get(sc["id"])):
                continue
            contents.append(
                f"Scene {i}: {sc['title'] or 'Untitled'} ({sc['time'] or 'time not given'})\n"
                f"Lyrics/dialogue: {sc['lyrics'] or 'none'}\nAnimation: {sc['animation'] or 'not given'}"
            )
            if imgs.get(sc["id"]):
                contents += [f"Thumbnail for scene {i}:", *to_parts([imgs[sc["id"]]])]
        if pasted.strip():
            contents.append("Scenes pasted as text:\n" + pasted.strip())
        try:
            run_pipeline(contents, asset_text, env_text, asset_files, notes)
        except Exception as e:  # noqa: BLE001
            st.error(f"Generation failed: {e}")
    render_outputs()


pg = st.navigation(
    [
        st.Page(page_storyboard, title="Storyboard image", icon="🖼️", default=True),
        st.Page(page_scenes, title="Scene-wise input", icon="🎞️"),
    ]
)
pg.run()
