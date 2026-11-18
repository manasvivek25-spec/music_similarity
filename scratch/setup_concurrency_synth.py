from pathlib import Path
import numpy as np
import soundfile as sf

sr = 22050
t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
y = (0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 554.37 * t) + 0.2 * np.sin(2 * np.pi * 659.25 * t)).astype(np.float32)
p_audio = Path("temp_uploads/concurrency_synth_3s.wav")
sf.write(str(p_audio), y, sr)

# Now update test_concurrency_stress.py to use concurrency_synth_3s.wav
p_script = Path("scratch/test_concurrency_stress.py")
text = p_script.read_text(encoding="utf-8")

text = text.replace(
    'VALID_AUDIO_PATH = REPO_ROOT / "temp_uploads/short_clip_7s.wav"',
    'VALID_AUDIO_PATH = REPO_ROOT / "temp_uploads/concurrency_synth_3s.wav"'
)

# Also update result summary checking to handle both top-level and nested schemas
old_summary_sync = '''                "result_summary": {
                    "similarity": data.get("similarity_score"),
                    "p_value": data.get("p_value"),
                    "score_tier": data.get("creative_score", {}).get("score_tier") if data.get("creative_score") else None,
                    "has_verdict": "overall_verdict" in data
                }'''

new_summary_sync = '''                "result_summary": {
                    "similarity": data.get("similarity_score") or (data.get("similarity", {}).get("similarity_score") if isinstance(data.get("similarity"), dict) else None),
                    "p_value": data.get("p_value") or (data.get("statistical_significance", {}).get("p_value") if isinstance(data.get("statistical_significance"), dict) else None),
                    "score_tier": data.get("creative_score", {}).get("score_tier") if isinstance(data.get("creative_score"), dict) else None,
                    "has_verdict": "overall_verdict" in data
                }'''

old_summary_async = '''            "result_summary": {
                "similarity": result_data.get("similarity_score") if result_data else None,
                "p_value": result_data.get("p_value") if result_data else None,
                "has_verdict": "overall_verdict" in result_data if result_data else False
            }'''

new_summary_async = '''            "result_summary": {
                "similarity": (result_data.get("similarity_score") or (result_data.get("similarity", {}).get("similarity_score") if isinstance(result_data.get("similarity"), dict) else None)) if result_data else None,
                "p_value": (result_data.get("p_value") or (result_data.get("statistical_significance", {}).get("p_value") if isinstance(result_data.get("statistical_significance"), dict) else None)) if result_data else None,
                "has_verdict": "overall_verdict" in result_data if result_data else False
            }'''

text = text.replace(old_summary_sync, new_summary_sync)
text = text.replace(old_summary_async, new_summary_async)

p_script.write_text(text, encoding="utf-8")
print("Successfully generated concurrency_synth_3s.wav and updated scratch/test_concurrency_stress.py")

