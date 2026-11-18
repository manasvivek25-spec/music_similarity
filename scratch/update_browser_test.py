from pathlib import Path

p = Path("scratch/test_browser_e2e_interaction.py")
text = p.read_text(encoding="utf-8").replace("\r\r\n", "\n").replace("\r\n", "\n")

target1 = """            # TEST 2: Genuine Audio Upload & Full 7-Feature Execution
            print("\\n--- TEST 2: Genuine Audio File Upload & Live Pipeline Execution ---", flush=True)"""

replacement1 = """            # TEST 2: Genuine Audio Upload & Full 7-Feature Execution
            print("\\n--- TEST 2: Genuine Audio File Upload & Live Pipeline Execution ---", flush=True)
            test2_start_log_idx = len(console_logs)
            test2_start_exc_idx = len(exceptions)"""

target2 = """            error_logs = [log for log in console_logs if log.get("type") == "error"]

            report = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "target_url": TARGET_URL,
                "browser": "Google Chrome Headless (CDP)",
                "test_1_zero_byte": {
                    "status": "PASSED",
                    "file": str(ZERO_BYTE_PATH),
                    "error_state_detected": zero_byte_result
                },
                "test_2_full_pipeline_e2e": {
                    "status": "PASSED",
                    "file": str(VALID_AUDIO_PATH),
                    "progress_stages_recorded": len(progress_history),
                    "progress_history_sample": progress_history,
                    "cards_verification": {
                        "card1_similarity_engine": final_cards_state["card1"],
                        "card2_statistical_gateway": final_cards_state["card2"],
                        "card3_creative_novelty": final_cards_state["card3"],
                        "card4_origin_classifier": final_cards_state["card4"],
                        "card5_musical_context": final_cards_state["card5"],
                        "card6_mimic_identification": final_cards_state["card6"],
                        "card7_suggestor_evidence": final_cards_state["card7"]
                    }
                },
                "browser_health": {
                    "console_log_count": len(console_logs),
                    "console_error_count": len(error_logs),
                    "runtime_exception_count": len(exceptions),
                    "console_errors": error_logs,
                    "runtime_exceptions": exceptions
                },
                "overall_status": "PASSED" if len(error_logs) == 0 and len(exceptions) == 0 else "FAILED"
            }"""

replacement2 = """            test1_error_logs = [log for log in console_logs[:test2_start_log_idx] if log.get("type") == "error"]
            test2_error_logs = [log for log in console_logs[test2_start_log_idx:] if log.get("type") == "error"]
            test2_exceptions = exceptions[test2_start_exc_idx:]

            report = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "target_url": TARGET_URL,
                "browser": "Google Chrome Headless (CDP)",
                "test_1_zero_byte": {
                    "status": "PASSED",
                    "file": str(ZERO_BYTE_PATH),
                    "error_state_detected": zero_byte_result,
                    "expected_error_logged": len(test1_error_logs) > 0
                },
                "test_2_full_pipeline_e2e": {
                    "status": "PASSED",
                    "file": str(VALID_AUDIO_PATH),
                    "progress_stages_recorded": len(progress_history),
                    "progress_history_sample": progress_history,
                    "cards_verification": {
                        "card1_similarity_engine": final_cards_state["card1"],
                        "card2_statistical_gateway": final_cards_state["card2"],
                        "card3_creative_novelty": final_cards_state["card3"],
                        "card4_origin_classifier": final_cards_state["card4"],
                        "card5_musical_context": final_cards_state["card5"],
                        "card6_mimic_identification": final_cards_state["card6"],
                        "card7_suggestor_evidence": final_cards_state["card7"]
                    }
                },
                "browser_health": {
                    "total_console_logs": len(console_logs),
                    "test1_expected_error_count": len(test1_error_logs),
                    "test2_unexpected_error_count": len(test2_error_logs),
                    "test2_runtime_exception_count": len(test2_exceptions),
                    "test2_console_errors": test2_error_logs,
                    "test2_runtime_exceptions": test2_exceptions
                },
                "overall_status": "PASSED" if len(test2_error_logs) == 0 and len(test2_exceptions) == 0 and final_cards_state is not None else "FAILED"
            }"""

assert target1 in text, "Target 1 not found"
assert target2 in text, "Target 2 not found"

text = text.replace(target1, replacement1, 1)
text = text.replace(target2, replacement2, 1)

with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write(text)

print("Successfully updated scratch/test_browser_e2e_interaction.py")

