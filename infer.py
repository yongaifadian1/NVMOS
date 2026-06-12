#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from nvmos import NVMOSPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the quality score of a marked NV event in speech.")
    parser.add_argument("--audio", required=True, help="Path to an input audio file.")
    parser.add_argument("--text", required=True, help="Text containing the target NV tag, e.g. 'I am fine. [laugh]'.")
    parser.add_argument("--tag", default=None, help="Optional target tag without brackets. If omitted, inferred from text.")
    parser.add_argument("--model-id", default="maimai11/NVMOS", help="Hugging Face repo containing NVMOS config and scorer weights.")
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave unset for auto.")
    parser.add_argument("--audio-encoder", default=None, help="Override SPEAR model id/path.")
    parser.add_argument("--text-encoder", default=None, help="Override XLM-R model id/path.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download files; use local HF cache only.")
    args = parser.parse_args()

    pipe = NVMOSPipeline(
        model_id=args.model_id,
        device=args.device,
        audio_encoder=args.audio_encoder,
        text_encoder=args.text_encoder,
        local_files_only=args.local_files_only,
    )
    result = pipe.predict(args.audio, args.text, args.tag)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
